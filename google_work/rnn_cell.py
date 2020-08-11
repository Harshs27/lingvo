# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RNN cells (e.g., LSTM, GRU) that the Lingvo model uses."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import summary_utils
from six.moves import range
from six.moves import zip
# from lingvo.core import bn_layers
from google3.third_party.tensorflow.python.util import deprecation as tf_deprecation  # pylint: disable=g-direct-tensorflow-import


def _HistogramSummary(name, v):
  """Adds a histogram summary for 'v' into the default tf graph."""
  summary_utils.histogram(name, tf.cast(v, tf.float32))


RNN_CELL_WT = 'rnn_cell_weight_variable'


class RNNCell(quant_utils.QuantizableLayer):
  # pylint: disable=line-too-long
  """RNN cells.

  RNNCell represents recurrent state in a `.NestedMap`.

  `zero_state(theta, batch_size)` returns the initial state, which is defined
  by each subclass. From the state, each subclass defines `GetOutput()`
  to extract the output tensor.

  `RNNCell.FProp` defines the forward function::

      (theta, state0, inputs) -> state1, extras

  All arguments and return values are `.NestedMap`. Each subclass defines
  what fields these `.NestedMap` are expected to have. `extras` is a
  `.NestedMap` containing some intermediate results `FProp` computes to
  facilitate the backprop.

  `zero_state(theta, batch_size)`, `state0` and `state1` are all compatible
  `.NestedMap` (see `.NestedMap.IsCompatible`).
  I.e., they have the same keys recursively. Furthermore, the corresponding
  tensors in these `.NestedMap` have the same shape and dtype.
  """
  # pylint: enable=line-too-long

  @classmethod
  def Params(cls):
    p = super(RNNCell, cls).Params()
    p.Define('inputs_arity', 1,
             'number of tensors expected for the inputs.act to FProp.')
    p.Define('num_input_nodes', 0, 'Number of input nodes.')
    p.Define(
        'num_output_nodes', 0,
        'Number of output nodes. If num_hidden_nodes is 0, also used as '
        'cell size.')
    p.Define(
        'reset_cell_state', False,
        ('Set True to support resetting cell state in scenarios where multiple '
         'inputs are packed into a single training example. The RNN layer '
         'should provide reset_mask inputs in addition to act and padding if '
         'this flag is set.'))
    p.Define(
        'zero_state_init_params', py_utils.DefaultRNNCellStateInit(),
        'Parameters that define how the initial state values are set '
        'for each cell. Must be one of the static functions defined in '
        'py_utils.RNNCellStateInit.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes RnnCell."""
    super(RNNCell, self).__init__(params)
    assert not self.params.vn.per_step_vn, (
        'We do not support per step VN in RNN cells.')

  def _VariableCollections(self):
    return [RNN_CELL_WT, '%s_vars' % (self.__class__.__name__)]

  def zero_state(self, theta, batch_size):
    """Returns the initial state given the batch size."""
    raise NotImplementedError('Abstract method')

  def GetOutput(self, state):
    """Returns the output value given the current state."""
    raise NotImplementedError('Abstract method')

  def batch_size(self, inputs):
    """Given the inputs, returns the batch size."""
    raise NotImplementedError('Abstract method')

  def FProp(self, theta, state0, inputs):
    """Forward function.

    The default implementation here assumes the cell forward
    function is composed of two functions::

        _Gates(_Mix(theta, state0, inputs), theta, state0, inputs)

    The result of `_Mix` is stashed in `extras` to facilitate backprop.

    `_ResetState` is optionally applied if `reset_cell_state` is True. The RNN
    layer should provide `reset_mask` inputs in addition to other inputs.
    `reset_mask` inputs are expected to be 0 at timesteps where state0 should be
    reset to default (zeros) before running `_Mix()` and `_Gates()`, and 1
    otherwise. This is meant to support use cases like packed inputs, where
    multiple samples are fed in a single input example sequence, and need to be
    masked from each other. For example, if the two examples packed together
    are ['good', 'day'] -> ['guten-tag'] and ['thanks'] -> ['danke']
    to produce ['good', 'day', 'thanks'] -> ['guten-tag', 'danke'], the
    source reset_masks would be [1, 1, 0] and target reset masks would be
    [1, 0]. These ids are meant to enable masking computations for
    different examples from each other.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      state0: The previous recurrent state. A `.NestedMap`.
      inputs: The inputs to the cell. A `.NestedMap`.

    Returns:
      A tuple (state1, extras).
      - state1: The next recurrent state. A `.NestedMap`.
      - extras: Intermediate results to faciliate backprop. A `.NestedMap`.
    """
    assert isinstance(inputs.act, list)
    assert self.params.inputs_arity == len(inputs.act)
    if self.params.reset_cell_state:
      state0_modified = self._ResetState(state0.DeepCopy(), inputs)
    else:
      state0_modified = state0
    xmw = self._Mix(theta, state0_modified, inputs)
    state1 = self._Gates(xmw, theta, state0_modified, inputs)
    return state1, py_utils.NestedMap()

  def _ZoneOut(self,
               prev_v,
               cur_v,
               padding_v,
               zo_prob,
               is_eval,
               random_uniform,
               qt=None,
               qdomain=''):
    """Apply ZoneOut regularlization to cur_v.

    Implements ZoneOut regularization as described in
    https://arxiv.org/abs/1606.01305

    Args:
      prev_v: A tensor, values from the previous timestep.
      cur_v: A tensor, values from the current timestep.
      padding_v: A tensor, the paddings vector for the cur timestep.
      zo_prob: A float, probability at which to apply ZoneOut regularization.
      is_eval: A bool, whether or not in eval mode.
      random_uniform: a tensor of random uniform numbers. This can be None if
        zo_prob=0.0
      qt: A string, name of the qtensor for zone out math.
      qdomain: A string, name of the qdomain for quantized zone out math.

    Returns:
      cur_v after ZoneOut regularization has been applied.
    """
    prev_v = tf.convert_to_tensor(prev_v)
    cur_v = tf.convert_to_tensor(cur_v)
    padding_v = tf.convert_to_tensor(padding_v)
    if zo_prob == 0.0:
      # Special case for when ZoneOut is not enabled.
      return py_utils.ApplyPadding(padding_v, cur_v, prev_v)

    if is_eval:
      # We take expectation in the eval mode.
      #
      fns = self.fns
      # This quantized mixed operation should probably occur as fused kernel to
      # avoid quantized-math rounding errors. Current accuracy has not been
      # verified.
      prev_weight = self.QWeight(zo_prob, domain=qdomain)
      new_weight = self.QWeight(1.0 - prev_weight, domain=qdomain)
      if qt is None:
        mix_prev = tf.multiply(tf.fill(tf.shape(prev_v), prev_weight), prev_v)
        mix_curr = tf.multiply(tf.fill(tf.shape(cur_v), new_weight), cur_v)
        mix = tf.add(mix_prev, mix_curr)
      else:
        mix_prev = fns.qmultiply(
            self.QWeight(
                tf.fill(tf.shape(prev_v), prev_weight), domain=qdomain),
            prev_v,
            qt=qt)
        mix_curr = fns.qmultiply(
            self.QWeight(tf.fill(tf.shape(cur_v), new_weight), domain=qdomain),
            cur_v,
            qt=qt)
        mix = fns.qadd(mix_prev, mix_curr, qt=qt)

      # If padding_v is 1, it always carries over the previous state.
      return py_utils.ApplyPadding(padding_v, mix, prev_v)
    else:
      assert random_uniform is not None
      random_uniform = py_utils.HasShape(random_uniform, tf.shape(prev_v))
      zo_p = tf.cast(random_uniform < zo_prob, padding_v.dtype)
      zo_p += padding_v
      # If padding_v is 1, we always carry over the previous state.
      zo_p = tf.minimum(zo_p, 1.0)
      zo_p = tf.stop_gradient(zo_p)
      return py_utils.ApplyPadding(zo_p, cur_v, prev_v)


class LSTMCellSimple(RNNCell):
  """Simple LSTM cell.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(LSTMCellSimple, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    p.Define('forget_gate_bias', 0.0, 'Bias to apply to the forget gate.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('enable_lstm_bias', True, 'Enable the LSTM Cell bias.')
    p.Define(
        'couple_input_forget_gates', False,
        'Whether to couple the input and forget gates. Just like '
        'tf.contrib.rnn.CoupledInputForgetGateLSTMCell')
    p.Define('apply_pruning', False, 'Whether to prune the weights while '
             'training')
    p.Define('apply_pruning_to_projection', False,
             'Whether to prune the projection matrix while '
             'training')
    p.Define('gradient_pruning', False, 'Whether to gradient prune the model')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Initialization parameters for bias')
    p.Define('init_distribution', 'uniform', 'Distribution for initialization.')
    p.Define('input_scale', 2.0, 'Scale of the input matrix.')
    p.Define('spectral_radius', 0.9, 'Spectral radius of this layer.')
    p.Define('input_sparsity', 0.3, 'Input sparsity of this layer.')
    p.Define('hidden_sparsity', 0.3, 'Hidden sparsity of this layer.')
    p.Define('trainable', False, 'Trainable params or not.')
    p.Define('train_scale', False, 'Train input scale or not')
    p.Define('train_radius', False, 'Train spectral radius or not')
    p.Define('init_scale', 500, 'dummy params not used')
    p.Define('init_radius', 2.0, 'dummy params not used')

    # Non-default quantization behaviour.
    p.qdomain.Define('weight', None, 'Quantization for the weights')
    p.qdomain.Define('c_state', None, 'Quantization for the c-state.')
    p.qdomain.Define('m_state', None, 'Quantization for the m-state.')
    p.qdomain.Define('fullyconnected', None,
                     'Quantization for fully connected node.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCellSimple."""
    super(LSTMCellSimple, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None

    assert p.cell_value_cap is None or p.qdomain.default is None
    self.TrackQTensor(
        'zero_m',
        'm_output',
        'm_output_projection',
        'm_zoneout',
        domain='m_state')
    self.TrackQTensor(
        'zero_c',
        'mixed',
        'c_couple_invert',
        'c_input_gate',
        'c_forget_gate',
        'c_output_gate',
        'c_zoneout',
        domain='c_state')
    self.TrackQTensor('add_bias', domain='fullyconnected')

    with tf.variable_scope(p.name) as scope:
      if p.train_scale:
        scale_pc = py_utils.WeightParams([],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      if p.init_distribution == 'uniform':
        if p.train_scale:
          scale_t = 1.0
        else:
          scale_t = p.input_scale

        if p.train_radius:
          radius_t = 1.0
        else:
          radius_t = p.spectral_radius

        init_lstm = py_utils.WeightInit.LSTMUniform(
            input_scale=scale_t,
            spectral_radius=radius_t,
            input_sparsity=p.input_sparsity,
            hidden_sparsity=p.hidden_sparsity,
            input_nodes=p.num_input_nodes)
      elif p.init_distribution == 'gaussian':
        init_lstm = py_utils.WeightInit.LSTMGaussian(
            input_scale=p.input_scale,
            spectral_radius=p.spectral_radius,
            input_sparsity=p.input_sparsity,
            hidden_sparsity=p.hidden_sparsity,
            input_nodes=p.num_input_nodes)
      else:
        init_lstm = p.params_init
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              p.num_input_nodes + self.output_size,
              self.num_gates * self.hidden_size
          ],
          init=init_lstm,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN, trainable=p.trainable)
      if p.apply_pruning:
        mask_pc = py_utils.WeightParams(wm_pc.shape,
                                        py_utils.WeightInit.Constant(1.0),
                                        p.dtype)
        threshold_pc = py_utils.WeightParams([],
                                             py_utils.WeightInit.Constant(0.0),
                                             tf.float32)
        self.CreateVariable('mask', mask_pc, theta_fn=None, trainable=False)
        self.CreateVariable(
            'threshold', threshold_pc, theta_fn=None, trainable=False)
        # for gradient based pruning
        # gradient and weight snapshots
        grad_pc = py_utils.WeightParams(wm_pc.shape,
                                        py_utils.WeightInit.Constant(0.0),
                                        p.dtype)
        if p.gradient_pruning:
          self.CreateVariable(
              'gradient', grad_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              'old_weight', grad_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              'old_old_weight', grad_pc, theta_fn=None, trainable=False)

          py_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                           self.vars.threshold,
                                           self.vars.gradient,
                                           self.vars.old_weight,
                                           self.vars.old_old_weight)
        else:
          py_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                           self.vars.threshold)
      if p.num_hidden_nodes:
        w_proj = py_utils.WeightParams(
            shape=[self.hidden_size, self.output_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'w_proj', w_proj, self.AddGlobalVN, trainable=p.trainable)
        if p.apply_pruning_to_projection:
          proj_mask_pc = py_utils.WeightParams(
              w_proj.shape, py_utils.WeightInit.Constant(1.0), p.dtype)
          proj_threshold_pc = py_utils.WeightParams(
              [], py_utils.WeightInit.Constant(0.0), tf.float32)
          self.CreateVariable(
              'proj_mask', proj_mask_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              'proj_threshold', proj_threshold_pc, trainable=False)
          # for gradient based pruning
          # gradient and weight snapshots
          proj_grad_pc = py_utils.WeightParams(
              w_proj.shape, py_utils.WeightInit.Constant(0.0), p.dtype)
          if p.gradient_pruning:
            self.CreateVariable('proj_gradient', proj_grad_pc, trainable=False)
            self.CreateVariable(
                'proj_old_weight', proj_grad_pc, trainable=False)
            self.CreateVariable(
                'proj_old_old_weight', proj_grad_pc, trainable=False)
            py_utils.AddToPruningCollections(self.vars.w_proj,
                                             self.vars.proj_mask,
                                             self.vars.proj_threshold,
                                             self.vars.proj_gradient,
                                             self.vars.proj_old_weight,
                                             self.vars.proj_old_old_weight)
          else:
            py_utils.AddToPruningCollections(self.vars.w_proj,
                                             self.vars.proj_mask,
                                             self.vars.proj_threshold)
      if p.enable_lstm_bias:
        bias_pc = py_utils.WeightParams(
            shape=[self.num_gates * self.hidden_size],
            init=p.bias_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN, trainable=True)

      # Collect some stats.
      w = self.vars.wm
      if p.couple_input_forget_gates:
        i_i, f_g, o_g = tf.split(
            value=w, num_or_size_splits=self.num_gates, axis=1)
      else:
        i_i, i_g, f_g, o_g = tf.split(
            value=w, num_or_size_splits=self.num_gates, axis=1)
        _HistogramSummary(scope.name + '/wm_i_g', i_g)
      _HistogramSummary(scope.name + '/wm_i_i', i_i)
      _HistogramSummary(scope.name + '/wm_f_g', f_g)
      _HistogramSummary(scope.name + '/wm_o_g', o_g)

      if p.train_scale:
        _HistogramSummary(scope.name + '/input_scale', self.vars.scale)
      if p.train_radius:
        _HistogramSummary(scope.name + '/radius', self.vars.radius)

      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  @property
  def num_gates(self):
    return 3 if self.params.couple_input_forget_gates else 4

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    if p.is_inference:
      zero_m = self.QTensor('zero_m', zero_m)
      zero_c = self.QTensor('zero_c', zero_c)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def _GetBias(self, theta):
    """Gets the bias vector to add.

    Includes adjustments like forget_gate_bias. Use this instead of the 'b'
    variable directly as including adjustments in this way allows const-prop
    to eliminate the adjustments at inference time.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.

    Returns:
      The bias vector.
    """
    p = self.params
    if p.enable_lstm_bias:
      b = theta.b
    else:
      b = tf.zeros([self.num_gates * self.hidden_size], dtype=p.dtype)
    if p.forget_gate_bias != 0.0:
      # Apply the forget gate bias directly to the bias vector.
      if not p.couple_input_forget_gates:
        # Normal 4 gate bias (i_i, i_g, f_g, o_g).
        adjustment = (
            tf.ones([4, self.hidden_size], dtype=p.dtype) * tf.expand_dims(
                tf.constant([0., 0., p.forget_gate_bias, 0.], dtype=p.dtype),
                axis=1))
      else:
        # 3 gates with coupled input/forget (i_i, f_g, o_g).
        adjustment = (
            tf.ones([3, self.hidden_size], dtype=p.dtype) * tf.expand_dims(
                tf.constant([0., p.forget_gate_bias, 0.], dtype=p.dtype),
                axis=1))
      adjustment = tf.reshape(adjustment, [self.num_gates * self.hidden_size])
      b += adjustment

    return b

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    if self.params.apply_pruning:
      wm = self.QWeight(tf.multiply(theta.wm, theta.mask, 'masked_weights'))
    else:
      wm = self.QWeight(theta.wm)

    # scale input part of weight matrix with input_scale
    # and hidden part with spectral radius
    input_dim0 = self.params.num_input_nodes
    hidden_dim0 = self.output_size
    i, h = tf.split(wm, [input_dim0, hidden_dim0], 0)
    if self.params.train_scale:
      i = tf.multiply(theta.scale, i)
    if self.params.train_radius:
      h = tf.multiply(theta.radius, h)
    wm = tf.concat([i, h], 0)

    concat = tf.concat(inputs.act + [state0.m], 1)
    # Defer quantization until after adding in the bias to support fusing
    # matmul and bias add during inference.
    return tf.matmul(concat, wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    i_i, i_g, f_g, o_g = self._RetrieveAndSplitGates(xmw, theta)
    return self._GatesInternal(theta, state0, inputs, i_i, i_g, f_g, o_g)

  def _RetrieveAndSplitGates(self, xmw, theta):
    p = self.params
    b = self.QWeight(tf.expand_dims(self._GetBias(theta), 0), domain='fc')
    xmw = self.fns.qadd(xmw, b, qt='add_bias')
    gates = tf.split(value=xmw, num_or_size_splits=self.num_gates, axis=1)
    if p.couple_input_forget_gates:
      gates = gates[0], None, gates[1], gates[2]
    return gates

  def _GatesInternal(self, theta, state0, inputs, i_i, i_g, f_g, o_g):
    p = self.params
    fns = self.fns
    if not p.couple_input_forget_gates:
      assert i_g is not None
      forget_gate = fns.qmultiply(tf.sigmoid(f_g), state0.c, qt='c_input_gate')
      # Sigmoid / tanh calls are not quantized under the assumption they share
      # the range with c_input_gate and c_forget_gate.
      input_gate = fns.qmultiply(
          tf.sigmoid(i_g), tf.tanh(i_i), qt='c_forget_gate')
      new_c = fns.qadd(forget_gate, input_gate, qt='c_output_gate')
    else:
      assert i_g is None
      # Sigmoid / tanh calls are not quantized under the assumption they share
      # the range with c_input_gate and c_forget_gate.
      forget_gate = fns.qmultiply(tf.sigmoid(f_g), state0.c, qt='c_input_gate')

      # input_gate = tanh(i_i) - tanh(i_i) * tf.sigmoid(f_g)
      # equivalent to (but more stable in fixed point):
      # (1.0 - sigmoid(f_g)) * tanh(i_i)
      tanh_i_i = tf.tanh(i_i)
      input_gate = fns.qsubtract(
          tanh_i_i,
          fns.qmultiply(tanh_i_i, tf.sigmoid(f_g), qt='c_couple_invert'),
          qt='c_forget_gate')

      new_c = fns.qadd(forget_gate, input_gate, qt='c_output_gate')

    new_c = self._ProcessNewC(theta, new_c)

    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
    if p.output_nonlinearity:
      new_m = fns.qmultiply(tf.sigmoid(o_g), tf.tanh(new_c), qt='m_output')
    else:
      new_m = fns.qmultiply(tf.sigmoid(o_g), new_c, qt='m_output')
    if p.num_hidden_nodes:
      if p.apply_pruning_to_projection:
        w_proj = self.QWeight(
            tf.multiply(theta.w_proj, theta.proj_mask, 'masked_projection'),
            domain='m_state')
      else:
        w_proj = self.QWeight(theta.w_proj, domain='m_state')

      new_m = fns.qmatmul(new_m, w_proj, qt='m_output_projection')

    # Apply Zoneout.
    return self._ApplyZoneOut(state0, inputs, new_c, new_m)

  def _ProcessNewC(self, theta, new_c):
    return new_c

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply Zoneout and returns the updated states."""
    p = self.params
    if p.zo_prob > 0.0:
      assert not py_utils.use_tpu(), (
          'LSTMCellSimple does not support zoneout on TPU. Switch to '
          'LSTMCellSimpleDeterministic instead.')
      c_random_uniform = tf.random.uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random.uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(
        state0.c,
        new_c,
        self.QRPadding(inputs.padding),
        p.zo_prob,
        self.do_eval,
        c_random_uniform,
        qt='c_zoneout',
        qdomain='c_state')
    new_m = self._ZoneOut(
        state0.m,
        new_m,
        self.QRPadding(inputs.padding),
        p.zo_prob,
        self.do_eval,
        m_random_uniform,
        qt='m_zoneout',
        qdomain='m_state')
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LSTMCellGrouped(RNNCell):
  """LSTM cell with groups.

  Grouping: based on "Factorization tricks for LSTM networks".
  https://arxiv.org/abs/1703.10722.

  Shuffling: adapted from "ShuffleNet: An Extremely Efficient Convolutional
  Neural Network for Mobile Devices". https://arxiv.org/abs/1707.01083.

  theta:

  - groups: a list of child LSTM cells.

  state:

    A `.NestedMap` containing 'groups', a list of `.NestedMap`, each with:

    - m: the lstm output. [batch, cell_nodes // num_groups]
    - c: the lstm cell state. [batch, cell_nodes // num_groups]

  inputs:

  -  act: a list of input activations. [batch, input_nodes]
  -  padding: the padding. [batch, 1].
  -  reset_mask: optional 0/1 float input to support packed input training.
     Shape [batch, 1]
  """

  @classmethod
  def Params(cls, child_cell_cls=LSTMCellSimple):
    p = super(LSTMCellGrouped, cls).Params()
    p.Define('child_lstm_tpl', child_cell_cls.Params(),
             'Template of child LSTM cells.')
    p.Define('num_hidden_nodes', 0, 'Number of hidden nodes.')
    p.Define(
        'split_inputs', True, 'If true, split the inputs into N groups. '
        'If false, each group gets all inputs.')
    p.Define('num_groups', 0, 'Number of LSTM cell groups.')
    p.Define('num_shuffle_shards', 1,
             'If > 1, number of shards for cross-group shuffling.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCellGrouped."""
    super(LSTMCellGrouped, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert p.num_input_nodes > 0
    assert p.num_output_nodes > 0
    assert p.num_groups > 0
    assert p.num_shuffle_shards > 0
    assert p.num_input_nodes % p.num_groups == 0
    assert p.num_output_nodes % (p.num_shuffle_shards * p.num_groups) == 0

    with tf.variable_scope(p.name):
      child_params = []
      for i in range(p.num_groups):
        child_p = self.params.child_lstm_tpl.Copy()
        child_p.name = 'group_%d' % i
        assert child_p.num_input_nodes == 0
        assert child_p.num_output_nodes == 0
        if p.split_inputs:
          child_p.num_input_nodes = p.num_input_nodes // p.num_groups
        else:
          child_p.num_input_nodes = p.num_input_nodes
        child_p.num_output_nodes = p.num_output_nodes // p.num_groups
        child_p.num_hidden_nodes = p.num_hidden_nodes // p.num_groups
        child_p.reset_cell_state = p.reset_cell_state
        child_params.append(child_p)
      self.CreateChildren('groups', child_params)

  def batch_size(self, inputs):
    return self.groups[0].batch_size(inputs)

  def zero_state(self, theta, batch_size):
    return py_utils.NestedMap(groups=[
        child.zero_state(child_theta, batch_size)
        for child, child_theta in zip(self.groups, theta.groups)
    ])

  # TODO(rpang): avoid split and concat between layers with the same number of
  # groups, if necessary.
  def GetOutput(self, state):
    p = self.params
    # Assuming that GetOutput() is stateless, we can just use the first child.
    outputs = [
        child.GetOutput(child_state)
        for child, child_state in zip(self.groups, state.groups)
    ]
    split_output = []
    # Split each output to num_shuffle_shards.
    for output in outputs:
      split_output.extend(
          py_utils.SplitRecursively(output, p.num_shuffle_shards))
    # Shuffle and concatenate shards.
    return py_utils.ConcatRecursively(self._ShuffleShards(split_output))

  def FProp(self, theta, state0, inputs):
    """Forward function.

    Splits state0 and inputs into N groups (N=num_groups), runs child
    LSTM cells on each group, and concatenates the outputs with optional
    shuffling between groups.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      state0: The previous recurrent state. A `.NestedMap`.
      inputs: The inputs to the cell. A `.NestedMap`.

    Returns:
      A tuple (state1, extras).
      - state1: The next recurrent state. A list.
      - extras: An empty `.NestedMap`.
    """
    p = self.params
    if p.split_inputs:
      split_inputs_act = py_utils.SplitRecursively(inputs.act, p.num_groups)
    else:
      split_inputs_act = [inputs.act] * p.num_groups
    state1 = py_utils.NestedMap(groups=[])
    for child, child_theta, child_state0, child_inputs_act in zip(
        self.groups, theta.groups, state0.groups, split_inputs_act):
      child_inputs = inputs.copy()
      child_inputs.act = child_inputs_act
      child_state1, child_extras = child.FProp(child_theta, child_state0,
                                               child_inputs)
      assert not child_extras
      state1.groups.append(child_state1)
    return state1, py_utils.NestedMap()

  def _ShuffleShards(self, shards):
    """Shuffles shards across groups.

    Args:
      shards: a list of length num_shuffle_shards (S) * num_groups (G). The
        first S shards belong to group 0, the next S shards belong to group 1,
        etc.

    Returns:
      A shuffled list of shards such that shards from each input group are
      scattered across output groups.

      For example, if we have 3 groups, each with 4 shards:

      | Group 0: 0_0, 0_1, 0_2, 0_3
      | Group 1: 1_0, 1_1, 1_2, 1_3
      | Group 2: 2_0, 2_1, 2_2, 2_3

      The shuffled output will be:

      | Group 0: 0_0, 1_1, 2_2, 0_3
      | Group 1: 1_0, 2_1, 0_2, 1_3
      | Group 2: 2_0, 0_1, 1_2, 2_3
    """
    p = self.params
    assert len(shards) == (p.num_shuffle_shards * p.num_groups)
    shuffled_shards = []
    for group_i in range(p.num_groups):
      for shuffle_i in range(p.num_shuffle_shards):
        shuffled_shards.append(shards[(
            (group_i + shuffle_i) % p.num_groups) * p.num_shuffle_shards +
                                      shuffle_i])
    return shuffled_shards


# TODO(yonghui): Merge this cell with the LSTMCellSimple cell.
class LSTMCellSimpleDeterministic(LSTMCellSimple):
  """Same as LSTMCellSimple, except this cell is completely deterministic."""

  @classmethod
  def Params(cls):
    p = super(LSTMCellSimpleDeterministic, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LSTMCell."""
    super(LSTMCellSimpleDeterministic, self).__init__(params)
    p = self.params
    assert p.name
    with tf.variable_scope(p.name):
      _, self._step_counter = py_utils.CreateVariable(
          name='lstm_step_counter',
          params=py_utils.WeightParams([], py_utils.WeightInit.Constant(0),
                                       tf.int64),
          trainable=False)
      vname = self._step_counter.name
      self._prng_seed = tf.constant(
          py_utils.GenerateSeedFromName(vname), dtype=tf.int64)
      if p.random_seed:
        self._prng_seed += p.random_seed

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = tf.zeros((batch_size, self.output_size),
                      dtype=py_utils.FPropDtype(p))
    zero_c = tf.zeros((batch_size, self.hidden_size),
                      dtype=py_utils.FPropDtype(p))
    if p.is_inference:
      zero_m = self.QTensor('zero_m', zero_m)
      zero_c = self.QTensor('zero_c', zero_c)

    # The first random seed changes for different layers and training steps.
    random_seed1 = self._prng_seed + self._step_counter
    # The second random seed changes for different unroll time steps.
    random_seed2 = tf.constant(0, dtype=tf.int64)
    random_seeds = tf.stack([random_seed1, random_seed2])
    return py_utils.NestedMap(m=zero_m, c=zero_c, r=random_seeds)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply Zoneout and returns the updated states."""
    p = self.params
    random_seed1 = state0.r[0]
    random_seed2 = state0.r[1]
    if p.zo_prob > 0.0:
      # Note(yonghui): It seems that currently TF only supports int64 as the
      # random seeds, however, TPU will support int32 as the seed.
      # TODO(yonghui): Fix me for TPU.
      c_seed = tf.stack([random_seed1, 2 * random_seed2])
      m_seed = tf.stack([random_seed1, 2 * random_seed2 + 1])
      if py_utils.use_tpu():
        c_random_uniform = tf.random.stateless_uniform(
            py_utils.GetShape(new_c, 2), tf.cast(c_seed, tf.int32))
        m_random_uniform = tf.random.stateless_uniform(
            py_utils.GetShape(new_m, 2), tf.cast(m_seed, tf.int32))
      else:
        c_random_uniform = tf.random.stateless_uniform(
            py_utils.GetShape(new_c, 2), c_seed)
        m_random_uniform = tf.random.stateless_uniform(
            py_utils.GetShape(new_m, 2), m_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(
        state0.c,
        new_c,
        inputs.padding,
        p.zo_prob,
        self.do_eval,
        c_random_uniform,
        qt='zero_c',
        qdomain='c_state')
    new_m = self._ZoneOut(
        state0.m,
        new_m,
        inputs.padding,
        p.zo_prob,
        self.do_eval,
        m_random_uniform,
        qt='zero_m',
        qdomain='m_state')
    # TODO(yonghui): stop the proliferation of tf.stop_gradient
    r = tf.stop_gradient(tf.stack([random_seed1, random_seed2 + 1]))
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    r.set_shape(state0.r.shape)
    return py_utils.NestedMap(m=new_m, c=new_c, r=r)

  def PostTrainingStepUpdate(self, global_step):
    """Update the global_step value."""
    p = self.params
    with tf.name_scope(p.name):
      summary_utils.scalar('step_counter', self._step_counter)
    return self._step_counter.assign(tf.cast(global_step, tf.int64))


class QuantizedLSTMCell(RNNCell):
  """Simplified LSTM cell used for quantized training.

  There is no forget_gate_bias, no output_nonlinearity and no bias. Right now
  only clipping is performed.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - cap: the cell value cap.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(QuantizedLSTMCell, cls).Params()
    p.Define('cc_schedule', quant_utils.LinearClippingCapSchedule.Params(),
             'Clipping cap schedule.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes QuantizedLSTMCell."""
    super(QuantizedLSTMCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              p.num_input_nodes + p.num_output_nodes, 4 * p.num_output_nodes
          ],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      self.CreateChild('cc_schedule', p.cc_schedule)

      # Collect some stats
      i_i, i_g, f_g, o_g = tf.split(
          value=self.vars.wm, num_or_size_splits=4, axis=1)
      _HistogramSummary(scope.name + '/wm_i_i', i_i)
      _HistogramSummary(scope.name + '/wm_i_g', i_g)
      _HistogramSummary(scope.name + '/wm_f_g', f_g)
      _HistogramSummary(scope.name + '/wm_o_g', o_g)

      self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    i_i, i_g, f_g, o_g = tf.split(value=xmw, num_or_size_splits=4, axis=1)

    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    new_c = self.cc_schedule.ApplyClipping(theta.cc_schedule, new_c)
    new_m = tf.sigmoid(o_g) * new_c

    # Respect padding.
    new_m = state0.m * inputs.padding + new_m * (1 - inputs.padding)
    new_c = state0.c * inputs.padding + new_c * (1 - inputs.padding)

    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LayerNormalizedLSTMCell(RNNCell):
  """DEPRECATED: use LayerNormalizedLSTMCellSimple instead.

  Simple LSTM cell with layer normalization.

  Implements normalization scheme as described in
  https://arxiv.org/pdf/1607.06450.pdf

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(LayerNormalizedLSTMCell, cls).Params()
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('forget_gate_bias', 0.0, 'Bias to apply to the forget gate.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    p.Define('cc_schedule', None, 'Clipping cap schedule.')
    p.Define('use_fused_layernorm', False, 'Whether to use fused layernorm.')
    p.Define('init_distribution', 'uniform', 'Distribution for initialization.')
    p.Define('input_scale', 2.0, 'Scale of the input matrix.')
    p.Define('spectral_radius', 0.9, 'Spectral radius of this layer.')
    p.Define('input_sparsity', 0.3, 'Input sparsity of this layer.')
    p.Define('hidden_sparsity', 0.3, 'Hidden sparsity of this layer.')
    p.Define('trainable', False, 'Trainable params or not.')
    p.Define('train_scale', False, 'Train input scale or not')
    p.Define('train_radius', False, 'Train spectral radius or not')
    p.Define('init_scale', 500, 'dummy params not used')
    p.Define('init_radius', 2.0, 'dummy params not used')
    return p

  @tf_deprecation.deprecated(
      date=None,
      instructions='New models should use LayerNormalizedLSTMCellSimple.')
  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCell."""
    super(LayerNormalizedLSTMCell, self).__init__(params)
    params = self.params
    if not isinstance(params.cell_value_cap, (int, float)):
      raise ValueError('Cell value cap must of type int or float!')

    with tf.variable_scope(params.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[
              params.num_input_nodes + params.num_output_nodes,
              4 * params.num_output_nodes
          ],
          init=params.params_init,
          dtype=params.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)
      # This bias variable actually packs the initial lstm bias variables as
      # well as various layer norm scale and bias variables. We pack multiple
      # variables into one so that we can still unroll this lstm using the FRNN
      # layer defined in layers.py.
      bias_pc = py_utils.WeightParams(
          shape=[4 * params.num_output_nodes + 4 * params.num_output_nodes],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=params.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      if params.cc_schedule:
        self.CreateChild('cc_schedule', params.cc_schedule)

      # Collect some stats
      i_i, i_g, f_g, o_g = tf.split(
          value=self.vars.wm, num_or_size_splits=4, axis=1)
      _HistogramSummary(scope.name + '/wm_i_i', i_i)
      _HistogramSummary(scope.name + '/wm_i_g', i_g)
      _HistogramSummary(scope.name + '/wm_f_g', f_g)
      _HistogramSummary(scope.name + '/wm_o_g', o_g)
      # TODO(yonghui): Add more summaries here.

      self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_output_nodes

  def zero_state(self, theta, batch_size):
    p = self.params
    return py_utils.NestedMap(
        m=py_utils.InitRNNCellState([batch_size, p.num_output_nodes],
                                    init=p.zero_state_init_params,
                                    dtype=py_utils.FPropDtype(p),
                                    is_eval=self.do_eval),
        c=py_utils.InitRNNCellState([batch_size, p.num_output_nodes],
                                    init=p.zero_state_init_params,
                                    dtype=py_utils.FPropDtype(p),
                                    is_eval=self.do_eval))

  def GetOutput(self, state):
    return state.m

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def _Mix(self, theta, state0, inputs):
    if not isinstance(inputs.act, list):
      raise ValueError('Input activations must be of list type!')
    return py_utils.Matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    # Unpack the variables (weight and bias) into individual variables.
    params = self.params

    def BiasSlice(dim, num_dims, start_ind):
      s = []
      for i in range(num_dims):
        s.append(theta.b[start_ind + i * dim:start_ind + (i + 1) * dim])
      start_ind += dim * num_dims
      return s, start_ind

    # Unpack the bias variable.
    slice_start = 0
    bias_lstm, slice_start = BiasSlice(params.num_output_nodes, 4, slice_start)
    ln_scale, slice_start = BiasSlice(params.num_output_nodes, 4, slice_start)
    assert slice_start == 8 * params.num_output_nodes

    def _LayerNorm(x, last_dim):
      """Normalize the last dimension."""
      if params.use_fused_layernorm:
        counts, means_ss, variance_ss, _, = tf.nn.sufficient_statistics(
            x, axes=[last_dim], keepdims=True)
        mean, variance = tf.nn.normalize_moments(counts, means_ss, variance_ss,
                                                 None)
      else:
        mean = tf.reduce_mean(x, axis=[last_dim], keepdims=True)
        variance = tf.reduce_mean(
            tf.square(x - mean), axis=[last_dim], keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance + params.layer_norm_epsilon)

    state_split = tf.split(xmw, num_or_size_splits=4, axis=1)
    for i in range(4):
      state_split[i] = _LayerNorm(state_split[i], 1) * tf.expand_dims(
          ln_scale[i] + 1.0, 0) + tf.expand_dims(bias_lstm[i], 0)

    i_i, i_g, f_g, o_g = state_split

    if params.forget_gate_bias != 0.0:
      f_g += params.forget_gate_bias
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)

    # Clip the cell states to reasonable value.
    if params.cc_schedule:
      cap = self.cc_schedule.GetState(theta.cc_schedule)
    else:
      cap = params.cell_value_cap
    new_c = py_utils.clip_by_value(new_c, -cap, cap)

    if params.output_nonlinearity:
      new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    else:
      new_m = tf.sigmoid(o_g) * new_c

    if params.zo_prob > 0.0:
      c_random_uniform = tf.random.uniform(
          tf.shape(new_c), seed=params.random_seed)
      m_random_uniform = tf.random.uniform(
          tf.shape(new_m), seed=params.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(state0.c, new_c, inputs.padding, params.zo_prob,
                          self.do_eval, c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, inputs.padding, params.zo_prob,
                          self.do_eval, m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class LayerNormalizedLSTMCellSimple(LSTMCellSimple):
  """An implementation of layer normalized LSTM based on LSTMCellSimple.

  Implements normalization scheme as described in
  https://arxiv.org/pdf/1607.06450.pdf

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(LayerNormalizedLSTMCellSimple, cls).Params()
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCellSimple."""
    super(LayerNormalizedLSTMCellSimple, self).__init__(params)
    p = self.params

    add_biases = ['add_bias_{}'.format(i) for i in range(self.num_gates)]
    self.TrackQTensor(*add_biases, domain='fullyconnected')

    with tf.variable_scope(p.name):
      ln_scale_pc = py_utils.WeightParams(
          shape=[self.num_gates * self.hidden_size],
          init=py_utils.WeightInit.Constant(1.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('ln_scale', ln_scale_pc, self.AddGlobalVN)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""

    def _LayerNorm(x):
      """Applies layer normalization on the last dimension of 'x'.

      Args:
        x: activation tensor, where the last dimension represents channels.

      Returns:
        Layer normalized 'x', with the same shape as the input.
      """
      mean = tf.reduce_mean(x, axis=-1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance + p.layer_norm_epsilon)

    p = self.params
    b = self.QWeight(tf.expand_dims(self._GetBias(theta), 0), domain='fc')

    bs = tf.split(b, num_or_size_splits=self.num_gates, axis=1)
    ln_scales = tf.split(
        theta.ln_scale, num_or_size_splits=self.num_gates, axis=0)
    gates = tf.split(xmw, num_or_size_splits=self.num_gates, axis=1)

    for i in range(self.num_gates):
      # i_g is None when p.couple_input_forget_gates is True.
      if gates[i] is not None:
        gates[i] = _LayerNorm(gates[i]) * tf.expand_dims(ln_scales[i], 0)
        gates[i] = self.fns.qadd(gates[i], bs[i], qt='add_bias_{}'.format(i))

    if not p.couple_input_forget_gates:
      i_i, i_g, f_g, o_g = gates
    else:
      i_i, i_g, f_g, o_g = gates[0], None, gates[1], gates[2]
    return self._GatesInternal(theta, state0, inputs, i_i, i_g, f_g, o_g)


class LayerNormalizedLSTMCellLean(RNNCell):
  """A very lean layer normalized LSTM cell.

  This version is around 20% faster on TPU than LayerNormalizedLSTMCellSimple as
  it avoids certain reshape ops which are not free on TPU.

  Note, this version doesn't support all the options as implemented in
  LayerNormalizedLSTMCellSimple, such as quantization, zoneout regularization
  and etc.

  For the overlapping options, an incomplete list of differences from
  LayerNormalizedLSTMCellSimple include:
  - c_state is layer-normalized for computing new_m (if enable_ln_on_c=True)
  - ln_scale has a fixed offset of 1.

  Please use the other version if you even need those options.

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(LayerNormalizedLSTMCellLean, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqrt against.')
    p.Define(
        'enable_ln_on_c', True,
        'Whether to apply layer normalization on state.c. '
        'If false, LayerNormalizedLSTMCellLean will behave exactly as '
        'LayerNormalizedLSTMCellSimple.')
    p.Define(
        'cell_value_cap', None, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    p.Define('enable_lstm_bias', False, 'Enable the LSTM Cell bias.')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Initialization parameters for bias')
    p.Define('use_ln_bias', True, 'If to include a bias term for layer norm.')

    # TODO(yonghui): Get rid of the following two params.
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes LayerNormalizedLSTMCellLean."""
    super(LayerNormalizedLSTMCellLean, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    assert p.output_nonlinearity
    assert p.zo_prob == 0.0
    if p.cell_value_cap is not None and not isinstance(p.cell_value_cap,
                                                       (int, float)):
      raise ValueError(
          'p.cell_value_cap should be a int/float if not None, but got {}'
          .format(p.cell_value_cap))

    with tf.variable_scope(p.name):
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes + self.output_size, 4 * self.hidden_size],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      if p.num_hidden_nodes:
        w_proj = py_utils.WeightParams(
            shape=[self.hidden_size, self.output_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)

      if p.enable_lstm_bias:
        bias_pc = py_utils.WeightParams(
            shape=[4 * self.hidden_size],
            init=p.bias_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      pc = py_utils.WeightParams(
          shape=[self.hidden_size],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      ln_gates = ['i_g', 'i_i', 'f_g', 'o_g']
      if p.enable_ln_on_c:
        ln_gates += ['c']
      for ln_name in ln_gates:
        self.CreateVariable('ln_scale_' + ln_name, pc, self.AddGlobalVN)
        if p.use_ln_bias:
          self.CreateVariable('bias_' + ln_name, pc, self.AddGlobalVN)

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    mixed = tf.matmul(tf.concat(inputs.act + [state0.m], 1), theta.wm)
    return mixed

  def _LayerNormGate(self, theta, gate_name, x):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      theta: a NestedMap of layer params.
      gate_name: the name of the gate, e.g., 'i_i', 'f_g', 'c', etc.
      x: activation tensor, where the last dimension represents channels.

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    if gate_name == 'c' and not p.enable_ln_on_c:
      return x
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.math.rsqrt(variance + p.layer_norm_epsilon)
    scale = theta['ln_scale_%s' % gate_name] + 1.0
    if p.use_ln_bias:
      bias = theta['bias_%s' % gate_name]
      return normed * scale + bias
    else:
      return normed * scale

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    i_i, i_g, f_g, o_g = tf.split(value=xmw, num_or_size_splits=4, axis=1)
    i_i = self._LayerNormGate(theta, 'i_i', i_i)
    i_g = self._LayerNormGate(theta, 'i_g', i_g)
    f_g = self._LayerNormGate(theta, 'f_g', f_g)
    o_g = self._LayerNormGate(theta, 'o_g', o_g)
    if p.enable_lstm_bias:
      # LayerNormalizedLSTMCellLean applies biases after LN.
      biases = tf.split(theta.b, num_or_size_splits=4)
      i_i += biases[0]
      i_g += biases[1]
      f_g += biases[2]
      o_g += biases[3]
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
    # new_c_normed is only used for computing 'new_m'. We use the un-normalized
    # new_cc as cell state to keep the residual property of lstm cell.
    new_c_normed = self._LayerNormGate(theta, 'c', new_c)
    new_m = tf.sigmoid(o_g) * tf.tanh(new_c_normed)

    if p.num_hidden_nodes:
      new_m = tf.matmul(new_m, theta.w_proj)

    # Now take care of padding.
    padding = inputs.padding
    new_m = py_utils.ApplyPadding(padding, new_m, state0.m)
    new_c = py_utils.ApplyPadding(padding, new_c, state0.c)

    return py_utils.NestedMap(m=new_m, c=new_c)


class DoubleProjectionLSTMCell(RNNCell):
  """A layer normalized LSTM cell that support input and output projections.

  Note, this version doesn't support all the options as implemented in
  LayerNormalizedLSTMCellSimple, like quantization, zoneout regularization,
  etc. Please use the other version if you need those options and do not need
  input projection.

  It also uses separate variables for weight matrices between gates
  ('wm_{i_i, i_g, f_g, o_g}') instead of a single variable ('wm'). This allows
  the initialization to use the default GeoMeanXavier().

  state:

  - m: the lstm output. [batch, cell_nodes]
  - c: the lstm cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1]
  """

  @classmethod
  def Params(cls):
    p = super(DoubleProjectionLSTMCell, cls).Params()
    p.Define(
        'num_input_hidden_nodes', 0,
        'Project all inputs, include m, to a hidden vector this size before '
        'projecting to num_gates * |c|. Must be > 0.')
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqrt against.')
    p.Define('enable_ln_on_c', True,
             'Whether to apply layer normalization on state.c.')
    p.params_init = py_utils.WeightInit.GeoMeanXavier()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(DoubleProjectionLSTMCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    with tf.variable_scope(p.name):

      def _WeightInit(shape):
        return py_utils.WeightParams(
            shape=shape,
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())

      self.CreateVariable(
          'w_input_proj',
          _WeightInit(
              [p.num_input_nodes + self.output_size, p.num_input_hidden_nodes]),
          self.AddGlobalVN)

      self.CreateVariable('w_output_proj',
                          _WeightInit([self.hidden_size, self.output_size]),
                          self.AddGlobalVN)

      for gate_name in self.gates:
        self.CreateVariable(
            'wm_%s' % gate_name,
            _WeightInit([p.num_input_hidden_nodes, self.hidden_size]),
            self.AddGlobalVN)

      pc = py_utils.WeightParams(
          shape=[self.hidden_size],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      ln_gates = self.gates
      if p.enable_ln_on_c:
        ln_gates += ['c']
      for ln_name in ln_gates:
        self.CreateVariable('ln_scale_' + ln_name, pc, self.AddGlobalVN)
        self.CreateVariable('bias_' + ln_name, pc, self.AddGlobalVN)

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  @property
  def gates(self):
    return ['i_g', 'i_i', 'f_g', 'o_g']

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def _ProcessInputProj(self, theta, input_proj):
    return input_proj

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    concat = tf.concat(inputs.act + [state0.m], 1)
    input_proj = tf.matmul(concat, theta.w_input_proj)
    input_proj = self._ProcessInputProj(theta, input_proj)
    gate_map = {}
    for gate_name in self.gates:
      g = tf.matmul(input_proj, theta.get('wm_%s' % gate_name))
      g = self._LayerNormGate(theta, gate_name, g)
      gate_map[gate_name] = g
    return gate_map

  def _LayerNormGate(self, theta, gate_name, x):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      theta: a NestedMap of layer params.
      gate_name: the name of the gate, e.g., 'i_i', 'f_g', 'c', etc.
      x: activation tensor, where the last dimension represents channels.

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    if gate_name == 'c' and not p.enable_ln_on_c:
      return x
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.math.rsqrt(variance + p.layer_norm_epsilon)
    scale = theta['ln_scale_%s' % gate_name] + 1.0
    bias = theta['bias_%s' % gate_name]
    return normed * scale + bias

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    new_c = tf.sigmoid(xmw['f_g']) * state0.c + tf.sigmoid(
        xmw['i_g']) * tf.tanh(xmw['i_i'])
    new_c_normed = self._LayerNormGate(theta, 'c', new_c)
    new_m = tf.sigmoid(xmw['o_g']) * tf.tanh(new_c_normed)
    new_m = tf.matmul(new_m, theta.w_output_proj)

    # Now take care of padding.
    padding = inputs.padding
    new_m = py_utils.ApplyPadding(padding, new_m, state0.m)
    new_c = py_utils.ApplyPadding(padding, new_c, state0.c)

    return py_utils.NestedMap(m=new_m, c=new_c)


class ConvLSTMCell(RNNCell):
  """Convolution LSTM cells.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the lstm output. cell_shape
  - c: the lstm cell state. cell_shape

  inputs:

  - act: a list of input activations. input_shape.
  - padding: the padding. [batch].
  """

  @classmethod
  def Params(cls):
    p = super(ConvLSTMCell, cls).Params()
    p.Define(
        'inputs_shape', [None, None, None, None],
        'The shape of the input. It should be a list/tuple of size four.'
        ' Elements are in the order of batch, height, width, channel.')
    p.Define(
        'cell_shape', [None, None, None, None],
        'The cell shape. It should be a list/tuple of size four.'
        ' Elements are in the order of batch, height, width, channel.'
        ' Height and width of cell_shape should match that of'
        ' inputs_shape.')
    p.Define(
        'filter_shape', [None, None],
        'Shape of the convolution filter. This should be a pair, in the'
        ' order height and width.')
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('output_nonlinearity', True,
             'Whether or not to apply tanh non-linearity on lstm output.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ConvLSTMCell."""
    assert isinstance(params, hyperparams.Params)
    super(ConvLSTMCell, self).__init__(params)

    p = self.params
    # assert p.reset_cell_state is False, ('ConvLSTMCell currently does not '
    #                                      'support resetting cell state.')
    assert p.inputs_shape[1] == p.cell_shape[1]
    assert p.inputs_shape[2] == p.cell_shape[2]
    assert isinstance(p.cell_value_cap, (int, float))

    in_channels = p.inputs_shape[3] + p.cell_shape[3]
    out_channels = p.cell_shape[3]
    with tf.variable_scope(p.name):
      # Define weights.
      var_shape = [
          p.filter_shape[0], p.filter_shape[1], in_channels, 4 * out_channels
      ]
      wm_pc = py_utils.WeightParams(
          shape=var_shape,
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)

      bias_pc = py_utils.WeightParams(
          shape=[4 * out_channels],
          init=py_utils.WeightInit.Constant(0.0),
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    height = p.inputs_shape[1]
    width = p.inputs_shape[2]
    out_channels = p.cell_shape[3]
    return py_utils.NestedMap(
        m=py_utils.InitRNNCellState(
            tf.stack([batch_size, height, width, out_channels]),
            init=p.zero_state_init_params,
            dtype=p.dtype,
            is_eval=self.do_eval),
        c=py_utils.InitRNNCellState(
            tf.stack([batch_size, height, width, out_channels]),
            init=p.zero_state_init_params,
            dtype=p.dtype,
            is_eval=self.do_eval))

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    # Concate on channels.
    xm = tf.concat(inputs.act + [state0.m], 3)
    # TODO(yonghui): Possibly change the data_format to NCHW to speed
    # up conv2d kernel on gpu.
    xmw = tf.nn.conv2d(xm, theta.wm, [1, 1, 1, 1], 'SAME', data_format='NHWC')
    return xmw

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # Bias is applied to channels.
    bias = tf.reshape(theta.b, [1, 1, 1, -1])
    i_i, i_g, f_g, o_g = tf.split(
        value=xmw + bias, num_or_size_splits=4, axis=3)
    new_c = tf.sigmoid(f_g) * state0.c + tf.sigmoid(i_g) * tf.tanh(i_i)
    # Clip the cell states to reasonable value.
    new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)
    if p.output_nonlinearity:
      new_m = tf.sigmoid(o_g) * tf.tanh(new_c)
    else:
      new_m = tf.sigmoid(o_g) * new_c
    padding = tf.reshape(inputs.padding, [-1, 1, 1, 1])
    new_c = state0.c * padding + new_c * (1.0 - padding)
    new_m = state0.m * padding + new_m * (1.0 - padding)
    if p.zo_prob > 0.0:
      c_random_uniform = tf.random.uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random.uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None
    new_c = self._ZoneOut(state0.c, new_c, padding, p.zo_prob, self.do_eval,
                          c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, padding, p.zo_prob, self.do_eval,
                          m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class SRUCell(RNNCell):
  """SRU cell.

  From this paper: https://arxiv.org/abs/1709.02755

  This is a simple implementation that can be used as a drop-in replacement for
  another RNN. It doesn't do the performance tricks that an SRU is capable of,
  like unrolling matrix computations over time. This is just a basic
  implementation. It does the 4-matrix implementation found in appendix C.

  theta:

  - wm: the parameter weight matrix. All gates combined.
  - b: the combined bias vector.

  state:

  - m: the sru output. [batch, cell_nodes]
  - c: the sru cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(SRUCell, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define(
        'cell_value_cap', 10.0, 'SRU cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap]. This can be a'
        ' scalar or a scalar tensor.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('couple_input_forget_gates', True,
             'Whether to couple the input and forget gates.')
    p.Define('apply_layer_norm', False, 'Apply layer norm to the variables')
    p.Define(
        'layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.'
        'value is necessary only if apply_layer_norm is True')
    p.Define('apply_pruning', False, 'Whether to prune the weights while'
             'training')
    p.Define('apply_pruning_to_projection', False,
             'Whether to prune the weights in the projection layer')
    p.Define('gradient_pruning', False, 'Whether to gradient prune the model')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Initialization parameters for bias')
    # Add cell-recursive vector into the SRU cells (arxiv.org/abs/1709.02755).
    p.Define(
        'pointwise_peephole', False, 'Whether c_{t-1} should be used to'
        'calculate gate values by aggregating gate calculations with its '
        'point-wise dot product with a weight vector.')
    p.Define(
        'hidden_scaling_factor', False,
        'scaling factor alpha for hidden layer. See details on alpha in'
        'section 3.2 of https://arxiv.org/pdf/1709.02755.pdf')
    p.Define(
        'uniform_heuristic_init', False,
        'When set to True, initialize the weight params with uniform '
        'distribution of [-sqrt(3/hidden_nodes), +sqrt(3/hidden_nodes)], aka'
        'UniformUnitScaling. This initialization has proven to help NLP tasks'
        'in arxiv.org/abs/1709.02755. This impacts 1) input weight matrices for'
        'gates, 2) projection weight matrices, and 3) cell recursion vectors.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes SRUCell."""
    super(SRUCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    # assert p.reset_cell_state is False, ('SRUCell currently does not support '
    #                                      'resetting cell state.')
    assert isinstance(p.cell_value_cap, (int, float))
    if p.uniform_heuristic_init:
      # Setting init = sqrt(3) / sqrt(hidden) * tf.uniform(-1, 1).
      p.params_init = py_utils.WeightInit.Uniform(
          scale=(math.sqrt(3.0 / float(self.hidden_size))))

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      wm_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, self.num_gates * self.hidden_size],
          init=p.params_init,
          dtype=p.dtype,
          collections=self._VariableCollections())

      self.CreateVariable('wm', wm_pc, self.AddGlobalVN)
      if p.apply_pruning:
        mask_pc = py_utils.WeightParams(wm_pc.shape,
                                        py_utils.WeightInit.Constant(1.0),
                                        p.dtype)
        threshold_pc = py_utils.WeightParams([],
                                             py_utils.WeightInit.Constant(0.0),
                                             tf.float32)
        self.CreateVariable('mask', mask_pc, theta_fn=None, trainable=False)
        self.CreateVariable(
            'threshold', threshold_pc, theta_fn=None, trainable=False)

        # for gradient based pruning
        # gradient and weight snapshots
        grad_pc = py_utils.WeightParams(wm_pc.shape,
                                        py_utils.WeightInit.Constant(0.0),
                                        p.dtype)
        if p.gradient_pruning:
          self.CreateVariable(
              'gradient', grad_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              'old_weight', grad_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              'old_old_weight', grad_pc, theta_fn=None, trainable=False)
          py_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                           self.vars.threshold,
                                           self.vars.gradient,
                                           self.vars.old_weight,
                                           self.vars.old_old_weight)
        else:
          py_utils.AddToPruningCollections(self.vars.wm, self.vars.mask,
                                           self.vars.threshold)
      bias_pc = py_utils.WeightParams(
          shape=[self.num_gates * self.hidden_size],
          init=p.bias_init,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('b', bias_pc, self.AddGlobalVN)

      if p.num_hidden_nodes:
        w_proj = py_utils.WeightParams(
            shape=[self.hidden_size, self.output_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('w_proj', w_proj, self.AddGlobalVN)
        if p.apply_pruning_to_projection:
          proj_mask_pc = py_utils.WeightParams(
              w_proj.shape, py_utils.WeightInit.Constant(1.0), p.dtype)
          proj_threshold_pc = py_utils.WeightParams(
              [], py_utils.WeightInit.Constant(0.0), tf.float32)
          self.CreateVariable(
              'proj_mask', proj_mask_pc, theta_fn=None, trainable=False)
          self.CreateVariable(
              'proj_threshold', proj_threshold_pc, trainable=False)
          # for gradient based pruning
          # gradient and weight snapshots
          proj_grad_pc = py_utils.WeightParams(
              w_proj.shape, py_utils.WeightInit.Constant(0.0), p.dtype)
          if p.gradient_pruning:
            self.CreateVariable('proj_gradient', proj_grad_pc, trainable=False)
            self.CreateVariable(
                'proj_old_weight', proj_grad_pc, trainable=False)
            self.CreateVariable(
                'proj_old_old_weight', proj_grad_pc, trainable=False)
            py_utils.AddToPruningCollections(self.vars.w_proj,
                                             self.vars.proj_mask,
                                             self.vars.proj_threshold,
                                             self.vars.proj_gradient,
                                             self.vars.proj_old_weight,
                                             self.vars.proj_old_old_weight)
          else:
            py_utils.AddToPruningCollections(self.vars.w_proj,
                                             self.vars.proj_mask,
                                             self.vars.proj_threshold)
      # TODO(yuansg): b/136014373 investigate the layer norm initialization and
      # implementation, try skipping LP regularization on layer norm and bias.
      if p.apply_layer_norm:
        f_t_ln_scale = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('f_t_ln_scale', f_t_ln_scale, self.AddGlobalVN)
        r_t_ln_scale = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('r_t_ln_scale', r_t_ln_scale, self.AddGlobalVN)
        c_t_ln_scale = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('c_t_ln_scale', c_t_ln_scale, self.AddGlobalVN)
        if not p.couple_input_forget_gates:
          i_t_ln_scale = py_utils.WeightParams(
              shape=[self.hidden_size],
              init=py_utils.WeightInit.Constant(1.0),
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable('i_t_ln_scale', i_t_ln_scale, self.AddGlobalVN)

      if p.pointwise_peephole:
        f_t_vector_cell = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('f_t_vector_cell', f_t_vector_cell,
                            self.AddGlobalVN)
        r_t_vector_cell = py_utils.WeightParams(
            shape=[self.hidden_size],
            init=p.params_init,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('r_t_vector_cell', r_t_vector_cell,
                            self.AddGlobalVN)
        if not p.couple_input_forget_gates:
          i_t_vector_cell = py_utils.WeightParams(
              shape=[self.hidden_size],
              init=p.params_init,
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable('i_t_vector_cell', i_t_vector_cell,
                              self.AddGlobalVN)

      # Collect some stats.
      if p.couple_input_forget_gates:
        x_t2, resized, f_t, r_t = tf.split(
            value=self.vars.wm, num_or_size_splits=self.num_gates, axis=1)
      else:
        x_t2, resized, i_t, f_t, r_t = tf.split(
            value=self.vars.wm, num_or_size_splits=self.num_gates, axis=1)
        _HistogramSummary(scope.name + '/wm_i_t', i_t)
      _HistogramSummary(scope.name + '/wm_x_t2', x_t2)
      _HistogramSummary(scope.name + '/wm_resized', resized)
      _HistogramSummary(scope.name + '/wm_f_t', f_t)
      _HistogramSummary(scope.name + '/wm_r_t', r_t)

      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  @property
  def num_gates(self):
    return 4 if self.params.couple_input_forget_gates else 5

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def LayerNorm(self, theta, gate_name, x, bias):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      theta: a NestedMap of layer params.
      gate_name: the name of the gate, e.g., 'i_i', 'f_g', 'c', etc.
      x: activation tensor, where the last dimension represents channels.
      bias: the bias tensor of the gate.

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    if p.apply_layer_norm:
      mean = tf.reduce_mean(x, axis=[1], keepdims=True)
      centered = x - mean
      variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
      normed = centered * tf.math.rsqrt(variance + p.layer_norm_epsilon)
      scale = theta['%s_ln_scale' % gate_name]
      x = normed * scale
    return x + bias

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    if self.params.apply_pruning:
      wm = tf.multiply(theta.wm, theta.mask)
    else:
      wm = theta.wm
    return py_utils.Matmul(tf.concat(inputs.act, 1), wm)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    if p.couple_input_forget_gates:
      x_t2, resized, f_t, r_t = tf.split(
          value=xmw, num_or_size_splits=4, axis=1)
      b_t2, b_resized, b_f, b_r = tf.split(
          value=tf.expand_dims(theta.b, 0), num_or_size_splits=4, axis=1)
      if p.pointwise_peephole:
        f_t = f_t + tf.multiply(state0.c, theta.f_t_vector_cell)
      f_t = self.LayerNorm(theta, 'f_t', f_t, b_f)
      f_t = tf.nn.sigmoid(f_t)
      i_t = 1.0 - f_t
    else:
      x_t2, resized, i_t, f_t, r_t = tf.split(
          value=xmw + tf.expand_dims(theta.b, 0), num_or_size_splits=5, axis=1)
      b_t2, b_resized, b_i, b_f, b_r = tf.split(
          value=tf.expand_dims(theta.b, 0), num_or_size_splits=5, axis=1)
      if p.pointwise_peephole:
        f_t = f_t + tf.multiply(state0.c, theta.f_t_vector_cell)
        i_t = i_t + tf.multiply(state0.c, theta.i_t_vector_cell)
      f_t = self.LayerNorm(theta, 'f_t', f_t, b_f)
      f_t = tf.nn.sigmoid(f_t)
      i_t = self.LayerNorm(theta, 'i_t', i_t, b_i)
      i_t = tf.nn.sigmoid(i_t)

    if p.pointwise_peephole:
      r_t = r_t + tf.multiply(state0.c, theta.r_t_vector_cell)
    r_t = self.LayerNorm(theta, 'r_t', r_t, b_r)
    r_t = tf.nn.sigmoid(r_t)

    c_t = f_t * state0.c + i_t * x_t2
    c_t = self.LayerNorm(theta, 'c_t', c_t, 0)

    resized = tf.add(resized, b_resized)
    x_t2 = tf.add(x_t2, b_t2)
    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      c_t = py_utils.clip_by_value(c_t, -p.cell_value_cap, p.cell_value_cap)
    # Calculate state outputs.
    g_c_t = tf.nn.tanh(c_t)
    # Apply scaling factor if needed.
    alpha = 1.0
    if p.hidden_scaling_factor:
      # For the derivations of alpha please refer to variance computation of
      # hidden cells h with respect to the variance of input x in appendix A.3
      # https://arxiv.org/pdf/1709.02755.pdf.
      alpha = tf.sqrt(1.0 + tf.exp(p.bias_init.scale * 2.0))
    h_t = r_t * g_c_t + (1.0 - r_t) * resized * alpha

    if p.num_hidden_nodes:
      if p.apply_pruning_to_projection:
        w_proj = tf.multiply(theta.w_proj, theta.proj_mask)
      else:
        w_proj = theta.w_proj
      h_t = tf.matmul(h_t, w_proj)

    return self._ApplyZoneOut(state0, inputs, c_t, h_t)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply ZoneOut and returns updated states."""
    p = self.params
    if p.zo_prob > 0.0:
      assert not py_utils.use_tpu(), (
          'SRUCell does not support zoneout on TPU yet.')
      c_random_uniform = tf.random.uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random.uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob,
                          self.do_eval, c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob,
                          self.do_eval, m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class QRNNPoolingCell(RNNCell):
  """This implements just the "pooling" part of a quasi-RNN or SRU.

  From these papers:

  - https://arxiv.org/abs/1611.01576
  - https://arxiv.org/abs/1709.02755

  The pooling part implements gates for recurrence. These architectures split
  the transform (conv or FC) from the gating/recurrent part. This cell can
  do either the quasi-RNN style or SRU style pooling operation based on params.

  If you want all of the functionality in one RNN cell, use `SRUCell` instead.

  theta:

    Has the trainable zero state. Other weights are done outside the recurrent
    loop.

  state:

  - m: the qrnn output. [batch, cell_nodes]
  - c: the qrnn cell state. [batch, cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes * num_rnn_matrices]
  - padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(QRNNPoolingCell, cls).Params()
    p.Define(
        'cell_value_cap', 10.0, 'LSTM cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('pooling_formula', 'INVALID',
             'Options: quasi_ifo, sru. Which pooling math to use')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes quasi-RNN Cell."""
    super(QRNNPoolingCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    # assert p.reset_cell_state is False, ('QRNNPoolingCell currently does not '
    #                                      'support resetting cell state.')
    assert p.pooling_formula in ('quasi_ifo', 'sru')
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None

    self._timestep = -1

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=p.dtype,
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, p.num_output_nodes),
                                       init=p.zero_state_init_params,
                                       dtype=p.dtype,
                                       is_eval=self.do_eval)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def GetOutput(self, state):
    return state.m

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    # Just do identity. The convolution part of the QRNN has to be done earlier.
    return inputs.act

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    if p.pooling_formula == 'quasi_ifo':
      z_t, i_t, f_t, o_t = tf.split(
          value=tf.concat(xmw, 1), num_or_size_splits=4, axis=1)
      # Quasi-RNN "ifo" pooling
      c_t = f_t * state0.c + i_t * z_t
      h_t = o_t * c_t
    elif p.pooling_formula == 'sru':
      x_t2, resized, f_t, r_t = tf.split(
          value=tf.concat(xmw, 1), num_or_size_splits=4, axis=1)
      c_t = f_t * state0.c + (1.0 - f_t) * x_t2
      # TODO(otaviogood): Optimization - Since state doesn't depend on these
      # ops, they can be moved outside the loop.
      g_c_t = tf.nn.tanh(c_t)
      h_t = r_t * g_c_t + (1.0 - r_t) * resized
    else:
      raise ValueError('Invalid pooling_formula: ', p.pooling_formula)

    new_c = c_t
    new_m = h_t

    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)

    # Apply Zoneout.
    return self._ApplyZoneOut(state0, inputs, new_c, new_m)

  def _ApplyZoneOut(self, state0, inputs, new_c, new_m):
    """Apply Zoneout and returns the updated states."""
    p = self.params
    if p.zo_prob > 0.0:
      c_random_uniform = tf.random.uniform(tf.shape(new_c), seed=p.random_seed)
      m_random_uniform = tf.random.uniform(tf.shape(new_m), seed=p.random_seed)
    else:
      c_random_uniform = None
      m_random_uniform = None

    new_c = self._ZoneOut(state0.c, new_c, inputs.padding, p.zo_prob,
                          self.do_eval, c_random_uniform)
    new_m = self._ZoneOut(state0.m, new_m, inputs.padding, p.zo_prob,
                          self.do_eval, m_random_uniform)
    new_c.set_shape(state0.c.shape)
    new_m.set_shape(state0.m.shape)
    return py_utils.NestedMap(m=new_m, c=new_c)


class GRUCell(RNNCell):
  """Gated Recurrent Unit cell.

  implemented: layer normalization, gru_biasing, gru_cell cap,
  not yet implemented: pruning, quantization, zone-out (enforced to 0.0 now)
  reference: https://arxiv.org/pdf/1412.3555.pdf

  theta:

  - w_n: the parameter weight matrix for the input block.
  - w_u: the parameter weight matrix for the update gate
  - w_r: the parameter weight matrix for the reset gate
  - b_n: the bias vector for the input block
  - b_u: the bias vector for the update gate
  - b_r: the bias vector for the reset gate

  state:

  - m: the GRU output. [batch, output_cell_nodes]
  - c: the GRU cell state. [batch, hidden_cell_nodes]

  inputs:

  - act: a list of input activations. [batch, input_nodes]
  - padding: the padding. [batch, 1].
  - reset_mask: optional 0/1 float input to support packed input training.
    Shape [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(GRUCell, cls).Params()
    p.Define(
        'num_hidden_nodes', 0, 'Number of projection hidden nodes '
        '(see https://arxiv.org/abs/1603.08042). '
        'Set to 0 to disable projection.')
    p.Define(
        'cell_value_cap', 10.0, 'GRU cell values are capped to be within '
        ' [-cell_value_cap, +cell_value_cap] if the value is not None. '
        'It can be a scalar, a scalar tensor or None. When set to None, '
        'no capping is applied.')
    p.Define('enable_gru_bias', False, 'Enable the GRU Cell bias.')
    p.Define('bias_init', py_utils.WeightInit.Constant(0.0),
             'Initialization parameters for GRU Cell bias')
    p.Define('zo_prob', 0.0,
             'If > 0, applies ZoneOut regularization with the given prob.')
    p.Define('apply_layer_norm', True, 'Apply layer norm to the variables')
    p.Define(
        'layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.'
        'value is necessary only if apply_layer_norm is True')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes GRUCell."""
    super(GRUCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert isinstance(p.cell_value_cap,
                      (int, float)) or p.cell_value_cap is None
    assert p.zo_prob == 0.0

    def CreateVarHelper(variable_name, shape_to_init, params_to_init):
      """Utility function to initialize variables.

      Args:
        variable_name: the name of the variable
        shape_to_init: shape of the variables to be initialized.
        params_to_init: p.params_init, p.bias_init, or otherwise specified
      Returns:
        initialized variable with name "$variable_name".
      """
      return self.CreateVariable(
          variable_name,
          py_utils.WeightParams(
              shape=shape_to_init,
              init=params_to_init,
              dtype=p.dtype,
              collections=self._VariableCollections()), self.AddGlobalVN)

    with tf.variable_scope(p.name):
      # Define weights.
      # Weight for block input
      CreateVarHelper('w_n',
                      [p.num_input_nodes + self.output_size, self.hidden_size],
                      p.params_init)
      # Weight for update gate
      CreateVarHelper('w_u',
                      [p.num_input_nodes + self.output_size, self.hidden_size],
                      p.params_init)
      # Weight for reset gate
      CreateVarHelper('w_r',
                      [p.num_input_nodes + self.output_size, self.output_size],
                      p.params_init)

      if p.num_hidden_nodes:
        # Set up projection matrix
        CreateVarHelper('w_proj', [self.hidden_size, self.output_size],
                        p.params_init)
        CreateVarHelper('b_proj', [self.output_size], p.bias_init)

      if p.enable_gru_bias:
        # Bias for the block input
        CreateVarHelper('b_n', [self.hidden_size], p.bias_init)
        # Bias for update gate
        CreateVarHelper('b_u', [self.hidden_size], p.bias_init)
        # Bias for the reset gate
        CreateVarHelper('b_r', [self.output_size], p.bias_init)

      if p.apply_layer_norm:
        assert p.layer_norm_epsilon is not None
        ln_unit = py_utils.WeightInit.Constant(0.0)
        CreateVarHelper('bn_ln_scale', [self.hidden_size], ln_unit)
        CreateVarHelper('bu_ln_scale', [self.hidden_size], ln_unit)
        CreateVarHelper('br_ln_scale', [self.output_size], ln_unit)

      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  @property
  def hidden_size(self):
    return self.params.num_hidden_nodes or self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = py_utils.InitRNNCellState((batch_size, self.output_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    zero_c = py_utils.InitRNNCellState((batch_size, self.hidden_size),
                                       init=p.zero_state_init_params,
                                       dtype=py_utils.FPropDtype(p),
                                       is_eval=self.do_eval)
    return py_utils.NestedMap(m=zero_m, c=zero_c)

  def _ResetState(self, state, inputs):
    state.m = inputs.reset_mask * state.m
    state.c = inputs.reset_mask * state.c
    return state

  def GetOutput(self, state):
    return state.m

  def LayerNorm(self, x, scale):
    """Applies layer normalization on the last dimension of 'x'.

    Args:
      x: activation tensor, where the last dimension represents channels.
      scale: the scale tensor of the layer normalization

    Returns:
      Layer normalized 'x', with the same shape as the input.
    """
    p = self.params
    mean = tf.reduce_mean(x, axis=[1], keepdims=True)
    centered = x - mean
    variance = tf.reduce_mean(tf.square(centered), axis=[1], keepdims=True)
    normed = centered * tf.math.rsqrt(variance + p.layer_norm_epsilon)
    return normed * scale

  def FProp(self, theta, state0, inputs):
    """Forward function.

    GRU has coupled reset gate in the candidate actiavation function for output.
    See equation 5 and above in https://arxiv.org/pdf/1412.3555.pdf.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      state0: The previous recurrent state. A `.NestedMap`.
      inputs: The inputs to the cell. A `.NestedMap`.

    Returns:
      A tuple (state1, extras).
      - state1: The next recurrent state. A `.NestedMap`.
      - extras: Intermediate results to faciliate backprop. A `.NestedMap`.
    """

    p = self.params
    assert isinstance(inputs.act, list)

    # Update all gates
    # Compute r_g. r_g has size [batch, output]
    r_g = tf.matmul(tf.concat(inputs.act + [state0.m], 1), theta.w_r)
    if p.apply_layer_norm:
      r_g = self.LayerNorm(r_g, theta.br_ln_scale + 1.0)
    if p.enable_gru_bias:
      r_g = r_g + theta.b_r
    r_g = tf.sigmoid(r_g)

    # Compute u_g and n_g. Both have size [batch, hidden].
    # u_g has size [batch, hidden]
    u_g = tf.matmul(tf.concat(inputs.act + [state0.m], 1), theta.w_u)
    # size of n_g is [batch, hidden]
    n_g = tf.matmul(
        tf.concat(inputs.act + [tf.multiply(r_g, state0.m)], 1), theta.w_n)
    if p.apply_layer_norm:
      u_g = self.LayerNorm(u_g, theta.bu_ln_scale + 1.0)
      n_g = self.LayerNorm(n_g, theta.bn_ln_scale + 1.0)
    if p.enable_gru_bias:  # Add biases to u_g and n_g if needed
      u_g = u_g + theta.b_u
      n_g = n_g + theta.b_n

    u_g = tf.sigmoid(u_g)
    n_g = tf.tanh(n_g)

    new_c = (1.0 - u_g) * (state0.c) + u_g * n_g

    # Clip the cell states to reasonable value.
    if p.cell_value_cap is not None:
      new_c = py_utils.clip_by_value(new_c, -p.cell_value_cap, p.cell_value_cap)

    # Apply non-linear output is necessary
    new_m = new_c
    # Apply projection matrix if necessary
    if p.num_hidden_nodes:
      new_m = tf.matmul(new_m, theta.w_proj) + theta.b_proj
    # Apply padding.
    new_m = py_utils.ApplyPadding(inputs.padding, new_m, state0.m)
    new_c = py_utils.ApplyPadding(inputs.padding, new_c, state0.c)
    return py_utils.NestedMap(m=new_m, c=new_c), py_utils.NestedMap()


class RandomVanillaRNN(RNNCell):
  """ESN cell.

  state:
    m: the lstm output. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(RandomVanillaRNN, cls).Params()
    p.Define('output_mode', 'non_linear', 'non_linear, linear, or mixture')
    p.Define('linear_percent', 0.0, 'In output_mode=mixture, the percentage of '
             'linear dimensions.')
    p.Define('leak_weight', 1.0, 'Weight of the leak connection.')
    p.Define('init_distribution', 'uniform', 'Distribution for initialization.')
    p.Define('input_scale', 2.0, 'Scale of the input matrix.')
    p.Define('spectral_radius', 0.9, 'Spectral radius of this layer.')
    p.Define('input_sparsity', 0.8, 'Percentage of 0 values in the '
             'reservoir.')
    p.Define('hidden_sparsity', 0.2, 'Percentage of 0 values in the '
             'reservoir.')
    p.Define('train_scale', False, 'Train input scale or not')
    p.Define('train_radius', False, 'Train spectral radius or not')
    p.Define('init_scale', 500, 'dummy params not used')
    p.Define('init_radius', 2.0, 'dummy params not used')
    p.Define('trainable', False, 'Train matrices or not')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    p.Define('layer_norm', False, 'Apply layer normalization or not')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNCell."""
    super(RandomVanillaRNN, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params

    if p.init_distribution == 'gaussian':
      init_esnin = py_utils.WeightInit.ESNInGaussian(scale=p.input_sparsity)
      init_esnres = py_utils.WeightInit.ESNResGaussian(scale=p.hidden_sparsity)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity, input_scale=scale_t)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity, spectral_radius=radius_t)
    else:
      init_esnin = p.params_init
      init_esnres = p.params_init

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      # Input
      w_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable)

      # Reservoir
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable)

      if p.train_scale:
        _HistogramSummary(scope.name + '/input_scale', self.vars.wscale)
      if p.train_radius:
        _HistogramSummary(scope.name + '/spectral_radius', self.vars.wradius)
      # bias_pc = w_pc.Copy()
      # bias_pc.shape = [p.num_output_nodes]
      # bias_pc.init = p.params_init
      # self.CreateVariable('b', bias_pc, trainable=False)
      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def GetOutput(self, state):
    return state.m

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = tf.zeros((batch_size, self.output_size), dtype=p.dtype)
    return py_utils.NestedMap(m=zero_m)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    i = theta.win
    h = theta.wr
    out = (py_utils.Matmul(inputs.act[0], i) + py_utils.Matmul(state0.m, h))
    return out

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim]
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      new_m = tf.tanh(value)
    elif p.output_mode == 'linear':
      new_m = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      new_m = tf.concat([linear_part, nonlinear_part], axis=1)

    new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    return py_utils.NestedMap(m=new_m)


class FeedbackESNCell(RNNCell):
  """ESN cell.

  state:
    m: the lstm output. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(FeedbackESNCell, cls).Params()
    p.Define('output_mode', 'non_linear', 'non_linear, linear, or mixture')
    p.Define('linear_percent', 0.0, 'In output_mode=mixture, the percentage of '
             'linear dimensions.')
    p.Define('leak_weight', 1.0, 'Weight of the leak connection.')

    p.Define('input_scale', 2.0, 'Scale of the input matrix.')
    p.Define('param_g', 0.9, 'Sclaing parameter g')
    p.Define('param_j', 0.9, 'Sclaing parameter j')

    p.Define('spectral_radius', 0.9, 'spectral radius')

    p.Define('input_sparsity', 0.8, 'Percentage of 0 values in the '
             'reservoir.')
    p.Define('hidden_sparsity', 0.8, 'Percentage of 0 values in the '
             'reservoir.')

    p.Define('train_scale', False, 'Train input scale or not')
    p.Define('train_radius', False, 'Train radius or not')
    p.Define('train_g', False, 'Train spectral radius or not')
    p.Define('train_j', False, 'Train spectral radius or not')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNCell."""
    super(FeedbackESNCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    init_esnin = py_utils.WeightInit.ESNInUniform(scale=p.input_sparsity)
    init_esnres = py_utils.WeightInit.ESNResUniform(scale=p.hidden_sparsity)
    init_u = py_utils.WeightInit.GaussianSqrtDim()
    # init_v = py_utils.WeightInit.GaussianSqrtDim()
    init_v = py_utils.WeightInit.UniformUnitScaling()

    with tf.variable_scope(p.name) as scope:
      # Define weights.

      scale_pc = py_utils.WeightParams([],
                                       py_utils.WeightInit.Constant(
                                           p.input_scale), tf.float32,
                                       self._VariableCollections())
      if p.train_scale:
        self.CreateVariable('wscale', scale_pc, theta_fn=None, trainable=True)
      else:
        self.CreateVariable('wscale', scale_pc, theta_fn=None, trainable=False)

      radius_pc = py_utils.WeightParams([],
                                        py_utils.WeightInit.Constant(
                                            p.spectral_radius), tf.float32,
                                        self._VariableCollections())
      if p.train_radius:
        self.CreateVariable('wradius', radius_pc, theta_fn=None, trainable=True)
      else:
        self.CreateVariable(
            'wradius', radius_pc, theta_fn=None, trainable=False)

      g_pc = py_utils.WeightParams([], py_utils.WeightInit.Constant(p.param_g),
                                   tf.float32, self._VariableCollections())
      if p.train_g:
        self.CreateVariable('wg', g_pc, theta_fn=None, trainable=True)
      else:
        self.CreateVariable('wg', g_pc, theta_fn=None, trainable=False)

      j_pc = py_utils.WeightParams([], py_utils.WeightInit.Constant(p.param_j),
                                   tf.float32, self._VariableCollections())
      if p.train_j:
        self.CreateVariable('wj', j_pc, theta_fn=None, trainable=True)
      else:
        self.CreateVariable('wj', j_pc, theta_fn=None, trainable=False)

      # Input
      w_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=False)

      # Hidden
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=False)

      # u and v
      u_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, 1],
          init=init_u,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('feed_u', u_pc, trainable=True)

      v_pc = py_utils.WeightParams(
          shape=[1, p.num_output_nodes],
          init=init_v,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('feed_v', v_pc, trainable=True)

      if p.train_scale:
        _HistogramSummary(scope.name + '/input_scale', self.vars.wscale)
      if p.train_radius:
        _HistogramSummary(scope.name + '/spectral_radius', self.vars.wradius)
      if p.train_g:
        _HistogramSummary(scope.name + '/param_g', self.vars.wg)
      if p.train_j:
        _HistogramSummary(scope.name + '/param_j', self.vars.wj)

      # bias_pc = w_pc.Copy()
      # bias_pc.shape = [p.num_output_nodes]
      # bias_pc.init = p.params_init
      # self.CreateVariable('b', bias_pc, trainable=False)
      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def GetOutput(self, state):
    return state.m

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = tf.zeros((batch_size, self.output_size), dtype=p.dtype)
    return py_utils.NestedMap(m=zero_m)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    inc = tf.multiply(theta.wscale, theta.win)
    wh = tf.multiply(theta.wradius, theta.wr)
    p = tf.multiply(theta.wg, wh)
    i = tf.multiply(theta.wj, tf.eye(self.params.num_output_nodes))
    # d = tf.matmul(theta.feed_u, tf.transpose(theta.feed_v))
    d = tf.matmul(theta.feed_u, theta.feed_v)
    h = p + i + d
    return py_utils.Matmul(inputs.act[0], inc) + py_utils.Matmul(state0.m, h)

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim]
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      new_m = tf.tanh(value)
    elif p.output_mode == 'linear':
      new_m = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      new_m = tf.concat([linear_part, nonlinear_part], axis=1)

    new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    return py_utils.NestedMap(m=new_m)


class ESNCell(RNNCell):
  """ESN cell.

  state:
    m: the lstm output. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(ESNCell, cls).Params()
    p.Define('output_mode', 'non_linear', 'non_linear, linear, or mixture')
    p.Define('linear_percent', 0.0, 'In output_mode=mixture, the percentage of '
             'linear dimensions.')
    p.Define('leak_weight', 1.0, 'Weight of the leak connection.')
    p.Define('init_distribution', 'uniform', 'Distribution for initialization.')
    p.Define('input_scale', 2.0, 'Scale of the input matrix.')
    p.Define('spectral_radius', 0.9, 'Spectral radius of this layer.')
    p.Define('input_sparsity', 0.8, 'Percentage of 0 values in the '
             'reservoir.')
    p.Define('hidden_sparsity', 0.2, 'Percentage of 0 values in the '
             'reservoir.')
    p.Define('train_scale', False, 'Train input scale or not')
    p.Define('train_radius', False, 'Train spectral radius or not')
    p.Define('init_scale', 500, 'dummy params not used')
    p.Define('init_radius', 2.0, 'dummy params not used')
    p.Define('trainable', False, 'Train matrices or not')
    p.Define('layer_norm_epsilon', 1e-8, 'Tiny value to guard rsqr against.')
    p.Define('layer_norm', False, 'Apply layer normalization or not')
    p.Define('wr_sigma', None, 'Largest singular value of reservoir matrix')
    p.Define('win_sigma', None, 'Largest singular value of input matrix')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNCell."""
    super(ESNCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    millis = int(round(time.time() * 1000))
    if p.init_distribution == 'gaussian':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInGaussian(
          scale=p.input_sparsity, input_scale=scale_t, wt_seed=millis)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResGaussian(
          scale=p.hidden_sparsity, spectral_radius=radius_t, wt_seed=millis + 1)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity, input_scale=scale_t, wt_seed=millis + 2)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity, spectral_radius=radius_t, wt_seed=millis + 3)

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      if p.train_scale:
        scale_pc = py_utils.WeightParams([],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      # Input matrix
      w_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable)

      # Reservoir (Hidden-Hidden matrix)
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable)

      if p.layer_norm:
        # Set up layer normalization variables
        bias_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=params.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN, trainable=True)

        ln_scale_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'ln_scale', ln_scale_pc, self.AddGlobalVN, trainable=True)

      if p.train_scale:
        summary_utils.scalar(scope.name + '/input_scale',
                             tf.cast(self.vars.scale, tf.float32))
      if p.train_radius:
        summary_utils.scalar(scope.name + '/radius',
                             tf.cast(self.vars.radius, tf.float32))

      # bias_pc = w_pc.Copy()
      # bias_pc.shape = [p.num_output_nodes]
      # bias_pc.init = p.params_init
      # self.CreateVariable('b', bias_pc, trainable=False)
      self._timestep = -1

  @property
  def output_size(self):
    return self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def GetOutput(self, state):
    return state.m

  def zero_state(self, theta, batch_size):
    p = self.params
    zero_m = tf.zeros((batch_size, self.output_size), dtype=p.dtype)
    return py_utils.NestedMap(m=zero_m)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    p = self.params
    if p.train_scale:
      i = tf.multiply(theta.scale, theta.win)
    else:
      i = theta.win
    if self.params.train_radius:
      h = tf.multiply(theta.radius, theta.wr)
    else:
      h = theta.wr
    out = (py_utils.Matmul(inputs.act[0], i) + py_utils.Matmul(state0.m, h))

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    if self.params.layer_norm:
      out = _LayerNorm(out)
      # Apply shift and gain parameters
      out = out * theta.ln_scale + theta.b

    return out

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim]
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      new_m = tf.tanh(value)
    elif p.output_mode == 'linear':
      new_m = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      new_m = tf.concat([linear_part, nonlinear_part], axis=1)

    new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    return py_utils.NestedMap(m=new_m)


class ESNCellactNN(ESNCell):
  """ESN cell with neural network as activation functions."""

  @classmethod
  def Params(cls):
    p = super(ESNCellactNN, cls).Params()
    p.Define('weight_topology_res', 'random',
             'types=random/scr-l,u,b,bd/eye/chain.')
    p.Define('weight_topology_in', 'random', 'types=random/scr-l,u. Input wt')
    p.Define('hidden_dim_actnn', 5, 'Num of Hidden units in activation NN.')
    p.Define('gx_input_dim', 1, 'Num input units in actNN: g(x), g(x, tanh(x))')
    p.Define('nn_layer2', False, 'True: use 2 layer NN')
    p.Define('trainable_in', False, 'Train input matrix or not')
    p.Define('trainable_res', False, 'Train res matrix or not')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNCellactNN."""
    super(ESNCellactNN, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    # same seed for random matrix and random mask
    millis = int(round(time.time() * 1000))
    if p.init_distribution == 'gaussian':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInGaussian(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.weight_topology_in,
          get_mask=False,
          wt_seed=millis + 25)
      if p.trainable_in:
        init_esnin_mask = py_utils.WeightInit.ESNInGaussian(
            scale=p.input_sparsity,
            input_scale=scale_t,
            weight_topology_in=p.weight_topology_in,
            get_mask=True,
            wt_seed=millis + 25)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResGaussian(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.weight_topology_res,
          wt_seed=millis + 50)
      if p.trainable_res:
        init_esnres_mask = py_utils.WeightInit.ESNResGaussian(
            scale=p.hidden_sparsity,
            spectral_radius=radius_t,
            weight_topology_res=p.weight_topology_res,
            get_mask=True,
            wt_seed=millis + 50)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.weight_topology_in,
          wt_seed=millis + 75)
      if p.trainable_in:
        init_esnin_mask = py_utils.WeightInit.ESNInUniform(
            scale=p.input_sparsity,
            input_scale=scale_t,
            weight_topology_in=p.weight_topology_in,
            get_mask=True,
            wt_seed=millis + 75)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.weight_topology_res,
          wt_seed=millis + 100)
      if p.trainable_res:
        init_esnres_mask = py_utils.WeightInit.ESNResUniform(
            scale=p.hidden_sparsity,
            spectral_radius=radius_t,
            weight_topology_res=p.weight_topology_res,
            get_mask=True,
            wt_seed=millis + 100)

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      if p.train_scale:
        scale_pc = py_utils.WeightParams([],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      # Input matrix
      w_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable_in)

      if p.trainable_in:
        # Input matrix mask
        w_pc_mask = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin_mask,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('win_mask', w_pc_mask, trainable=False)

      # Reservoir (Hidden-Hidden matrix)
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable_res)

      if p.trainable_res:
        # Reservoir (Hidden-Hidden matrix mask)
        w_pc_mask = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_esnres_mask,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('wr_mask', w_pc_mask, trainable=False)

      if p.layer_norm:
        # Set up layer normalization variables
        bias_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=params.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN, trainable=True)

        ln_scale_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'ln_scale', ln_scale_pc, self.AddGlobalVN, trainable=True)

      if p.train_scale:
        summary_utils.scalar(scope.name + '/input_scale',
                             tf.cast(self.vars.scale, tf.float32))
      if p.train_radius:
        summary_utils.scalar(scope.name + '/radius',
                             tf.cast(self.vars.radius, tf.float32))
      self._timestep = -1
      # initializing the activation function
      self.init_actNN_layer_functions(p, scope)

  def init_actNN_layer_functions(self, p, scope):
    """Function implemented using import layers ."""
    hidden_size = p.hidden_dim_actnn
    # layer 1
    fc1 = layers.FCLayer.Params().Set(
        input_dim=p.gx_input_dim,
        output_dim=hidden_size,
        activation='RELU',  # RELU/TANH
        trainable=True)

    self.CreateChild('fc1', fc1)
    if p.nn_layer2:
      # layer 2
      fc2 = layers.FCLayer.Params().Set(
          input_dim=hidden_size,
          output_dim=hidden_size,
          activation='RELU',  # NONE/SIGMOID
          trainable=True)
      self.CreateChild('fc2', fc2)
    # final layer
    fcn = layers.FCLayer.Params().Set(
        input_dim=hidden_size,
        output_dim=1,
        activation='NONE',  # NONE/SIGMOID
        trainable=True)
    self.CreateChild('fcn', fcn)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    p = self.params
    if p.train_scale:
      i = tf.multiply(theta.scale, theta.win)
    else:
      i = theta.win
    if self.params.train_radius:
      h = tf.multiply(theta.radius, theta.wr)
    else:
      h = theta.wr
    # masking if train= True
    if p.trainable_in:
      i = i * theta.win_mask
    if p.trainable_res:
      h = h * theta.wr_mask
    # add the masking operation with the topology. h = (HxH) * mask (binary)
    out = (py_utils.Matmul(inputs.act[0], i) + py_utils.Matmul(state0.m, h))

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    if self.params.layer_norm:
      out = _LayerNorm(out)
      # Apply shift and gain parameters
      out = out * theta.ln_scale + theta.b
      # print('OUT: ', out.shape) # (?, 512)
    return out

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim] ; out_dim = hidden state dim
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      new_m = tf.tanh(value)
    elif p.output_mode == 'linear':
      new_m = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      new_m = tf.concat([linear_part, nonlinear_part], axis=1)
    elif p.output_mode == 'actNN':  # neural network based activation
      if p.gx_input_dim == 1:  # y = g(x)
        in_act = tf.expand_dims(value, -1)  # [batch, out_dim, 1]
      elif p.gx_input_dim == 2:  # y = g(x, tanh(x))
        in_act_x = tf.expand_dims(value, -1)
        in_act_tanhx = tf.tanh(in_act_x)
        in_act = tf.concat([in_act_x, in_act_tanhx],
                           axis=-1)  # [batch,out_dim,2]
      hidden_act = self.fc1.FProp(theta.fc1, in_act)
      if p.nn_layer2:
        hidden_act = self.fc2.FProp(theta.fc2, hidden_act)
      out_act = self.fcn.FProp(theta.fcn, hidden_act)

      new_m = tf.squeeze(out_act, axis=-1)  # reduce the extra dim introduced
    new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    return py_utils.NestedMap(m=new_m)


class EntrywiseRNN(ESNCellactNN):
  """Entrywise RNN design."""

  @classmethod
  def Params(cls):
    p = super(EntrywiseRNN, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes EntrywiseRNN."""
    super(EntrywiseRNN, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    p.output_mode = 'non_linear'
    # same seed for random matrix and random mask
    millis = int(round(time.time() * 1000))
    if p.init_distribution == 'gaussian':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInGaussian(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.weight_topology_in,
          get_mask=False,
          wt_seed=millis + 25)
      if p.trainable_in:
        init_esnin_mask = py_utils.WeightInit.ESNInGaussian(
            scale=p.input_sparsity,
            input_scale=scale_t,
            weight_topology_in=p.weight_topology_in,
            get_mask=True,
            wt_seed=millis + 25)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius

      init_esnres = py_utils.WeightInit.ESNResGaussian(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.weight_topology_res,
          wt_seed=millis + 50)
      if p.trainable_res:
        init_esnres_mask = py_utils.WeightInit.ESNResGaussian(
            scale=p.hidden_sparsity,
            spectral_radius=radius_t,
            weight_topology_res=p.weight_topology_res,
            get_mask=True,
            wt_seed=millis + 50)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.weight_topology_in,
          wt_seed=millis + 75)
      if p.trainable_in:
        init_esnin_mask = py_utils.WeightInit.ESNInUniform(
            scale=p.input_sparsity,
            input_scale=scale_t,
            weight_topology_in=p.weight_topology_in,
            get_mask=True,
            wt_seed=millis + 75)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.weight_topology_res,
          wt_seed=millis + 100)
      if p.trainable_res:
        init_esnres_mask = py_utils.WeightInit.ESNResUniform(
            scale=p.hidden_sparsity,
            spectral_radius=radius_t,
            weight_topology_res=p.weight_topology_res,
            get_mask=True,
            wt_seed=millis + 100)

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      if p.train_scale:
        scale_pc = py_utils.WeightParams([],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      # Input matrix
      w_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable_in)

      if p.trainable_in:
        # Input matrix mask
        w_pc_mask = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin_mask,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('win_mask', w_pc_mask, trainable=False)

      # Reservoir (Hidden-Hidden matrix)
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable_res)

      if p.trainable_res:
        # Reservoir (Hidden-Hidden matrix mask)
        w_pc_mask = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_esnres_mask,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('wr_mask', w_pc_mask, trainable=False)

      if p.layer_norm:
        # Set up layer normalization variables
        bias_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=params.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN, trainable=True)

        ln_scale_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'ln_scale', ln_scale_pc, self.AddGlobalVN, trainable=True)

      if p.train_scale:
        summary_utils.scalar(scope.name + '/input_scale',
                             tf.cast(self.vars.scale, tf.float32))
      if p.train_radius:
        summary_utils.scalar(scope.name + '/radius',
                             tf.cast(self.vars.radius, tf.float32))

      self._timestep = -1
      # initializing the activation function
      self.init_actNN_layer_functions(p, scope)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    p = self.params
    if p.train_scale:
      i = tf.multiply(theta.scale, theta.win)
    else:
      i = theta.win
    if self.params.train_radius:
      h = tf.multiply(theta.radius, theta.wr)
    else:
      h = theta.wr
    # masking if train= True
    if p.trainable_in:
      i = i * theta.win_mask
    if p.trainable_res:
      h = h * theta.wr_mask

    in1 = py_utils.Matmul(inputs.act[0], i)
    in1 = tf.expand_dims(in1, -1)  # [batch, out_dim, 1]
    in2 = py_utils.Matmul(state0.m, h)
    in2 = tf.expand_dims(in2, -1)  # [batch, out_dim, 1]
    in_net = tf.concat([in1, in2], axis=-1)  # [batch,out_dim,2]
    hidden1 = self.fc1.FProp(theta.fc1, in_net)
    if p.nn_layer2:
      hidden1 = self.fc2.FProp(theta.fc2, hidden1)
    out = self.fcn.FProp(theta.fcn, hidden1)
    out = tf.squeeze(out, axis=-1)

    # print('CHECK: out ', out)
    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    if self.params.layer_norm:
      out = _LayerNorm(out)
      # Apply shift and gain parameters
      out = out * theta.ln_scale + theta.b

    return out


class ESNcellBasisForloop(ESNCell):
  """ESN cell which ensembles of all the scrl basis.

     1. Use the same topology for w_in and w_res. TODO: REMOVE all for loops
     2. Learn the scaling and eigenvalue scalar for basis.
     3. NO layer normalization
     4. Combine different topologies using non-linear functions.
     5. Faster implementation using the Roll function + Hadamard product.
  """

  @classmethod
  def Params(cls):
    p = super(ESNcellBasisForloop, cls).Params()
    p.Define('init_weight_vector', 'random_basis', 'types=random_basis')
    p.Define('hidden_dim_actnn', 5, 'Num of Hidden units in activation NN.')
    p.Define('nn_layer2', False, 'True: use 2 layer NN')
    p.Define('trainable_in', False, 'Train input matrix or not')
    p.Define('trainable_res', False, 'Train res matrix or not')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNcellBasisForloop."""
    super(ESNcellBasisForloop, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    self.num_basis = p.num_output_nodes  # number of basis
    self.num_tp = self.num_basis
    print('ESN Basis cell called: ', self.num_basis)

    # same seed for random matrix and random mask
    millis = int(round(time.time() * 1000)) + 1000
    if p.init_distribution == 'gaussian':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInGaussian(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.init_weight_vector,
          wt_seed=millis + 25)  # pass seed for random topology
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResGaussian(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.init_weight_vector,
          wt_seed=millis + 50)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.init_weight_vector,
          wt_seed=millis + 75)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.init_weight_vector,
          wt_seed=millis + 100)

    if p.layer_norm:
      init_bias = py_utils.WeightInit.Constant(0.0)
      init_ln_scale = py_utils.WeightInit.Constant(1.0)

    with tf.variable_scope(p.name) as scope:
      # Input matrix
      if p.num_input_nodes <= p.num_output_nodes:
        # to take care of non-square input matrix
        repeat_in_dim = 1
      else:
        repeat_in_dim = math.ceil(p.num_input_nodes / p.num_output_nodes)

      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes * repeat_in_dim, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable_in)

      # Reservoir (Hidden-Hidden matrix)
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable_res)

      if p.train_scale:
        scale_pc = py_utils.WeightParams([1, p.num_output_nodes],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([1, p.num_output_nodes],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      if p.layer_norm:
        # Set up layer normalization variables
        bias_pc = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_bias,  # py_utils.WeightInit.Constant(0.0),
            dtype=params.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN, trainable=True)

        ln_scale_pc = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_ln_scale,  # py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'ln_scale', ln_scale_pc, self.AddGlobalVN, trainable=True)

      if p.train_scale:  # to be done: this should be in loop over TPs
        summary_utils.scalar(scope.name + '/input_scale',
                             tf.cast(self.vars['scale'], tf.float32))
      if p.train_radius:
        summary_utils.scalar(scope.name + '/radius',
                             tf.cast(self.vars['radius'], tf.float32))

      self._timestep = -1
      # initializing the activation function
      self.ensemble_NN(p, scope)

  def ensemble_NN(self, p, scope):
    """Function implemented using import layers ."""
    hidden_size = p.hidden_dim_actnn
    # layer 1
    fc1 = layers.FCLayer.Params().Set(
        input_dim=self.num_basis,
        output_dim=hidden_size,
        activation='RELU',  # RELU/TANH
        trainable=True)
    self.CreateChild('fc1', fc1)
    if p.nn_layer2:
      # layer 2
      fc2 = layers.FCLayer.Params().Set(
          input_dim=hidden_size,
          output_dim=hidden_size,
          activation='RELU',  # NONE/SIGMOID
          trainable=True)
      self.CreateChild('fc2', fc2)
    # final layer
    fcn = layers.FCLayer.Params().Set(
        input_dim=hidden_size,
        output_dim=1,
        activation='NONE',  # NONE/SIGMOID
        trainable=True)
    self.CreateChild('fcn', fcn)

  def _Mix(self, theta, state0, inputs):
    # Note: theta has access to all the CreateVariables.
    assert isinstance(inputs.act, list)
    p = self.params

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    def ReservoirOutBatch(hidden_vec, res_wt, res_dim):
      hidden_vec_batch = []
      for _ in range(res_dim):
        hidden_vec = tf.roll(hidden_vec, shift=[0, 1], axis=[0, 1])
        hidden_vec_batch.append(tf.expand_dims(hidden_vec, -1))
      hidden_vec_batch = tf.concat(hidden_vec_batch, axis=-1)
      return hidden_vec_batch * res_wt  # [?, H, num_basis]

    def ProjectionOutBatch(input_vec, in_wt, in_dim, res_dim):
      """Get projection output."""
      if res_dim >= in_dim:  # case I:  res_dim >= in_dim
        # concat inputs to match the res_sim
        repeat = math.ceil(res_dim / in_dim) - 1
        print('repeat: ', repeat)
        for i in range(repeat):
          if i == repeat - 1:
            input_vec = tf.concat([
                input_vec, input_vec[:, :(res_dim - repeat * (in_dim - 1) - 1)]
            ],
                                  axis=-1)
          else:
            input_vec = tf.concat([input_vec, input_vec], axis=-1)
        input_vec_batch = []
        # rotate the input_vec
        for _ in range(res_dim):
          input_vec = tf.roll(input_vec, shift=[0, 1], axis=[-2, -1])
          input_vec_batch.append(tf.expand_dims(input_vec, -1))
        input_vec_batch = tf.concat(input_vec_batch, axis=-1)
        print('final in vec: ', input_vec_batch, in_wt)
        proj_vec = input_vec_batch * in_wt
      else:  # Case II: res_dim < in_dim -->  FIX
        # 1. reapeat the input res_dim times with rot
        # 2. Do the hadamard product 3. Reshape & tf.reduce ; return HxH
        repeat = math.ceil(in_dim / res_dim) - 1
        print('repeat num :', repeat, res_dim, in_dim, input_vec.shape)
        # # Step 1. pad zeros
        # Step 1. concat the Input with rotation
        input_vec_batch = []
        for _ in range(res_dim):
          input_vec = tf.roll(input_vec, shift=[0, 1], axis=[-2, -1])
          input_vec_batch.append(tf.expand_dims(input_vec, -1))
        input_vec_batch = tf.concat(input_vec_batch, axis=-1)  # B x I x H
        # Step 2. The Hadamard product
        proj_vec_temp = input_vec_batch * in_wt  # B x I x H
        print('intermediate shape: ', proj_vec_temp.shape)
        # Step 3. Reshape the proj_vec = B x H x H x (I/H)
        pv_final1 = proj_vec_temp[:, :res_dim, :res_dim]  # B x H x H
        for r in range(1, in_dim // res_dim):
          pv_final1 = pv_final1 + proj_vec_temp[:, r * res_dim:(r + 1) *
                                                res_dim, :res_dim]
        print('pv final1 shape: ', pv_final1.shape)
        if in_dim % res_dim != 0:  # last part, pad zeros and add
          pv_final2 = proj_vec_temp[:, res_dim * (in_dim // res_dim):in_dim, :
                                    res_dim]  # B x (I-H*r) x H
          paddings = tf.constant([[1, 1, 1], [1, 1, 2]])
          pv_final2 = tf.pad(pv_final2, paddings, mode='CONSTANT', name=None)
        proj_vec = pv_final2  # pv_final1 + pv_final2

      return proj_vec

    if p.train_scale:  # to be done : scale each of the t vectors.
      i = tf.multiply(theta['scale'], theta['win'])
    else:
      i = theta['win']
    if self.params.train_radius:
      h = tf.multiply(theta['radius'], theta['wr'])
    else:
      h = theta['wr']

    res_out = ReservoirOutBatch(state0.m, h, p.num_output_nodes)
    proj_out = ProjectionOutBatch(inputs.act[0], i, p.num_input_nodes,
                                  p.num_output_nodes)
    out = proj_out + res_out

    for t in range(self.num_basis):
      if self.params.layer_norm:
        out = _LayerNorm(out)
        out = out * theta['ln_scale' + str(t)] + theta['b' + str(t)]

    # print('OUT: ', out.shape) # (?, H, num_basis)
    return out

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim] ; out_dim = hidden state dim
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      in_act = tf.tanh(value)  # [?, H, num_basis]
    elif p.output_mode == 'linear':
      in_act = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      in_act = tf.concat([linear_part, nonlinear_part], axis=1)

    # apply the ensemble NN
    hidden_act = self.fc1.FProp(theta.fc1, in_act)
    if p.nn_layer2:
      hidden_act = self.fc2.FProp(theta.fc2, hidden_act)
    out_act = self.fcn.FProp(theta.fcn, hidden_act)

    new_m = tf.squeeze(out_act, axis=-1)  # reduce the extra dim introduced
    new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    return py_utils.NestedMap(m=new_m)


class ESNcellBasis3D(ESNcellBasisForloop):
  """ESN cell using a 3-way Tensor."""

  @classmethod
  def Params(cls):
    p = super(ESNcellBasis3D, cls).Params()
    p.Define('weight_topology_res', 'basis3D',
             'types=random/scr-l,u,b,bd/eye/rot.')
    p.Define('weight_topology_in', 'basis3D', 'types=random/scr-l,u. Input wt')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNcellBasis3D."""
    super(ESNcellBasis3D, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    self.num_basis = p.num_output_nodes  # number of basis
    self.num_tp = self.num_basis
    print('ESN Basis3D cell called: ', self.num_basis)
    # same seed for random matrix and random mask
    millis = int(round(time.time() * 1000))
    if p.init_distribution == 'gaussian':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInGaussian(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.weight_topology_in,
          get_mask=False,
          wt_seed=millis + 25)
      if p.trainable_in:
        init_esnin_mask = py_utils.WeightInit.ESNInGaussian(
            scale=p.input_sparsity,
            input_scale=scale_t,
            weight_topology_in=p.weight_topology_in,
            get_mask=True,
            wt_seed=millis + 25)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResGaussian(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.weight_topology_res,
          wt_seed=millis + 50)
      if p.trainable_res:
        init_esnres_mask = py_utils.WeightInit.ESNResGaussian(
            scale=p.hidden_sparsity,
            spectral_radius=radius_t,
            weight_topology_res=p.weight_topology_res,
            get_mask=True,
            wt_seed=millis + 50)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.weight_topology_in,
          wt_seed=millis + 75)
      if p.trainable_in:
        init_esnin_mask = py_utils.WeightInit.ESNInUniform(
            scale=p.input_sparsity,
            input_scale=scale_t,
            weight_topology_in=p.weight_topology_in,
            get_mask=True,
            wt_seed=millis + 75)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.weight_topology_res,
          wt_seed=millis + 100)
      if p.trainable_res:
        init_esnres_mask = py_utils.WeightInit.ESNResUniform(
            scale=p.hidden_sparsity,
            spectral_radius=radius_t,
            weight_topology_res=p.weight_topology_res,
            get_mask=True,
            wt_seed=millis + 100)

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      if p.train_scale:
        scale_pc = py_utils.WeightParams([1, self.num_basis],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([1, self.num_basis],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      # Input matrix
      w_pc = py_utils.WeightParams(
          shape=[p.num_input_nodes, p.num_output_nodes, self.num_basis],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable_in)

      if p.trainable_in:
        # Input matrix mask
        w_pc_mask = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes, self.num_basis],
            init=init_esnin_mask,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('win_mask', w_pc_mask, trainable=False)

      # Reservoir (Hidden-Hidden matrix)
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes, self.num_basis],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable_res)

      if p.trainable_res:
        # Reservoir (Hidden-Hidden matrix mask)
        w_pc_mask = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes, self.num_basis],
            init=init_esnres_mask,
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('wr_mask', w_pc_mask, trainable=False)

      if p.layer_norm:
        # Set up layer normalization variables
        bias_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(0.0),
            dtype=params.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('b', bias_pc, self.AddGlobalVN, trainable=True)

        ln_scale_pc = py_utils.WeightParams(
            shape=[1, p.num_output_nodes],
            init=py_utils.WeightInit.Constant(1.0),
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'ln_scale', ln_scale_pc, self.AddGlobalVN, trainable=True)

      if p.train_scale:
        summary_utils.scalar(scope.name + '/input_scale',
                             tf.cast(self.vars.scale, tf.float32))
      if p.train_radius:
        summary_utils.scalar(scope.name + '/radius',
                             tf.cast(self.vars.radius, tf.float32))

      self._timestep = -1
      # initializing the activation function
      self.ensemble_NN(p, scope)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    p = self.params
    if p.train_scale:
      i = tf.multiply(theta.scale, theta.win)
    else:
      i = theta.win
    if self.params.train_radius:
      h = tf.multiply(theta.radius, theta.wr)
    else:
      h = theta.wr
    # masking if train= True
    if p.trainable_in:
      i = i * theta.win_mask
    if p.trainable_res:
      h = h * theta.wr_mask

    def ReservoirOutBatch3D(hidden_vec, res_wt):
      # hidden_vec = BxH and res_wt = HxHxH
      res_wt = tf.transpose(res_wt, perm=[2, 1, 0])  # num_basis x H x H
      res_out = tf.matmul(res_wt, tf.transpose(hidden_vec))
      res_out = tf.transpose(res_out, perm=[2, 1, 0])
      return res_out  # [B, H, num_basis]

    def ProjectionOutBatch3D(input_vec, in_wt):
      # input_vec = BxI and in_wt = IxHxH
      proj_out = tf.matmul(
          tf.transpose(in_wt, perm=[2, 1, 0]), tf.transpose(input_vec))
      proj_out = tf.transpose(proj_out, perm=[2, 1, 0])
      return proj_out  # [B, H, num_basis]

    res_out = ReservoirOutBatch3D(state0.m, h)
    proj_out = ProjectionOutBatch3D(inputs.act[0], i)
    out = proj_out + res_out

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    if self.params.layer_norm:
      out = _LayerNorm(out)
      # Apply shift and gain parameters
      out = out * theta.ln_scale + theta.b
    return out  # (?, H, num_basis)


class ESNcellBasis(ESNcellBasis3D):
  """ESN cell which ensembles of all the scrl basis.

     1. Use the same topology for w_in and w_res.
     2. Learn the scaling and eigenvalue scalar for basis.
     3. NO layer normalization
     4. Combine different topologies using non-linear functions.
     5. Faster implementation using the Roll function + Hadamard product.
  """

  @classmethod
  def Params(cls):
    p = super(ESNcellBasis, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNcellBasis."""
    super(ESNcellBasis, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    self.num_basis = p.num_output_nodes  # number of basis
    self.num_tp = self.num_basis
    print('ESN Basis cell called: ', self.num_basis)

    # same seed for random matrix and random mask
    millis = int(round(time.time() * 1000)) + 1000
    if p.init_distribution == 'gaussian':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInGaussian(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.init_weight_vector,
          wt_seed=millis + 25)  # pass seed for random topology
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResGaussian(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.init_weight_vector,
          wt_seed=millis + 50)
    elif p.init_distribution == 'uniform':
      if p.train_scale:
        scale_t = 1.0
      else:
        scale_t = p.input_scale
      init_esnin = py_utils.WeightInit.ESNInUniform(
          scale=p.input_sparsity,
          input_scale=scale_t,
          weight_topology_in=p.init_weight_vector,
          wt_seed=millis + 75)
      if p.train_radius:
        radius_t = 1.0
      else:
        radius_t = p.spectral_radius
      init_esnres = py_utils.WeightInit.ESNResUniform(
          scale=p.hidden_sparsity,
          spectral_radius=radius_t,
          weight_topology_res=p.init_weight_vector,
          wt_seed=millis + 100)

    with tf.variable_scope(p.name) as scope:
      # Input matrix
      if p.num_input_nodes <= p.num_output_nodes:
        # to take care of non-square input matrix
        win_dim = p.num_output_nodes
      else:
        win_dim = p.num_input_nodes

      w_pc = py_utils.WeightParams(
          shape=[win_dim, p.num_output_nodes],
          init=init_esnin,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('win', w_pc, trainable=p.trainable_in)

      # Reservoir (Hidden-Hidden matrix)
      w_pc = py_utils.WeightParams(
          shape=[p.num_output_nodes, p.num_output_nodes],
          init=init_esnres,
          dtype=p.dtype,
          collections=self._VariableCollections())
      self.CreateVariable('wr', w_pc, trainable=p.trainable_res)

      if p.train_scale:
        scale_pc = py_utils.WeightParams([1, p.num_output_nodes],
                                         py_utils.WeightInit.Constant(
                                             p.input_scale), tf.float32,
                                         self._VariableCollections())
        self.CreateVariable('scale', scale_pc, theta_fn=None, trainable=True)
      if p.train_radius:
        radius_pc = py_utils.WeightParams([1, p.num_output_nodes],
                                          py_utils.WeightInit.Constant(
                                              p.spectral_radius), tf.float32,
                                          self._VariableCollections())
        self.CreateVariable('radius', radius_pc, theta_fn=None, trainable=True)

      if p.train_scale:  # to be done: this should be in loop over TPs
        summary_utils.scalar(scope.name + '/input_scale',
                             tf.cast(self.vars['scale'], tf.float32))
      if p.train_radius:
        summary_utils.scalar(scope.name + '/radius',
                             tf.cast(self.vars['radius'], tf.float32))

      self._timestep = -1
      # initializing the activation function
      self.ensemble_NN(p, scope)

  def _Mix(self, theta, state0, inputs):
    # Note: theta has access to all the CreateVariables.
    assert isinstance(inputs.act, list)
    p = self.params

    def AllShifts(hidden_state, h):
      # H = hidden_vec.shape[-1]
      repeated = tf.tile(hidden_state, [1, h + 1])
      out = tf.reshape(repeated, [-1, h, h + 1])[:, :, :h]
      return out

    def ReservoirOut(hidden_vec, res_wt, res_dim):
      # print('shapes: ', hidden_vec.shape, res_wt.shape)
      return AllShifts(hidden_vec, res_dim) * res_wt

    def ProjectionOut(input_vec, in_wt, in_dim, res_dim):
      """Projection output vector."""
      if res_dim >= in_dim:  # case I:  res_dim >= in_dim
        # concat inputs to match the res_sim
        repeat = math.ceil(res_dim / in_dim) - 1
        for i in range(repeat):
          if i == repeat - 1:
            input_vec = tf.concat(
                [input_vec, input_vec[:, :(res_dim - (repeat * in_dim))]],
                axis=-1)
          else:
            input_vec = tf.concat([input_vec, input_vec], axis=-1)
        input_vec_batch = AllShifts(input_vec, res_dim)
        proj_vec = input_vec_batch * in_wt
      else:  # Case II: res_dim < in_dim
        print('Case res_dim < in_dim not implemented', res_dim, in_dim)
      return proj_vec

    if p.train_scale:  # to be done: scale each of the t vectors.
      i = tf.multiply(theta['scale'], theta['win'])
    else:
      i = theta['win']
    if self.params.train_radius:
      h = tf.multiply(theta['radius'], theta['wr'])
    else:
      h = theta['wr']
    res_out = ReservoirOut(state0.m, h, p.num_output_nodes)
    proj_out = ProjectionOut(inputs.act[0], i, p.num_input_nodes,
                             p.num_output_nodes)
    out = proj_out + res_out
    # print('OUT************: ', out.shape)  # (?, H, num_basis)
    return out


class ESNcellTopology(ESNCell):
  """ESN cell which also ensembles between different topologies.

     1. Use the same topology for w_in and w_res.
     2. Learn the scaling and eigenvalue scalar for each topology.
     3. Have an option for layer normalization for each topology.
     4. Combine different topologies using non-linear functions.
     5. [TODO] Faster implementation using the Roll function + Hadamard product

  state:
    m: the lstm output. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(ESNcellTopology, cls).Params()
    p.Define('weight_topology_res', ['scrl', 'scru', 'eye'],
             'types=random/scr-l,u,b,bd/eye/chain.')
    p.Define('weight_topology_in', ['scrl', 'scru', 'eye'],
             'types=random/scr-l,u. Input wt')
    p.Define('hidden_dim_actnn', 5, 'Num of Hidden units in activation NN.')
    p.Define('nn_layer2', False, 'True: use 2 layer NN')
    p.Define('trainable_in', False, 'Train input matrix or not')
    p.Define('trainable_res', False, 'Train res matrix or not')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes ESNcellTopology."""
    super(ESNcellTopology, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert len(p.weight_topology_res) == len(p.weight_topology_in)
    self.num_tp = len(p.weight_topology_res)  # number of ensemble topologies
    print('ESN Topology cell called: Total topologies: ', self.num_tp)
    # Initialize the weights for different topologies
    init_esnin, init_esnin_mask = [], []
    init_esnres, init_esnres_mask = [], []
    init_bias, init_ln_scale = [], []
    for t in range(self.num_tp):
      # same seed for random matrix and random mask
      millis = int(round(time.time() * 1000)) + t * 1000
      if p.init_distribution == 'gaussian':
        if p.train_scale:
          scale_t = 1.0
        else:
          scale_t = p.input_scale
        init_esnin.append(
            py_utils.WeightInit.ESNInGaussian(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t],
                get_mask=False,
                wt_seed=millis + 25))  # pass seed for random topology
        if p.trainable_in:
          init_esnin_mask.append(
              py_utils.WeightInit.ESNInGaussian(
                  scale=p.input_sparsity,
                  input_scale=scale_t,
                  weight_topology_in=p.weight_topology_in[t],
                  get_mask=True,
                  wt_seed=millis + 25))
        if p.train_radius:
          radius_t = 1.0
        else:
          radius_t = p.spectral_radius
        init_esnres.append(
            py_utils.WeightInit.ESNResGaussian(
                scale=p.hidden_sparsity,
                spectral_radius=radius_t,
                weight_topology_res=p.weight_topology_res[t],
                wt_seed=millis + 50))
        if p.trainable_res:
          init_esnres_mask.append(
              py_utils.WeightInit.ESNResGaussian(
                  scale=p.hidden_sparsity,
                  spectral_radius=radius_t,
                  weight_topology_res=p.weight_topology_res[t],
                  get_mask=True,
                  wt_seed=millis + 50))
      elif p.init_distribution == 'uniform':
        if p.train_scale:
          scale_t = 1.0
        else:
          scale_t = p.input_scale
        init_esnin.append(
            py_utils.WeightInit.ESNInUniform(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t],
                wt_seed=millis + 75))
        if p.trainable_in:
          init_esnin_mask.append(
              py_utils.WeightInit.ESNInUniform(
                  scale=p.input_sparsity,
                  input_scale=scale_t,
                  weight_topology_in=p.weight_topology_in[t],
                  get_mask=True,
                  wt_seed=millis + 75))
        if p.train_radius:
          radius_t = 1.0
        else:
          radius_t = p.spectral_radius
        init_esnres.append(
            py_utils.WeightInit.ESNResUniform(
                scale=p.hidden_sparsity,
                spectral_radius=radius_t,
                weight_topology_res=p.weight_topology_res[t],
                wt_seed=millis + 100))
        if p.trainable_res:
          init_esnres_mask.append(
              py_utils.WeightInit.ESNResUniform(
                  scale=p.hidden_sparsity,
                  spectral_radius=radius_t,
                  weight_topology_res=p.weight_topology_res[t],
                  get_mask=True,
                  wt_seed=millis + 100))

      if p.layer_norm:
        init_bias.append(py_utils.WeightInit.Constant(0.0))
        init_ln_scale.append(py_utils.WeightInit.Constant(1.0))

    with tf.variable_scope(p.name) as scope:
      for t, (in_tp, res_tp) in enumerate(
          zip(p.weight_topology_in, p.weight_topology_res)):
        # Define weights.
        print(t, in_tp, res_tp)
        if p.train_scale:
          scale_pc = py_utils.WeightParams([],
                                           py_utils.WeightInit.Constant(
                                               p.input_scale), tf.float32,
                                           self._VariableCollections())
          self.CreateVariable(
              'scale' + str(t), scale_pc, theta_fn=None, trainable=True)
        if p.train_radius:
          radius_pc = py_utils.WeightParams([],
                                            py_utils.WeightInit.Constant(
                                                p.spectral_radius), tf.float32,
                                            self._VariableCollections())
          self.CreateVariable(
              'radius' + str(t), radius_pc, theta_fn=None, trainable=True)

        # Input matrix
        w_pc = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'win' + in_tp + str(t), w_pc, trainable=p.trainable_in)

        if p.trainable_in:
          # Input matrix mask
          w_pc_mask = py_utils.WeightParams(
              shape=[p.num_input_nodes, p.num_output_nodes],
              init=init_esnin_mask[t],
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'win_mask' + in_tp + str(t), w_pc_mask, trainable=False)

        # Reservoir (Hidden-Hidden matrix)
        w_pc = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_esnres[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'wr' + res_tp + str(t), w_pc, trainable=p.trainable_res)

        if p.trainable_res:
          # Reservoir (Hidden-Hidden matrix mask)
          w_pc_mask = py_utils.WeightParams(
              shape=[p.num_output_nodes, p.num_output_nodes],
              init=init_esnres_mask[t],
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'wr_mask' + res_tp + str(t), w_pc_mask, trainable=False)

        if p.layer_norm:
          # Set up layer normalization variables
          bias_pc = py_utils.WeightParams(
              shape=[1, p.num_output_nodes],
              init=init_bias[t],  # py_utils.WeightInit.Constant(0.0),
              dtype=params.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'b' + str(t), bias_pc, self.AddGlobalVN, trainable=True)

          ln_scale_pc = py_utils.WeightParams(
              shape=[1, p.num_output_nodes],
              init=init_ln_scale[t],  # py_utils.WeightInit.Constant(1.0),
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'ln_scale' + str(t),
              ln_scale_pc,
              self.AddGlobalVN,
              trainable=True)

        if p.train_scale:  # to be done: this should be in loop over TPs
          summary_utils.scalar(scope.name + '/input_scale',
                               tf.cast(self.vars['scale' + str(t)], tf.float32))
        if p.train_radius:
          summary_utils.scalar(
              scope.name + '/radius',
              tf.cast(self.vars['radius' + str(t)], tf.float32))

      self._timestep = -1
      if self.num_tp > 1:
        # initializing the ensemble function
        self.ensemble_NN(p, scope)

  def ensemble_NN(self, p, scope):
    """Function implemented using import layers ."""
    hidden_size = p.hidden_dim_actnn
    # layer 1
    fc1 = layers.FCLayer.Params().Set(
        input_dim=self.num_tp,
        output_dim=hidden_size,
        activation='RELU',  # RELU/TANH
        trainable=True)
    self.CreateChild('fc1', fc1)
    if p.nn_layer2:
      # layer 2
      fc2 = layers.FCLayer.Params().Set(
          input_dim=hidden_size,
          output_dim=hidden_size,
          activation='RELU',  # NONE/SIGMOID
          trainable=True)
      self.CreateChild('fc2', fc2)
    # final layer
    fcn = layers.FCLayer.Params().Set(
        input_dim=hidden_size,
        output_dim=1,
        activation='NONE',  # NONE/SIGMOID
        trainable=True)
    self.CreateChild('fcn', fcn)

  def _Mix(self, theta, state0, inputs):
    # Note: theta has access to all the CreateVariables.
    assert isinstance(inputs.act, list)
    p = self.params

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    # Concatenate the output of various topologies. (?, H, numTP)
    out = []
    for t in range(self.num_tp):
      in_tp, res_tp = p.weight_topology_in[t], p.weight_topology_res[t]
      if p.train_scale:
        # i = tf.multiply(theta.scale[t], theta.win[t])
        i = tf.multiply(theta['scale' + str(t)], theta['win' + in_tp + str(t)])
      else:
        i = theta['win' + in_tp + str(t)]
      if self.params.train_radius:
        h = tf.multiply(theta['radius' + str(t)], theta['wr' + res_tp + str(t)])
      else:
        h = theta['wr' + res_tp + str(t)]
      if p.trainable_in:
        # if in_tp in ['scrbd']:
        i = i * theta['win_mask' + in_tp + str(t)]

      if p.trainable_res:
        # if res_tp in ['scrbd']:
        h = h * theta['wr_mask' + res_tp + str(t)]

      outt = (py_utils.Matmul(inputs.act[0], i) + py_utils.Matmul(state0.m, h))

      if self.params.layer_norm:
        outt = _LayerNorm(outt)
        # Apply shift and gain parameters
        outt = outt * theta['ln_scale' + str(t)] + theta['b' + str(t)]
      out.append(tf.expand_dims(outt, -1))  # [?, H, 1]
    # convert the output list to tensor
    out = tf.concat(out, axis=-1)
    # print('OUT: ', out.shape) # (?, H, numTP)
    return out

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim] ; out_dim = hidden state dim
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      out_act = tf.tanh(value)  # [?, H, numTP]
    elif p.output_mode == 'linear':
      out_act = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      out_act = tf.concat([linear_part, nonlinear_part], axis=1)

    if self.num_tp > 1:
      # apply the ensemble NN
      hidden_act = self.fc1.FProp(theta.fc1, out_act)
      if p.nn_layer2:
        hidden_act = self.fc2.FProp(theta.fc2, hidden_act)
      out_act = self.fcn.FProp(theta.fcn, hidden_act)

    new_m = tf.squeeze(out_act, axis=-1)  # reduce the extra dim introduced
    new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    return py_utils.NestedMap(m=new_m)


class ESNcellSpectral(ESNcellTopology):
  """ESN cell with spectral learning.

  state:
    m: the lstm output. [batch, cell_nodes]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(ESNcellSpectral, cls).Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes. KEEP train_radius and train_scale = False."""
    super(ESNcellSpectral, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert len(p.weight_topology_res) == len(p.weight_topology_in)
    self.num_tp = len(p.weight_topology_res)  # number of ensemble topologies
    print('Spectral ensemble cell called: TOTAL topologies: ', self.num_tp)
    # Initialize the weights for different topologies
    init_esnin_u, init_esnin_s, init_esnin_v = [], [], []
    init_esnres_x, init_esnres_lamb = [], []
    init_bias, init_ln_scale = [], []
    for t in range(self.num_tp):
      # same seed for random matrix and random mask
      millis = int(round(time.time() * 1000)) + t * 1000
      if p.init_distribution == 'uniform':
        if p.train_scale:
          scale_t = 1.0
        else:
          scale_t = p.input_scale
          # Init U Sigma V: note, same seed used (taking care of random case)
        init_esnin_u.append(
            py_utils.WeightInit.ESNInUniform(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t] + '_U',
                wt_seed=millis))

        init_esnin_s.append(
            py_utils.WeightInit.ESNInUniform(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t] + '_S',
                wt_seed=millis))

        init_esnin_v.append(
            py_utils.WeightInit.ESNInUniform(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t] + '_V',
                wt_seed=millis))

        if p.train_radius:
          radius_t = 1.0
        else:
          radius_t = p.spectral_radius

        # init X Lambda X^T
        init_esnres_x.append(
            py_utils.WeightInit.ESNResUniform(
                scale=p.hidden_sparsity,
                spectral_radius=radius_t,
                weight_topology_res=p.weight_topology_res[t] + '_X',
                wt_seed=millis + 100))

        init_esnres_lamb.append(
            py_utils.WeightInit.ESNResUniform(
                scale=p.hidden_sparsity,
                spectral_radius=radius_t,
                weight_topology_res=p.weight_topology_res[t] + '_L',
                wt_seed=millis + 100))

        if p.layer_norm:
          init_bias.append(py_utils.WeightInit.Constant(0.0))
          init_ln_scale.append(py_utils.WeightInit.Constant(1.0))

    with tf.variable_scope(p.name) as scope:
      # Define weights.
      for t, (in_tp, res_tp) in enumerate(
          zip(p.weight_topology_in, p.weight_topology_res)):
        print(t, in_tp, res_tp)
        # considering Sigma = PxP where P=min(M, N)
        w_pc = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin_u[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('win_U' + in_tp + str(t), w_pc, trainable=False)

        w_pc = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin_s[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'win_S' + in_tp + str(t), w_pc,
            trainable=True)  # learn sigma = vector

        w_pc = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin_v[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('win_V' + in_tp + str(t), w_pc, trainable=False)

        # Reservoir (Hidden-Hidden matrix)
        w_pc = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_esnres_x[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('wr_X' + res_tp + str(t), w_pc, trainable=False)

        w_pc = py_utils.WeightParams(
            shape=[p.num_output_nodes, p.num_output_nodes],
            init=init_esnres_lamb[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable('wr_L' + res_tp + str(t), w_pc, trainable=True)

#         if p.layer_norm:
#           # Set up layer normalization variables
#           bias_pc = py_utils.WeightParams(
#               shape=[1, p.num_output_nodes],
#               init=init_bias[t],  # py_utils.WeightInit.Constant(0.0),
#               dtype=params.dtype,
#               collections=self._VariableCollections())
#           self.CreateVariable(
#               'ln_b_' + str(t), bias_pc, self.AddGlobalVN, trainable=True)

#           ln_scale_pc = py_utils.WeightParams(
#               shape=[1, p.num_output_nodes],
#               init=init_ln_scale[t],  # py_utils.WeightInit.Constant(1.0),
#               dtype=p.dtype,
#               collections=self._VariableCollections())
#           self.CreateVariable(
#               'ln_scale_' + str(t),
#               ln_scale_pc,
#               self.AddGlobalVN,
#               trainable=True)

      self._timestep = -1
      # if self.num_tp > 1:
      #   # initializing the ensemble NN function
      #   self.ensemble_NN(p, scope)

  def _Mix(self, theta, state0, inputs):
    assert isinstance(inputs.act, list)
    p = self.params

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    def ReservoirOut(hidden_vec, x, lambda_vec):
      # hidden vec = B x H, x = HxH
      wr = tf.matmul(x, tf.matmul(tf.linalg.diag(lambda_vec), tf.transpose(x)))
      return py_utils.Matmul(hidden_vec, wr)

    def ProjectionOut(input_vec, svd_u, sigma, svd_v):
      # inputvec = BxN , Win = MxN, svd_u = MxP, S = PxP,
      # svd_v = NxP where P=min(M, N)
      win = tf.matmul(svd_u,
                      tf.matmul(tf.linalg.diag(sigma), svd_v, adjoint_b=True))
      return py_utils.Matmul(input_vec, win)

    # Concatenate the output of various topologies. (?, H, numTP)
    out = []
    for t in range(self.num_tp):
      in_tp, res_tp = p.weight_topology_in[t], p.weight_topology_res[t]
      term1 = ProjectionOut(inputs.act[0], theta['win_U' + in_tp + str(t)],
                            theta['win_S' + in_tp + str(t)],
                            theta['win_V' + in_tp + str(t)])
      term2 = ReservoirOut(state0.m, theta['wr_X' + res_tp + str(t)],
                           theta['wr_L' + res_tp + str(t)])
      outt = term1 + term2

      if self.params.layer_norm:
        outt = _LayerNorm(outt)
        # Apply shift and gain parameters
        # outt = outt * theta['ln_scale_' + str(t)] + theta['ln_b_' + str(t)]
        outt = outt * theta['ln_scale' + str(t)] + theta['b' + str(t)]
        # print('OUT: ', out.shape) # (?, 512)
      out.append(tf.expand_dims(outt, -1))  # [?, H, 1]
    # convert the output list to tensor
    out = tf.concat(out, axis=-1)
    return out


class HigherOrderESNTopologyCell(ESNCell):
  """Higher Order ESN cell with different topologies.

     1. Use the same topology for w_in and w_res.
     2. Learn the scaling and eigenvalue scalar for each topology.
     3. Have an option for layer normalization for each topology.

  state:
    m: the HO-ESN output. [batch, cell_nodes, cell_nodes, ...]

  inputs:
    act: a list of input activations. [batch, input_nodes]
    padding: the padding. [batch, 1].
  """

  @classmethod
  def Params(cls):
    p = super(HigherOrderESNTopologyCell, cls).Params()
    p.Define('weight_topology_res', ['lap_chain'],
             'types=random/scr-l,u,b,bd/eye/chain.')
    p.Define('weight_topology_in', ['lap_chain'],
             'types=random/scr-l,u. Input wt')
    p.Define('hidden_dim_actnn', 5, 'Num of Hidden units in activation NN.')
    p.Define('nn_layer2', False, 'True: use 2 layer NN')
    p.Define('trainable_in', False, 'Train input matrix or not')
    p.Define('trainable_res', False, 'Train res matrix or not')
    p.Define('order', 2, 'The order of ESN')
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Initializes HigherOrderESNTopologyCell."""
    super(HigherOrderESNTopologyCell, self).__init__(params)
    assert isinstance(params, hyperparams.Params)
    p = self.params
    assert len(p.weight_topology_res) == len(p.weight_topology_in)
    self.num_tp = len(p.weight_topology_res)  # number of ensemble topologies
    print('ESN Topology cell called: Total topologies: ', self.num_tp)
    # Initialize the weights for different topologies
    init_esnin, init_esnin_mask = [], []
    init_esnres, init_esnres_mask = [], []
    init_bias, init_ln_scale = [], []
    for t in range(self.num_tp):
      # same seed for random matrix and random mask
      millis = int(round(time.time() * 1000)) + t * 1000
      if p.init_distribution == 'gaussian':
        if p.train_scale:
          scale_t = 1.0
        else:
          scale_t = p.input_scale
        init_esnin.append(
            py_utils.WeightInit.ESNInGaussian(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t],
                get_mask=False,
                wt_seed=millis + 25))  # pass seed for random topology
        if p.trainable_in:
          init_esnin_mask.append(
              py_utils.WeightInit.ESNInGaussian(
                  scale=p.input_sparsity,
                  input_scale=scale_t,
                  weight_topology_in=p.weight_topology_in[t],
                  get_mask=True,
                  wt_seed=millis + 25))
        if p.train_radius:
          radius_t = 1.0
        else:
          radius_t = p.spectral_radius

        init_esnres_od = {}
        for od in range(p.order):
          init_esnres_od[od] = py_utils.WeightInit.ESNResGaussian(
              scale=p.hidden_sparsity,
              spectral_radius=radius_t,
              weight_topology_res=p.weight_topology_res[t],
              wt_seed=millis + 50)
        init_esnres.append(init_esnres_od)  # init_enres[t][o]

        if p.trainable_res:
          init_esnres_mask_od = {}
          for od in range(p.order):
            init_esnres_mask_od[od] = py_utils.WeightInit.ESNResGaussian(
                scale=p.hidden_sparsity,
                spectral_radius=radius_t,
                weight_topology_res=p.weight_topology_res[t],
                get_mask=True,
                wt_seed=millis + 50)
          init_esnres_mask.append(init_esnres_mask_od)

      elif p.init_distribution == 'uniform':
        if p.train_scale:
          scale_t = 1.0
        else:
          scale_t = p.input_scale
        init_esnin.append(
            py_utils.WeightInit.ESNInUniform(
                scale=p.input_sparsity,
                input_scale=scale_t,
                weight_topology_in=p.weight_topology_in[t],
                wt_seed=millis + 75))
        if p.trainable_in:
          init_esnin_mask.append(
              py_utils.WeightInit.ESNInUniform(
                  scale=p.input_sparsity,
                  input_scale=scale_t,
                  weight_topology_in=p.weight_topology_in[t],
                  get_mask=True,
                  wt_seed=millis + 75))
        if p.train_radius:
          radius_t = 1.0
        else:
          radius_t = p.spectral_radius
        init_esnres_od = {}
        for od in range(p.order):
          init_esnres_od[od] = py_utils.WeightInit.ESNResGaussian(
              scale=p.hidden_sparsity,
              spectral_radius=radius_t,
              weight_topology_res=p.weight_topology_res[t],
              wt_seed=millis + 50)
        init_esnres.append(init_esnres_od)  # init_enres[t][o]
        if p.trainable_res:
          init_esnres_mask_od = {}
          for od in range(p.order):
            init_esnres_mask_od[od] = py_utils.WeightInit.ESNResGaussian(
                scale=p.hidden_sparsity,
                spectral_radius=radius_t,
                weight_topology_res=p.weight_topology_res[t],
                get_mask=True,
                wt_seed=millis + 50)
          init_esnres_mask.append(init_esnres_mask_od)

      if p.layer_norm:
        init_bias.append(py_utils.WeightInit.Constant(0.0))
        init_ln_scale.append(py_utils.WeightInit.Constant(1.0))

    with tf.variable_scope(p.name) as scope:
      for t, (in_tp, res_tp) in enumerate(
          zip(p.weight_topology_in, p.weight_topology_res)):
        # Define weights.
        print(t, in_tp, res_tp)
        if p.train_scale:
          scale_pc = py_utils.WeightParams([],
                                           py_utils.WeightInit.Constant(
                                               p.input_scale), tf.float32,
                                           self._VariableCollections())
          self.CreateVariable(
              'scale' + str(t), scale_pc, theta_fn=None, trainable=True)
        if p.train_radius:
          for od in range(p.order):
            radius_pc = py_utils.WeightParams(
                [], py_utils.WeightInit.Constant(p.spectral_radius), tf.float32,
                self._VariableCollections())
            self.CreateVariable(
                'radius' + str(t) + str(od),
                radius_pc,
                theta_fn=None,
                trainable=True)

        # Input matrix
        w_pc = py_utils.WeightParams(
            shape=[p.num_input_nodes, p.num_output_nodes],
            init=init_esnin[t],
            dtype=p.dtype,
            collections=self._VariableCollections())
        self.CreateVariable(
            'win' + in_tp + str(t), w_pc, trainable=p.trainable_in)

        if p.trainable_in:
          # Input matrix mask
          w_pc_mask = py_utils.WeightParams(
              shape=[p.num_input_nodes, p.num_output_nodes],
              init=init_esnin_mask[t],
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'win_mask' + in_tp + str(t), w_pc_mask, trainable=False)

        # Reservoir (Hidden-Hidden matrix)
        for od in range(p.order):
          w_pc = py_utils.WeightParams(
              shape=[p.num_output_nodes, p.num_output_nodes],
              init=init_esnres[t][od],
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'wr' + res_tp + str(t) + str(od), w_pc, trainable=p.trainable_res)

          if p.trainable_res:
            # Reservoir (Hidden-Hidden matrix mask)
            w_pc_mask = py_utils.WeightParams(
                shape=[p.num_output_nodes, p.num_output_nodes],
                init=init_esnres_mask[t][od],
                dtype=p.dtype,
                collections=self._VariableCollections())
            self.CreateVariable(
                'wr_mask' + res_tp + str(t) + str(od),
                w_pc_mask,
                trainable=False)

        if p.layer_norm:
          # Set up layer normalization variables
          bias_pc = py_utils.WeightParams(
              shape=[1, p.num_output_nodes],
              init=init_bias[t],  # py_utils.WeightInit.Constant(0.0),
              dtype=params.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'b' + str(t), bias_pc, self.AddGlobalVN, trainable=True)

          ln_scale_pc = py_utils.WeightParams(
              shape=[1, p.num_output_nodes],
              init=init_ln_scale[t],  # py_utils.WeightInit.Constant(1.0),
              dtype=p.dtype,
              collections=self._VariableCollections())
          self.CreateVariable(
              'ln_scale' + str(t),
              ln_scale_pc,
              self.AddGlobalVN,
              trainable=True)

        if p.train_scale:  # to be done: this should be in loop over TPs
          summary_utils.scalar(scope.name + '/input_scale',
                               tf.cast(self.vars['scale' + str(t)], tf.float32))
        if p.train_radius:
          for od in range(p.order):
            summary_utils.scalar(
                scope.name + '/radius',
                tf.cast(self.vars['radius' + str(t) + str(od)], tf.float32))

      self._timestep = -1
      if self.num_tp > 1:
        # initializing the ensemble function
        self.ensemble_NN(p, scope)

  @property
  def output_size(self):
    return self.params.num_output_nodes

  def batch_size(self, inputs):
    return tf.shape(inputs.act[0])[0]

  def GetOutput(self, state):
    return state['m' + str(0)]

  def zero_state(self, theta, batch_size):
    p = self.params
    # zero_m = tf.zeros((batch_size, self.output_size), dtype=p.dtype)  # m
    previous_states = py_utils.NestedMap()
    # previous_states.m = zero_m  # h(t-1)
    for od in range(0, p.order):
      var_name = 'm' + str(od)
      previous_states[var_name] = tf.zeros((batch_size, self.output_size),
                                           dtype=p.dtype)
    return previous_states

  def ensemble_NN(self, p, scope):
    """Function implemented using import layers ."""
    hidden_size = p.hidden_dim_actnn
    # layer 1
    fc1 = layers.FCLayer.Params().Set(
        input_dim=self.num_tp,
        output_dim=hidden_size,
        activation='RELU',  # RELU/TANH
        trainable=True)
    self.CreateChild('fc1', fc1)
    if p.nn_layer2:
      # layer 2
      fc2 = layers.FCLayer.Params().Set(
          input_dim=hidden_size,
          output_dim=hidden_size,
          activation='RELU',  # NONE/SIGMOID
          trainable=True)
      self.CreateChild('fc2', fc2)
    # final layer
    fcn = layers.FCLayer.Params().Set(
        input_dim=hidden_size,
        output_dim=1,
        activation='NONE',  # NONE/SIGMOID
        trainable=True)
    self.CreateChild('fcn', fcn)

  def _Mix(self, theta, state0, inputs):
    # Note: theta has access to all the CreateVariables.
    assert isinstance(inputs.act, list)
    p = self.params

    def _LayerNorm(x):
      mean = tf.reduce_mean(x, axis=1, keepdims=True)
      variance = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
      return (x - mean) * tf.math.rsqrt(variance +
                                        self.params.layer_norm_epsilon)

    # Concatenate the output of various topologies. (?, H, numTP)
    out = []
    for t in range(self.num_tp):
      in_tp, res_tp = p.weight_topology_in[t], p.weight_topology_res[t]
      if p.train_scale:
        # i = tf.multiply(theta.scale[t], theta.win[t])
        i = tf.multiply(theta['scale' + str(t)], theta['win' + in_tp + str(t)])
      else:
        i = theta['win' + in_tp + str(t)]

      h = {}  # the hidden state matrices
      for od in range(p.order):  # each order will have its own radius
        if self.params.train_radius:
          h[od] = tf.multiply(theta['radius' + str(t) + str(od)],
                              theta['wr' + res_tp + str(t) + str(od)])
        else:
          h[od] = theta['wr' + res_tp + str(t) + str(od)]

      # multiply by mask if training
      if p.trainable_in:
        # if in_tp in ['scrbd']:
        i = i * theta['win_mask' + in_tp + str(t)]

      if p.trainable_res:  # apply the mask
        for od in range(p.order):
          h[od] = h[od] * theta['wr_mask' + res_tp + str(t) + str(od)]

      outt = py_utils.Matmul(inputs.act[0], i)

      # outt_res = py_utils.Matmul(state0.m, h[0])
      for od in range(0, p.order):  # previous time steps
        outt += py_utils.Matmul(state0['m' + str(od)], h[od])

      # outt = outt_in + outt_res

      if self.params.layer_norm:
        outt = _LayerNorm(outt)
        # Apply shift and gain parameters
        outt = outt * theta['ln_scale' + str(t)] + theta['b' + str(t)]
      out.append(tf.expand_dims(outt, -1))  # [?, H, 1]
    # convert the output list to tensor
    out = tf.concat(out, axis=-1)
    # print('OUT: ', out.shape) # (?, H, numTP)
    return out

  def _Gates(self, xmw, theta, state0, inputs):
    """Compute the new state."""
    p = self.params
    # value: [batch, out_dim] ; out_dim = hidden state dim
    # value = xmw + tf.expand_dims(theta.b, 0)
    value = xmw
    if p.output_mode == 'non_linear':
      out_act = tf.tanh(value)  # [?, H, numTP]
    elif p.output_mode == 'linear':
      out_act = value
    elif p.output_mode == 'mixture':
      linear_units = int(p.linear_percent * p.num_output_nodes)
      linear_part = tf.slice(value, [0, 0], [-1, linear_units])
      nonlinear_part = tf.tanh(tf.slice(value, [0, linear_units], [-1, -1]))
      out_act = tf.concat([linear_part, nonlinear_part], axis=1)

    if self.num_tp > 1:
      # apply the ensemble NN
      hidden_act = self.fc1.FProp(theta.fc1, out_act)
      if p.nn_layer2:
        hidden_act = self.fc2.FProp(theta.fc2, hidden_act)
      out_act = self.fcn.FProp(theta.fcn, hidden_act)

    new_m = tf.squeeze(out_act, axis=-1)  # reduce the extra dim introduced
    # new_m = state0.m * (1.0 - p.leak_weight) + new_m * p.leak_weight
    # a for loop to update the new states.
    previous_states = py_utils.NestedMap()
    previous_states['m' + str(0)] = new_m  # h(t-1)
    for od in range(0, p.order - 1):
      new_var = 'm' + str(od + 1)
      old_var = 'm' + str(od)
      previous_states[new_var] = state0[old_var]
    return previous_states
