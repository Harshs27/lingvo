# Lint as: python2, python3
"""Machine translation models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
from lingvo import compat as tf
from lingvo.core import base_decoder
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import batch_major_attention
from lingvo.core import cluster_factory
from lingvo.core import hyperparams
from lingvo.core import layers
from lingvo.core import layers_with_gpipe
from lingvo.core import metrics
from lingvo.core import ml_perf_bleu_metric
from lingvo.core import py_utils
from lingvo.core import summary_utils
from lingvo.tasks.mt import model as lingvo_model
import numpy as np
import six
from six.moves import range
from six.moves import zip
from google3.learning.brain.research.babelfish import beam_search_tpu_helper
from google3.learning.brain.research.babelfish import inference_registry
from google3.learning.brain.research.babelfish import metrics as bf_metrics
from google3.learning.brain.research.babelfish import moe_builder
from google3.learning.brain.research.babelfish import moe_layers
from google3.learning.brain.research.babelfish import ops
from google3.learning.brain.research.babelfish import py_utils as bbf_py_utils
from google3.learning.brain.research.babelfish.experimental.sys import flat_beam_search
from google3.learning.brain.research.babelfish.experimental.sys import packed_greedy_search
from google3.learning.brain.research.babelfish.mt import decoder
from google3.learning.brain.research.babelfish.mt import encoder
from google3.learning.brain.research.babelfish.mt import layers as mt_layers
from google3.learning.brain.research.babelfish.speech import decoder_utils
from google3.learning.brain.research.babelfish.speech import emission_delay_loss


def _SelectTopOne(topk_items, batch_size, k, batch_first=False):
  """Selects the top-1 item for each top-k list in the batch.

  Args:
    topk_items: tensor of shape [k * batch_size, ...] of top-k lists
    batch_size: numer of k-best lists in the batch.
    k: the size of each k-best list.
    batch_first: boolean that determines the layout of topk_items
      .True:  the i'th item of the j'th batch is at index [j * k + i].
      .False: the i'th item of the j'th batch is at index [i * batch_size + j].

  Returns:
    A tensor of shape [batch, ...] containing the top-1 entry of each list.
  """
  if k == 1:
    return topk_items
  elif batch_first:
    return tf.gather(topk_items, tf.range(batch_size) * k)
  else:
    return tf.gather(topk_items, tf.range(batch_size))


def _ProbsToDelay(emit_probs, target_padding):
  """Transform emit_probs into a delay vector, to help with metrics.

  Args:
    emit_probs: emission probabilities [max_target_len, max_source_len]
    target_padding: padding vector [max_target_len]

  Returns:
    delay vector [target_len]
  """
  max_source_len = emit_probs.shape[1]
  position_delay = (1 + np.arange(max_source_len)).astype(emit_probs.dtype)
  delay = np.sum(emit_probs * np.reshape(position_delay, [1, -1]), axis=1)
  target_len = np.sum(1 - target_padding).astype(np.int32)
  return delay[:target_len]


def ApplyWaitK(topk_ids,
               topk_lens,
               tokens_waited,
               src_is_complete,
               wait_k,
               emission_rate,
               float_dtype=tf.float32,
               eos_id=2):
  """Truncate decoder output according to wait_k schedule.

  Args:
    topk_ids: ids from beam search including EOS; int tensor [batch,
      num_hyps_per_beam, max_target_length].
    topk_lens: lengths from beam search; int tensor [batch, num_hyps_per_beam].
    tokens_waited: source lengths less any fake EOS; int tensor [batch].
    src_is_complete: 1 - complete source, 0 - partial; int tensor [batch].
    wait_k: Number of source tokens to wait before emission. wait_k=0 switches
      off all waiting behavior; int tensor [batch].
    emission_rate: Number of target tokens to emit for each source token, after
      waiting for wait_k time; float scalar.
    float_dtype: dtype for floats.
    eos_id: id for target EOS; int scalar.

  Returns:
    topk_ids: ids after truncation for wait-k. If EOS was truncated, then it
      is added back; int tensor [batch, num_hyps_per_beam, max_target_length].
    topk_lens: lengths of topk_ids; int tensor [batch, num_hyps_per_beam].
  """
  wait_k = py_utils.with_dependencies(
      [py_utils.assert_greater_equal(wait_k, tf.zeros_like(wait_k))], wait_k)

  batch = py_utils.GetShape(topk_lens)[0]
  num_hyps_per_beam = py_utils.GetShape(topk_lens)[1]
  max_target_length = py_utils.GetShape(topk_ids)[2]

  topk_lens = topk_lens - 1  # Suppress EOS during truncation.

  # Calculate truncated target lengths.
  #
  num_emitting = tf.maximum(0, tokens_waited - tf.maximum(0, wait_k - 1))
  # Tile to [batch, num_hyps_per_beam].
  num_emitting = tf.tile(
      tf.reshape(num_emitting, [batch, 1]), [1, num_hyps_per_beam])
  if emission_rate:
    max_dec_len = tf.cast(
        tf.cast(num_emitting, float_dtype) * emission_rate, topk_lens.dtype)
  else:
    max_dec_len = tf.cast(num_emitting > 0, topk_lens.dtype) * topk_lens
  dec_len = tf.minimum(max_dec_len, topk_lens)
  # Ignore all that work if source was complete or wait_k==0.
  no_wait_k = tf.math.logical_or(
      tf.equal(wait_k, 0), tf.cast(src_is_complete, tf.bool))

  no_wait_k = tf.tile(
      tf.reshape(no_wait_k, [batch, 1]),
      [1, num_hyps_per_beam])  # Tile to [batch, num_hyps_per_beam]
  dec_len = tf.where(no_wait_k, topk_lens, dec_len)

  dec_len = dec_len + 1  # Add EOS back in.

  # Truncate target.
  #
  new_topk_lens = dec_len
  wo_eos_mask = tf.sequence_mask(dec_len - 1, max_target_length, topk_ids.dtype)
  w_eos_mask = tf.sequence_mask(dec_len, max_target_length, topk_ids.dtype)
  new_topk_ids = wo_eos_mask * topk_ids + (
      # Place EOS symbol at the correct spot.
      tf.ones_like(topk_ids) * (w_eos_mask - wo_eos_mask) * eos_id)
  return new_topk_ids, new_topk_lens


class MTBaseModel(lingvo_model.MTBaseModel):
  """Base Class for NMT models."""

  @classmethod
  def Params(cls):
    p = super(MTBaseModel, cls).Params()

    p.Define('separator', None, 'Separator to remove after decoding.')
    p.Define(
        'sep_id', -1, 'Get sentence following the "sep_id"th separator.'
        'If negative, retrieve the "sep_id"th to last sentence.')

    p.Define(
        'prod_subgraph_retranslation', True,
        'If True, add retranslation capabilities to the "prod" subgraph. This '
        'may a cause a small (~1%) latency regression.')

    p.Define(
        'scoring_subgraph_normalization', 'simple_length',
        'Specify how to normalize scores in InferenceSubgraph_Scoring. '
        'Choose from [simple_length, simple_length_and_coverage, '
        'beam_length, beam_length_and_coverage].')

    p.Define(
        'lang_id_list', None, 'list of language ids. This list is required for'
        'MASS trained models to pass task ids for src and tgt languages at'
        'inference time. In this list index(lang_id) = task_id.')

    ep = p.eval
    ep.Define('add_rouge_to_eval_metrics', False,
              'Whether to include rouge among evaluation metrics')
    ep.Define('rouge_version', 'v1', 'Either v1 or v2.')
    ep.Define('add_m2_to_eval_metrics', False,
              'Whether to include GEC M2 among evaluation metrics.')
    ep.Define('inference_source_language', 'en',
              'Default source language to use in Inference().')
    ep.Define('inference_target_language', 'es',
              'Default target language to use in Inference().')
    ep.Define(
        'logging_interval', 1,
        'Log src, tgt, and hyps to INFO every x examples. Set to a large number'
        '(maxed out at batch_size) reduces logging and speeds up decoding.')
    ep.Define('ml_perf_metrics_only', False,
              'Whether to only calculate MLPerf metrics.')

    tp = p.train
    tp.vn_start_step = 20000
    tp.vn_std = 0.0
    tp.learning_rate = 2e-4
    tp.clip_gradient_norm_to_value = 5.0
    tp.grad_norm_to_clip_to_zero = 10000.0

    tp.lr_schedule.start_step = 400000
    tp.lr_schedule.half_life_steps = 100000
    tp.start_up_delay_steps = 500
    tp.optimizer.epsilon = 8e-7

    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTBaseModel, self).__init__(params)
    p = self.params
    if p.scoring_subgraph_normalization not in [
        'simple_length', 'simple_length_and_coverage', 'beam_length',
        'beam_length_and_coverage'
    ]:
      raise ValueError('Invalid p.scoring_subgraph_normalization: {}'.format(
          p.scoring_subgraph_normalization))

  def Inference(self):
    """Constructs the inference subgraphs.

    Supported subgraphs:
      - 'default'
      - '[src]->[tgt]'
      - 'prod'
      - 'scoring' (if implemented for given model)

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    p = self.params
    subgraphs = dict()
    with tf.name_scope('inference'):
      src = p.eval.inference_source_language
      tgt = p.eval.inference_target_language
      subgraphs['default'] = self._InferenceSubgraph_Default(src, tgt)
      subgraphs['%s->%s' % (src, tgt)] = subgraphs['default']
      subgraphs['prod'] = self._InferenceSubgraph_Prod()

      # Add the scoring subgraph if implemented.
      scoring_graph = self._InferenceSubgraph_Scoring()
      if scoring_graph:
        subgraphs['scoring'] = scoring_graph
      try:
        subgraphs['retranslation'] = self._InferenceSubgraph_Retranslation()
      except (NotImplementedError,) as ex:
        print('failed to generate the (optional?) retranslation subgraph: ', ex)

    return subgraphs

  def _InferenceSubgraph_Default(self, src, tgt):
    p = self.params

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):
      # TODO(yonghui): Make sure the following is the same preprocessing as we
      # do in preparing our NMT training data.
      src_strings = tf.placeholder(tf.string, shape=[None], name='src_strings')
      upp_tokenized_src = ops.upp_tokenize(src_strings, language=src)
      _, src_ids, src_paddings = self.input_generator.StringsToIds(
          upp_tokenized_src,
          external_max_length=p.decoder.target_seq_len,
          key=self._GetTokenizerKeyToUse('src'))

      # Truncate paddings at the end.
      max_seq_length = tf.cast(
          tf.round(tf.reduce_max(tf.reduce_sum(1.0 - src_paddings, 1))),
          dtype=tf.int32)
      src_paddings = py_utils.with_dependencies([
          py_utils.assert_equal(
              tf.constant(True, tf.bool),
              tf.reduce_all(src_paddings[:, max_seq_length:] > 0.5))
      ], src_paddings)
      src_ids = src_ids[:, :max_seq_length]
      src_paddings = src_paddings[:, :max_seq_length]
      # Last step, reverse the source sequence if it is training an reversed
      # order model.
      if not self.input_generator.natural_order_model:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids = ops.reverse_ids(src_ids, slen)

      src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      if p.lang_id_list is not None and src in p.lang_id_list:
        src_input_map['task_ids'] = tf.fill(
            tf.shape(src_ids),
            tf.where(tf.equal(p.lang_id_list, src))[0, 0],
            name='src_task_ids')

      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      if p.lang_id_list is not None and tgt in p.lang_id_list:
        encoder_outputs['target_task_ids'] = tf.fill(
            tf.shape(src_ids[:, 0]),
            tf.dtypes.cast(
                tf.where(tf.equal(p.lang_id_list, tgt))[0, 0], dtype=tf.int32),
            name='tgt_task_ids')

      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      topk_decoded = self.input_generator.IdsToStrings(
          topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))

      # last step, runs the upp untokenizer
      topk_decoded = ops.upp_detokenize(topk_decoded, language=tgt)

      feeds = py_utils.NestedMap({'src_strings': src_strings})
      fetches = py_utils.NestedMap({
          'topk_decoded': topk_decoded,
          'topk_hyps': topk_hyps,
          'topk_scores': topk_scores,
          'src_ids': src_ids,
      })

      return fetches, feeds

  # Builds an inference subgraph similar to _InferenceSubgraph_Default with
  # src_ids and src_paddings expected to be passed in as feed data. Returns
  # the topk hypotheses, target token IDs and target sentence lengths. This
  # subgraph allows for running the preprocessor and postprocessor outside of
  # the Babelfish inference stack.
  def _InferenceSubgraph_Prod(self):
    p = self.params

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):
      src_ids = tf.placeholder(tf.int32, shape=None, name='src_ids')
      src_paddings = tf.placeholder(tf.float32, shape=None, name='src_paddings')
      batch_size = tf.shape(src_ids)[0]

      source_languages = tf.placeholder(
          tf.string, shape=None, name='source_languages')
      target_languages = tf.placeholder(
          tf.string, shape=None, name='target_languages')

      if p.input.natural_order_model:
        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      else:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids_rev = ops.reverse_ids(src_ids, slen)
        src_input_map = py_utils.NestedMap(
            ids=src_ids_rev, paddings=src_paddings)

      if p.lang_id_list is not None:
        src_input_map['task_ids'] = tf.tile(
            tf.where(tf.equal(p.lang_id_list, source_languages))[:, 1,
                                                                 tf.newaxis],
            [1, tf.shape(src_ids)[1]],
            name='src_task_ids')

      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)

      tgt_ids = None
      tgt_paddings = None
      tgt_bias = None
      if p.prod_subgraph_retranslation:
        # Extra fields for biasing beam search.
        tgt_ids = tf.placeholder_with_default(
            tf.zeros([batch_size, 0], dtype=tf.int32),
            shape=None,
            name='tgt_ids')
        tgt_paddings = tf.placeholder_with_default(
            tf.ones([batch_size, 0], dtype=tf.float32),
            shape=None,
            name='tgt_paddings')
        tgt_bias = tf.placeholder_with_default(
            tf.zeros([batch_size, 1], dtype=tf.float32),
            shape=[None, 1],
            name='tgt_bias')
        tgt_input_map = py_utils.NestedMap(
            labels=tgt_ids,
            paddings=tgt_paddings,
            weights=tgt_bias * (1 - tgt_paddings))
        encoder_outputs['targets'] = tgt_input_map

      if p.lang_id_list is not None:
        encoder_outputs['target_task_ids'] = tf.dtypes.cast(
            tf.where(tf.equal(p.lang_id_list, target_languages))[:, 1],
            dtype=tf.int32)

      if p.prod_subgraph_retranslation:
        decoder_outs = self.dec.BeamSearchDecodeBiased(encoder_outputs)
      else:
        decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

      # Reshape topk_ids from 2D to 3D so we can iterate across hyps for a given
      # source:
      # topk_ids: [batch, number of hyps, max num sequence ids]
      topk_ids = tf.reshape(
          topk_ids, [batch_size, p.decoder.beam_search.num_hyps_per_beam, -1])

      # Reshape topk_lens from 1D to 2D so we can iterate lens for the nbest
      # for each source:
      # topk_lens: [batch, number of hyps]
      topk_lens = tf.reshape(
          topk_lens, [batch_size, p.decoder.beam_search.num_hyps_per_beam])

      feeds = py_utils.NestedMap({
          'src_ids': src_ids,
          'src_paddings': src_paddings,
          'source_languages': source_languages,
          'target_languages': target_languages,
      })
      if p.prod_subgraph_retranslation:
        feeds.update({
            'tgt_ids': tgt_ids,
            'tgt_paddings': tgt_paddings,
            'tgt_bias': tgt_bias,
        })

      fetches = py_utils.NestedMap({
          'topk_hyps': topk_hyps,
          'topk_ids': topk_ids,
          'topk_lens': topk_lens,
          'topk_scores': topk_scores,
      })

      return fetches, feeds

  def _InferenceSubgraph_Scoring(self):
    """Builds an inference graph that produces scores for the given targets.

    Each model has to implement it accordingly.

    Scores shoud be length normalized summed log probabilities.
    """
    pass

  def _InferenceSubgraph_Retranslation(self):
    raise NotImplementedError('Retranslation subgraph not implemented for ',
                              self.params.cls.__name__)

  def _GatherLogProbsByIds(self, ids, log_probs):
    """Gathers log_probs corresponding to ids.

    Args:
      ids: int tensor of [time, batch].
      log_probs: tensor of [time, batch, vocab].

    Returns:
      id_log_probs: tensor of [time, batch], where id_log_probs[t, b] =
        log_probs[t, b, ids[t, b]].
    """
    py_utils.assert_equal(tf.rank(ids), 2)
    time = tf.shape(ids)[0]
    batch = tf.shape(ids)[1]
    # [time, batch]: [[0, ..., 0], [1, ..., 1], ..., [time - 1, ..., time - 1]].
    dim0 = tf.tile(tf.expand_dims(tf.range(time), axis=-1), [1, batch])
    # [time, batch]: [[0, 1, ..., batch - 1], ..., [0, ..., batch - 1]].
    dim1 = tf.tile(tf.expand_dims(tf.range(batch), axis=0), [time, 1])
    # [time, batch, 3].
    indices = tf.stack(
        [tf.cast(dim0, dtype=ids.dtype),
         tf.cast(dim1, dtype=ids.dtype), ids],
        axis=-1)
    return tf.gather_nd(log_probs, indices)

  def _BeamSearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      encoder_outputs = self.dec.AddExtraDecodingInfo(encoder_outputs,
                                                      input_batch.tgt)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      return py_utils.RunOnTpuHost(self._ProcessBeamSearchDecodeOut,
                                   input_batch, encoder_outputs, decoder_outs)

  def _GreedySearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      encoder_outputs = self.dec.AddExtraDecodingInfo(encoder_outputs,
                                                      input_batch.tgt)
      decoder_outs = self.dec.GreedySearchDecode(encoder_outputs)
      return decoder_outs

  def _ProcessBeamSearchDecodeOut(self, input_batch, encoder_outputs,
                                  decoder_outs):
    """Tensorflow processing of decoder outputs before Python post-processing.

    Args:
      input_batch: same input_batch passed to _BeamSearchDecode, a NestedMap.
      encoder_outputs: outputs from Encoder, a NestedMap.
      decoder_outs: output of dec.BeamSearchDecode, a `BeamSearchDecodeOutput`.

    Returns:
      a dictionary of processed decoder outputs.
    """
    del encoder_outputs
    p = self.params
    topk_hyps = decoder_outs.topk_hyps
    topk_ids = decoder_outs.topk_ids
    topk_lens = decoder_outs.topk_lens
    topk_scores = decoder_outs.topk_scores

    # NOTE: Source sequence ids are read in reversed order from NMTExample
    # if we are training a reversed order model. In that case we reverse it
    # back to natural order. Its last id is </s> and we ignore it.
    slen = tf.cast(
        tf.round(tf.reduce_sum(1 - input_batch.src.paddings, 1) - 1),
        dtype=tf.int32)

    # Note: srcs should not contain trailing EOS token.
    srcs = input_batch.src.get('strs', None)
    if srcs is not None:
      tf.logging.info('Using input_batch.src.strs')
    elif self.input_generator.natural_order_model:
      srcs = self.input_generator.IdsToStrings(
          input_batch.src.ids, slen, self._GetTokenizerKeyToUse('src'))
    else:
      srcs = self.input_generator.IdsToStrings(
          ops.reverse_ids(input_batch.src.ids, slen), slen,
          self._GetTokenizerKeyToUse('src'))

    topk_decoded = self.input_generator.IdsToStrings(
        topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
    topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))
    topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

    # Note: refs should not contain trailing EOS token.
    refs = input_batch.tgt.get('strs', None)
    if refs is not None:
      tf.logging.info('Using input_batch.tgt.strs')
    else:
      refs = self.input_generator.IdsToStrings(
          input_batch.tgt.labels,
          tf.cast(
              tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
              dtype=tf.int32), self._GetTokenizerKeyToUse('tgt'))
    hyps = topk_decoded

    def ComputeTranslationQuality(metric):
      """Computes translation quality scores for all hyps."""
      scores = ops.per_sentence_translation_quality(
          tf.reshape(hyps, [-1]),
          tf.reshape(
              tf.tile(
                  tf.reshape(refs, [-1, 1]),
                  [1, p.decoder.beam_search.num_hyps_per_beam]), [-1]),
          metric=metric,
          max_ngram=4)
      return tf.reshape(scores, tf.shape(topk_hyps))

    gleu = ComputeTranslationQuality('gleu')

    ret_dict = {
        'target_ids': input_batch.tgt.ids,
        'target_labels': input_batch.tgt.labels,
        'target_weights': input_batch.tgt.weights,
        'target_paddings': input_batch.tgt.paddings,
        'sources': srcs,
        'targets': refs,
        'topk_ids': topk_ids,
        'topk_decoded': topk_decoded,
        'topk_lens': topk_lens,
        'topk_scores': topk_scores,
        'gleu': gleu,
    }
    return ret_dict

  def _RemoveSeparator(self, input_text):
    """Retrieve one sentence based on a separator token."""
    p = self.params

    input_ = input_text.split()
    if not input_:
      return input_text
    sep_pos = [int(token == p.separator) for token in input_]
    cumsum = np.cumsum(sep_pos)
    sep_id = p.sep_id if p.sep_id >= 0 else max(cumsum) + p.sep_id + 1
    out = [word for ii, word in enumerate(input_) if cumsum[ii] == sep_id]
    out = out[1:] if out[0] == p.separator else out
    out = ' '.join(out)
    return out

  def _PostProcessBeamSearchDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post processes the output from _BeamSearchDecode."""
    p = self.params
    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    targets = dec_out_dict['targets']
    sources = dec_out_dict['sources']
    gleu = dec_out_dict['gleu']

    num_samples = len(targets)
    assert num_samples == len(topk_decoded), ('%s vs %s' % (num_samples,
                                                            len(topk_decoded)))
    assert num_samples == len(sources)
    assert num_samples == len(gleu)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)

    key_value_pairs = []
    for i in range(num_samples):
      src, tgt = sources[i], targets[i]
      tgt = self._RemoveSeparator(tgt) if p.separator else tgt
      if i % p.eval.logging_interval == 0:
        tf.logging.info('source: %s', src)
        tf.logging.info('  target: %s', tgt)
      hyps = topk_decoded[i]
      assert p.decoder.beam_search.num_hyps_per_beam == len(hyps)
      info = [src, tgt]
      for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
        hyp_str = self._RemoveSeparator(hyp_str) if p.separator else hyp_str
        if i % p.eval.logging_interval == 0:
          tf.logging.info('  %f: %s', score, hyp_str)
        hyp_gleu = gleu[i][n]
        info += [(hyp_str, score, hyp_gleu)]
        # Only aggregate scores of the top hypothesis.
        if n == 0:
          if p.eval.ml_perf_metrics_only:
            dec_metrics_dict['ml_perf_bleu'].Update(tgt, hyp_str)
          else:
            dec_metrics_dict['gleu'].Update(hyp_gleu, 1)
            dec_metrics_dict['corpus_bleu'].Update(tgt, hyp_str)
            dec_metrics_dict['canonical_bleu'].Update(tgt, hyp_str, src)
          # Update rouge metrics
          if p.eval.add_rouge_to_eval_metrics:
            dec_metrics_dict['f1_rouge_1'].Update(tgt, hyp_str)
            dec_metrics_dict['f1_rouge_2'].Update(tgt, hyp_str)
            dec_metrics_dict['f1_rouge_L'].Update(tgt, hyp_str)
          if p.eval.add_m2_to_eval_metrics:
            dec_metrics_dict['m2_f0.5'].Update(tgt, hyp_str, src)
            dec_metrics_dict['m2_precision'].Update(tgt, hyp_str, src)
            dec_metrics_dict['m2_recall'].Update(tgt, hyp_str, src)
      key_value_pairs.append((src, '%s' % info))

    return key_value_pairs

  def CreateDecoderMetrics(self):
    p = self.params
    if hasattr(p.input, 'target_language') and p.input.target_language:
      target_language = p.input.target_language
    else:
      target_language = p.eval.inference_target_language

    if p.eval.ml_perf_metrics_only:
      decoder_metrics = {
          'ml_perf_bleu': ml_perf_bleu_metric.MlPerfBleuMetric(),
          'num_samples_in_batch': metrics.AverageMetric(),
      }
    else:
      decoder_metrics = {
          'num_samples_in_batch': metrics.AverageMetric(),
          'gleu': metrics.AverageMetric(),
          'corpus_bleu': metrics.CorpusBleuMetric(),
          'canonical_bleu': bf_metrics.CanonicalBleuMetric(target_language),
      }

    if p.eval.add_rouge_to_eval_metrics:
      if p.eval.rouge_version == 'v1':
        decoder_metrics['f1_rouge_1'] = bf_metrics.RougeMetric(
            ngram=1, is_f_version=True, ignore_unk=True)
        decoder_metrics['f1_rouge_2'] = bf_metrics.RougeMetric(
            ngram=2, is_f_version=True, ignore_unk=True)
        decoder_metrics['f1_rouge_L'] = bf_metrics.RougeMetric(
            ngram=0, is_f_version=True, ignore_unk=True)
      elif p.eval.rouge_version == 'v2':
        decoder_metrics['f1_rouge_1'] = bf_metrics.RougeMetricV2(
            rouge_type='rouge1', use_stemmer=True)
        decoder_metrics['f1_rouge_2'] = bf_metrics.RougeMetricV2(
            rouge_type='rouge2', use_stemmer=True)
        decoder_metrics['f1_rouge_L'] = bf_metrics.RougeMetricV2(
            rouge_type='rougeL', use_stemmer=True)
      else:
        raise ValueError('p.eval.rouge_version must be either v1 or v2.')

    if p.eval.add_m2_to_eval_metrics:
      decoder_metrics['m2_f0.5'] = bf_metrics.M2FBetaMetric(target_language)
      decoder_metrics['m2_precision'] = bf_metrics.M2PrecisionMetric(
          target_language)
      decoder_metrics['m2_recall'] = bf_metrics.M2RecallMetric(target_language)

    return decoder_metrics


@inference_registry.AllowClassParamsOverride
class MTModelClassifier(MTBaseModel):
  """Encoder + Classifier."""

  @classmethod
  def Params(cls):
    p = super(MTModelClassifier, cls).Params()
    p.encoder = encoder.MTEncoderV1.Params()
    p.decoder = decoder.MTBaseClassifier.Params()
    p.Define(
        'source_seq_len', 256,
        'Maximum sequence length for padding source sentences. Other MT models'
        'use target_seq_len from decoder to get this information.')
    p.Define(
        'freeze_encoder', False,
        'Whether to hold encoder fixed (no gradient updates). This setting may '
        'be useful when encoder has been pre-trained on much larger data and '
        'we wish only to learn a classifier on top of the pre-trained '
        'representation.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTModelClassifier, self).__init__(params)
    p = self.params
    assert p.name

  def ComputePredictions(self, theta, batch):
    encoder_outputs = self.enc.FProp(theta.enc, batch.src)
    if self.params.freeze_encoder:
      encoder_outputs.encoded = tf.stop_gradient(encoder_outputs.encoded)
      encoder_outputs.padding = tf.stop_gradient(encoder_outputs.padding)
    return self.dec.ComputePredictions(theta.dec, encoder_outputs, batch.tgt)

  def Decode(self, input_batch):
    """Constructs the inference graph."""
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      decoder_outs = self.dec.FPropDefaultTheta(encoder_outputs,
                                                input_batch.tgt).metrics
      slen = tf.cast(
          tf.round(tf.reduce_sum(1 - input_batch.src.paddings, 1) - 1),
          dtype=tf.int32)
      if self.input_generator.natural_order_model:
        srcs = self.input_generator.IdsToStrings(input_batch.src.ids, slen)
      else:
        srcs = self.input_generator.IdsToStrings(
            ops.reverse_ids(input_batch.src.ids, slen), slen)

      ret_dict = {
          'target_ids': input_batch.tgt.ids,
          'target_labels': input_batch.tgt.labels,
          'target_weights': input_batch.tgt.weights,
          'target_paddings': input_batch.tgt.paddings,
          'sources': srcs,
          'predicted_label': decoder_outs['predicted_label'],
          'scores': decoder_outs['scores']
      }

    return ret_dict

  def CreateDecoderMetrics(self):
    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
    }
    return decoder_metrics

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    sources = dec_out_dict['sources']
    targets = dec_out_dict['target_labels']

    num_samples = len(targets)
    assert num_samples == len(sources)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)
    key_value_pairs = []
    for i in range(num_samples):
      src, tgt = sources[i], targets[i]

      tf.logging.info('source: %s', src)
      # Convert target label from an array of size 1 to a single element.
      tgt = tgt[0]
      tf.logging.info('target_label: %d', tgt)
      if dec_out_dict['predicted_label'] is not None:
        # predicted_label[0]: the labels with max logits
        pred_id = int(dec_out_dict['predicted_label'][0][i])
        tf.logging.info('predicted_label: %d', pred_id)
      if dec_out_dict['scores'] is not None:
        # scores[0]: log_prob of each example over num_classes
        # We return the log_prob of the predicted class for example 'i'.
        score = dec_out_dict['scores'][0][i][pred_id]
        tf.logging.info('scores: %f', score)
      info = [src, tgt, pred_id, score]

      # `src` may be a pair for multi-source model.
      key_value_pairs.append((str(src), '%s' % info))
    return key_value_pairs

  def _InferenceSubgraph_Default(self, src, tgt):
    p = self.params

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):
      src_strings = tf.placeholder(tf.string, shape=[None], name='src_strings')
      upp_tokenized_src = ops.upp_tokenize(src_strings, language=src)
      _, src_ids, src_paddings = self.input_generator.StringsToIds(
          upp_tokenized_src,
          external_max_length=p.source_seq_len,
          key=self._GetTokenizerKeyToUse('src'))

      # Truncate paddings at the end.
      max_seq_length = tf.cast(
          tf.round(tf.reduce_max(tf.reduce_sum(1.0 - src_paddings, 1))),
          dtype=tf.int32)
      src_paddings = py_utils.with_dependencies(
          [py_utils.assert_greater(src_paddings[:, max_seq_length:], 0.5)],
          src_paddings)
      src_ids = src_ids[:, :max_seq_length]
      src_paddings = src_paddings[:, :max_seq_length]
      # Last step, reverse the source sequence if it is training an reversed
      # order model.
      if not self.input_generator.natural_order_model:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids = ops.reverse_ids(src_ids, slen)

      src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)

      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.Decode(encoder_outputs)

      topk_labels = decoder_outs['topk_labels'][0]
      topk_probs = decoder_outs['topk_probs'][0]
      logits = decoder_outs['logits'][0]

      feeds = py_utils.NestedMap({'src_strings': src_strings})
      fetches = py_utils.NestedMap({
          'topk_labels': topk_labels,
          'topk_probs': topk_probs,
          'logits': logits,
      })

      return fetches, feeds

  # Builds an inference subgraph similar to _InferenceSubgraph_Default with
  # src_ids and src_paddings expected to be passed in as feed data. Returns
  # the topk hypotheses, target token IDs and target sentence lengths. This
  # subgraph allows for running the preprocessor and postprocessor outside of
  # the Babelfish inference stack.
  def _InferenceSubgraph_Prod(self):
    p = self.params

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):
      src_ids = tf.placeholder(tf.int32, shape=None, name='src_ids')
      src_paddings = tf.placeholder(tf.float32, shape=None, name='src_paddings')

      if p.input.natural_order_model:
        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      else:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids_rev = ops.reverse_ids(src_ids, slen)
        src_input_map = py_utils.NestedMap(
            ids=src_ids_rev, paddings=src_paddings)

      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.Decode(encoder_outputs)

      topk_labels = decoder_outs['topk_labels'][0]
      topk_probs = decoder_outs['topk_probs'][0]
      logits = decoder_outs['logits'][0]

      feeds = py_utils.NestedMap({
          'src_ids': src_ids,
          'src_paddings': src_paddings
      })
      fetches = py_utils.NestedMap({
          'topk_labels': topk_labels,
          'topk_probs': topk_probs,
          'logits': logits,
      })

      return fetches, feeds


class MTModelClassifierWithReversal(MTModelClassifier):
  """Encoder + Classifier with Reversal Layer."""

  @classmethod
  def Params(cls):
    p = super(MTModelClassifierWithReversal, cls).Params()
    p.decoder = decoder.MTBaseClassifierWithReversal.Params()
    return p


class MTModelSequenceLabeler(MTBaseModel):
  """MT encoder and Sequence Tagging Decoder with Span F1 Metric."""

  @classmethod
  def Params(cls):
    p = super(MTModelSequenceLabeler, cls).Params()
    p.encoder = encoder.MTEncoderV1.Params()
    p.decoder = decoder.FFSequenceLabelingDecoder.Params()
    p.Define('encoding_scheme', 'iob1', 'Encoding scheme used for tags')
    p.Define(
        'freeze_encoder', False,
        'Whether to hold encoder fixed (no gradient updates). This setting may '
        'be useful when encoder has been pre-trained on much larger data and '
        'we wish only to learn a classifier on top of the pre-trained '
        'representation.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTModelSequenceLabeler, self).__init__(params)
    p = self.params
    assert p.name

  def ComputePredictions(self, theta, batch):

    encoder_outputs = self.enc.FProp(theta.enc, batch.src)

    if self.params.freeze_encoder:
      encoder_outputs.encoded = tf.stop_gradient(encoder_outputs.encoded)
      encoder_outputs.padding = tf.stop_gradient(encoder_outputs.padding)
    return self.dec.ComputePredictions(theta.dec, encoder_outputs, batch.tgt)

  def Decode(self, input_batch):
    """Constructs the inference graph."""
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      decoder_outs = self.dec.FPropDefaultTheta(encoder_outputs,
                                                input_batch.tgt)

      slen = tf.cast(
          tf.round(
              tf.reduce_sum(1 - tf.transpose(encoder_outputs.padding), 1) - 1),
          dtype=tf.int32)
      tlen = tf.cast(
          tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
          dtype=tf.int32)

      if self.input_generator.natural_order_model:
        srcs = self.input_generator.IdsToStrings(
            input_batch.src.ids, slen, self._GetTokenizerKeyToUse('src'))
      else:
        srcs = self.input_generator.IdsToStrings(
            ops.reverse_ids(input_batch.src.ids, slen), slen,
            self._GetTokenizerKeyToUse('src'))

      refs = self.input_generator.IdsToStrings(
          input_batch.tgt.labels, tlen, self._GetTokenizerKeyToUse('tgt'))

      pred_labels = self.input_generator.IdsToStrings(
          tf.cast(decoder_outs[0]['predicted_label'][0], dtype=tf.int32), tlen,
          self._GetTokenizerKeyToUse('tgt'))

      ret_dict = {
          'target_ids': input_batch.tgt.ids,
          'target_labels': input_batch.tgt.labels,
          'target_weights': input_batch.tgt.weights,
          'target_paddings': input_batch.tgt.paddings,
          'sources': srcs,
          'targets': refs,
          'predicted_labels': pred_labels,
          'scores': decoder_outs[0]['scores'][0]
      }

    return ret_dict

  def CreateDecoderMetrics(self):
    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'ner_f1_score': bf_metrics.SpanF1Metric(self.params.encoding_scheme)
    }
    return decoder_metrics

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    sources = dec_out_dict['sources']
    targets = dec_out_dict['targets']
    preds = dec_out_dict['predicted_labels']
    preds_score = dec_out_dict['scores']

    num_samples = len(targets)
    assert num_samples == len(sources)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)
    key_value_pairs = []
    for i in range(num_samples):
      src, tgt, pred = [
          six.ensure_str(x) for x in (sources[i], targets[i], preds[i])
      ]
      psc = preds_score[i]

      tf.logging.info('source: %s', src)
      tf.logging.info('target: %t', tgt)
      tf.logging.info('predicted: %s, %f', tgt, psc)

      info = [src, tgt, pred, psc]

      # If there are special tokens in prediction or target, replace them by
      # "O" so that they are not accounted into scoring.
      tgt = tgt.replace('<unk>', 'O').replace('<s>', 'O').replace('</s>', 'O')
      pred = pred.replace('<unk>', 'O').replace('<s>', 'O').replace('</s>', 'O')

      tgtsp, predsp = tgt.split(), pred.split()
      dec_metrics_dict['ner_f1_score'].Update(tgtsp, predsp)
      key_value_pairs.append((str(src), '%s' % info))

    return key_value_pairs


class MTModelJointSemanticParsing(MTModelSequenceLabeler):
  """Joint Semantic Parsing Model with accuracy and F1 metric."""

  def CreateDecoderMetrics(self):
    decoder_metrics = {
        'num_samples_in_batch':
            metrics.AverageMetric(),
        'classification_accuracy':
            metrics.AverageMetric(),
        'tagging_span_f1_score':
            bf_metrics.SpanF1Metric(self.params.encoding_scheme)
    }
    return decoder_metrics

  def PostProcessDecodeOut(self, dec_out_dict, dec_metrics_dict):
    sources = dec_out_dict['sources']
    targets = dec_out_dict['targets']
    preds = dec_out_dict['predicted_labels']

    num_samples = len(targets)
    assert num_samples == len(sources)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)
    key_value_pairs = []
    for i in range(num_samples):
      src, tgt, pred = [
          six.ensure_str(x) for x in (sources[i], targets[i], preds[i])
      ]

      tf.logging.info('source: %s', src)
      tf.logging.info('target: %t', tgt)
      tf.logging.info('predicted: %s', pred)

      info = [src, tgt, pred]

      # If there are special tokens in prediction or target, replace them by
      # "O" so that they are not accounted into scoring.
      tgt = tgt.replace('<unk>', 'O').replace('<s>', 'O').replace('</s>', 'O')
      pred = pred.replace('<unk>', 'O').replace('<s>', 'O').replace('</s>', 'O')

      tgtsp, predsp = tgt.split(), pred.split()
      dec_metrics_dict['classification_accuracy'].Update(tgtsp[0] == predsp[0])
      dec_metrics_dict['tagging_span_f1_score'].Update(tgtsp[1:], predsp[1:])
      key_value_pairs.append((str(src), '%s' % info))

    return key_value_pairs


@inference_registry.AllowClassParamsOverride
class MTModelV1(MTBaseModel):
  """MT model version 1."""

  @classmethod
  def Params(cls):
    p = super(MTModelV1, cls).Params()
    p.encoder = encoder.MTEncoderV1.Params()
    p.decoder = decoder.MTDecoderV1.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTModelV1, self).__init__(params)
    p = self.params
    assert p.name

  def _InferenceSubgraph_Scoring(self):
    """Builds an inference graph that produces scores for the given targets.

    Scores are sum_logprobs.

    The feed needs to contain:
     src: src_ids and src_paddings (src ids)
     trg: trg_ids and trg_labels (where ids must start with <s> and labels
      must end with </s>); trg_paddings

    Returns:
     The following elements are contained in the fetches.
     sum_logprobs: per sentence summed log probs.
     norm_sent_sumlog_probs: length normalized sum_logprobs.
     log_probs: per sentence: log probs per word/unit.
    """
    p = self.params
    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    num_classes = p.decoder.softmax.num_classes

    softmax_dim = p.decoder.rnn_cell_dim
    if (('feed_attention_context_vec_to_softmax' in p.decoder and
         p.decoder.feed_attention_context_vec_to_softmax) or
        ('softmax_uses_attention' in p.decoder and
         p.decoder.softmax_uses_attention)):
      softmax_dim = p.decoder.rnn_cell_dim + p.decoder.source_dim
      ######### For ESN experiments #########
    if p.decoder.softmax_in_dim != 0:
      softmax_dim = p.decoder.softmax_in_dim
    with tf.name_scope('inference'):
      # Handle inputs.
      src_ids = tf.placeholder(tf.int32, shape=None, name='src_ids')
      src_paddings = tf.placeholder(tf.float32, shape=None, name='src_paddings')

      if p.input.natural_order_model:
        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      else:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids_rev = ops.reverse_ids(src_ids, slen)
        src_input_map = py_utils.NestedMap(
            ids=src_ids_rev, paddings=src_paddings)

      trg_ids = tf.placeholder(tf.int32, shape=None, name='trg_ids')
      trg_labels = tf.placeholder(tf.int32, shape=None, name='trg_labels')
      trg_paddings = tf.placeholder(tf.float32, shape=None, name='trg_paddings')
      trg_input_map = py_utils.NestedMap(
          ids=trg_ids, labels=trg_labels, paddings=trg_paddings)

      # Get scores.
      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)

      predictions = self.dec.ComputePredictions(self.theta.dec, encoder_outputs,
                                                trg_input_map)
      assert isinstance(predictions, py_utils.NestedMap)
      decoder_output = predictions.softmax_input

      decoder_output_orig_shape = py_utils.GetShape(decoder_output)
      logits_shape = tf.concat([[decoder_output_orig_shape[0]],
                                [decoder_output_orig_shape[1]], [num_classes]],
                               0)

      # Flatten for softmax calculation
      softmax_input = tf.reshape(decoder_output, [-1, softmax_dim])
      logits = self.dec.softmax.Logits(self.theta.dec.softmax, [softmax_input])
      log_probs = tf.nn.log_softmax(logits)
      log_probs_reshaped = tf.reshape(log_probs, logits_shape)

      log_probs = tf.transpose(log_probs_reshaped, perm=[1, 0, 2])
      sent_log_probs = self._GatherLogProbsByIds(trg_labels, log_probs)
      weights = 1 - trg_paddings
      sent_sumlog_probs = tf.reduce_sum(sent_log_probs * weights, axis=-1)

      length_normalization = 0
      coverage_penalty = 0
      target_seq_length_ratio = p.decoder.beam_search.target_seq_length_ratio
      norm_sent_sumlog_probs = sent_sumlog_probs
      if 'simple_length' in p.scoring_subgraph_normalization:
        tf.logging.info('Doing simple length normalization.')
        norm_sent_sumlog_probs = (
            norm_sent_sumlog_probs / tf.reduce_sum(weights, axis=-1))
      if 'beam_length' in p.scoring_subgraph_normalization:
        length_normalization = p.decoder.beam_search.length_normalization
      if 'coverage' in p.scoring_subgraph_normalization:
        coverage_penalty = p.decoder.beam_search.coverage_penalty
      if length_normalization > 0 or coverage_penalty > 0:
        tf.logging.info(
            'Doing normalization as in beam search, with '
            'length_normalization = %s and coverage_penalty = %s',
            length_normalization, coverage_penalty)
        norm_sent_sumlog_probs = decoder_utils.NormalizeScores(
            scores=norm_sent_sumlog_probs,
            atten_probs=predictions.attention.probs,
            source_paddings=src_paddings,
            target_paddings=trg_paddings,
            length_normalization=length_normalization,
            coverage_penalty=coverage_penalty,
            target_seq_length_ratio=target_seq_length_ratio)

      feeds = py_utils.NestedMap({
          'src_ids': src_ids,
          'src_paddings': src_paddings,
          'trg_ids': trg_ids,
          'trg_labels': trg_labels,
          'trg_paddings': trg_paddings
      })

      fetches = py_utils.NestedMap({
          'sum_logprobs': sent_sumlog_probs,
          'norm_sum_logprobs': norm_sent_sumlog_probs,
          'log_probs': sent_log_probs
      })

      return fetches, feeds


class ConvSeq2SeqModel(MTBaseModel):
  """Fully Convolutional Seq2Seq Model.

  Implements Convolutional Sequence to Sequence Learning:
  https://arxiv.org/abs/1705.03122
  """

  @classmethod
  def Params(cls):
    p = super(ConvSeq2SeqModel, cls).Params()
    p.encoder = encoder.ConvSeq2SeqEncoder.Params()
    p.decoder = decoder.ConvSeq2SeqDecoder.Params()
    p.train.Define('scale_encoder_grads', 1.0,
                   'Scale encoder, excluding embeddings, by this value.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(ConvSeq2SeqModel, self).__init__(params)

    p = self.params
    assert p.encoder.emb_dim == p.decoder.source_dim

    # list of gradients to be scaled by a given factor
    vs = self.vars
    self._vars_for_grad_scaling = py_utils.NestedMap(
        lst=[vs.enc.proj, vs.enc.source_enc_proj, vs.enc.gated_conv]).Flatten()
    self._selected_grad_scaling_factor = p.train.scale_encoder_grads

  def AdjustGradients(self, var_grads):
    """Scale selected gradients by given factor."""

    def AddScaleFactor(var_grad):
      var_grad_with_scale = py_utils.VarGrad(*var_grad)
      if var_grad.var in self._vars_for_grad_scaling:
        var_grad_with_scale.scale = self._selected_grad_scaling_factor
      else:
        var_grad_with_scale.scale = 1.0
      return var_grad_with_scale

    return py_utils.ApplyGradMultiplier(var_grads.Transform(AddScaleFactor))


@inference_registry.AllowClassParamsOverride
class TransformerModel(MTBaseModel):
  """Transformer Model.

  Implements Attention is All You Need:
  https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super(TransformerModel, cls).Params()
    p.encoder = encoder.TransformerEncoder.Params()
    p.decoder = decoder.TransformerDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(TransformerModel, self).__init__(params)
    p = self.params
    assert p.encoder.model_dim == p.decoder.source_dim

  def _InferenceSubgraph_Scoring(self):
    """Builds an inference graph that produces scores for the given targets.

    Scores are sum_logprobs.

    The feed needs to contain:
     src: src_ids and src_paddings (src ids)
     trg: trg_ids and trg_labels (where ids must start with <s> and labels
      must end with </s>); trg_paddings

    Returns:
     The following elements are contained in the fetches.
     sum_logprobs: per sentence summed log probs.
     norm_sent_sumlog_probs: length normalized sum_logprobs.
     log_probs: per sentence: log probs per word/unit.
    """
    p = self.params
    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    num_classes = p.decoder.softmax.num_classes
    softmax_dim = p.decoder.model_dim

    with tf.name_scope('inference'):
      # Handle inputs.
      src_ids = tf.placeholder(tf.int32, shape=None, name='src_ids')
      src_paddings = tf.placeholder(tf.float32, shape=None, name='src_paddings')

      source_languages = tf.placeholder(
          tf.string, shape=None, name='source_languages')
      target_languages = tf.placeholder(
          tf.string, shape=None, name='target_languages')

      if p.input is None or p.input.natural_order_model:
        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      else:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids_rev = ops.reverse_ids(src_ids, slen)
        src_input_map = py_utils.NestedMap(
            ids=src_ids_rev, paddings=src_paddings)

      if p.lang_id_list is not None:
        src_input_map['task_ids'] = tf.tile(
            tf.where(tf.equal(p.lang_id_list, source_languages))[:, 1,
                                                                 tf.newaxis],
            [1, tf.shape(src_ids)[1]],
            name='src_task_ids')

      trg_ids = tf.placeholder(tf.int32, shape=None, name='trg_ids')
      trg_labels = tf.placeholder(tf.int32, shape=None, name='trg_labels')
      trg_paddings = tf.placeholder(tf.float32, shape=None, name='trg_paddings')
      trg_input_map = py_utils.NestedMap(
          ids=trg_ids, labels=trg_labels, paddings=trg_paddings)

      if p.lang_id_list is not None:
        trg_input_map['task_ids'] = tf.tile(
            tf.dtypes.cast(
                tf.where(tf.equal(p.lang_id_list,
                                  target_languages))[:, 1, tf.newaxis],
                dtype=tf.int32), [1, tf.shape(trg_ids)[1]],
            name='trg_task_ids')

      # Get scores.
      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)

      decoder_output = self.dec.ComputePredictions(
          self.theta.dec, encoder_outputs, trg_input_map)

      if isinstance(decoder_output, py_utils.NestedMap):
        decoder_output = decoder_output.softmax_input

      decoder_output_orig_shape = py_utils.GetShape(decoder_output)
      logits_shape = tf.concat([[decoder_output_orig_shape[0]],
                                [decoder_output_orig_shape[1]], [num_classes]],
                               0)

      # Flatten for softmax calculation
      softmax_input = tf.reshape(decoder_output, [-1, softmax_dim])
      logits = self.dec.softmax.Logits(self.theta.dec.softmax, [softmax_input])
      log_probs = tf.nn.log_softmax(logits)
      log_probs_reshaped = tf.reshape(log_probs, logits_shape)

      log_probs = tf.transpose(log_probs_reshaped, perm=[1, 0, 2])
      sent_log_probs = self._GatherLogProbsByIds(trg_labels, log_probs)
      weights = 1 - trg_paddings
      sent_sumlog_probs = tf.reduce_sum(sent_log_probs * weights, axis=-1)
      norm_sent_sumlog_probs = (
          sent_sumlog_probs / tf.reduce_sum(weights, axis=-1))

      feeds = py_utils.NestedMap({
          'src_ids': src_ids,
          'src_paddings': src_paddings,
          'trg_ids': trg_ids,
          'trg_labels': trg_labels,
          'trg_paddings': trg_paddings,
          'source_languages': source_languages,
          'target_languages': target_languages
      })

      fetches = py_utils.NestedMap({
          'sum_logprobs': sent_sumlog_probs,
          'norm_sum_logprobs': norm_sent_sumlog_probs,
          'log_probs': sent_log_probs
      })

      return fetches, feeds


@inference_registry.AllowClassParamsOverride
class BucketingTransformerModel(TransformerModel):
  """Transformer Model with bucketing performance optimization.

  Wrapper around standard transformer model.
  """

  @classmethod
  def Params(cls):
    p = super(BucketingTransformerModel, cls).Params()
    p.Define('bucket_lengths', None,
             'A list of length used for Transformer bucketing,')
    return p

  def FPropTower(self, theta, input_batch):
    p = self.params

    # Compute the maximum sequence length based on paddings.
    max_src_len = tf.cast(
        tf.reduce_max(tf.reduce_sum(1.0 - input_batch.src.paddings, 1)),
        tf.int32)
    max_tgt_len = tf.cast(
        tf.reduce_max(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1)),
        tf.int32)
    max_len = tf.maximum(max_src_len, max_tgt_len)

    def _SwitchValue():
      return tf.reduce_min(
          tf.where(
              tf.less_equal(max_len, tf.constant(p.bucket_lengths,
                                                 max_len.dtype))))

    def _MakeBranch(seq_len):
      """Make FpropTower branch for each case."""

      def _FPropTowerBranch(nested_input):
        """The body of the FpropTower for each case."""
        theta = nested_input.theta
        actual_input_batch = nested_input.input_batch

        def _ResizeInput(x):
          if py_utils.GetRank(x) == 2:
            bs = py_utils.GetShape(x)[0]
            x = tf.reshape(tf.slice(x, [0, 0], [bs, seq_len]), [bs, seq_len])
            return x
          else:
            return x

        resized_input_batch = actual_input_batch.Transform(_ResizeInput)
        predictions = self.ComputePredictions(theta, resized_input_batch)
        temp_metrics, per_example = self.ComputeLoss(theta, predictions,
                                                     resized_input_batch)
        return py_utils.NestedMap(metrics=temp_metrics, per_example=per_example)

      return _FPropTowerBranch

    nested_input = py_utils.NestedMap(theta=theta, input_batch=input_batch)
    fns = [_MakeBranch(bucket_len) for bucket_len in p.bucket_lengths]
    nested_output = bbf_py_utils.SwitchCases(
        switch=_SwitchValue(), fns=fns, xs=nested_input)
    return nested_output.metrics, nested_output.per_example


@inference_registry.AllowClassParamsOverride
class OnlineBTTransformerModel(TransformerModel):
  """Transformer base model with online back-translation.

  Wrapper around standard transformer.
  """

  @classmethod
  def Params(cls):
    p = super(OnlineBTTransformerModel, cls).Params()
    p.Define('sos_id', 1, 'Start of sentence token id in vocab.')
    p.Define('eos_id', 2, 'End of sentence token id in vocab.')
    p.Define('pure_greedy_decode_during_onlinbt', False,
             'Use pure greedy decode during online back translation.')
    p.Define(
        'bt_num_hyps_per_beam', 1, 'Num of hyps per beam for back-translation. '
        'If > 1, randomly sample a translation from bt_num_hyps_per_beam '
        'hypotheses.')
    return p

  def _OnlineBackTrans(self, input_batch):
    p = self.params
    batch_size = py_utils.GetShape(input_batch.src.ids)[0]
    source_max_len = py_utils.GetShape(input_batch.src.ids)[1]
    target_max_len = py_utils.GetShape(input_batch.tgt.ids)[1]
    input_batch.src.task_ids = tf.tile(
        tf.expand_dims(input_batch.src.task_ids[:, 0], 1), [1, source_max_len])
    input_batch.tgt.task_ids = tf.tile(
        tf.expand_dims(input_batch.tgt.task_ids[:, 0], 1), [1, target_max_len])

    with cluster_factory.SetEval(True):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      # Task ids need to be set for forward translation task. This is required
      # by AddExtraDecodinginfo.
      encoder_outputs = self.dec.AddExtraDecodingInfo(encoder_outputs,
                                                      input_batch.tgt)
      # Greedy decode:
      # BeamSearchDecode with num_hypes_per_beam=1 is a simulation of pure
      # greedy.
      if p.pure_greedy_decode_during_onlinbt:
        new_src_ids, new_src_lens, _ = self.dec.GreedySearchDecode(
            encoder_outputs)
      else:
        num_hyps = p.bt_num_hyps_per_beam
        decoder_outs = self.dec.BeamSearchDecode(
            encoder_outputs, num_hyps_per_beam_override=num_hyps)
        # Make new batch
        new_src_ids = decoder_outs.topk_ids
        new_src_lens = decoder_outs.topk_lens

        if num_hyps > 1:
          # Randomly sample a translation from candidate hypotheses.
          indices = tf.random.uniform([batch_size],
                                      maxval=num_hyps,
                                      seed=p.random_seed,
                                      dtype=tf.int32)
          # Sentence offset
          indices += tf.range(
              0, num_hyps * batch_size, delta=num_hyps, dtype=tf.int32)
          new_src_ids = tf.gather(new_src_ids, indices)
          new_src_lens = tf.gather(new_src_lens, indices)

    # Account for empty sequences (assumes padding is 0)
    new_src_ids = tf.where(
        tf.equal(new_src_lens, 0),
        tf.concat([
            tf.fill([batch_size, 1], p.eos_id),
            tf.zeros([batch_size, source_max_len - 1], tf.int32)
        ],
                  axis=1), new_src_ids)
    new_src_lens = tf.where(
        tf.equal(new_src_lens, 0), tf.ones([batch_size], tf.int32),
        new_src_lens)

    # Summarize percentage of empty sentences in a batch.
    percent_empty = tf.reduce_sum(
        tf.cast(tf.equal(new_src_lens, 0), dtype=tf.int32)) / batch_size
    summary_utils.scalar('percent_empty_sent_in_bt', percent_empty)

    return new_src_ids, new_src_lens

  def _MakeNewBatch(self, input_batch):
    """Prepare new batches for online back-translation.

    Uses input_batch to compose bt_batch that can be used to train a
    back-translation task. First runs beam search (which can either be greedy,
    or when bt_num_hyps_per_beam > 1, do sampling) by feeding
    input_batch.src and using the tasks specified in input_batch.tgt.task_ids.
    The decoder result becomes bt_batch.src. while input_batch.src becomes
    bt_batch.tgt. bt_batch.src.task_ids is set to input_batch.tgt_ids.

    Args:
      input_batch: A `.NestedMap` object containing input tensors to this model.

    Returns:
      A `.NestedMap` object containing updated batches.
    """
    p = self.params
    assert hasattr(input_batch.src, 'task_ids')
    assert hasattr(input_batch.tgt, 'task_ids')

    new_src_ids, new_src_lens = self._OnlineBackTrans(input_batch)

    # Need to add segment_ids/segment_pos since still added to unpacked input
    bt_batch = py_utils.NestedMap()
    bt_batch.src = py_utils.NestedMap(
        ids=new_src_ids,
        paddings=(1 - tf.sequence_mask(
            new_src_lens,
            new_src_ids.shape[1],
            dtype=input_batch.src.paddings.dtype)),
        segment_ids=tf.sequence_mask(
            new_src_lens,
            new_src_ids.shape[1],
            dtype=input_batch.src.paddings.dtype),
        segment_pos=tf.pad(
            tf.ragged.range(new_src_lens).to_tensor(),
            [[0, 0], [0,
                      (new_src_ids.shape[1] - tf.reduce_max(new_src_lens))]]),
        # Task id is now from target side.
        task_ids=input_batch.tgt.task_ids)

    bt_batch.tgt = py_utils.NestedMap(
        ids=tf.pad(
            input_batch.src.ids, [[0, 0], [1, 0]],
            constant_values=p.sos_id)[:, :-1],
        labels=input_batch.src.ids,
        paddings=input_batch.src.paddings,
        segment_ids=input_batch.src.segment_ids,
        segment_pos=input_batch.src.segment_pos,
        # Task id is now from source side.
        task_ids=input_batch.src.task_ids,
        weights=(1 - input_batch.src.paddings))

    if 'strs' in input_batch.src:
      bt_batch.src['strs'] = self.input_generator.IdsToStrings(
          new_src_ids, new_src_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      bt_batch.tgt['strs'] = input_batch.src.strs

    updated_batch = py_utils.NestedMap(
        src=py_utils.NestedMap(), tgt=py_utils.NestedMap())

    if hasattr(input_batch, 'is_bt_task'):
      # Only update batch for back-translation tasks
      for st in ('src', 'tgt'):
        for b in input_batch[st]:
          updated_batch[st][b] = tf.where(input_batch.is_bt_task,
                                          bt_batch[st][b], input_batch[st][b])
    else:
      updated_batch = input_batch

    return updated_batch

  def FPropTower(self, theta, input_batch):
    if not self.do_eval:
      # Create new batch
      new_batch = self._MakeNewBatch(input_batch)
    else:
      new_batch = input_batch
    # Use newly created batch to compute loss
    return super(OnlineBTTransformerModel, self).FPropTower(theta, new_batch)


@inference_registry.AllowClassParamsOverride
class TransformerBatchMajorModel(MTBaseModel):
  """Transformer Model with batch major encoder and decoder.

  Implements Attention is All You Need:
  https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super(TransformerBatchMajorModel, cls).Params()
    p.encoder = encoder.TransformerBatchMajorEncoder.Params()
    p.decoder = decoder.TransformerBatchMajorDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    """Transformer batch-major model constructor.

    Args:
      params: Params used to construct this model.

    Raises:
      ValueError: If the decoder source_dim is different from the encoder
        model_dim.
    """
    super(TransformerBatchMajorModel, self).__init__(params)
    p = self.params
    if p.encoder.model_dim != p.decoder.source_dim:
      raise ValueError('The source_dim of Transformer decoder must be the '
                       'same as the model_dim of Transformer encoder.')


@inference_registry.AllowClassParamsOverride
class HybridModel(MTBaseModel):
  """Hybrid models combine the BNMT and Transformer models.

  BNMT and Transformer encoders and decoders can be combined. A typical
  combination is a Transformer encoder and a BNMT (V2) decoder.
  """

  @classmethod
  def Params(cls):
    """Placeholder params -- will be changed in most use-cases.

    We specify the params here, but in most usecases we will overwrite
    and modify the encoder/decoder settings. Ideally, HybridModel should be
    an abstract class and depending on the actual hybrid model configuration
    we have different implementations.

    Returns:
     Params for a hybrid model with encoder and decoder set to a default.
    """
    p = super(HybridModel, cls).Params()
    p.encoder = encoder.MTEncoderV1.Params()
    p.decoder = decoder.TransformerDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(HybridModel, self).__init__(params)

  def CreateDecoderMetrics(self):
    p = self.params
    if hasattr(p.input, 'target_language') and p.input.target_language:
      target_language = p.input.target_language
    else:
      target_language = p.eval.inference_target_language
    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'gleu': metrics.AverageMetric(),
        'corpus_bleu': metrics.CorpusBleuMetric(),
        'canonical_bleu': bf_metrics.CanonicalBleuMetric(target_language),
    }

    if p.eval.add_m2_to_eval_metrics:
      decoder_metrics['m2_f0.5'] = bf_metrics.M2FBetaMetric(target_language)
      decoder_metrics['m2_precision'] = bf_metrics.M2PrecisionMetric(
          target_language)
      decoder_metrics['m2_recall'] = bf_metrics.M2RecallMetric(target_language)

    return decoder_metrics

  def _InferenceSubgraph_Scoring(self):
    """Builds an inference graph that produces scores for the given targets.

    Scores are sum_logprobs.

    The feed needs to contain:
     src: src_ids and src_paddings (src ids)
     trg: trg_ids and trg_labels (where ids must start with <s> and labels
      must end with </s>); trg_paddings

    Returns:
     The following elements are contained in the fetches.
     sum_logprobs: per sentence summed log probs.
     norm_sent_sumlog_probs: length normalized sum_logprobs.
     log_probs: per sentence: log probs per word/unit.
    """
    p = self.params
    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    num_classes = p.decoder.softmax.num_classes

    softmax_dim = p.decoder.rnn_cell_dim
    if (('feed_attention_context_vec_to_softmax' in p.decoder and
         p.decoder.feed_attention_context_vec_to_softmax) or
        ('softmax_uses_attention' in p.decoder and
         p.decoder.softmax_uses_attention)):
      softmax_dim = p.decoder.rnn_cell_dim + p.decoder.source_dim

    with tf.name_scope('inference'):
      # Handle inputs.
      src_ids = tf.placeholder(tf.int32, shape=None, name='src_ids')
      src_paddings = tf.placeholder(tf.float32, shape=None, name='src_paddings')

      if p.input.natural_order_model:
        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      else:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids_rev = ops.reverse_ids(src_ids, slen)
        src_input_map = py_utils.NestedMap(
            ids=src_ids_rev, paddings=src_paddings)

      trg_ids = tf.placeholder(tf.int32, shape=None, name='trg_ids')
      trg_labels = tf.placeholder(tf.int32, shape=None, name='trg_labels')
      trg_paddings = tf.placeholder(tf.float32, shape=None, name='trg_paddings')
      trg_input_map = py_utils.NestedMap(
          ids=trg_ids, labels=trg_labels, paddings=trg_paddings)

      # Get scores.
      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)

      decoder_output = self.dec.ComputePredictions(
          self.theta.dec, encoder_outputs, trg_input_map)

      # TODO(colincherry) Change Hybrid ComputePredictions to return NestedMap
      if isinstance(decoder_output, py_utils.NestedMap):
        decoder_output = decoder_output.softmax_input

      decoder_output_orig_shape = py_utils.GetShape(decoder_output)
      logits_shape = tf.concat([[decoder_output_orig_shape[0]],
                                [decoder_output_orig_shape[1]], [num_classes]],
                               0)

      # Flatten for softmax calculation
      softmax_input = tf.reshape(decoder_output, [-1, softmax_dim])
      logits = self.dec.softmax.Logits(self.theta.dec.softmax, [softmax_input])
      log_probs = tf.nn.log_softmax(logits)
      log_probs_reshaped = tf.reshape(log_probs, logits_shape)

      log_probs = tf.transpose(log_probs_reshaped, perm=[1, 0, 2])
      sent_log_probs = tf.cast(
          self._GatherLogProbsByIds(trg_labels, log_probs), dtype=p.dtype)
      weights = 1 - trg_paddings
      sent_sumlog_probs = tf.reduce_sum(sent_log_probs * weights, axis=-1)
      norm_sent_sumlog_probs = (
          sent_sumlog_probs / tf.reduce_sum(weights, axis=-1))

      feeds = py_utils.NestedMap({
          'src_ids': src_ids,
          'src_paddings': src_paddings,
          'trg_ids': trg_ids,
          'trg_labels': trg_labels,
          'trg_paddings': trg_paddings
      })

      fetches = py_utils.NestedMap({
          'sum_logprobs': sent_sumlog_probs,
          'norm_sum_logprobs': norm_sent_sumlog_probs,
          'log_probs': sent_log_probs
      })

      return fetches, feeds


class ConvolutionalCharModel(MTBaseModel):
  """Model from Fully character NMT paper https://arxiv.org/abs/1610.03017."""

  @classmethod
  def Params(cls):
    p = super(ConvolutionalCharModel, cls).Params()
    p.encoder = encoder.ConvHighwayBiLSTMEncoder.Params()
    p.decoder = decoder.MTDecoderV1.Params()
    return p


@inference_registry.AllowClassParamsOverride
class MultiSourceModel(MTBaseModel):
  """Multi-Source Translation Model."""

  @classmethod
  def Params(cls):
    p = super(MultiSourceModel, cls).Params()
    p.encoder = encoder.MultiSourceEncoder.Params()
    p.decoder = decoder.MultiSourceDecoder.Params()

    p.Define(
        'inference_source_name', '',
        'Specify name for the main source. MultiSourceModel treats all sources '
        'identically during training. However, the prod stack assumes there is '
        'one main source input. This is used to specify the multi-source name '
        'for the main input.')

    ep = p.eval
    ep.Define(
        'canonical_bleu_source_list', [],
        'List of source names to compute canonical bleu for.'
        'If left empty, computes for all sources.')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MultiSourceModel, self).__init__(params)
    if params.inference_source_name:
      assert params.inference_source_name in params.input.source_names

  def _BeamSearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      # NOTE: Nmt training example proto contains source sequence ids
      # in the reversed order. Hence, we reverse it here. Its last id
      # is </s> and we ignore it.

      srcs = {}
      for lang, batch_src in input_batch.src.items():
        slen = tf.cast(
            tf.round(tf.reduce_sum(1 - batch_src.paddings, 1) - 1),
            dtype=tf.int32)
        if not self.input_generator.natural_order_model:
          # It works if you set the tokenizer_dict to map from lang to
          # tokenizer. If you don't set your own tokenizer_dict, only 'src',
          # 'tgt' are supported. For unknown lang, default tokenizer will be
          # used.
          srcs[lang] = self.input_generator.IdsToStrings(
              ops.reverse_ids(batch_src.ids, slen), slen,
              self._GetTokenizerKeyToUse(lang))
        else:
          srcs[lang] = self.input_generator.IdsToStrings(
              batch_src.ids, slen, self._GetTokenizerKeyToUse(lang))

      topk_decoded = self.input_generator.IdsToStrings(
          topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))
      topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

      refs = self.input_generator.IdsToStrings(
          input_batch.tgt.labels,
          tf.cast(
              tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
              dtype=tf.int32), self._GetTokenizerKeyToUse('tgt'))
      hyps = topk_decoded

      def ComputeTranslationQuality(metric):
        """Computes translation quality scores for all hyps."""
        scores = ops.per_sentence_translation_quality(
            tf.reshape(hyps, [-1]),
            tf.reshape(
                tf.tile(
                    tf.reshape(refs, [-1, 1]),
                    [1, p.decoder.beam_search.num_hyps_per_beam]), [-1]),
            metric=metric,
            max_ngram=4)
        return tf.reshape(scores, tf.shape(topk_hyps))

      gleu = ComputeTranslationQuality('gleu')

      ret_dict = {
          'target_ids': input_batch.tgt.ids,
          'target_labels': input_batch.tgt.labels,
          'target_weights': input_batch.tgt.weights,
          'target_paddings': input_batch.tgt.paddings,
          'sources': srcs,
          'targets': refs,
          'topk_decoded': topk_decoded,
          'topk_lens': topk_lens,
          'topk_scores': topk_scores,
          'gleu': gleu,
      }
      return ret_dict

  def _PostProcessBeamSearchDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post processes the output from _BeamSearchDecode."""
    p = self.params
    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    targets = dec_out_dict['targets']
    allsources = dec_out_dict['sources']
    gleu = dec_out_dict['gleu']

    num_samples = len(targets)
    assert num_samples == len(topk_decoded), ('%s vs %s' % (num_samples,
                                                            len(topk_decoded)))
    assert num_samples == len(gleu)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)

    key_value_pairs = []

    for i in range(num_samples):
      src, tgt = {lang: sources[i]
                  for lang, sources in allsources.items()}, targets[i]
      for lang in sorted(src):
        tf.logging.info('source[%s]: %s', lang, src[lang])
      tf.logging.info('  target: %s', tgt)
      hyps = topk_decoded[i]
      assert p.decoder.beam_search.num_hyps_per_beam == len(hyps)
      info = [src, tgt]
      for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
        tf.logging.info('  %f: %s', score, hyp_str)
        hyp_gleu = gleu[i][n]
        info += [(hyp_str, score, hyp_gleu)]
        # Only aggregate scores of the top hypothesis.
        if n == 0:
          dec_metrics_dict['gleu'].Update(hyp_gleu, 1)
          dec_metrics_dict['corpus_bleu'].Update(tgt, hyp_str)
          # CanonicalBleu metric depends on source sentence.
          # Therefore it is computed separately for each source.
          canonical_bleu_sources = p.eval.canonical_bleu_source_list
          if not canonical_bleu_sources:
            canonical_bleu_sources = src
          for lang in canonical_bleu_sources:
            dec_metrics_dict['canonical_bleu_%s' % lang].Update(
                tgt, hyp_str, src[lang])
          # Update rouge metrics
          if p.eval.add_rouge_to_eval_metrics:
            dec_metrics_dict['f1_rouge_1'].Update(tgt, hyp_str)
            dec_metrics_dict['f1_rouge_2'].Update(tgt, hyp_str)
            dec_metrics_dict['f1_rouge_L'].Update(tgt, hyp_str)
      key_value_pairs.append((str(src), '%s' % info))
    return key_value_pairs

  def CreateDecoderMetrics(self):
    p = self.params
    if hasattr(p.input, 'target_language') and p.input.target_language:
      target_language = p.input.target_language
    else:
      target_language = p.eval.inference_target_language

    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'gleu': metrics.AverageMetric(),
        'corpus_bleu': metrics.CorpusBleuMetric(),
    }

    source_names = p.eval.canonical_bleu_source_list
    if not source_names:
      source_names = set(
          source_name for source_names, _, _ in p.encoder.encoder_tpls
          for source_name in source_names.split(','))
    for lang in source_names:
      decoder_metrics['canonical_bleu_%s' % lang] = (
          bf_metrics.CanonicalBleuMetric(target_language))

    if p.eval.add_rouge_to_eval_metrics:
      decoder_metrics['f1_rouge_1'] = bf_metrics.RougeMetric(
          ngram=1, is_f_version=True, ignore_unk=True)
      decoder_metrics['f1_rouge_2'] = bf_metrics.RougeMetric(
          ngram=2, is_f_version=True, ignore_unk=True)
      decoder_metrics['f1_rouge_L'] = bf_metrics.RougeMetric(
          ngram=0, is_f_version=True, ignore_unk=True)

    return decoder_metrics

  def _InferenceSubgraph_Default(self, src, tgt):
    p = self.params

    # Currently we need to specify p.inference_source_name, which corresponds
    # to the source sentence during inference and maps to one of the src_names
    # used while training the multi-source model.
    assert p.inference_source_name in p.input.source_names

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):
      # TODO(yonghui): Make sure the following is the same preprocessing as we
      # do in preparing our NMT training data.
      def PrepareSource(src_name, inference_source_name=''):
        """Creates string placeholder and converts string to ids and padding."""
        src_strings = tf.placeholder(
            tf.string, shape=[None], name='%s_strings' % src_name)
        upp_tokenized_src = ops.upp_tokenize(src_strings, language=src)
        tokenizer_key = self._GetTokenizerKeyToUse('src')
        if not tokenizer_key and inference_source_name:
          tokenizer_key = self._GetTokenizerKeyToUse(inference_source_name)
        _, src_ids, src_paddings = self.input_generator.StringsToIds(
            upp_tokenized_src,
            external_max_length=p.decoder.target_seq_len,
            key=tokenizer_key)

        # Truncate paddings at the end.
        max_seq_length = tf.cast(
            tf.round(tf.reduce_max(tf.reduce_sum(1.0 - src_paddings, 1))),
            dtype=tf.int32)
        src_paddings = py_utils.with_dependencies([
            py_utils.assert_equal(
                tf.constant(True, tf.bool),
                tf.reduce_all(src_paddings[:, max_seq_length:] > 0.5))
        ], src_paddings)
        src_ids = src_ids[:, :max_seq_length]
        src_paddings = src_paddings[:, :max_seq_length]
        # Last step, reverse the source sequence if it is training an reversed
        # order model.
        if not self.input_generator.natural_order_model:
          slen = tf.cast(
              tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1),
              dtype=tf.int32)
          src_ids = ops.reverse_ids(src_ids, slen)

        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
        return src_input_map, src_strings

      feed_dict = {}
      src_input_map = py_utils.NestedMap()
      for src_name in p.input.source_names:
        placeholder_src_name = src_name
        if src_name == p.inference_source_name:
          placeholder_src_name = 'src'
        src_input_map[src_name], src_string_placeholder = PrepareSource(
            placeholder_src_name, p.inference_source_name)
        feed_dict['%s_strings' % placeholder_src_name] = src_string_placeholder

      src_ids = src_input_map[p.inference_source_name].ids

      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      topk_decoded = self.input_generator.IdsToStrings(
          topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))

      # last step, runs the upp untokenizer
      topk_decoded = ops.upp_detokenize(topk_decoded, language=tgt)

      feeds = py_utils.NestedMap(feed_dict)
      fetches = py_utils.NestedMap({
          'topk_decoded': topk_decoded,
          'topk_hyps': topk_hyps,
          'topk_scores': topk_scores,
          'src_ids': src_ids,
      })

      return fetches, feeds

  # Builds an inference subgraph with src_ids and src_paddings expected to be
  # passed in as feed data. Additionally generates ids and paddings feeds for
  # all other sources specified in params.input.source_names. For example, for
  # a source named 'de' it will generate feeds for 'de_ids' and 'de_paddings'.
  # Returns the topk hypotheses, target token IDs and target sentence lengths.
  # This subgraph allows for running the preprocessor and postprocessor outside
  # of the Babelfish inference stack.
  def _InferenceSubgraph_Prod(self):
    p = self.params

    # Currently we need to specify p.inference_source_name, which corresponds
    # to the source sentence during inference and maps to one of the src_names
    # used while training the multi-source model.
    if p.inference_source_name not in p.input.source_names:
      raise ValueError('Inference source should be present in source_names.')

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):

      def PrepareSource(src_name):
        """Creates ids and paddings placeholders and reverses if necessary."""
        src_ids = tf.placeholder(tf.int32, shape=None, name='%s_ids' % src_name)
        src_paddings = tf.placeholder(
            tf.float32, shape=None, name='%s_paddings' % src_name)

        if p.input.natural_order_model:
          src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
        else:
          slen = tf.cast(
              tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1),
              dtype=tf.int32)
          src_ids_rev = ops.reverse_ids(src_ids, slen)
          src_input_map = py_utils.NestedMap(
              ids=src_ids_rev, paddings=src_paddings)
        return src_input_map

      # This assumes that the same source names specified in the input_generator
      # are specified during inference.
      feed_dict = {}
      src_input_map = py_utils.NestedMap()
      for src_name in p.input.source_names:
        placeholder_src_name = src_name
        if src_name == p.inference_source_name:
          placeholder_src_name = 'src'
        src_input_map[src_name] = PrepareSource(placeholder_src_name)
        feed_dict['%s_ids' % placeholder_src_name] = src_input_map[src_name].ids
        feed_dict['%s_paddings' %
                  placeholder_src_name] = src_input_map[src_name].paddings

      src_ids = src_input_map[p.inference_source_name].ids

      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))
      batch_size = tf.shape(src_ids)[0]

      # Reshape topk_ids from 2D to 3D so we can iterate across hyps for a given
      # source:
      # topk_ids: [batch, number of hyps, max num sequence ids]
      topk_ids = tf.reshape(
          topk_ids, [batch_size, p.decoder.beam_search.num_hyps_per_beam, -1])

      # Reshape topk_lens from 1D to 2D so we can iterate lens for the nbest
      # for each source:
      # topk_lens: [batch, number of hyps]
      topk_lens = tf.reshape(
          topk_lens, [batch_size, p.decoder.beam_search.num_hyps_per_beam])

      feeds = py_utils.NestedMap(feed_dict)

      fetches = py_utils.NestedMap({
          'topk_hyps': topk_hyps,
          'topk_ids': topk_ids,
          'topk_lens': topk_lens,
          'topk_scores': topk_scores,
      })

      return fetches, feeds


class MTGraphModel(MTBaseModel):
  """MT model using graph encoder and/or graph decoder."""

  @classmethod
  def Params(cls):
    p = super(MTGraphModel, cls).Params()
    p.Define(
        'atten_bundle', [], 'A list of hyperparams specifying the '
        'attention pair between graph encoder and decoders. Only valid '
        'when at least one of the encoder/decoder is graph '
        'encoder/decoder.')

    p.Define('atten_pair', hyperparams.Params(), 'Specifies one attention '
             'pair.')
    ap = p.atten_pair
    ap.Define(
        'enc_layer_ids', [], 'A list enumerating the ids of encoder '
        'layers whose encodings will paid attention to. If not specified '
        'then the sum of output layers will be used as encodings.')
    ap.Define(
        'dec_layer_ids', [], 'A list enumerating the ids of decoder '
        'layers whose encodings will paid attention to (for now only '
        'a single decoder query node is supported, so the list can '
        'contain only 1 entry).')

    p.encoder = encoder.GraphEncoder.Params()
    p.decoder = decoder.GraphDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    if params.encoder.cls == encoder.GraphEncoder:
      # Figure out which layers' encodings we need to collect.
      id_set = set()
      for ap in params.atten_bundle:
        for i in ap.enc_layer_ids:
          id_set.add(i)
      params.encoder.enc_layer_ids = id_set

    if params.decoder.cls == decoder.GraphDecoder:
      params.decoder.atten_bundle = params.atten_bundle

    super(MTGraphModel, self).__init__(params)
    assert self.params.name


@inference_registry.AllowClassParamsOverride
class EnsembleModel(MTBaseModel):
  """Builds a ensemble model with EnsembleEncoder and EnsembleDecoder."""

  @classmethod
  def Params(cls):
    """Placeholder params -- will be changed in most use-cases.

    We specify the params here, but in most usecases we will overwrite
    and modify the encoder/decoder settings. The default config trains a model
    with ensemble of 2.

    Returns:
     Params for a ensemble model with encoder and decoder set to a default.
    """
    p = super(EnsembleModel, cls).Params()
    p.encoder = encoder.EnsembleEncoder.Params()
    p.encoder.encoder_tpls = [encoder.MTEncoderV1.Params() for _ in range(2)]
    p.decoder = decoder.EnsembleDecoder.Params()
    p.decoder.decoder_tpls = [decoder.MTDecoderV1.Params() for _ in range(2)]
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(EnsembleModel, self).__init__(params)
    p = self.params
    assert len(p.encoder.encoder_tpls) == len(p.decoder.decoder_tpls)


@inference_registry.AllowClassParamsOverride
class GPipeTransformerModel(MTBaseModel):
  """Transformer Model using GPipe.

  Implements Attention is All You Need:
  https://arxiv.org/abs/1706.03762
  """

  @classmethod
  def Params(cls):
    p = super(GPipeTransformerModel, cls).Params()
    p.Define('stack', layers_with_gpipe.GPipeTransformerStack.Params(),
             'GPipeTransformerStack Layer params.')

    p.Define('init_decoder_step_ids', False,
             'Initializes beam search with first target id instead of <s>.')
    p.Define('use_task_ids', False,
             'If task ids are to be used as input to the model.')

    # Dummy Params for compatibility with standard interfaces.
    p.encoder = base_layer.BaseLayer.Params()
    p.encoder.Define('packed_input', False, 'Unused param for compatibility.')
    p.decoder = base_decoder.BaseBeamSearchDecoder.Params()
    p.decoder.beam_search.batch_major_state = False
    return p

  @classmethod
  def CommonParams(cls,
                   vocab_size,
                   model_dim,
                   hidden_dim=1024,
                   num_heads=8,
                   num_layers=6,
                   num_micro_batches=1,
                   num_shards=16,
                   input_dropout_prob=0.0,
                   residual_dropout_prob=0.1,
                   atten_dropout_prob=0.0,
                   relu_dropout_prob=0.0,
                   label_smoothing_uncertainty=0.1,
                   is_transparent=False,
                   add_unnormalized_input=False,
                   normalize_encoder=False,
                   use_ff_fixup=False,
                   packed_input=False,
                   atten_hidden_dim=0,
                   splits=1,
                   micro_batch_size=None,
                   num_experts=None,
                   expert_capacity_dim=None,
                   batch_major=False,
                   num_encoder_layers=None,
                   num_decoder_layers=None,
                   encoder_hidden_dim=None,
                   decoder_hidden_dim=None,
                   encoder_vocab_size=None,
                   decoder_vocab_size=None,
                   logits_clipping=0.0,
                   use_task_ids=False,
                   moe_type=None,
                   num_dec_tasks=0,
                   num_enc_tasks=0):
    """Common setup for Transformer language models.

    Args:
      vocab_size: vocab size.
      model_dim: model dimension.
      hidden_dim: hidden dimension of feed-forward inner layer.
      num_heads: number of attention heads.
      num_layers: number of layers in the transformer LM.
      num_micro_batches: number of micro batches for GPipe.
      num_shards: num_shards for softmax. Assert vocab_size % num_shards == 0
      input_dropout_prob: dropout prob to the sums of the token embeddings and
        the position embeddings.
      residual_dropout_prob: dropout prob to the output of each sub-layer before
        it is added to the sub-layer input.
      atten_dropout_prob: dropout prob to the attention weights in each
        Transformer attention sub-layer.
      relu_dropout_prob: dropout prob to the inner layer output (ReLU
        activation) in each Transformer feed-forward sub-layer.
      label_smoothing_uncertainty: Label smoothing uncertainty.
      is_transparent: If True, encoder outputs a merger of embeddings and layer
        outputs.
      add_unnormalized_input: In transformer attention, add unnormalized input.
      normalize_encoder: To layer-normalize encoder outputs.
      use_ff_fixup: To apply layer scaled initialization to FF layer.
      packed_input: If True, assumes multiple training samples per input.
      atten_hidden_dim: attention hidden dim size, If 0, default is model_dim.
      splits: Number of splits or list of integers specifying the ending index
        for each split. Must always be in ascending order and last index should
        be num_layers. Example, for a model with 6 encoder and decoder layers
        splits could be [3, 6, 9, 12]. Split 1 will have the 3 encoder layers,
        split 2 will have the next 3, split 3 will have the first 3 decoder
        layers and split 4 the final 3 decoder layers. Both Embeddings are
        placed with split 1, and softmax with split 4.
      micro_batch_size: Size of each micro batches. If None, the value will be
        computed from batch_size // num_micro_batches.
      num_experts: Number of Experts. None to use regular feedforward layer.
      expert_capacity_dim: Number of examples per minibatch(group) per expert.
      batch_major: use batch major transformer layer implementation.
      num_encoder_layers: to set a different number of layers for the encoder.
      num_decoder_layers: to set a different number of layers for the decoder.
      encoder_hidden_dim: to set different feed-forward hidden dim for encoder.
      decoder_hidden_dim: to set different feed-forward hidden dim for decoder.
      encoder_vocab_size: to set different vocab_size for encoder.
      decoder_vocab_size: to set different vocab_size for decoder.
      logits_clipping: clipping magnitude for softmax logits.
      use_task_ids: If there is a task_ids attribute in the input batch.
      moe_type: Specifies type of MoE, if any. Must be one of None, 'top_2_moe',
        'static_moe' or 'static_moe_ln'. Defaults to None, which creates a
        FeedForwardLayer instead of MOE.
      num_dec_tasks: Specify number of tasks for task embeddings - added to
        every decoder token embedding as input. Setting to 0 disables.
      num_enc_tasks: Specify number of tasks for task embeddings - added to
        every encoder token embedding as input. Setting to 0 disables.

    Returns:
      A Params object containing the parameters that set up a Transformer LM.
    """
    p = cls.Params()
    p.name = 'transformer'
    use_static_moe = moe_type in ['static_moe', 'static_moe_ln']
    use_moe = moe_type in ['moe', 'sentence_moe', 'task_moe']
    if use_moe:
      assert num_experts and expert_capacity_dim
    if use_static_moe:
      assert num_experts and batch_major and not expert_capacity_dim
    assert moe_type in [
        None, 'moe', 'static_moe', 'static_moe_ln', 'sentence_moe', 'task_moe'
    ]

    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Xavier(1.0)
    emb_params_init = py_utils.WeightInit.Gaussian(1.0 / math.sqrt(model_dim))

    emb_params = p.stack.emb_tpl
    # Default config for the token embedding.
    emb_params.token_emb.use_matmul = True
    emb_params.token_emb.use_3d_weight_tensor = False
    emb_params.token_emb.vocab_size = encoder_vocab_size or vocab_size
    emb_params.target_vocab_size = decoder_vocab_size or vocab_size
    emb_params.token_emb.embedding_dim = model_dim
    emb_params.token_emb.params_init = emb_params_init
    emb_params.token_emb.vn = disable_vn

    # Default config for the position embedding.
    emb_params.position_emb.embedding_dim = model_dim
    emb_params.position_emb.trainable_scaling = False
    emb_params.position_emb.vn = disable_vn
    # Embedding layers pass task_ids to followup transformer layers.
    emb_params.ret_task_ids = use_static_moe

    emb_params.input_dropout_prob = input_dropout_prob
    emb_params.packed_input = packed_input
    # Use task embeddings if num_tasks > 0.
    # TODO(huangyp,snehakudugunta): Figout out why task embeddings failed
    # when static_moe is used.
    p.use_task_ids = use_task_ids
    if use_task_ids and num_enc_tasks and not use_static_moe:
      emb_params.enc_task_emb = emb_params.token_emb.Copy()
      emb_params.enc_task_emb.vocab_size = num_enc_tasks
    if use_task_ids and num_dec_tasks and not use_static_moe:
      emb_params.dec_task_emb = emb_params.token_emb.Copy()
      emb_params.dec_task_emb.vocab_size = num_dec_tasks

    p.stack.model_dim = model_dim
    p.stack.splits = splits
    p.stack.micro_batch_size = micro_batch_size
    p.stack.num_micro_batches = num_micro_batches

    # Default config for the transformer encoder layers.
    num_encoder_layers = num_encoder_layers or num_layers
    p.stack.num_encoder_layers = num_encoder_layers
    p.stack.normalize_encoder = normalize_encoder
    p.stack.packed_input = packed_input
    p.stack.is_transparent = is_transparent

    if batch_major:
      p.stack.batch_dim = 0
      assert not is_transparent
      encoder_tpl = batch_major_attention.GPipeTransformerLayer.Params()
      encoder_tpl.packed_input = packed_input
      encoder_tpl.input_dim = model_dim
      encoder_tpl.output_dim = model_dim
      encoder_tpl.tr_atten_tpl.input_dim = model_dim
      encoder_tpl.tr_atten_tpl.num_heads = num_heads
      encoder_tpl.tr_atten_tpl.residual_dropout_prob = residual_dropout_prob
      encoder_tpl.tr_atten_tpl.atten_dropout_prob = atten_dropout_prob
      encoder_tpl.tr_atten_tpl.add_unnormalized_input = add_unnormalized_input
      encoder_tpl.tr_atten_tpl.atten_tpl.packed_input = packed_input
      encoder_tpl.tr_atten_tpl.hidden_dim = atten_hidden_dim
      p.stack.encoder_tpl = encoder_tpl
    else:
      encoder_tpl = p.stack.encoder_tpl
      encoder_tpl.source_dim = model_dim
      encoder_tpl.has_aux_atten = False
      encoder_tpl.tr_atten_tpl.source_dim = model_dim
      encoder_tpl.tr_atten_tpl.num_attention_heads = num_heads
      encoder_tpl.tr_atten_tpl.residual_dropout_prob = residual_dropout_prob
      encoder_tpl.tr_atten_tpl.atten_dropout_prob = atten_dropout_prob
      encoder_tpl.tr_atten_tpl.params_init = default_params_init
      encoder_tpl.tr_atten_tpl.atten_hidden_dim = atten_hidden_dim
      encoder_tpl.tr_atten_tpl.vn = disable_vn
      encoder_tpl.tr_atten_tpl.add_unnormalized_input = add_unnormalized_input
      encoder_tpl.tr_atten_tpl.atten_tpl.num_attention_heads = num_heads
      encoder_tpl.tr_atten_tpl.atten_tpl.enable_ctx_pre_proj = True
      encoder_tpl.tr_atten_tpl.atten_tpl.enable_ctx_post_proj = True
      encoder_tpl.tr_atten_tpl.atten_tpl.context_dim = model_dim
      encoder_tpl.tr_atten_tpl.atten_tpl.vn = disable_vn

    if moe_type == 'static_moe_ln':
      encoder_tpl.tr_fflayer_tpl = (
          batch_major_attention.StaticMOEFeedForwardLayer.Params().Set(
              num_experts=num_experts,
              output_dim=model_dim,
              use_task_ids=use_task_ids))
    encoder_tpl.tr_fflayer_tpl.input_dim = model_dim
    encoder_tpl.tr_fflayer_tpl.hidden_dim = encoder_hidden_dim or hidden_dim
    if use_moe:
      if moe_type == 'moe':
        encoder_tpl.tr_fflayer_tpl.fflayer_tpl = (
            mt_layers.MoEPositionWiseFeedForwardNetworks.Params().Set(
                experts_dim=num_experts,
                gating_tpl=mt_layers.MoETop2GatingLayer.Params().Set(
                    expert_capacity_dim=expert_capacity_dim)))
      elif moe_type == 'sentence_moe':
        encoder_tpl.tr_fflayer_tpl.fflayer_tpl = (
            mt_layers.MoESentenceWiseFeedForwardNetworks.Params().Set(
                experts_dim=num_experts,
                gating_tpl=mt_layers.MoESentenceTop2GatingLayer.Params().Set(
                    expert_capacity_dim=expert_capacity_dim,
                    embedding_fn=moe_layers.GetSentenceEmbeddings)))
      elif moe_type == 'task_moe':
        encoder_tpl.tr_fflayer_tpl.fflayer_tpl = (
            mt_layers.MoESentenceWiseFeedForwardNetworks.Params().Set(
                experts_dim=num_experts,
                gating_tpl=mt_layers.MoESentenceTop2GatingLayer.Params().Set(
                    expert_capacity_dim=expert_capacity_dim,
                    embedding_fn=moe_layers.GetTaskEmbeddings)))
    elif moe_type == 'static_moe':
      assert num_experts and batch_major and not expert_capacity_dim
      encoder_tpl.tr_fflayer_tpl.fflayer_tpl = (
          mt_layers.StaticMoEFeedForwardNetworks.Params().Set(
              num_experts=num_experts))
    else:
      pass

    encoder_tpl.tr_fflayer_tpl.residual_dropout_prob = residual_dropout_prob
    encoder_tpl.tr_fflayer_tpl.relu_dropout_prob = relu_dropout_prob
    encoder_tpl.tr_fflayer_tpl.params_init = default_params_init
    encoder_tpl.tr_fflayer_tpl.vn = disable_vn
    if use_ff_fixup:
      encoder_tpl.tr_fflayer_tpl.params_init = (
          py_utils.WeightInit.XavierWithFixupParams(1.0, num_encoder_layers, 2))

    # Default config for the transformer decoder layers.
    num_decoder_layers = num_decoder_layers or num_layers
    p.stack.num_decoder_layers = num_decoder_layers
    p.stack.decoder_tpl = encoder_tpl.Copy()
    decoder_tpl = p.stack.decoder_tpl
    decoder_tpl.tr_fflayer_tpl.hidden_dim = decoder_hidden_dim or hidden_dim
    decoder_tpl.has_aux_atten = True
    decoder_tpl.mask_self_atten = True

    if use_ff_fixup:
      decoder_tpl.tr_fflayer_tpl.params_init = (
          py_utils.WeightInit.XavierWithFixupParams(1.0, num_decoder_layers, 2))

    # Softmax Layer.
    decoder_vocab_size = decoder_vocab_size or vocab_size
    p.stack.softmax_tpl.Set(
        input_dim=model_dim,
        num_classes=decoder_vocab_size,
        num_shards=num_shards,
        params_init=emb_params_init)
    if label_smoothing_uncertainty > 0:
      p.stack.label_smoothing = layers.UniformLabelSmoother.Params()
      p.stack.label_smoothing.num_classes = decoder_vocab_size
      p.stack.label_smoothing.uncertainty = label_smoothing_uncertainty
    if logits_clipping:
      p.stack.softmax_tpl.logits_abs_max = logits_clipping

    p.decoder.target_seq_len = 300
    p.decoder.beam_search.length_normalization = 0.5
    p.decoder.beam_search.coverage_penalty = 0.0
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(GPipeTransformerModel, self).__init__(params)
    p = self.params
    p.stack.packed_input = p.stack.packed_input and not self.do_eval
    p.stack.state_dtype = p.dtype
    if p.fprop_dtype:
      p.stack.state_dtype = p.fprop_dtype
    emb_p = p.stack.emb_tpl
    assert emb_p.target_vocab_size == p.stack.softmax_tpl.num_classes
    assert emb_p.token_emb.embedding_dim == emb_p.position_emb.embedding_dim
    self.enc.FPropDefaultTheta = self._EncoderFPropDefaultTheta
    self.dec.BeamSearchDecode = self._DecoderBeamSearchDecode
    self.dec.ComputePredictions = self.DecoderComputePredictions
    self.dec.ExtendStep = self.ExtendStep

    with tf.variable_scope(p.name):
      self.CreateChild('stack', p.stack)

  def _PrepareInputs(self, input_batch):
    p = self.params
    paddings = input_batch.paddings
    input_ids = py_utils.with_dependencies([
        py_utils.assert_shape_match(
            tf.shape(input_batch.ids), tf.shape(paddings)),
        py_utils.assert_equal(tf.rank(input_batch.ids), 2)
    ], input_batch.ids)
    segment_pos = None
    segment_id = None
    if p.stack.packed_input:
      segment_pos = input_batch.segment_pos
      segment_id = input_batch.segment_ids
      if p.stack.batch_dim:
        # [seq_len, batch]
        segment_pos = tf.transpose(segment_pos)
        segment_id = tf.transpose(segment_id)
    if p.stack.batch_dim:  # time-major
      # [seq_len, batch]
      input_ids = tf.transpose(input_ids)
      paddings = tf.transpose(paddings)
    task_ids = None
    if p.use_task_ids:
      task_ids = input_batch.task_ids
      if p.stack.batch_dim:  # time-major
        task_ids = tf.transpose(task_ids)
    return input_ids, paddings, segment_id, segment_pos, task_ids

  def FPropTower(self, theta, input_batch):
    """Forward propagation through one tower of the model.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A dict containing metrics pairs.
    """
    p = self.params
    with tf.name_scope(p.name):
      (src_input_ids, src_paddings, src_segment_ids, src_segment_pos,
       src_task_ids) = self._PrepareInputs(input_batch.src)
      (tgt_input_ids, tgt_paddings, tgt_segment_ids, tgt_segment_pos,
       tgt_task_ids) = self._PrepareInputs(input_batch.tgt)
      tgt_labels = input_batch.tgt.labels
      tgt_weights = input_batch.tgt.weights
      if p.stack.batch_dim:  # time-major
        tgt_labels = tf.transpose(input_batch.tgt.labels)
        tgt_weights = tf.transpose(input_batch.tgt.weights)
      per_example_xent, logits = self.stack.FProp(
          theta.stack, src_input_ids, src_paddings, tgt_input_ids, tgt_paddings,
          src_segment_ids, tgt_segment_ids, tgt_labels, tgt_weights,
          src_segment_pos, tgt_segment_pos, src_task_ids, tgt_task_ids)

      total_weights = tf.reduce_sum(tgt_weights)
      loss = tf.reduce_sum(per_example_xent * tgt_weights) / total_weights

    ret_dict = {
        'loss': (loss, total_weights),
        'log_pplx': (loss, total_weights)
    }
    per_example_loss = {}
    per_example_loss['loss'] = tf.reduce_sum(per_example_xent * tgt_weights, 0)
    # NOTE: tf.argmax is not implemented for the JF backend, see b/36093673
    # Skip the fraction_of_correct_next_step_preds during training.
    if self.do_eval:
      correct_preds = tf.cast(
          tf.equal(tf.cast(tf.argmax(logits, 2), tf.int32), tgt_labels),
          p.dtype)
      correct_next_preds = tf.reduce_sum(correct_preds * tgt_weights)
      num_preds = tf.reduce_sum(tgt_weights)
      accuracy = tf.identity(
          correct_next_preds / num_preds,
          name='fraction_of_correct_next_step_preds')
      ret_dict['fraction_of_correct_next_step_preds'] = (accuracy, num_preds)
    return ret_dict, per_example_loss

  def DecoderComputePredictions(self, theta, encoder_outputs, targets):
    source_encs = encoder_outputs.encoded
    source_paddings = encoder_outputs.padding
    src_segment_id = encoder_outputs.segment_id

    (tgt_ids, tgt_paddings, tgt_segment_ids, _,
     tgt_task_ids) = self._PrepareInputs(targets)
    tgt_encs = self.stack.DecoderEmbedFPropDefaultTheta(tgt_ids, tgt_task_ids)
    for decoder_l in self.stack.GetDecoders():
      layer_outs = decoder_l.FProp(decoder_l.theta, source_encs,
                                   source_paddings, tgt_encs, tgt_paddings,
                                   src_segment_id, tgt_segment_ids, None, None)
      tgt_encs = layer_outs[2]
    return tgt_encs

  def _MaybeAppendAdditionalDecodingInputs(self, encoder_outputs, input_batch):
    """Adds additional information required for decoding to encoder_outputs."""
    p = self.params
    if p.use_task_ids:
      encoder_outputs['task_ids'] = tf.tile(
          input_batch.tgt.task_ids[:, 0],
          [p.decoder.beam_search.num_hyps_per_beam])
    if p.init_decoder_step_ids:
      encoder_outputs['init_step_ids'] = input_batch.tgt.ids[:, 0]
    return encoder_outputs

  def _BeamSearchDecode(self, input_batch):
    p = self.params
    with tf.name_scope('fprop'), tf.name_scope(p.name):
      encoder_outputs = self.enc.FPropDefaultTheta(input_batch.src)
      encoder_outputs = self._MaybeAppendAdditionalDecodingInputs(
          encoder_outputs, input_batch)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      return self._ProcessBeamSearchDecodeOut(input_batch, encoder_outputs,
                                              decoder_outs)

  def ExtendStep(self, theta, encoder_outputs, new_ids, t, prefix_states):
    p = self.params
    source_encs = encoder_outputs.encoded
    source_paddings = encoder_outputs.padding
    if not p.stack.batch_dim:
      # For batch-major model, first transpose the time-major encoder_outputs.
      source_encs = tf.transpose(source_encs, [1, 0, 2])
      source_paddings = tf.transpose(source_paddings)
    # Embedding layer
    # [batch, time, model_dim]
    tgt_input_ids = new_ids
    # Make a copy of the input.
    out_prefix_states = prefix_states.Pack(prefix_states.Flatten())
    tgt_task_ids = None
    if p.use_task_ids:
      tgt_task_ids = encoder_outputs.task_ids
    layer_in = self.stack.DecoderEmbedFPropDefaultTheta(tgt_input_ids,
                                                        tgt_task_ids, t)

    for (i, decoder_l) in enumerate(self.stack.GetDecoders()):
      layer_prefix_states = prefix_states['decoder_%i' % i]
      if p.stack.batch_dim:  # time-major
        layer_out, _, updated_prefix_states = decoder_l.ExtendStep(
            decoder_l.theta, layer_in, layer_prefix_states, source_encs,
            source_paddings,
            t if p.decoder.beam_search.name == 'tpu_beam_search' else
            None)  # tpu_beam_search = False
      else:
        layer_out, updated_prefix_states = decoder_l.ExtendStep(
            decoder_l.theta,
            layer_in,
            source_encs,
            source_paddings,
            layer_prefix_states,
            t,
            task_id=tgt_task_ids)
      out_prefix_states['decoder_%i' % i] = updated_prefix_states
      layer_in = layer_out

    if not p.stack.batch_dim:  # batch-major
      layer_out = tf.squeeze(layer_out, 1)
    return layer_out, out_prefix_states

  def _InitBeamSearchStateBatchMajorCallback(self, theta, encoder_outputs,
                                             num_hyps_per_beam):
    p = self.params

    # [source_batch, source_time, dim]
    source_encs = encoder_outputs.encoded
    target_batch = py_utils.GetShape(source_encs)[1] * num_hyps_per_beam
    source_time = py_utils.GetShape(source_encs)[0]
    target_time = p.decoder.target_seq_len

    log_probs = tf.zeros([target_batch, p.stack.softmax_tpl.num_classes],
                         dtype=py_utils.FPropDtype(p))
    # Dummy attention probs
    atten_probs = (
        tf.ones([target_batch, source_time], dtype=py_utils.FPropDtype(p)) /
        tf.cast(source_time, py_utils.FPropDtype(p)))
    initial_results = py_utils.NestedMap(
        log_probs=log_probs, atten_probs=atten_probs)
    dim = p.stack.decoder_tpl.tr_atten_tpl.hidden_dim
    if not dim:
      dim = p.stack.model_dim
    num_heads = p.stack.decoder_tpl.tr_atten_tpl.num_heads
    # If per-head dim is less than 128, make the cached shape 128 to avoid
    # padding and more efficient interpolation in beamsearch.
    if dim // num_heads < 128 and dim % 128 == 0:
      num_heads = dim // 128

    def _GenStates():
      return py_utils.NestedMap({
          'key':
              tf.zeros([target_time, target_batch, num_heads, dim // num_heads],
                       dtype=py_utils.FPropDtype(p)),
          'value':
              tf.zeros([target_time, target_batch, num_heads, dim // num_heads],
                       dtype=py_utils.FPropDtype(p)),
      })

    prefix_states = py_utils.NestedMap({
        'decoder_%d' % l: _GenStates()
        for l in range(p.stack.num_decoder_layers)
    })

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(0)
    })

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    """Returns initial beams search states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      num_hyps_per_beam: An int, number hyps to keep for source sentence.

    Returns:
      A tuple (initial_results, states).
        initial_results: a `.NestedMap` of initial results.
          atten_probs:
            The initial attention probs, of shape [tgt_batch, src_len].
        states: a `.NestedMap` of initial model states.
          prefix_states: A `.NestedMap` object, containing already preprocessed
            source_vecs and source_contexts for each decoder layer. The shape of
            the contained tensors is [0, num_hyps, model_dim]. The first
            dimension is for the decoding step; 0 is used as a placeholder.
          time_step: A scalar, representing the decoding time step.
    """
    p = self.params

    source_encs = encoder_outputs.encoded
    num_hyps = py_utils.GetShape(source_encs)[1] * num_hyps_per_beam
    source_len = py_utils.GetShape(source_encs)[0]

    # Dummy attention probs
    atten_probs = tf.ones([num_hyps, source_len],
                          dtype=py_utils.FPropDtype(p)) / tf.cast(
                              source_len, dtype=py_utils.FPropDtype(p))
    initial_results = py_utils.NestedMap({
        'log_probs':
            tf.zeros([num_hyps, self.params.stack.softmax_tpl.num_classes],
                     dtype=py_utils.FPropDtype(p)),
        'atten_probs':
            atten_probs
    })

    if p.init_decoder_step_ids:
      initial_results['step_ids'] = tf.expand_dims(
          tf.tile(
              input=encoder_outputs.init_step_ids,
              multiples=[num_hyps_per_beam]), 1)

    atten_hidden_dim = p.stack.decoder_tpl.tr_atten_tpl.atten_hidden_dim
    if not atten_hidden_dim:
      atten_hidden_dim = p.stack.model_dim

    if p.decoder.beam_search.name == 'tpu_beam_search':
      seq_len = p.decoder.target_seq_len
    else:
      seq_len = 0

    prefix_states = py_utils.NestedMap({
        'decoder_%d' % decoder: py_utils.NestedMap({
            'key':
                tf.zeros([seq_len, num_hyps, atten_hidden_dim],
                         dtype=py_utils.FPropDtype(p)),
            'value':
                tf.zeros([seq_len, num_hyps, atten_hidden_dim],
                         dtype=py_utils.FPropDtype(p)),
        }) for decoder in range(p.stack.num_decoder_layers)
    })

    return initial_results, py_utils.NestedMap({
        'prefix_states': prefix_states,
        'time_step': tf.constant(0)
    })

  def _PreBeamSearchStepBatchMajorCallback(self, theta, encoder_outputs,
                                           new_ids, states, num_hyps_per_beam):
    p = self.params

    target_time = p.decoder.target_seq_len
    dim = p.stack.decoder_tpl.tr_atten_tpl.hidden_dim
    if not dim:
      dim = p.stack.model_dim
    num_heads = p.stack.decoder_tpl.tr_atten_tpl.num_heads
    # If per-head dim is less than 128, make the cached shape 128 to avoid
    # padding and more efficient interpolation in beamsearch.
    per_head_dim = dim // num_heads
    if per_head_dim < 128 and dim % 128 == 0:
      num_heads = dim // 128
      per_head_dim = 128
    num_hyps = py_utils.GetShape(new_ids)[0]
    source_batch = num_hyps // num_hyps_per_beam

    new_states = states.Pack(states.Flatten())
    time_step = states.time_step
    prefix_states = states.prefix_states

    def _ReorderBeamsIn(x):
      x = tf.reshape(x, [
          target_time, num_hyps_per_beam, source_batch, num_heads, per_head_dim
      ])
      x = tf.transpose(x, [0, 2, 1, 3, 4])
      return tf.reshape(x, [target_time, num_hyps, num_heads, per_head_dim])

    def _ReorderBeamsOut(x):
      x = tf.reshape(x, [
          target_time, source_batch, num_hyps_per_beam, num_heads, per_head_dim
      ])
      x = tf.transpose(x, [0, 2, 1, 3, 4])
      return tf.reshape(x, [target_time, num_hyps, num_heads, per_head_dim])

    # The inputs are ordered as num_hyps_per_beam by beam size,
    # which needs to be transposed for the layer computation.
    # [num_hyps_per_beam, source_batch, 1]
    new_ids = tf.reshape(new_ids, [num_hyps_per_beam, source_batch, 1])
    # [source_batch, num_hyps_per_beam, 1]
    new_ids = tf.transpose(new_ids, [1, 0, 2])
    # [source_batch * num_hyps_per_beam, 1]
    new_ids = tf.reshape(new_ids, [-1, 1])
    # [target_time, source_batch * num_hyps_per_beam, num_heads, dim_per_head]
    prefix_states = prefix_states.Transform(_ReorderBeamsIn)

    softmax_input, updated_prefix_states = self.ExtendStep(
        theta, encoder_outputs, new_ids, time_step, prefix_states)

    # Transpose the outputs as beam size by num_hyps_per_beam to match the
    # beam search requirement.
    # [source_batch, num_hyps_per_beam, dim]
    softmax_input = tf.reshape(softmax_input,
                               [source_batch, num_hyps_per_beam, -1])
    # [num_hyps_per_beam, source_batch, dim]
    softmax_input = tf.transpose(softmax_input, [1, 0, 2])
    # [num_hyps_per_beam * source_batch, dim]
    softmax_input = tf.reshape(softmax_input, [num_hyps, -1])
    # [target_time, num_hyps_per_beam * source_batch, num_heads, dim_per_head]
    updated_prefix_states = updated_prefix_states.Transform(_ReorderBeamsOut)

    # [target_batch, vocab_size]
    logits = self.stack.Logits(theta.stack, [softmax_input])

    # Only return logits for the last ids
    log_probs = tf.nn.log_softmax(logits)

    # Dummy attention probs
    source_time = py_utils.GetShape(encoder_outputs.padding)[0]
    atten_probs = (
        tf.ones([num_hyps, source_time], dtype=py_utils.FPropDtype(p)) /
        tf.cast(source_time, py_utils.FPropDtype(p)))

    bs_results = py_utils.NestedMap({
        'log_probs': log_probs,
        'atten_probs': atten_probs,
    })

    new_states.prefix_states = updated_prefix_states
    new_states.time_step = time_step + 1

    return bs_results, new_states

  def _PreBeamSearchStepCallback(self, theta, encoder_outputs, step_ids, states,
                                 num_hyps_per_beam):
    """Returns logits for sampling ids and the next model states.

    Args:
      theta: A `.NestedMap` object containing weights' values of this layer and
        its children layers.
      encoder_outputs: a NestedMap computed by encoder.
      step_ids: A tensor of shape [tgt_batch, 1].
      states: A `.NestedMap` of tensors representing states that the clients
        would like to keep track of for each of the active hyps.
      num_hyps_per_beam: Beam size.

    Returns:
      A tuple (results, out_states).
        results: A `.NestedMap` of beam search results.
          atten_probs:
            The updated attention probs, of shape [tgt_batch, src_len].
          log_probs:
            Log prob for each of the tokens in the target vocab. This is of
            shape [tgt_batch, vocab_size].
        out_states: A `.NestedMap`. The updated states.
           source_encs:
             A tensor of shape [src_batch, src_len, source_dim].
           source_paddings:
             A tensor of shape [src_batch, src_len].
           target_ids:
             Updated list of decoded ids. [num_hyps, Num of decoded ids].
    """
    p = self.params

    target_time = states.time_step
    prefix_states = states.prefix_states

    new_states = states.Pack(states.Flatten())

    layer_out, updated_prefix_states = self.ExtendStep(theta, encoder_outputs,
                                                       tf.squeeze(step_ids, 1),
                                                       target_time,
                                                       prefix_states)

    new_states.prefix_states = updated_prefix_states
    new_states.time_step = target_time + 1

    softmax_input = tf.reshape(layer_out, [-1, p.stack.softmax_tpl.input_dim])
    logits = self.stack.Logits(theta.stack, [softmax_input])

    num_hyps = py_utils.GetShape(step_ids)[0]
    source_len = py_utils.GetShape(encoder_outputs.encoded)[0]
    # [time * batch, num_classes] -> [time, batch, num_classes]
    logits = tf.reshape(logits, (-1, num_hyps, p.stack.softmax_tpl.num_classes))
    # [time, batch, num_classes] -> [batch, time, num_classes]
    logits = tf.transpose(logits, (1, 0, 2))

    # Dummy attention probs
    atten_probs = tf.ones([num_hyps, source_len],
                          dtype=py_utils.FPropDtype(p)) / tf.cast(
                              source_len, dtype=py_utils.FPropDtype(p))

    # Only return logits for the last ids
    log_probs = tf.nn.log_softmax(tf.squeeze(logits, axis=1))

    bs_results = py_utils.NestedMap({
        'atten_probs': atten_probs,
        'log_probs': log_probs,
    })

    return bs_results, new_states

  def _PostBeamSearchStepCallback(self, theta, encoder_outputs, new_step_ids,
                                  states):
    return states

  def _EncoderFPropDefaultTheta(self, input_batch):
    src_ids, src_enc_paddings, _, _, src_task_ids = self._PrepareInputs(
        input_batch)
    src_enc = self.stack.EncoderEmbedFPropDefaultTheta(src_ids, src_task_ids)
    if self.params.stack.batch_dim:  # time-major
      src_enc = self.stack.EncoderFPropDefaultTheta(src_enc, src_enc_paddings)
    else:
      src_enc = self.stack.EncoderFPropDefaultTheta(
          src_enc, src_enc_paddings, source_task_id=src_task_ids)
    return py_utils.NestedMap(encoded=src_enc, padding=src_enc_paddings)

  def _DecoderBeamSearchDecode(self,
                               encoder_outputs,
                               num_hyps_per_beam_override=0):
    if self.params.stack.batch_dim:  # time-major
      return self.dec.beam_search.BeamSearchDecode(
          self.theta, encoder_outputs, num_hyps_per_beam_override,
          self._InitBeamSearchStateCallback, self._PreBeamSearchStepCallback,
          self._PostBeamSearchStepCallback)
    # For batch-major encoder_outpus, first transpose them since
    # beam_search_helper.BeamSearchDecode assumes time-major encoder_outputs.
    encoder_outputs.encoded = tf.transpose(encoder_outputs.encoded, [1, 0, 2])
    encoder_outputs.padding = tf.transpose(encoder_outputs.padding)
    return self.dec.beam_search.BeamSearchDecode(
        self.theta, encoder_outputs, num_hyps_per_beam_override,
        self._InitBeamSearchStateBatchMajorCallback,
        self._PreBeamSearchStepBatchMajorCallback,
        self._PostBeamSearchStepCallback)

  def Inference(self):
    """Constructs the inference subgraphs.

    Supported subgraphs:
      - 'default'
      - '[src]->[tgt]'
      - 'prod'

    Returns:
      {'subgraph_name': (fetches, feeds)}
    """
    p = self.params
    subgraphs = dict()
    if p.init_decoder_step_ids or p.use_task_ids:
      raise NotImplementedError(
          'Currently inference does not support task_ids '
          'or initializing decoder step ids with special tokens.')
    with tf.name_scope('inference'):
      src = p.eval.inference_source_language
      tgt = p.eval.inference_target_language
      subgraphs['default'] = self._InferenceSubgraph_Default(src, tgt)
      subgraphs['%s->%s' % (src, tgt)] = subgraphs['default']
      subgraphs['prod'] = self._InferenceSubgraph_Prod()
    return subgraphs


def PartSentEvalParams():
  """Params to modify the part_sent_eval executable."""
  p = hyperparams.Params()
  p.Define('respect_emit_probs', True,
           'True: streaming models will stream instead of prefix decoding,')
  p.Define(
      'wait_k', 0,
      'Number of source tokens to see before emitting first target token.')
  p.Define(
      'emission_rate', 1.,
      'Limit number of beam search steps and allow incomplete (no terminating'
      'EOS) hypothesis based on length of source prefix. Only source tokens'
      'after wait_k are considered to determine max target length. Decode'
      'regularly if emission_rate=0 or source is not prefix.')
  p.Define(
      'append_eos', True,
      'Add EOS to truncated source prefix. If false, EOS may still be added if '
      'it is the next token in the original sequence.')
  p.Define(
      'decode_only_once', False, 'Ignore all other options and decode '
      'only the full source, padding hyps with empty strings.')
  return p


def MakeRetranslationModel(base_model_cls):
  """Returns a RetranslationModel class that subclasses  base_model_cls."""
  assert issubclass(base_model_cls, lingvo_model.MTBaseModel)

  @inference_registry.AllowClassParamsOverride
  class RetranslationModel(base_model_cls):
    """MT model for simultaneous translation given a target prefix."""

    @staticmethod
    def _DefineParams(params):
      p = params
      p.Define(
          'part_sent_eval_params', PartSentEvalParams(), 'Inference time '
          'parameters to be overridden to influence part_sent_eval; does '
          'NOT impact training.')
      return p

    @classmethod
    def Params(cls):
      params = super(RetranslationModel, cls).Params()
      params = RetranslationModel._DefineParams(params)
      return params

    @classmethod
    def Cast(cls, params):
      assert issubclass(cls, params.cls)
      params.cls = cls
      params = RetranslationModel._DefineParams(params)

      return params

    @base_layer.initializer
    def __init__(self, params):
      super(RetranslationModel, self).__init__(params)
      assert params.name
      assert params.input.natural_order_model

    def PartInputOp(self):
      """Define an input op to get the next mini-batch + segmented strings.

      Multiple references are not supported.

      Returns:
        A dict containing:
          .input_batch: A `.NestedMap` containing src & tgt ids & paddings.
          .source_strings: A list of tokenized, WPMed source strings [batch].
          .target_strings: A list of tokenized, WPMed target strings [batch].
      """
      input_batch = self.input_generator.GetPreprocessedInputBatch()
      source_lengths = tf.cast(
          tf.round(tf.reduce_sum(1 - input_batch.src.paddings, 1) - 1),
          dtype=tf.int32)
      source_strings = self.input_generator.IdsToSegmentedStrings(
          input_batch.src.ids, source_lengths, 'src')
      target_lengths = tf.cast(
          tf.round(tf.reduce_sum(1 - input_batch.tgt.paddings, 1) - 1),
          dtype=tf.int32)
      target_strings = self.input_generator.IdsToSegmentedStrings(
          input_batch.tgt.labels, target_lengths, 'tgt')
      return dict({
          'input_batch': input_batch,
          'source_strings': source_strings,
          'target_strings': target_strings,
      })

    def PartPostProcessOp(self):
      """Define an op to post process decoder output.

      Returns:
        fetches, feeds; containing:
        feeds:
          .topk_ids: Int output ids [batch, num_hyps_per_beam,
          max_target_length].
          .topk_lens: Int output lengths [batch, num_hyps_per_beam].
        fetches:
          .topk_decoded: Tokenized, WPM segmented output strings
            [batch, num_hyps_per_beam].
      """
      topk_ids = tf.placeholder(tf.int32, shape=None, name='topk_ids')
      topk_lens = tf.placeholder(tf.int32, shape=None, name='topk_lens')

      batch_size = tf.shape(topk_ids)[0]
      num_hyps_per_beam = tf.shape(topk_ids)[1]
      num_hyps = batch_size * num_hyps_per_beam
      topk_decoded = self.input_generator.IdsToSegmentedStrings(
          tf.reshape(topk_ids, [num_hyps, -1]),
          tf.reshape(topk_lens - 1, [num_hyps]), 'tgt')
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_lens))

      feeds = py_utils.NestedMap({
          'topk_ids': topk_ids,
          'topk_lens': topk_lens,
      })

      fetches = py_utils.NestedMap({
          'topk_decoded': topk_decoded,
      })

      return fetches, feeds

    def _InferenceSubgraph_Retranslation(self):
      """Production graph that allows input of a target prefix.

      Returns:
        fetches, feeds; containing:
        feeds:
          .src_ids: Int tensor [batch, max_source_length].
          .src_paddings: Float tensor [batch, max_source_length].
          .tgt_prefix_ids: Int tensor, prefix from an earlier iteration of
            decoding [batch, max_target_length].
          .tgt_prefix_paddings: Float tensor [batch, max_target_length].
          .is_complete_ids: Int tensor, 0 if source sentence is incomplete,
            1 if complete [batch, 1] (default all complete).
          .is_complete_paddings: Float tensor, dummy paddings [batch, 1]
            (default all unpadded; padding values are ignored in any case).
        fetches:
          .topk_hyps: Serialized hypothesis protos [batch, num_hyps_per_beam].
          .topk_ids: Int output ids [batch, num_hyps_per_beam,
          max_target_length].
          .topk_lens: Int output lengths [batch, num_hyps_per_beam].
          .topk_scores: Float output scores [batch, num_hyps_per_beam].
          .emit_probs: Float tensor, one-hot hard attention head for each target
            position [batch, max_target_length, max_source_length].
      """
      p = self.params

      if p.encoder.packed_input:
        raise NotImplementedError('Packed input not supported (b/120300847)')

      with tf.name_scope('inference'):
        src_ids = tf.placeholder(tf.int32, shape=None, name='src_ids')
        src_paddings = tf.placeholder(
            tf.float32, shape=None, name='src_paddings')
        tgt_prefix_ids = tf.placeholder_with_default(
            tf.zeros([tf.shape(src_ids)[0], 0], dtype=tf.int32),
            shape=None,
            name='tgt_prefix_ids')
        tgt_prefix_paddings = tf.placeholder_with_default(
            tf.ones([tf.shape(src_ids)[0], 0], dtype=tf.float32),
            shape=None,
            name='tgt_prefix_paddings')
        src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)

        # Note: is_complete_* are currently optional. If not given, we assume
        # all sequences are complete.
        is_complete_ids = tf.placeholder_with_default(
            tf.ones([py_utils.GetShape(src_ids)[0], 1], dtype=tf.int32),
            shape=[None, 1],
            name='is_complete_ids')
        is_complete_paddings = tf.placeholder_with_default(
            tf.zeros_like(is_complete_ids, dtype=tf.float32),
            shape=[None, 1],
            name='is_complete_paddings')
        bias = tf.placeholder_with_default(
            p.decoder.bias *
            tf.ones([py_utils.GetShape(src_ids)[0], 1], dtype=tf.float32),
            shape=[None, 1],
            name='bias')
        wait_k = tf.placeholder_with_default(
            p.part_sent_eval_params.wait_k *
            tf.ones([py_utils.GetShape(src_ids)[0], 1], dtype=tf.int32),
            shape=[None, 1],
            name='wait_k')

        encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)

        # Provide the tgt_prefix_ids as part of the tgt_input_map
        tgt_input_map = py_utils.NestedMap(
            labels=tgt_prefix_ids,
            paddings=tgt_prefix_paddings,
            weights=bias * (1 - tgt_prefix_paddings))
        encoder_outputs['targets'] = tgt_input_map

        decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

        # TODO(colincherry): Share more code with _ProcessBeamSearchDecodeOut

        topk_hyps = decoder_outs.topk_hyps
        topk_ids = decoder_outs.topk_ids
        topk_lens = decoder_outs.topk_lens
        topk_scores = decoder_outs.topk_scores

        # Reshape everyone to be [batch, number of hyps]
        topk_lens = tf.reshape(topk_lens, tf.shape(topk_hyps))
        topk_scores = tf.reshape(topk_scores, tf.shape(topk_hyps))

        # Reshape topk_ids from 2D to 3D so we can iterate across hyps for a
        # given source:
        # topk_ids: [batch, number of hyps, max num sequence ids]
        batch_size = tf.shape(src_ids)[0]
        topk_ids = tf.reshape(
            topk_ids, [batch_size, p.decoder.beam_search.num_hyps_per_beam, -1])

        input_lengths = tf.cast(
            tf.reduce_sum(1 - src_paddings, axis=1), tf.int32)
        if p.part_sent_eval_params.append_eos:
          input_lengths -= 1
        topk_ids, topk_lens = ApplyWaitK(topk_ids, topk_lens, input_lengths,
                                         tf.squeeze(is_complete_ids, axis=1),
                                         tf.squeeze(wait_k, axis=1),
                                         p.part_sent_eval_params.emission_rate,
                                         p.dtype, p.decoder.target_eos_id)

        feeds = py_utils.NestedMap({
            'src_ids': src_ids,
            'src_paddings': src_paddings,
            'tgt_prefix_ids': tgt_prefix_ids,
            'tgt_prefix_paddings': tgt_prefix_paddings,
            'is_complete_ids': is_complete_ids,
            'is_complete_paddings': is_complete_paddings,
            'bias': bias,
            'wait_k': wait_k,
        })

        fetches = py_utils.NestedMap({
            'topk_hyps': topk_hyps,
            'topk_ids': topk_ids,
            'topk_lens': topk_lens,
            'topk_scores': topk_scores,
        })

        if (hasattr(decoder_outs.other_states, 'accumulated_state') and
            hasattr(decoder_outs.other_states.accumulated_state.atten_states,
                    'emit_probs')):
          num_hyps_per_beam = tf.shape(decoder_outs.topk_hyps)[1]
          # Get accumulated emit_probs into the necessary shape.
          #
          # [target_len, num_hyps_per_beam * batch_size, source_len]
          # First position corresponds to the initial attention state, trim it.
          emit_probs = (
              decoder_outs.other_states.accumulated_state.atten_states
              .emit_probs[1:])
          # [batch_size * num_hyps_per_beam, target_len, source_len]
          emit_probs = tf.transpose(emit_probs, [1, 0, 2])
          # [batch_size, target_len, source_len]
          emit_probs = _SelectTopOne(
              emit_probs, batch_size, num_hyps_per_beam, batch_first=False)
          fetches['emit_probs'] = emit_probs

        return fetches, feeds

  return RetranslationModel


# TODO(navari): inherit from MTModelV1 and disconnect from RetransaltionModel
@inference_registry.AllowClassParamsOverride
class MTOnlineModel(MakeRetranslationModel(MTModelV1)):
  """MT model for online/simultaneous translation."""

  @classmethod
  def Params(cls):
    p = super(MTOnlineModel, cls).Params()
    p.encoder = encoder.MTEncoderUniRNN.Params()
    p.decoder = decoder.MTOnlineDecoder.Params()
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MTOnlineModel, self).__init__(params)
    p = self.params
    assert p.name
    assert p.input.natural_order_model

  def _ProcessBeamSearchDecodeOut(self, input_batch, encoder_outputs,
                                  decoder_outputs):
    p = self.params
    dec_out_dict = super(MTOnlineModel, self)._ProcessBeamSearchDecodeOut(
        input_batch, encoder_outputs, decoder_outputs)

    # Get just the top ids (as opposed to top k) for use downstream.
    batch = tf.shape(decoder_outputs.topk_hyps)[0]
    num_hyps_per_beam = tf.shape(decoder_outputs.topk_hyps)[1]
    # top_ids: [batch, max num sequence ids]
    dec_out_dict['top_ids'] = _SelectTopOne(
        decoder_outputs.topk_ids, batch, num_hyps_per_beam, batch_first=True)

    # emission_delay_loss calculation
    if (hasattr(self.dec, 'emission_delay_loss') and
        hasattr(decoder_outputs.other_states.accumulated_state.atten_states,
                'emit_probs')):
      state = decoder_outputs.other_states.accumulated_state

      # Get accumulated emit_probs into the necessary shape.
      #
      # [target_len, num_hyps_per_beam * batch, source_len]
      # The first position corresponds to the initial attention state, trim it.
      emit_probs = state.atten_states.emit_probs[1:]
      target_len = tf.shape(emit_probs)[0]
      # [batch * num_hyps_per_beam, target_len, source_len]
      emit_probs = tf.transpose(emit_probs, [1, 0, 2])
      # [batch, target_len, source_len]
      emit_probs = _SelectTopOne(
          emit_probs, batch, num_hyps_per_beam, batch_first=False)

      # alignments intentionally None.
      aligments = None

      # Generate paddings tensor.
      #
      # [batch * num_hyps_per_beam]
      topk_lens = decoder_outputs.topk_lens
      # [batch]
      lens = _SelectTopOne(
          topk_lens, batch, num_hyps_per_beam, batch_first=True)
      # Account for occasional empty outputs due to beam search failure.
      lens = tf.maximum(lens, 1)
      # [batch, target_len]
      paddings = 1 - tf.sequence_mask(lens, target_len, dtype=p.dtype)

      # Generate source_enc_len tensor.
      #
      # [source_len, batch] or NestedMap
      source_paddings = encoder_outputs.padding
      # [batch, source_len]
      if isinstance(source_paddings, py_utils.NestedMap):
        source_paddings = tf.transpose(source_paddings.Flatten()[0])
      else:
        source_paddings = tf.transpose(source_paddings)
      # [batch]
      source_enc_len = tf.cast(
          tf.round(tf.reduce_sum(1.0 - source_paddings, 1)), dtype=tf.int32)

      # Make delay loss inputs available downstream for debugging purposes
      dec_out_dict['emit_probs'] = (
          emission_delay_loss.AppendResidualProbability(emit_probs,
                                                        source_enc_len))
      dec_out_dict['emit_paddings'] = paddings
      dec_out_dict['source_enc_len'] = source_enc_len

      # Calculate per-sequence scores and add them to the output
      delay_metrics = self.dec.emission_delay_loss.ComputePerSequenceMetrics(
          emit_probs, aligments, paddings, source_enc_len)
      for metric, scores in delay_metrics.items():
        dec_out_dict[metric] = scores

    return dec_out_dict

  def _PostProcessBeamSearchDecodeOut(self, dec_out_dict, dec_metrics_dict):
    info_pairs = super(MTOnlineModel, self)._PostProcessBeamSearchDecodeOut(
        dec_out_dict, dec_metrics_dict)

    # Update any delay losses in the dec_metrics_dict
    if (hasattr(self.dec, 'emission_delay_loss') and
        'emit_probs' in dec_out_dict):
      emit_probs_array = dec_out_dict['emit_probs']
      paddings_array = dec_out_dict['emit_paddings']
      source_enc_len_array = dec_out_dict['source_enc_len']
      for metric_key in self.dec.emission_delay_loss.ListPerSequenceMetrics():
        # [batch]: Alread did top-1 extraction in dec_out_dict construction
        scores = dec_out_dict[metric_key]
        for score, emit_probs, paddings, source_enc_len in zip(
            scores, emit_probs_array, paddings_array, source_enc_len_array):
          dec_metrics_dict[metric_key].Update(score)
          if math.isnan(score):
            tf.logging.warning('nan in %s', metric_key)
            for a in emit_probs:
              tf.logging.warning('emit_probs: %s', a)
            tf.logging.warning('padidngs: %s', paddings)
            tf.logging.warning('source_enc_len: %s', source_enc_len)
      # Average lagging as a normal metric (not an emission_delay_loss)
      for emit_probs, paddings, source_len, finite_lagging_tf, ids in zip(
          emit_probs_array,
          paddings_array,
          source_enc_len_array,
          dec_out_dict['emission_delay/average_finite_lagging'],
          dec_out_dict['top_ids'],
      ):
        delay = _ProbsToDelay(emit_probs, paddings)
        target_len = len(delay)
        lagging = bf_metrics.CalculateAverageLagging(delay, source_len,
                                                     target_len)
        dec_metrics_dict['emission_delay/average_lagging'].Update(lagging)
        # Sanity check comparing np and tf average_finite_lagging
        finite_lagging = bf_metrics.CalculateAverageFiniteLagging(
            delay, source_len, target_len)
        if abs(finite_lagging_tf - finite_lagging) >= 1e-4:
          tf.logging.warning(
              'np (%s) and tf (%s) finite laggings are not equal.',
              finite_lagging, finite_lagging_tf)
        # Sanity check lengths: check for EOS (2) on last token
        if ids[target_len - 1] != 2:
          tf.logging.warning(
              'No EOS on last token for target length %s and id sequence %s',
              target_len, ids)

    return info_pairs

  def CreateDecoderMetrics(self):
    dec_metrics_dict = super(MTOnlineModel, self).CreateDecoderMetrics()

    # Add delay losses to dec_metrics_dict
    if (hasattr(self.dec, 'emission_delay_loss') and
        self.dec.emission_delay_loss):
      for metric_key in self.dec.emission_delay_loss.ListPerSequenceMetrics():
        dec_metrics_dict[metric_key] = metrics.AverageMetric()
      dec_metrics_dict['emission_delay/average_lagging'] = (
          metrics.AverageMetric())

    return dec_metrics_dict

  def _InferenceSubgraph_Default(self, src, tgt):
    # TODO(colincherry): Add a unit test.
    # TODO(colincherry): Find a way to refactor MTModelV1's InferenceSubgraph
    # so that we don't need to copy and paste every time we want to expose
    # an inference side-effect of one of its children.
    p = self.params

    if p.encoder.packed_input:
      raise NotImplementedError('Packed input not supported (b/120300847)')

    with tf.name_scope('inference'):
      # Removed UPP tokenizer from parent inference graph; user should manually
      # tokenize the input string so that whitespace delimits token boundaries.
      src_strings = tf.placeholder(tf.string, shape=[None], name='src_strings')
      _, src_ids, src_paddings = self.input_generator.StringsToIds(
          src_strings,
          external_max_length=p.decoder.target_seq_len,
          key=self._GetTokenizerKeyToUse('src'))

      # Truncate paddings at the end.
      max_seq_length = tf.cast(
          tf.round(tf.reduce_max(tf.reduce_sum(1.0 - src_paddings, 1))),
          dtype=tf.int32)
      src_paddings = py_utils.with_dependencies([
          py_utils.assert_equal(
              tf.constant(True, tf.bool),
              tf.reduce_all(src_paddings[:, max_seq_length:] > 0.5))
      ], src_paddings)
      src_ids = src_ids[:, :max_seq_length]
      src_paddings = src_paddings[:, :max_seq_length]
      # Last step, reverse the source sequence if it is training an reversed
      # order model.
      if not self.input_generator.natural_order_model:
        slen = tf.cast(
            tf.round(tf.reduce_sum(1.0 - src_paddings, 1) - 1), dtype=tf.int32)
        src_ids = ops.reverse_ids(src_ids, slen)

      src_input_map = py_utils.NestedMap(ids=src_ids, paddings=src_paddings)
      encoder_outputs = self.enc.FPropDefaultTheta(src_input_map)
      decoder_outs = self.dec.BeamSearchDecode(encoder_outputs)

      topk_hyps = decoder_outs.topk_hyps
      topk_ids = decoder_outs.topk_ids
      topk_lens = decoder_outs.topk_lens
      topk_scores = decoder_outs.topk_scores

      topk_decoded = self.input_generator.IdsToStrings(
          topk_ids, topk_lens - 1, self._GetTokenizerKeyToUse('tgt'))
      topk_decoded = tf.reshape(topk_decoded, tf.shape(topk_hyps))

      # last step, runs the upp untokenizer
      topk_decoded = ops.upp_detokenize(topk_decoded, language=tgt)

      feeds = py_utils.NestedMap({'src_strings': src_strings})
      fetches = py_utils.NestedMap({
          'topk_decoded': topk_decoded,
          'topk_hyps': topk_hyps,
          'topk_scores': topk_scores,
          'src_ids': src_ids,
      })

      if (hasattr(decoder_outs.other_states, 'accumulated_state') and
          hasattr(decoder_outs.other_states.accumulated_state.atten_states,
                  'emit_probs')):
        batch = tf.shape(decoder_outs.topk_hyps)[0]
        num_hyps_per_beam = tf.shape(decoder_outs.topk_hyps)[1]
        # Get accumulated emit_probs into the necessary shape.
        #
        # [target_len, num_hyps_per_beam * batch, source_len]
        # First position corresponds to the initial attention state, trim it.
        emit_probs = decoder_outs.other_states.accumulated_state.atten_states.emit_probs[
            1:]
        # [batch * num_hyps_per_beam, target_len, source_len]
        emit_probs = tf.transpose(emit_probs, [1, 0, 2])
        # [batch, target_len, source_len]
        emit_probs = _SelectTopOne(
            emit_probs, batch, num_hyps_per_beam, batch_first=False)
        fetches['emit_probs'] = emit_probs

      return fetches, feeds


# TODO(huangyp): hide _Split from model.py.
_Split = moe_builder.Split


class MoETransformerModel(base_model.BaseTask):
  """MoE TransformerModel with aux_loss."""

  @classmethod
  def Params(cls):
    p = super(MoETransformerModel, cls).Params()
    p.Define('builder', None, 'Mixture-of-Experts Builder.')
    p.Define('loss_coef', 0.01, 'Multiplier for Mixture-of-Experts aux_loss.')
    p.Define('label_smoothing', 0.1, 'Label smoothing.')

    # TODO(lepikhin): enable source/target sequence_length
    #   {'source': 256, 'target': 512}
    # and source/target vocab_size
    #   {'source': 65536, 'target': 32768}
    p.Define('vocab_size', None, 'Vocabulary size')
    p.Define('sequence_length', None, 'Sequence length.')
    p.Define(
        'max_length', 512,
        'Max sequence length. Second pos_emb Tensor dim is set to ' +
        'max_length.')

    p.Define('batch_size', None, 'Batch size.')
    p.Define('num_transformer_layers', None,
             'Number of blocks in MoEBuilder.{Decoder,Encoder}LayerStack.')

    p.Define(
        'moe', None,
        'True for Mixture-of-Experts, False for canonical Transformer model, '
        '"encoder" for Mixture-of-Experts in encoder only.')

    p.Define(
        'beam_search',
        beam_search_tpu_helper.BeamSearchTpuHelper.Params().Set(
            batch_major_state=True,
            length_normalization=0.8,
        ), 'BeamSearchTpuHelper params.')

    p.Define(
        'use_tgt_labels_size_as_loss_denominator', True,
        'False to use total number of non-padding tokens instead of '
        'fixed tgt_labels tensor size.')

    p.Define('logits_abs_max', None, 'Logits clipping.')
    p.Define(
        'z_loss', 1e-4, 'if z_loss is nonzero, we add a loss equal to '
        'z_loss * tf.math.square(tf.math.reduce_logsumexp(logits, -1))')
    return p

  @base_layer.initializer
  def __init__(self, params):
    super(MoETransformerModel, self).__init__(params)
    p = self.params

    builder = p.builder.Instantiate()

    with tf.variable_scope(p.name):
      src_vocab_size, tgt_vocab_size = p.vocab_size
      enc_emb = builder.Embedding('enc_emb', src_vocab_size)
      self.CreateChild('enc_emb', enc_emb)

      dec_emb = builder.Embedding('dec_emb', tgt_vocab_size)
      self.CreateChild('dec_emb', dec_emb)

      enc_pos_emb = builder.Embedding('enc_pos_emb', p.max_length)
      self.CreateChild('enc_pos_emb', enc_pos_emb)

      dec_pos_emb = builder.Embedding('dec_pos_emb', p.max_length)
      self.CreateChild('dec_pos_emb', dec_pos_emb)

      enc = builder.EncoderLayerStack('encoder', [
          builder.SelfAttention('self_attention'),
          builder.MoE('moe')
          if p.moe else builder.DenseReluDense('dense_relu_dense'),
          builder.SelfAttention('self_attention'),
          builder.DenseReluDense('dense_relu_dense'),
      ], p.num_transformer_layers)
      enc.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      self.CreateChild('enc', enc)

      dec = builder.DecoderLayerStack('decoder', [
          builder.DecSelfAttention('dec_self_attention'),
          builder.DecEncAttention('dec_enc_attention'),
          builder.MoE('moe', decoder=True) if p.moe and p.moe != 'encoder' else
          builder.DenseReluDense('dense_relu_dense', decoder=True),
          builder.DecSelfAttention('dec_self_attention'),
          builder.DecEncAttention('dec_enc_attention'),
          builder.DenseReluDense('dense_relu_dense', decoder=True),
      ], p.num_transformer_layers)

      dec.params_init = py_utils.WeightInit.Xavier(scale=1.0, seed=0)
      self.CreateChild('dec', dec)

      p.beam_search.batch_major_state = True
      self.CreateChild('beam_search', p.beam_search)

  def ComputePredictions(self, theta, input_batch):
    """Forward propagation through one tower of the model.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A dict containing metrics pairs.
    """
    p = self.params

    if (p.fprop_dtype is not None and p.fprop_dtype != p.dtype):
      tf.logging.info('MaybeCastToFPropDtype theta and input_batch')

      def MaybeCastToFPropDtype(x):
        if x.dtype.is_floating and x.dtype == p.dtype:
          return tf.cast(x, p.fprop_dtype)
        else:
          return x

      theta = theta.Transform(MaybeCastToFPropDtype)
      # InputGenerator should have taken care of this already...
      input_batch = input_batch.Transform(MaybeCastToFPropDtype)

    with tf.name_scope(p.name):
      # ops.text_packed:
      #   target_id_eos => tgt_labels
      #   target_bos_id => tgt_ids

      # TODO(lepikhin): figure which annotations are absolutely necessary
      def _MaybeSplit(x):
        return _Split(x, 0, p.builder.num_devices)

      x = self.enc_emb.FProp(theta.enc_emb, input_batch.src.ids)

      x += self.enc_pos_emb.FProp(theta.enc_pos_emb,
                                  input_batch.src.segment_pos)

      all_outputs = self.enc.FProp(
          theta.enc, x, input_batch.src.segment_ids,
          input_batch.src.segment_pos,
          tf.convert_to_tensor(0.0, py_utils.FPropDtype(p)))
      enc_output, aux_loss = all_outputs[0], all_outputs[-1]

      y = self.dec_emb.FProp(theta.dec_emb, input_batch.tgt.ids)
      y += self.dec_pos_emb.FProp(theta.dec_pos_emb,
                                  input_batch.tgt.segment_pos)
      all_outputs = self.dec.FProp(theta.dec, y, input_batch.tgt.segment_ids,
                                   input_batch.tgt.segment_pos, enc_output,
                                   input_batch.src.segment_ids,
                                   input_batch.src.segment_pos, aux_loss)
      dec_outputs, aux_loss = all_outputs[0], all_outputs[-1]
      model_dim = self.params.builder.model_dim
      dec_outputs *= (model_dim**-0.5)
      # TODO(lepikhin): we only support
      # shared_embedding_and_softmax_weights=True at the moment.
      softmax_weights = self.vars.dec_emb.w.embedding.read_value()
      if dec_outputs.dtype != softmax_weights:
        softmax_weights = tf.cast(softmax_weights, dec_outputs.dtype)
      logits = _MaybeSplit(
          tf.einsum('BLM,VM->BLV', _MaybeSplit(dec_outputs), softmax_weights))

      if p.logits_abs_max is not None:
        logits = _MaybeSplit(
            py_utils.clip_by_value(logits, -p.logits_abs_max, p.logits_abs_max))

      return logits, aux_loss  # TODO(lepikhin): better API for aux_loss

  def ComputeLoss(self, theta, predictions, input_batch):
    p = self.params
    _, vocab_size = p.vocab_size

    def _MaybeSplit(x):
      return _Split(x, 0, p.builder.num_devices)

    input_batch = input_batch.Transform(_MaybeSplit)

    with tf.name_scope(p.name):
      logits, aux_loss = predictions
      if 'soft_labels' in input_batch.tgt:
        tf.logging.info('using input_batch.tgt.soft_labels: %r',
                        input_batch.tgt.soft_labels)
        soft_labels = _MaybeSplit(input_batch.tgt.soft_labels)
      else:
        label_smoothing = self.params.label_smoothing
        off_value = label_smoothing / vocab_size
        on_value = 1.0 - label_smoothing + off_value
        tf.logging.info({'on_value': on_value, 'off_value': off_value})
        soft_labels = _MaybeSplit(
            tf.one_hot(
                input_batch.tgt.labels,
                vocab_size,
                on_value=on_value,
                off_value=off_value))

      xent = _MaybeSplit(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=tf.one_hot(input_batch.tgt.labels, vocab_size),
              logits=logits))

      loss = _MaybeSplit(
          tf.nn.softmax_cross_entropy_with_logits(
              labels=soft_labels, logits=logits))
      soft_labels_xent = loss

      if self.params.z_loss > 0.0:
        log_z = tf.math.reduce_logsumexp(logits, -1)
        z_loss_inc = self.params.z_loss * tf.math.square(log_z)
        loss += z_loss_inc

      non_padding = _MaybeSplit(
          tf.cast(
              tf.not_equal(input_batch.tgt.segment_ids, 0),
              py_utils.FPropDtype(self.params)))

      def _SplitLoss(x):
        return _Split(x, 0, p.builder.num_devices)

      per_token_loss = _SplitLoss(loss * non_padding)
      if self.params.z_loss:
        per_token_z_loss_inc = _SplitLoss(z_loss_inc * non_padding)

      if p.use_tgt_labels_size_as_loss_denominator:
        # E.g. loss is going to be tiny if inputs are not packed and only a
        # fraction of tgt_labels are non-padding.
        loss_denom = tf.reduce_sum(tf.ones_like(non_padding))
        per_example_loss_denom = tf.reduce_sum(tf.ones_like(non_padding), 1)
      else:
        loss_denom = tf.reduce_sum(non_padding)
        per_example_loss_denom = tf.reduce_sum(non_padding, 1)
      avg_loss = tf.reduce_sum(per_token_loss) / loss_denom
      avg_z_loss_inc = (tf.reduce_sum(per_token_z_loss_inc) /
                        loss_denom) if self.params.z_loss else 0.0

      soft_labels_xent = (
          tf.reduce_sum(_SplitLoss(soft_labels_xent * non_padding)) /
          tf.reduce_sum(non_padding))

      avg_loss += p.loss_coef * aux_loss
      # TODO(lepikhin): consider returning
      #   {'loss': (unnormalized per_token_loss, tf.reduce_sum(non_padding))}
      per_example_loss = {
          'loss': (tf.reduce_sum(per_token_loss, 1) / per_example_loss_denom)
      }
      eval_metrics = {
          'mean_xent': (tf.reduce_sum(_SplitLoss(xent * non_padding)) /
                        tf.reduce_sum(non_padding), tf.reduce_sum(non_padding)),
          'soft_labels_xent': (soft_labels_xent, tf.reduce_sum(non_padding)),
          'weight': (tf.reduce_sum(non_padding), 1.0),
          'loss': (avg_loss, 1.0),
          'aux_loss': (p.loss_coef * aux_loss, 1.0),
          'avg_z_loss_inc': (avg_z_loss_inc, 1.0),
      }
      eval_metrics.update(py_utils.GetTpuSummaryTensors())
      return eval_metrics, per_example_loss

  def DecodeWithTheta(self, theta, input_batch):
    """Decoding.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.

    Returns:
      A dict containing metrics pairs.
    """
    tf.logging.info('DecodeWithTheta input_batch=%r', input_batch)
    input_batch = input_batch.Transform(tf.identity)
    bs = self.DecodeIds(theta, input_batch)
    return py_utils.RunOnTpuHost(self._ProcessBeamSearchDecodeOut, input_batch,
                                 bs)

  def DecodeIds(self, theta, input_batch, use_flat_beam_search=True):
    """Decoding, excluding IdsToStrings conversion.

    Args:
      theta: A `.NestedMap` object containing variable values of this task
        copied to this tower's devices.
      input_batch: A `.NestedMap` object containing input tensors to this tower.
      use_flat_beam_search: use flat beam search

    Returns:
      A NestedMap with
        topk_id: int32 tensor of shape [batch_size * beam_size, max_len]
        topk_lens: int32 tensor of shape [batch_size * beam_size]
        topk_scores: int32 tensor of shape [batch_size * beam_size]

      (first dimension can be reshaped to [batch_size, beam_size])
    """
    p = self.params
    tf.logging.info('DecodeIds input_batch=%r', input_batch)
    with tf.name_scope(p.name):
      # Encoder fprop as in ComputePredictions
      x = self.enc_emb.FProp(theta.enc_emb, input_batch.src.ids)
      x += self.enc_pos_emb.FProp(theta.enc_pos_emb,
                                  input_batch.src.segment_pos)

      all_outputs = self.enc.FProp(theta.enc, x, input_batch.src.segment_ids,
                                   input_batch.src.segment_pos,
                                   tf.convert_to_tensor(0.0, tf.float32))
      enc_output = all_outputs[0]
      tf.logging.info('enc_output=%r', enc_output)

      if not use_flat_beam_search:
        # Convert 'enc_output' to format accepted by BeamSearchDecode()
        encoder_outputs = py_utils.NestedMap()
        encoder_outputs.encoded = tf.transpose(enc_output, [1, 0, 2])
        encoder_outputs.padding = tf.transpose(input_batch.src.paddings)
        encoder_outputs.src_ids = tf.transpose(input_batch.src.ids)
        encoder_outputs.segment_id = tf.transpose(input_batch.src.segment_ids)
        encoder_outputs.segment_pos = tf.transpose(input_batch.src.segment_pos)

        num_hyps_per_beam_override = 0
        bs = self.beam_search.BeamSearchDecode(
            theta,
            encoder_outputs,
            num_hyps_per_beam_override,
            self._InitBeamSearchStateCallback,
            self._PreBeamSearchStepCallback,
            self._PostBeamSearchStepCallback,
            max_steps=p.max_length)
        return bs

      if use_flat_beam_search:
        batch_size = int(input_batch.src.ids.shape[0])
        max_steps = p.beam_search.target_seq_len or p.max_length
        beam_size = p.beam_search.num_hyps_per_beam

        dec_state = moe_builder.StateLayer.InitState(
            self.dec, shape=[batch_size, beam_size, max_steps])

        flat_bs = flat_beam_search.flat_beam_search(
            batch_size,
            beam_size,
            max_steps,
            dec_callback=functools.partial(self._FlatBeamSearchCallback, theta,
                                           enc_output, input_batch),
            dec_state=dec_state,
            length_norm_alpha=p.beam_search.length_normalization)

        loop_vars, dec_state, nbest = flat_bs
        (topk_ids, topk_lens, topk_scores) = nbest
        del loop_vars, dec_state

        bs = py_utils.NestedMap()
        bs.topk_ids = tf.reshape(topk_ids, [batch_size * beam_size, -1])
        bs.topk_lens = tf.reshape(topk_lens, [-1])
        bs.topk_scores = tf.reshape(topk_scores, [-1])
        bs.topk_hyps_shape = tf.shape(topk_scores)
        return bs

  def _ProcessBeamSearchDecodeOut(self, input_batch, bs):
    p = self.params
    with tf.name_scope('spm_src'):
      srcs = self.input_generator.IdsToStrings(
          input_batch.src.ids,
          tf.cast(
              tf.round(tf.reduce_sum(1 - input_batch.src.paddings, 1) - 1),
              dtype=tf.int32), 'src')
    with tf.name_scope('spm_tgt'):
      topk_decoded = self.input_generator.IdsToStrings(
          bs.topk_ids, tf.nn.relu(bs.topk_lens - 1), 'tgt')
    with tf.name_scope('spm_refs'):
      refs = self.input_generator.IdsToStrings(
          input_batch.tgt.labels,
          tf.cast(
              tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
              dtype=tf.int32), 'tgt')
    with tf.name_scope('gleu'):
      gleu = ops.per_sentence_translation_quality(
          tf.reshape(topk_decoded, [-1]),
          tf.reshape(
              tf.tile(
                  tf.reshape(refs, [-1, 1]),
                  [1, p.beam_search.num_hyps_per_beam]), [-1]),
          metric='gleu',
          max_ngram=4)

    try:
      topk_hyps_shape = bs.topk_hyps_shape
    except AttributeError:
      topk_hyps_shape = tf.shape(bs.topk_hyps)

    ret_dict = {
        'target_ids':
            tf.identity(input_batch.tgt.ids, name='target_ids'),
        'target_labels':
            tf.identity(input_batch.tgt.labels, name='target_labels'),
        'target_weights':
            tf.identity(input_batch.tgt.weights, name='target_weights'),
        'target_paddings':
            tf.identity(input_batch.tgt.paddings, name='target_paddings'),
        'sources':
            tf.identity(srcs, name='sources'),
        'targets':
            tf.identity(refs, name='refs'),
        'topk_ids':
            tf.identity(bs.topk_ids, name='topk_ids'),
        'topk_decoded':
            tf.reshape(topk_decoded, topk_hyps_shape, name='topk_decoded'),
        'topk_lens':
            tf.identity(bs.topk_lens, name='topk_lens'),
        'topk_scores':
            tf.reshape(bs.topk_scores, topk_hyps_shape, name='tokp_scores'),
        'gleu':
            tf.reshape(gleu, topk_hyps_shape, name='gleu'),
    }
    try:
      ret_dict['topk_hyps'] = tf.identity(bs.topk_hyps, name='topk_hyps')
    except AttributeError:
      pass
    return ret_dict

  def GreedyDecodeIds(self, theta, input_batch):
    """Greedy decoding with packed inputs."""
    p = self.params
    tf.logging.info('GreedyDecodeIds input_batch=%r', input_batch)
    with tf.name_scope(p.name):
      # Encoder fprop as in ComputePredictions
      x = self.enc_emb.FProp(theta.enc_emb, input_batch.src.ids)
      x += self.enc_pos_emb.FProp(theta.enc_pos_emb,
                                  input_batch.src.segment_pos)

      all_outputs = self.enc.FProp(theta.enc, x, input_batch.src.segment_ids,
                                   input_batch.src.segment_pos,
                                   tf.convert_to_tensor(0.0, tf.float32))
      enc_output = all_outputs[0]
      tf.logging.info('enc_output=%r', enc_output)

      batch_size = int(input_batch.src.ids.shape[0])
      max_steps = p.beam_search.target_seq_len or p.max_length

      dec_state = moe_builder.StateLayer.InitState(
          self.dec, shape=[batch_size, 1, max_steps])

      gs = packed_greedy_search.packed_greedy_search(
          batch_size,
          max_steps,
          dec_callback=functools.partial(self._FlatBeamSearchCallback, theta,
                                         enc_output, input_batch),
          dec_state=dec_state,
          plan=input_batch.src.segment_ids)

      loop_vars, dec_state, sample = gs
      (tgt_id, tgt_label, tgt_segment_id, tgt_pos, logits) = sample
      del loop_vars, dec_state

      ret = py_utils.NestedMap()
      ret.tgt_ids = tgt_id
      ret.tgt_labels = tgt_label
      ret.tgt_segment_ids = tgt_segment_id
      ret.tgt_segment_pos = tgt_pos
      ret.logits = logits

      return ret

  def _FlatBeamSearchCallback(self, theta, enc_output, input_batch, tgt_id,
                              tgt_segment_id, tgt_pos, tgt_mask, dec_state, t):
    p = self.params

    tgt_id = tf.cast(tgt_id, tf.int32)
    tgt_segment_id = tf.cast(tgt_segment_id, tf.float32)
    tgt_pos = tf.cast(tgt_pos, tf.int32)

    y = self.dec_emb.FProp(theta.dec_emb, tgt_id)
    y += self.dec_pos_emb.FProp(theta.dec_pos_emb, tgt_pos)

    theta_with_state = moe_builder.StateLayer.UpdateTheta(
        self.dec, theta.dec, dec_state, t)
    moe_builder.OverrideLayer.Set('dec_self_attention_bias',
                                  (tgt_mask - 1.0) * 1e9)

    def _MaybeSplit(x):
      return _Split(x, 0, p.builder.num_devices)

    tgt_segment_id = _MaybeSplit(tgt_segment_id)

    aux_loss = tf.constant(0.)
    dec_outputs, _ = self.dec.FProp(theta_with_state, y, tgt_segment_id,
                                    tgt_pos, enc_output,
                                    input_batch.src.segment_ids,
                                    input_batch.src.segment_pos, aux_loss)

    model_dim = self.params.builder.model_dim
    dec_outputs *= (model_dim**-0.5)

    logits = _MaybeSplit(
        tf.einsum('BLM,VM->BLV', _MaybeSplit(dec_outputs),
                  self.vars.dec_emb.w.embedding.read_value()))

    logits = tf.nn.log_softmax(logits)

    # TODO(krikun): decoder will try to emit 'over_capaciy_1' summary tensors
    # into a global collection. The problem that we are here inside a decoder
    # while loop. An attempt to use summaries from inside this while loop at
    # top-level ComputeLoss will result in fail with error message
    # "Cannot use ... because they are in different while loops".
    #
    # As an workaround, clear __lingvo_tpu_summary_tensors collection here.
    # We will get any useful summaries from MoE layer, however.
    #
    # To do it the right way, we need to collect summary tensors from decoder
    # layer here, explicitly aggregate as part of dec_state, and put emit
    # aggregated summary tensors after the loop.
    tf.get_default_graph().clear_collection(py_utils._TPU_SUMMARY_TENSORS_KEY)  # pylint: disable=protected-access

    dec_state = moe_builder.StateLayer.UpdateState(self.dec, theta.dec,
                                                   dec_state)
    moe_builder.OverrideLayer.Clear()

    return logits, dec_state

  def _InitBeamSearchStateCallback(self, theta, encoder_outputs,
                                   num_hyps_per_beam):
    p = self.params
    tf.logging.info('_InitBeamSearchStateCallback encoder_outputs=%r',
                         encoder_outputs)
    tf.logging.info('_InitBeamSearchStateCallback num_hyps_per_beam=%r',
                         num_hyps_per_beam)

    target_batch = py_utils.GetShape(
        encoder_outputs.encoded)[1] * num_hyps_per_beam
    source_time = py_utils.GetShape(encoder_outputs.encoded)[0]
    target_vocab_size = p.vocab_size[1]
    target_time = p.max_length

    initial_results = py_utils.NestedMap()
    initial_results.log_probs = tf.zeros([target_batch, target_vocab_size])
    # dummy atten_probs
    initial_results.atten_probs = tf.ones([target_batch, source_time
                                          ]) / tf.cast(
                                              source_time, dtype=tf.float32)

    states = py_utils.NestedMap()
    states.time_step = tf.constant(0, dtype=tf.int32)
    states.dec = moe_builder.StateLayer.InitState(
        self.dec, shape=[target_batch, target_time])

    tf.logging.info('_InitBeamSearchStateCallback initial_results=%r',
                         initial_results)
    tf.logging.info('_InitBeamSearchStateCallback states=%r', states)
    tf.logging.info('_InitBeamSearchStateCallback states.dec=%r',
                         states.dec)
    return (initial_results, states)

  def _PreBeamSearchStepCallback(self, theta, encoder_outputs, step_ids, states,
                                 num_hyps_per_beam):
    p = self.params
    tf.logging.info('_PreBeamSearchStepCallback encoder_outputs=%r',
                         encoder_outputs)
    tf.logging.info('_PreBeamSearchStepCallback step_ids=%r', step_ids)
    tf.logging.info('_PreBeamSearchStepCallback states=%r', states)
    tf.logging.info('_PreBeamSearchStepCallback id(states)=%r', id(states))
    tf.logging.info('_PreBeamSearchStepCallback num_hyps_per_beam=%r',
                         num_hyps_per_beam)

    target_batch = py_utils.GetShape(
        encoder_outputs.encoded)[1] * num_hyps_per_beam
    source_time = py_utils.GetShape(encoder_outputs.encoded)[0]

    # TODO(lepikhin): figure which annotations are absolutely necessary
    def _MaybeSplit(x):
      return _Split(x, 0, p.builder.num_devices)

    def RepeatDim(x, n):
      return tf.tile(x, [n] + [1] * (len(x.shape) - 1))

    src_segment_ids = tf.transpose(encoder_outputs.segment_id)
    src_segment_pos = tf.transpose(encoder_outputs.segment_pos)
    src_segment_ids = RepeatDim(src_segment_ids, num_hyps_per_beam)
    src_segment_pos = RepeatDim(src_segment_pos, num_hyps_per_beam)
    enc_output = tf.transpose(encoder_outputs.encoded, [1, 0, 2])
    enc_output = RepeatDim(enc_output, num_hyps_per_beam)
    aux_loss = tf.constant(0.)

    with tf.name_scope('tgt_ids_t'):
      tgt_ids_t = step_ids
      tgt_ids_t = _MaybeSplit(tgt_ids_t)

    with tf.name_scope('tgt_segment_ids_t'):
      tgt_segment_ids_t = tf.ones([target_batch, 1])
      tgt_segment_ids_t = _MaybeSplit(tgt_segment_ids_t)

    with tf.name_scope('tgt_segment_pos_t'):
      tgt_segment_pos_t = (
          tf.ones([target_batch, 1], dtype=tf.int32) * states.time_step)
      tgt_segment_pos_t = _MaybeSplit(tgt_segment_pos_t)

    src_segment_ids = _MaybeSplit(src_segment_ids)
    src_segment_pos = _MaybeSplit(src_segment_pos)

    # Decoder fprop as in ComputePredictions
    y = self.dec_emb.FProp(theta.dec_emb, tgt_ids_t)
    y += self.dec_pos_emb.FProp(theta.dec_pos_emb, tgt_segment_pos_t)

    theta_with_state = moe_builder.StateLayer.UpdateTheta(
        self.dec, theta.dec, states.dec, t=states.time_step)
    tf.logging.info('theta_with_state=%r', theta_with_state)

    all_outputs = self.dec.FProp(theta_with_state, y, tgt_segment_ids_t,
                                 tgt_segment_pos_t, enc_output, src_segment_ids,
                                 src_segment_pos, aux_loss)
    tf.logging.info('all_outputs=%r', all_outputs)
    dec_outputs, aux_loss = all_outputs[0], all_outputs[-1]
    model_dim = self.params.builder.model_dim
    dec_outputs *= (model_dim**-0.5)
    logits = _MaybeSplit(
        tf.einsum('BLM,VM->BLV', _MaybeSplit(dec_outputs),
                  self.vars.dec_emb.w.embedding.read_value()))
    tf.logging.info('logits=%r', logits)

    logits = tf.nn.softmax(logits)

    bs_results = py_utils.NestedMap()
    bs_results.log_probs = tf.nn.log_softmax(tf.squeeze(logits, 1))
    # dummy atten_probs
    bs_results.atten_probs = tf.ones([target_batch, source_time]) / tf.cast(
        source_time, dtype=tf.float32)

    out_states = py_utils.NestedMap()
    out_states.time_step = states.time_step + 1
    out_states.dec = moe_builder.StateLayer.UpdateState(self.dec,
                                                        theta_with_state,
                                                        states.dec)

    tf.logging.info('_PreBeamSearchStepCallback bs_results=%r', bs_results)
    tf.logging.info('_PreBeamSearchStepCallback out_states=%r', out_states)
    return (bs_results, out_states)

  def _PostBeamSearchStepCallback(self, theta, encoder_outputs, new_step_ids,
                                  states):
    p = self.params
    tf.logging.info('_PostBeamSearchStepCallback encoder_outputs=%r',
                         encoder_outputs)
    tf.logging.info('_PostBeamSearchStepCallback new_step_ids=%r',
                         new_step_ids)
    tf.logging.info('_PostBeamSearchStepCallback states=%r', states)

    def _MaybeSplit(x):
      if x.shape.rank < 1:
        return x
      else:
        return _Split(x, 0, p.builder.num_devices)

    new_states = states.Transform(_MaybeSplit)
    tf.logging.info('_PostBeamSearchStepCallback new_states=%r',
                         new_states)

    return new_states

  def CreateDecoderMetrics(self):
    p = self.params
    if hasattr(p.input, 'target_language') and p.input.target_language:
      target_language = p.input.target_language
    else:
      target_language = p.eval.inference_target_language

    decoder_metrics = {
        'num_samples_in_batch': metrics.AverageMetric(),
        'gleu': metrics.AverageMetric(),
        'corpus_bleu': metrics.CorpusBleuMetric(),
        'canonical_bleu': bf_metrics.CanonicalBleuMetric(target_language),
    }

    return decoder_metrics

  def PostProcessDecodeOut(self, dec_out, dec_metrics):
    return self._PostProcessBeamSearchDecodeOut(dec_out, dec_metrics)

  def _PostProcessBeamSearchDecodeOut(self, dec_out_dict, dec_metrics_dict):
    """Post processes the output from _BeamSearchDecode."""
    p = self.params
    topk_scores = dec_out_dict['topk_scores']
    topk_decoded = dec_out_dict['topk_decoded']
    targets = dec_out_dict['targets']
    sources = dec_out_dict['sources']
    gleu = dec_out_dict['gleu']

    num_samples = len(targets)
    assert num_samples == len(topk_decoded), ('%s vs %s' %
                                              (num_samples, len(topk_decoded)))
    assert num_samples == len(sources)
    assert num_samples == len(gleu)
    dec_metrics_dict['num_samples_in_batch'].Update(num_samples)

    key_value_pairs = []
    for i in range(num_samples):
      src, tgt = sources[i], targets[i]
      if i < 10:
        tf.logging.info('source: %s', src)
        tf.logging.info('  target: %s', tgt)
      hyps = topk_decoded[i]
      assert p.beam_search.num_hyps_per_beam == len(hyps)
      info = [src, tgt]
      for n, (score, hyp_str) in enumerate(zip(topk_scores[i], hyps)):
        if i < 10:
          tf.logging.info('  %f: %s', score, hyp_str)
        hyp_gleu = gleu[i][n]
        info += [(hyp_str, score, hyp_gleu)]
        # Only aggregate scores of the top hypothesis.
        if n == 0:
          dec_metrics_dict['gleu'].Update(hyp_gleu, 1)
          dec_metrics_dict['corpus_bleu'].Update(tgt, hyp_str)
          dec_metrics_dict['canonical_bleu'].Update(tgt, hyp_str, src)
      key_value_pairs.append((src, '%s' % info))
    return key_value_pairs
