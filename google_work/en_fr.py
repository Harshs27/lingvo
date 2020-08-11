# Lint as: python2, python3
"""Models for Wmt'14 En->Fr dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from lingvo import compat as tf
from lingvo.core import attention
from lingvo.core import base_model_params
from lingvo.core import layers
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import quant_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import schedule
from six.moves import range
from google3.learning.brain.research.babelfish import attention as bf_attention
from google3.learning.brain.research.babelfish import layers as bf_layers
from google3.learning.brain.research.babelfish import model_helper
from google3.learning.brain.research.babelfish import model_registry
from google3.learning.brain.research.babelfish import optimizer as bf_optimizer
from google3.learning.brain.research.babelfish import rnn_layers as bf_rnn_layers
from google3.learning.brain.research.babelfish import tokenizers
from google3.learning.brain.research.babelfish.mt import base_config
from google3.learning.brain.research.babelfish.mt import decoder
from google3.learning.brain.research.babelfish.mt import encoder
from google3.learning.brain.research.babelfish.mt import encoder_model
from google3.learning.brain.research.babelfish.mt import input_generator
from google3.learning.brain.research.babelfish.mt import input_policy
from google3.learning.brain.research.babelfish.mt import layers as mt_layers
from google3.learning.brain.research.babelfish.mt import model

FLAGS = tf.flags.FLAGS


@model_registry.RegisterSingleTaskModel
class WmtEnFr(base_model_params.SingleTaskModelParams):
  """Params for WMT'14 En->Fr."""

  DATADIR = '/placer/prod/home/brain-speech-exp/babelfish/wmt14_en_fr_wpm32k'

  def Train(self, params=None):
    p = base_config.InitTrainDatasetParams(params=params)
    p.is_nmt_example = False
    p.file_pattern = os.path.join(
        self.DATADIR, 'train-split-backward-maxlen200-?????-of-00036')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 36302505
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98]
    p.bucket_batch_limit = [128] * 7 + [64]
    return p

  def Dev(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    # Dev set = concatenated newstest2012 and newstest2013, no pre-processing.
    p.file_pattern = os.path.join(
        self.DATADIR, 'dev-split-backward-maxlen200-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 6002
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 6 + [64, 32, 16]
    return p

  def Test(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(
        self.DATADIR, 'test-split-backward-maxlen200-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 3003
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  def Test0_10(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(self.DATADIR, 'corpus.test0_10.nmt')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.bucket_upper_bound = [30]
    p.bucket_batch_limit = [128]
    p.num_samples = 191
    return p

  def Test10_30(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(self.DATADIR, 'corpus.test10_30.nmt')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.bucket_upper_bound = [50]
    p.bucket_batch_limit = [128]
    p.num_samples = 1601
    return p

  def Test30_50(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(self.DATADIR, 'corpus.test30_50.nmt')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.bucket_upper_bound = [70]
    p.bucket_batch_limit = [128]
    p.num_samples = 960
    return p

  def Test50_70(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(self.DATADIR, 'corpus.test50_70.nmt')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.bucket_upper_bound = [90]
    p.bucket_batch_limit = [128]
    p.num_samples = 226
    return p

  def Test70_90(self):
    p = base_config.InitTestDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(self.DATADIR, 'corpus.test70_90.nmt')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.bucket_upper_bound = [110]
    p.bucket_batch_limit = [30]
    p.num_samples = 25
    return p

  def Task(self):
    p = model.MTModelV1.Params()
    p.name = 'wmt14_en_fr'

    p.eval.inference_source_language = 'en'
    p.eval.inference_target_language = 'fr'
    p.eval.samples_per_summary = 7500

    p.train.lr_schedule.start_step = 800000
    p.train.lr_schedule.half_life_steps = 100000
    p.train.lr_schedule.min = 0.1

    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrT2T(WmtEnFr):
  """Params for WMT'14 En->Fr with tensor2tensor SubwordTextEncoder.

  Used t2t text_encoder.SubwordTextEncoder(DATADIR + 'vocab.enfr.32768').
  """

  # See README.txt in DATADIR.
  DATADIR = '/cns/vz-d/home/nmt/rs=6.3/data/t2t_data_enfr_wmt32k'

  def Train(self):
    p = base_config.InitTrainDatasetParams(vocab_size=2**15)
    p.file_pattern = os.path.join(
        self.DATADIR, 'translate_enfr_wmt32k-train-nmtexample-?????-of-00100')
    p.tokenizer = tokenizers.T2TTokenizer.Params()
    p.tokenizer.t2t_vocab = os.path.join(self.DATADIR, 'vocab.enfr.32768')
    # fileutil cat /cns/vz-d/home/nmt/rs=6.3/data/t2t_data_enfr_wmt32k/
    # vocab.enfr.32768 | wc -l
    # 27665
    # Note: only 27665 subtokens. T2T uses approximate vocab size, so 2*15
    # is a guidance for t2t_datagen that can produce larger or smaller vocab.
    #
    # No filtering of train data:
    # 3244152 commoncrawl.fr-en.{en,fr}
    # 2007723 training/europarl-v7.fr-en.{en,fr}
    # 183251 training/news-commentary-v9.fr-en.{en,fr}
    # 22520376 giga-fren.release2.fixed.{en,fr}.gz"
    # 12886831 un/undoc.2000.fr-en.{en,fr}
    #
    # Total: 40842333
    p.num_samples = 40842333
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 256]
    p.bucket_batch_limit = [128] * 7 + [64] + [32]
    return p

  def Dev(self):
    p = base_config.InitTestDatasetParams(vocab_size=2**15)
    # Dev set newstest2013, no pre-processing.
    p.file_pattern = os.path.join(
        self.DATADIR, 'translate_enfr_wmt32k-dev-nmtexample-00000-of-00001')
    p.tokenizer = tokenizers.T2TTokenizer.Params()
    p.tokenizer.t2t_vocab = os.path.join(self.DATADIR, 'vocab.enfr.32768')
    p.num_samples = 3000
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200, 256]
    p.bucket_batch_limit = [128] * 8 + [32] + [32]
    return p

  def Test(self):
    p = base_config.InitTestDatasetParams(vocab_size=2**15)
    # Dev set newstest2014, no pre-processing.
    p.file_pattern = os.path.join(
        self.DATADIR,
        'translate_enfr_wmt32k-newstest2014-fren-nmtexample-00000-of-00001')
    p.tokenizer = tokenizers.T2TTokenizer.Params()
    p.tokenizer.t2t_vocab = os.path.join(self.DATADIR, 'vocab.enfr.32768')
    p.num_samples = 3003
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200, 256]
    p.bucket_batch_limit = [128] * 8 + [32] + [32]
    return p

  def Task(self):
    p = model.MTModelV1.Params()
    p.name = 'wmt14_en_fr'

    p.encoder.emb.vocab_size = 2**15
    p.decoder.emb.vocab_size = 2**15
    p.decoder.softmax.num_classes = 2**15

    p.eval.inference_source_language = 'en'
    p.eval.inference_target_language = 'fr'
    p.eval.samples_per_summary = 7500

    p.train.lr_schedule.start_step = 800000
    p.train.lr_schedule.half_life_steps = 100000
    p.train.lr_schedule.min = 0.1

    return p


class WmtEnFrForJFBase(WmtEnFr):
  """Params for WMT'14 En->Fr running on JF."""

  def Task(self):
    p = super(WmtEnFrForJFBase, self).Task()
    # Uses the simple embedding layer on JF.
    p.encoder.emb = model_helper.ChangeToSimpleEmbedding(p.encoder.emb)
    p.decoder.emb = model_helper.ChangeToSimpleEmbedding(p.decoder.emb)
    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)
    return p


# https://mldash.corp.google.com/experiments/1129002880321234268
@model_registry.RegisterSingleTaskModel
class WmtEnFrForJFv000(WmtEnFrForJFBase):
  """Params for WMT'14 En->Fr running on JF."""

  def Train(self):
    p = super(WmtEnFrForJFv000, self).Train()
    p = model_helper.FixateInputShape(p, 128, 50, 50)
    return p

  def Task(self):
    p = super(WmtEnFrForJFv000, self).Task()
    p.train.optimizer = optimizer.SGD.Params()
    p.train.learning_rate = 0.3
    return p


# https://mldash.corp.google.com/experiments/1201089793051984591
@model_registry.RegisterSingleTaskModel
class WmtEnFrForJFv001(WmtEnFrForJFBase):
  """Params for WMT'14 En->Fr running on JF."""

  def Train(self):
    p = super(WmtEnFrForJFv001, self).Train()
    p = model_helper.FixateInputShape(p, 64, 100, 100)
    return p

  def Task(self):
    p = super(WmtEnFrForJFv001, self).Task()
    p.train.optimizer = optimizer.SGD.Params()
    p.train.learning_rate = 0.3
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrSync(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFr().Task()
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=300, decay_start=400000, decay_end=1200000, min=0.1))
    return p


# max bleu 40.15 @ 73.k steps
# Sync replica training with 16 towers (4 workers, 4 gpus each).
# logpplx 1.157 @20k 36.7
#         1.075 @40k 38.0
#         1.056 @50k
#         0.995 @70k 39.5
#         0.989 @90k
@model_registry.RegisterSingleTaskModel
class WmtEnFrSyncDropout(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFr().Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)
    enc = p.encoder
    enc.lstm_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        output_nonlinearity=False,
        params_init=default_params_init)
    enc.dropout_prob = 0.2

    dec = p.decoder
    rnn_cell_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        output_nonlinearity=False,
        params_init=default_params_init)
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.dropout_prob = 0.2

    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=800000, decay_end=1600000, min=0.1))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


# https://mldash.corp.google.com/experiments/4172909550472566681#scalars
#
# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/4172909550472566681#scalars
#
# logpplx 1.06  @30k bleu 38.0
#         0.98  @60k      39.9
#         0.97  @90k      40.0
#
# Steps/sec = 0.23
@model_registry.RegisterSingleTaskModel
class WmtEnFrBiEncSyncDropout(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFr().Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)
    p.encoder = encoder.MTEncoderBiRNN.Params()
    enc = p.encoder
    enc.lstm_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        num_output_nodes=512,
        output_nonlinearity=False,
        params_init=default_params_init)
    enc.dropout_prob = 0.2

    dec = p.decoder
    rnn_cell_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        output_nonlinearity=False,
        params_init=default_params_init)
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.dropout_prob = 0.2

    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.learning_rate = 1e-4
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=1200000, decay_end=3600000, min=0.2))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/3685692074079619478#scalars
#
# logpplx 1.054    @30k bleu  38.47
#         0.9689   @60k       40.15
#         0.9548   @90k       40.25
#         0.9526   @120k      40.27
#
# Steps/sec = 0.23
@model_registry.RegisterSingleTaskModel
class WmtEnFrBiEncSyncDropoutBigger(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFr().Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)
    p.encoder = encoder.MTEncoderBiRNN.Params()
    enc = p.encoder
    enc.num_lstm_layers = 6
    enc.lstm_cell_size = 1024
    enc.encoder_out_dim = 1024
    enc.lstm_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        output_nonlinearity=False,
        params_init=default_params_init)
    enc.dropout_prob = 0.2

    dec = p.decoder
    rnn_cell_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        output_nonlinearity=False,
        params_init=default_params_init)
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.dropout_prob = 0.2

    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.learning_rate = 1e-4
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=1200000, decay_end=3600000, min=0.5))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/1420184617947116474#scalars
#
# logpplx 2.350    @30k   bleu  38.86
#         2.269    @60k         40.28
#         2.252    @90k         40.50
#         2.250    @120k        40.60
# highest ckpt:    @130.3k      40.70
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrBiEncSyncDropoutBiggerLS(WmtEnFrBiEncSyncDropoutBigger):
  """Params for WMT'14 En->Fr BNMT V2 with label smoothing."""

  def Task(self):
    p = WmtEnFrBiEncSyncDropoutBigger().Task()
    p.decoder.label_smoothing = layers.UniformLabelSmoother.Params()
    p.decoder.label_smoothing.num_classes = 32000
    p.decoder.label_smoothing.uncertainty = 0.1
    return p


# This version is BNMT v2.0.
# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/2280051893584492006#scalars
#
# logpplx 2.390    @30k   bleu         38.79
#         2.299    @60k                40.14
#         2.275    @90k                40.44
#         2.271    @120k               40.57
# highest ckpt:                        40.71
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrV2(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFr().Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)

    p.encoder = encoder.MTEncoderBiRNN.Params()
    enc = p.encoder
    enc.num_lstm_layers = 6
    enc.lstm_cell_size = 1024
    enc.encoder_out_dim = 1024
    enc.lstm_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=1024,
        params_init=default_params_init)
    enc.dropout_prob = 0.2

    dec = p.decoder
    rnn_cell_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=1024,
        params_init=default_params_init)
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.dropout_prob = 0.2
    dec.label_smoothing = layers.UniformLabelSmoother.Params().Set(
        num_classes=32000,
        uncertainty=0.1)
    dec.beam_search.num_hyps_per_beam = 16

    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.learning_rate = 1e-4
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=1200000, decay_end=3600000, min=0.5))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


# BNMT v2.0 with quantized training.
@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFr().Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)
    cc_schedule = quant_utils.LinearClippingCapSchedule.Params().Set(
        start_step=0,
        end_step=20000,
        start_cap=8.0,
        end_cap=1.0)

    p.encoder = encoder.MTEncoderBiRNN.Params()
    enc = p.encoder
    enc.num_lstm_layers = 6
    enc.lstm_cell_size = 1024
    enc.encoder_out_dim = 1024
    enc.lstm_tpl = rnn_cell.QuantizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        params_init=default_params_init)
    enc.lstm_tpl.cc_schedule = cc_schedule.Copy()
    enc.cc_schedule = cc_schedule.Copy()
    enc.dropout_prob = 0.2

    dec = p.decoder
    rnn_cell_tpl = rnn_cell.QuantizedLSTMCell.Params().Set(
        num_output_nodes=1024,
        params_init=default_params_init)
    rnn_cell_tpl.cc_schedule = cc_schedule.Copy()
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.cc_schedule = cc_schedule.Copy()
    dec.dropout_prob = 0.2
    dec.softmax.logits_abs_max = 20.0
    dec.label_smoothing = layers.UniformLabelSmoother.Params().Set(
        num_classes=32000,
        uncertainty=0.1)
    dec.beam_search.num_hyps_per_beam = 16

    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.learning_rate = 1e-4
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=512000, decay_end=3600000, min=0.5))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


# BNMT v2.0 using previous attention with quantized training.
@model_registry.RegisterSingleTaskModel
class WmtEnFrV2PrevCtxQuantized(WmtEnFrV2Quantized):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = WmtEnFrV2Quantized().Task()
    p.encoder.proj_tpl.weight_norm = True
    p.decoder.use_prev_atten_ctx = True
    return p


# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/1521900276526734251#scalars
#
#                               num_hyps_per_beam=8  num_hyps_per_beam=20
# logpplx 2.358    @30k   bleu         38.73
#         2.268    @60k                40.54
#         2.247    @90k                40.91               40.92
#         2.244    @120k               40.95               40.89
# highest ckpt:                        41.13 (@83.54k)     41.23 (@87.47k)
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrSyncMultiHead(WmtEnFrBiEncSyncDropoutBiggerLS):
  """Params for WMT'14 En->Fr in sync training.

  Note: this model is equivalent to base_config.BNMTV25Params().
  """

  def Task(self):
    p = super(WmtEnFrSyncMultiHead, self).Task()
    dec = p.decoder
    # Note: feed_attention_context_vec_to_softmax=True is important for
    # the training to be stable, and atten_dropout_prob=0.2 is important
    # for the model performance.
    iap = attention.AdditiveAttention.Params()
    dec.attention = attention.MultiHeadedAttention.Params().Set(
        source_dim=1024,
        hidden_dim=1024,
        query_dim=1024,
        context_dim=1024,
        num_attention_heads=4,
        inner_atten_params=iap,
        use_source_vec_as_attention_value=True,
        enable_ctx_pre_proj=False,
        enable_query_proj=True,
        atten_dropout_prob=0.2)
    dec.atten_rnn_cls = bf_rnn_layers.DRNNWithAttention
    dec.feed_attention_context_vec_to_softmax = True
    dec.beam_search.num_hyps_per_beam = 20
    return p


# https://mldash.corp.google.com/experiments/4056696981642219009
@model_registry.RegisterSingleTaskModel
class WmtEnFrSyncMultiHeadNoLabelSmoothing(WmtEnFrSyncMultiHead):
  """Params for WMT'14 En->Fr without label smoothing. Ablation #1."""

  def Task(self):
    p = super(WmtEnFrSyncMultiHeadNoLabelSmoothing, self).Task()
    p.decoder.label_smoothing = None
    return p


# https://mldash.corp.google.com/experiments/9070800794132316593
@model_registry.RegisterSingleTaskModel
class WmtEnFrSyncNoMultiHead(WmtEnFrSyncMultiHead):
  """Params for WMT'14 En->Fr without multi-head attn. Ablation #2."""

  def Task(self):
    p = super(WmtEnFrSyncNoMultiHead, self).Task()
    # no multi-head, with proj
    iap = attention.AdditiveAttention.Params()
    p.decoder.attention = attention.MultiHeadedAttention.Params().Set(
        source_dim=1024,
        hidden_dim=1024,
        query_dim=1024,
        context_dim=1024,
        num_attention_heads=1,
        inner_atten_params=iap,
        use_source_vec_as_attention_value=True,
        enable_ctx_pre_proj=False,
        enable_query_proj=True,
        atten_dropout_prob=0.2)
    return p


# https://mldash.corp.google.com/experiments/7259380815796491788
@model_registry.RegisterSingleTaskModel
class WmtEnFrSyncMultiHeadNoLayerNorm(WmtEnFrSyncMultiHead):
  """Params for WMT'14 En->Fr without layer norm. Ablation #3."""

  def Task(self):
    p = super(WmtEnFrSyncMultiHeadNoLayerNorm, self).Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)
    # Remove layer-norm in the encoder.
    p.encoder.lstm_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=1024,
        params_init=default_params_init)
    # Remove layer-norm in decoder and attn.
    rnn_cell_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=1024,
        params_init=default_params_init)
    p.decoder.rnn_cell_tpl = rnn_cell_tpl.Copy()
    p.decoder.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    return p


# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/1711835038660667426#scalars
# https://mldash.corp.google.com/experiments/5837413086584106834#scalars
#
#                               num_hyps_per_beam=8  num_hyps_per_beam=16
# logpplx 2.335    @30k   bleu         38.89
#         2.254    @60k                40.49
#         2.236    @90k                40.80               40.87
#         2.233    @120k               40.91               40.95
# highest ckpt:                        41.07 (@104.4k)     41.14 (@100.6k)
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrSyncMultiHeadDotProduct(WmtEnFrBiEncSyncDropoutBiggerLS):
  """Params for WMT'14 En->Fr in sync training."""

  def Task(self):
    p = super(WmtEnFrSyncMultiHeadDotProduct, self).Task()
    dec = p.decoder
    # Note: Both feed_attention_context_vec_to_softmax=True and label
    # smoothing are important for the model performance.
    dec.attention = attention.MultiHeadedAttention.Params().Set(
        source_dim=1024,
        hidden_dim=1024,
        query_dim=1024,
        context_dim=1024,
        num_attention_heads=8,
        use_source_vec_as_attention_value=False,
        enable_ctx_pre_proj=True)
    dec.atten_rnn_cls = bf_rnn_layers.DRNNWithAttention
    dec.feed_attention_context_vec_to_softmax = True
    dec.beam_search.num_hyps_per_beam = 16
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrConvSeq2Seq(WmtEnFr):
  """Params for WMT'14 En->Fr using convolutional seq2seq model.

  These settings are intended to replicate ones in the original paper.
  Discrepancies:
    - Vocabulary is 32k rather than 40k.
    - Learning regime is different.
    - 14 layers (they say 15 but describe only 14)
  """

  def Train(self):
    p = super(WmtEnFrConvSeq2Seq, self).Train()
    p.bucket_batch_limit = [64] * 8
    return p

  def Task(self):
    layer_dims = [512] * 5 + [768] * 4 + [1024] * 3 + [2048, 4096]
    filter_sizes = [3] * 12 + [1] * 2
    conv_dropout_prob = 0.1
    p = base_config.SetupConvSeq2Seq(
        name='wmt14_en_fr_cnn',
        inference_source_language='en',
        inference_target_language='fr',
        vocab_size=32000,
        conv_dropout_prob=conv_dropout_prob,
        layer_dims=layer_dims,
        filter_sizes=filter_sizes,
        per_word_loss=True,
        batch_norm=False,
        weight_norm=True,
        lr_schedule_start_step=800000,
        lr_schedule_half_life_step=800000)

    # Initializing various matrices...
    sdev_scale = math.sqrt(1.0 - conv_dropout_prob)

    conv_init = py_utils.WeightInit.GaussianSqrtDim(2.0 * sdev_scale)
    proj_init = py_utils.WeightInit.GaussianSqrtDim(sdev_scale)
    attn_init = py_utils.WeightInit.GaussianSqrtDim(1.0)

    p.encoder.conv_tpl.params_init = conv_init
    p.decoder.conv_tpl.params_init = conv_init

    p.encoder.proj_tpl.params_init = proj_init
    p.decoder.proj_tpl.params_init = proj_init

    p.decoder.atten_tpl.params_init = attn_init
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrConvSeq2SeqOpt(WmtEnFrConvSeq2Seq):
  """Optimal settings for conv seq2seq model on WMT'14 En-Fr."""

  def Task(self):
    layer_dims = [512] * 5 + [768] * 4 + [1024] * 3 + [2048, 4096]
    filter_sizes = [3] * 12 + [1] * 2
    p = base_config.SetupConvSeq2Seq(
        name='wmt14_en_fr_cnn_opt',
        inference_source_language='en',
        inference_target_language='fr',
        vocab_size=32000,
        input_dropout_prob=0.05,
        conv_dropout_prob=0.05,
        layer_dims=layer_dims,
        filter_sizes=filter_sizes,
        per_word_loss=True,
        scale_attn=False,
        batch_norm=False,
        weight_norm=True,
        lr_schedule_start_step=800000,
        lr_schedule_half_life_step=800000)

    p.train.learning_rate = 0.0003
    p.train.start_up_delay_steps = 500

    return p

  def Train(self):
    p = super(WmtEnFrConvSeq2SeqOpt, self).Train()
    p.bucket_batch_limit = [512, 341, 256, 170, 128, 85, 64, 42]
    p.bucket_upper_bound = [8, 12, 16, 24, 32, 48, 64, 96]
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrConvSeq2SeqOptBig(WmtEnFrConvSeq2SeqOpt):
  """ConvSeq2SeqOpt with extra proj layer and bigger input embeddinngs."""

  def Task(self):
    p = WmtEnFrConvSeq2SeqOpt().Task()
    emb_dim = 768
    p.encoder.emb_dim = emb_dim
    p.encoder.token_emb.embedding_dim = emb_dim
    p.encoder.position_emb.embedding_dim = emb_dim
    p.encoder.conv_dropout_prob = 0.1
    p.encoder.input_dropout_prob = 0.1

    p.decoder.token_emb.embedding_dim = emb_dim
    p.decoder.position_emb.embedding_dim = emb_dim
    p.decoder.source_dim = emb_dim
    p.decoder.output_dim = 512
    p.decoder.atten_tpl.source_dim = emb_dim
    p.decoder.atten_tpl.query_dim = emb_dim
    p.decoder.atten_tpl.hidden_dim = emb_dim
    p.decoder.conv_dropout_prob = 0.1
    p.decoder.input_dropout_prob = 0.1
    p.decoder.softmax_dropout_prob = 0.1
    p.decoder.pre_out_dim = 768

    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrConvSeq2SeqOptBigLS(WmtEnFrConvSeq2SeqOptBig):
  """ConvSeq2SeqOptBig with label smoothing."""

  def Task(self):
    p = WmtEnFrConvSeq2SeqOptBig().Task()
    p.decoder.label_smoothing = layers.UniformLabelSmoother.Params()
    p.decoder.label_smoothing.num_classes = 32000
    p.decoder.label_smoothing.uncertainty = 0.1
    p.decoder.beam_search.num_hyps_per_beam = 16
    return p


# Sync replica training with 16 towers (4 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/8621221427073648767#scalars
#
# logpplx 2.385    @50k bleu  36.99
#         2.337    @100k      38.00
#         2.304    @200k      39.02
#         2.290    @300k      39.25
# highest ckpt:    @582.1k    39.82
#
# Note: this job uses rpc2.
# steps/sec = ~1.5
# examples/sec = ~2300
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformer(WmtEnFr):
  """Params for WMT'14 En->Fr using the base transformer model."""

  def Train(self):
    p = super(WmtEnFrTransformer, self).Train()
    p.bucket_upper_bound = [8, 12, 16, 24, 32, 48, 64, 96]
    p.bucket_batch_limit = [512, 341, 256, 170, 128, 85, 64, 42]
    return p

  def Dev(self):
    p = super(WmtEnFrTransformer, self).Dev()
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [64] * 8 + [16]
    return p

  def Test(self):
    p = super(WmtEnFrTransformer, self).Test()
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [64] * 8 + [16]
    return p

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=32000,
        model_dim=512,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        learning_rate=1.0,
        warmup_steps=4000
        )

    p.eval.samples_per_summary = 7500
    return p


# TODO(lepikhin): move out of en_fr.py
#
# See g3doc.
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerWorkshop(base_model_params.SingleTaskModelParams):
  """TransformerBase GPU model."""

  VOCAB_SIZE = 32000
  DATADIR = '/cns/oz-d/home/nmt/rs=6.3/data/wmt14/en_fr'

  def Train(self):
    p = base_config.InitTrainDatasetParams()
    p.is_nmt_example = True
    p.file_pattern = os.path.join(self.DATADIR,
                                  'train.wpm32k.nmtexample-?????-of-?????')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 39984473
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98]
    p.bucket_batch_limit = [128] * 7 + [64]
    return p

  def Dev(self):  # newstest2013
    p = base_config.InitTestDatasetParams(vocab_size=self.VOCAB_SIZE)

    p.file_pattern = os.path.join(
        self.DATADIR, 'newstest2013-en_fr.wpm32k.nmtexample-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    p.num_samples = 3000
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  def Test(self):  # newstest2014
    p = base_config.InitTestDatasetParams(vocab_size=self.VOCAB_SIZE)

    p.file_pattern = os.path.join(
        self.DATADIR, 'newstest2014-en_fr.wpm32k.nmtexample-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    p.num_samples = 3003
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=512,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        learning_rate=1.0,
        warmup_steps=4000)

    p.eval.samples_per_summary = 7500
    return p


# Sync replica training with 16 towers (4 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/8953154344722550423#scalars
#
# logpplx 2.313    @50k bleu  37.28
#         2.218    @100k      39.96
#         2.193    @150k      40.51
# highest ckpt:    @186.9k    41.19
#
# Note: this job uses rpc2.
# steps/sec = ~0.41
# examples/sec = ~700
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBig(WmtEnFrTransformer):
  """Params for WMT'14 En->Fr using the Big transformer model."""

  def Train(self):
    p = super(WmtEnFrTransformerBig, self).Train()
    p.bucket_upper_bound = (
        [8, 10, 12, 14, 16, 20, 24, 28] + [32, 40, 48, 56, 64, 80, 96])
    p.bucket_batch_limit = ([512, 409, 341, 292, 256, 204, 170, 146] +
                            [128, 102, 85, 73, 64, 51, 42])
    return p

  def Dev(self):
    p = super(WmtEnFrTransformerBig, self).Dev()
    p.bucket_batch_limit = [16] * 8 + [4]
    return p

  def Test(self):
    p = super(WmtEnFrTransformerBig, self).Test()
    p.bucket_batch_limit = [16] * 8 + [4]
    return p

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer_big',
        vocab_size=32000,
        model_dim=1024,
        hidden_dim=8192,
        num_heads=16,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=40000
        )

    p.eval.samples_per_summary = 7500
    return p


# model      corpus BLEU        log_pplx     @steps
#            dev    test        dev    test  (dev)
#-----------------------------------------------------
# baseline   32.9   41.1/40.7+  2.52   2.16  175.4k
# 1-gram     30.0   36.7        2.97   2.52  176.4k
# 2-gram     32.2   40.3        2.58   2.21  183.7k
# 3-gram     32.4   40.4        2.56   2.19  180.3k
# 5-gram     32.7   41.1        2.53   2.17  200.4k *
# 7-gram     32.7   40.7        2.52   2.17  176.5k *
# 9-gram     32.8   40.8        2.52   2.17  166.9k *
#
# +: depending on exact checkpoint on test
#
# TODO(ciprianchelba): seems that the N is in fact the context length.
# make sure it is consistent with the standard N-gram definition, i.e. it also
# includes the predicted position.
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBig5gramDecoder(WmtEnFrTransformerBig):
  """N-gram decoder for the baseline transformer model above."""

  def Task(self):
    p = super(WmtEnFrTransformerBig5gramDecoder, self).Task()
    # Turn on N-gram masking in the decoder TransformerLayer.
    # Before doing so though copy the self-attention params to avoid
    # the auxiliary attention being masked as well.
    dec = p.decoder
    dec.trans_tpl.tr_aux_atten_tpl = dec.trans_tpl.tr_atten_tpl.Copy()
    dec.trans_tpl.mask_self_atten = True
    dec.trans_tpl.tr_atten_tpl.is_masked = True
    dec.trans_tpl.tr_atten_tpl.mask_ngram_order = 5
    dec.trans_tpl.tr_atten_tpl.mask_type = 'ngram'
    return p


# http://mldash/experiments/5709438560188237508
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerDeterministic(WmtEnFrTransformer):
  """WmtEnFrTransformer with deterministic dropouts (w/additional dropouts)."""

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=32000,
        model_dim=512,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=1.0,
        warmup_steps=4000)

    p.eval.samples_per_summary = 7500
    return base_config.SetupDeterministicTransformer(p)


# 4x4 GPU (caveat: gradient summation, not averaging), seems to have no effect
#   http://mldash/experiments/8331399739801177656
# Comparison with baseline and 16x16 JF run:
# http://mldash/compare?eidstrs=8331399739801177656,8953154344722550423,2440026616471570208
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAcc(WmtEnFrTransformerBig):
  """Gradient accumulation for 5 steps."""

  def Task(self):
    p = super(WmtEnFrTransformerBigAcc, self).Task()
    p.train.optimizer = optimizer.Accumulator.Params().Set(
        accum_steps=5, optimizer_tpl=p.train.optimizer)
    return p


# https://mldash.corp.google.com/experiments/3801544609909926108
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigNoLabelSmoothing(WmtEnFrTransformerBig):
  """Params for WMT'14 En->Fr without label smoothing. Ablation #1."""

  def Task(self):
    p = super(WmtEnFrTransformerBigNoLabelSmoothing, self).Task()
    p.decoder.label_smoothing = None
    return p


# https://mldash.corp.google.com/experiments/2785678758473952008
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigNoMultiHead(WmtEnFrTransformerBig):
  """Params for WMT'14 En->Fr without multi-headed attention. Ablation #2."""

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer_big_no_mha',
        vocab_size=32000,
        model_dim=1024,
        hidden_dim=8192,
        num_heads=1,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=40000
    )
    p.eval.samples_per_summary = 7500
    return p


# https://mldash.corp.google.com/experiments/3047747970299494141
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigNoWarmup(WmtEnFrTransformerBig):
  """Params for WMT'14 En->Fr without LR warmup. Ablation #3."""

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer_big_no_wu',
        vocab_size=32000,
        model_dim=1024,
        hidden_dim=8192,
        num_heads=16,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=0.000001
    )
    p.train.Set(
        learning_rate=3.0,
        optimizer=optimizer.Adam.ParamsB(),
        clip_gradient_norm_to_value=0.0,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=schedule.TransformerScheduleNoWarmUp.Params().Set(
            decay_start=4000, worker_replicas=1, model_dim=1024))
    p.eval.samples_per_summary = 7500
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrForTpuTest(base_model_params.SingleTaskModelParams):
  """Params for testing model training on TPU."""

  def Train(self):
    # For unit-tests, the data is also available at
    # datadir = os.path.join(
    #     FLAGS.test_srcdir,
    #     'google3/learning/brain/research/babelfish/ops/testdata')
    datadir = '/cns/oz-d/home/yonghui/tmp/tpu_debug/'
    p = input_generator.NmtInput.Params()
    p.is_nmt_example = True
    p.file_pattern = os.path.join(datadir, 'wmt_enfr_wpm_500_test.nmtexample')
    p.file_random_seed = 31415
    p.file_parallelism = 1
    p.bucket_upper_bound = [40]
    p.bucket_batch_limit = [8]
    p.tokenizer.wpm_model = os.path.join(datadir, 'wmt_enfr_wpm_500.wpm')
    p.source_max_length = 40
    p.target_max_length = 40
    p.tokenizer.vocab_size = 512
    p.pad_to_max_seq_length = True
    return p

  def Dev(self):
    return self.Train()

  def Test(self):
    return self.Train()

  def Task(self):
    p = model.MTModelV1.Params()
    p.name = 'test_mdl'

    # ep = encoder.MTEncoderV1.Params()
    ep = encoder.MTEncoderBiRNN.Params()
    ep.name = 'encoder'
    ep.emb = layers.SimpleEmbeddingLayer.Params()
    ep.emb.vocab_size = 512
    ep.emb.embedding_dim = 256
    ep.emb.use_matmul = True
    ep.lstm_cell_size = 4
    ep.num_lstm_layers = 1
    ep.encoder_out_dim = 4
    p.encoder = ep

    dp = decoder.MTDecoderV1.Params()
    dp.name = 'decoder'
    dp.source_dim = 4
    dp.emb = layers.SimpleEmbeddingLayer.Params()
    dp.emb.vocab_size = 512
    dp.emb.embedding_dim = 256
    dp.emb.use_matmul = False
    dp.rnn_cell_dim = 4
    dp.rnn_layers = 1
    dp.attention.hidden_dim = 2
    dp.softmax = layers.SimpleFullSoftmax.Params()
    dp.softmax.num_classes = 512
    dp.softmax.num_shards = 1
    p.decoder = dp

    p.eval.samples_per_summary = 10
    return p


# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/120631922441990058#scalars
#
# logpplx 2.347    @50k bleu  38.91
#         2.252    @100k      40.58
#         2.232    @150k      41.36
#         2.226    @200k      41.39
# highest ckpt:    @183.6k    41.71
#
# steps/sec = ~0.3
# examples/sec = ~1200
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerEncBNMTV25DecAdditive(WmtEnFr):
  """Params for WMT'14 En->Fr using transformer encoder, bnmt v2.5 decoder."""

  def Task(self):
    p = model.HybridModel.Params()
    p.name = 'wmt14_en_fr_xformer_enc_bnmtv25_dec'
    p.eval.samples_per_summary = 7500
    p.eval.inference_source_language = 'en'
    p.eval.inference_target_language = 'fr'

    p.encoder = WmtEnFrTransformerBig().Task().encoder
    p.decoder = WmtEnFrSyncMultiHead().Task().decoder

    p.train.Set(
        learning_rate=4.0,
        optimizer=optimizer.Adam.ParamsB(),
        clip_gradient_norm_to_value=0.0,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=schedule.TransformerSchedule.Params().Set(
            warmup_steps=50000, worker_replicas=1, model_dim=1024))
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    return p


# TODO(yonghui): convert this model into a canonical config.
class WmtEnFrV2QuantizedJF(base_model_params.SingleTaskModelParams):
  """Params for WMT'14 En->Fr in sync training."""

  DATADIR = ''
  VOCAB_SIZE = -1
  MODEL_DIM = -1

  def Train(self):
    p = base_config.InitTrainDatasetParams(vocab_size=self.VOCAB_SIZE)
    p.file_pattern = os.path.join(self.DATADIR,
                                  'train.mosestok.nmtexample-?????-of-00036')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wpm')
    p.num_samples = 39949297
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98]
    p.bucket_batch_limit = [128] * 7 + [64]
    return p

  def Dev(self):
    p = base_config.InitTestDatasetParams(vocab_size=self.VOCAB_SIZE)
    # newstest2013
    p.file_pattern = os.path.join(self.DATADIR,
                                  'dev.mosestok.nmtexample-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wpm')
    p.num_samples = 3000
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  def Test(self):
    p = base_config.InitTestDatasetParams(vocab_size=self.VOCAB_SIZE)
    # newstest2014
    p.file_pattern = os.path.join(self.DATADIR,
                                  'test.mosestok.nmtexample-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wpm')
    p.num_samples = 3003
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98, 200]
    p.bucket_batch_limit = [128] * 8 + [32]
    return p

  def Task(self):
    p = model.MTModelV1.Params()
    p.name = 'wmt14_en_fr'

    p.eval.inference_source_language = 'en'
    p.eval.inference_target_language = 'fr'
    p.eval.samples_per_summary = 7500

    default_params_init = py_utils.WeightInit.Uniform(0.04)
    cc_schedule = quant_utils.LinearClippingCapSchedule.Params().Set(
        start_step=0,
        end_step=20000,
        start_cap=8.0,
        end_cap=1.0)

    p.encoder = encoder.MTEncoderBiRNN.Params()
    enc = p.encoder
    # Config the encoder embedding layer.
    enc.emb.vocab_size = self.VOCAB_SIZE
    enc.emb.embedding_dim = self.MODEL_DIM
    # Config the encoder lstm layers.
    enc.num_lstm_layers = 6
    enc.lstm_cell_size = self.MODEL_DIM
    enc.encoder_out_dim = self.MODEL_DIM
    enc.lstm_tpl = rnn_cell.QuantizedLSTMCell.Params().Set(
        num_output_nodes=self.MODEL_DIM, params_init=default_params_init)
    enc.lstm_tpl.cc_schedule = cc_schedule.Copy()
    # Config the encoder projection layer.
    enc.proj_tpl.weight_norm = True
    # Other encoder configs.
    enc.cc_schedule = cc_schedule.Copy()
    enc.dropout_prob = 0.2

    dec = p.decoder
    dec.source_dim = self.MODEL_DIM
    # Config the decoder embedding layer.
    dec.emb.vocab_size = self.VOCAB_SIZE
    dec.emb.embedding_dim = self.MODEL_DIM
    # Config the decoder lstm layers and the attention model.
    dec.rnn_cell_dim = self.MODEL_DIM
    rnn_cell_tpl = rnn_cell.QuantizedLSTMCell.Params().Set(
        num_output_nodes=self.MODEL_DIM, params_init=default_params_init)
    rnn_cell_tpl.cc_schedule = cc_schedule.Copy()
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.attention.hidden_dim = self.MODEL_DIM
    # Config the softmax layer.
    dec.softmax.num_classes = self.VOCAB_SIZE
    dec.softmax.logits_abs_max = 20.0
    dec.label_smoothing = layers.UniformLabelSmoother.Params().Set(
        num_classes=self.VOCAB_SIZE, uncertainty=0.1)
    # Other misc configs.
    dec.cc_schedule = cc_schedule.Copy()
    dec.dropout_prob = 0.2
    # Beam search decoder configs.
    dec.beam_search.num_hyps_per_beam = 16

    # Update the encoder and decoder to use simple embedding.
    enc.emb = model_helper.ChangeToSimpleEmbedding(enc.emb)
    dec.emb = model_helper.ChangeToSimpleEmbedding(dec.emb)
    dec.softmax = model_helper.ChangeToSimpleSoftmax(dec.softmax)

    # Learning schedule.
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    # Comparing to the corresponding gpu model, the per-replica batch size on
    # TPU is only half of that on GPU. Hence we need to adjust the learning
    # schedule accordingly so that effective learning rate on tpu is the same as
    # that on gpu.
    p.train.learning_rate = 5.0e-5
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(
            warmup=256,
            decay_start=512000 * 2,
            decay_end=3600000 * 2,
            min=1.0,
            num_splits=32))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


# https://mldash.corp.google.com/experiments/4466291781428173528
# Trained using tpu, using topology=4x4
@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized16kSmallJF(WmtEnFrV2QuantizedJF):
  """Params for WMT'14 En->Fr in sync training: 16K vocab, 512 dim."""

  DATADIR = '/cns/is-d/home/fosterg/data/rs=6.3/wmt14c_en_fr/wpm16k'
  VOCAB_SIZE = 16384
  MODEL_DIM = 512

  def Train(self):
    p = super(WmtEnFrV2Quantized16kSmallJF, self).Train()
    p = model_helper.FixateInputShape(p, 64, 100, 100)
    return p


# https://mldash.corp.google.com/experiments/6273263748097382947
# Trained using tpu, using topology=4x4
@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized16kSmallBfloat16JF(WmtEnFrV2Quantized16kSmallJF):
  """Params for WMT'14 En->Fr in sync training with bfloat16."""

  def Train(self):
    p = super(WmtEnFrV2Quantized16kSmallBfloat16JF, self).Train()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrV2Quantized16kSmallBfloat16JF, self).Task()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformer16kJF(WmtEnFrV2Quantized16kSmallJF):
  """Params for WMT'14 En->Fr JF training using the small Transformer model."""

  DATADIR = '/cns/is-d/home/fosterg/data/rs=6.3/wmt14c_en_fr/wpm16k'
  VOCAB_SIZE = 16384
  MODEL_DIM = 512
  HIDDEN_DIM = 2048
  NUM_HEADS = 8

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=self.MODEL_DIM,
        hidden_dim=self.HIDDEN_DIM,
        num_heads=self.NUM_HEADS,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        learning_rate=1.0,
        warmup_steps=4000)

    p.eval.samples_per_summary = 7500

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformer16kJFPolicy(WmtEnFrTransformer16kJF):
  """Version of WmtEnFrTransformer16kJF for testing NmtExample.input_policy."""

  def Train(self):
    p = super(WmtEnFrTransformer16kJFPolicy, self).Train()
    p.file_pattern = [
        os.path.join(self.DATADIR, 'train.mosestok.nmtexample-0000?-of-00036'),
        os.path.join(self.DATADIR, 'train.mosestok.nmtexample-0001?-of-00036'),
        os.path.join(self.DATADIR, 'train.mosestok.nmtexample-0002?-of-00036'),
        os.path.join(self.DATADIR, 'train.mosestok.nmtexample-0003?-of-00036'),
    ]
    p.enable_input_policy = True
    p.policy = input_policy.UniformInputPolicy.Params().Set(num_classes=4)
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformer1024HidDim16kJF(WmtEnFrTransformer16kJF):
  """Same as WmtEnFrTransformer16kJF, but with 1024 HIDDEN_DIM."""

  HIDDEN_DIM = 1024


# Does not work without CUDA_VISIBLE_DEVICES instruction.
# export MODEL=mt.research.wmt14.en_fr.WmtEnFrTransformerTinyJF && \
# CUDA_VISIBLE_DEVICES="" \
# blaze-bin/learning/brain/research/babelfish/trainer/trainer_jf_local \
# --model=$MODEL \
# --norestricted_data_access \
# --worker_tpus=2 --run_locally=tpu --mode=sync --worker_split_size=1 \
# --model_params_override=train.max_steps:5 \
# --enqueue_max_steps=10 \
# --logdir=/usr/local/google/$MODEL
# --alsologtostderr
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerTinyJF(WmtEnFrTransformer16kJF):
  """Params for WMT'14 En->Fr JF local simulation with Transformer model."""

  MODEL_DIM = 4
  HIDDEN_DIM = 4
  NUM_HEADS = 2


@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized32kLargeJF(WmtEnFrV2QuantizedJF):
  """Params for WMT'14 En->Fr in sync training: 32K vocab, 1024 dim."""

  DATADIR = '/cns/is-d/home/fosterg/data/rs=6.3/wmt14c_en_fr/wpm32k'
  VOCAB_SIZE = 32768
  MODEL_DIM = 1024

  def Train(self):
    p = super(WmtEnFrV2Quantized32kLargeJF, self).Train()
    # Batch 128 overflows HBM.
    p = model_helper.FixateInputShape(p, 64, 100, 100)
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized32kLargeBfloat16JF(WmtEnFrV2Quantized32kLargeJF):
  """Params for WMT'14 En->Fr in sync training: 32K vocab, 1024 dim."""

  def Train(self):
    p = super(WmtEnFrV2Quantized32kLargeBfloat16JF, self).Train()
    if py_utils.use_tpu():
      p = model_helper.FixateInputShape(p, 128, 100, 100)
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrV2Quantized32kLargeBfloat16JF, self).Task()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized32kLargeJFTest(WmtEnFrV2Quantized32kLargeJF):
  """Params for WMT'14 En->Fr in sync training for continuous test."""

  DATADIR = '/cns/tp-d/home/syzhang/data/rs=6.3/wmt14c_en_fr/wpm32k/wpm32k'
  VOCAB_SIZE = 32768
  MODEL_DIM = 1024


@model_registry.RegisterSingleTaskModel
class WmtEnFrBnmtTiny(WmtEnFr):
  """For unittest only."""

  def Task(self):
    p = model.MTModelV1.Params()
    p.name = 'wmt14_en_fr'
    p.encoder.emb.vocab_size = 3
    p.encoder.emb.embedding_dim = 4
    p.encoder.emb.max_num_shards = 1
    p.encoder.lstm_cell_size = 3
    p.encoder.num_lstm_layers = 1
    p.decoder.source_dim = 3
    p.decoder.emb.vocab_size = 5
    p.decoder.emb.embedding_dim = 2
    p.decoder.emb.max_num_shards = 1
    p.decoder.rnn_cell_dim = 3
    p.decoder.rnn_layers = 1
    p.decoder.attention.hidden_dim = 3
    p.decoder.softmax.num_classes = 5
    p.decoder.softmax.num_shards = 1
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerTiny(WmtEnFr):
  """For unittest only."""

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=3,
        model_dim=4,
        hidden_dim=5,
        num_heads=1,
        num_layers=1,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.0,
        learning_rate=1.0,
        warmup_steps=4000)
    p.encoder.token_emb.max_num_shards = 1
    p.decoder.token_emb.max_num_shards = 1
    p.decoder.softmax.num_shards = 1
    return p


# https://mldash.corp.google.com/experiments/5437624096248807492#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformer8K(WmtEnFr):
  """For unittest only."""

  DATADIR = 'learning/brain/research/babelfish/ops/testdata'

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_wpm_8k_test',
        vocab_size=8000,
        model_dim=16,
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        learning_rate=0.1,
        warmup_steps=4000,
        inference_source_language='en',
        inference_target_language='fr',
        residual_dropout_prob=0.0,
        input_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        label_smoothing_uncertainty=0.1,
        is_transparent=False)
    p.encoder.token_emb.max_num_shards = 1
    p.decoder.token_emb.max_num_shards = 1
    p.decoder.softmax.num_shards = 1
    p.train.max_steps = 10000
    return p

  def Train(self):
    p = base_config.InitTrainDatasetParams()
    p.is_nmt_example = False
    p.file_pattern = os.path.join(self.DATADIR, 'wmt_enfr_wpm_8k_test.sstable')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wmt_enfr_wpm_8k_test')
    p.num_samples = 3003
    p.bucket_upper_bound = [10, 14, 19, 26, 36, 50, 70, 98]
    p.bucket_batch_limit = [128] * 7 + [64]
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrMultiColumnUnitTest(WmtEnFr):
  """For unittest only."""

  def Task(self):
    p = model.HybridModel.Params()
    p.name = 'wmt14_en_fr_mcolumn'
    p.eval.inference_source_language = 'en'
    p.eval.inference_target_language = 'fr'
    p.eval.samples_per_summary = 7500
    enc1_tpl = WmtEnFrBnmtTiny().Task().encoder
    enc2_tpl = WmtEnFrTransformerTiny().Task().encoder
    p.encoder = encoder.MultiColumnEncoder.Params().Set(
        enable_post_proj=True,
        post_proj_dim=5,
        encoder_dims=[3, 3],
        encoder_tpls=[('enc1', enc1_tpl), ('enc2', enc2_tpl)])
    p.decoder = WmtEnFrTransformerTiny().Task().decoder
    return p


# Sync replica training with 32 towers (8 workers, 4 gpus each).
# https://mldash.corp.google.com/experiments/865695999312541551
#
#                  @50k bleu  40.78
#                  @100k      41.20
#                  @150k      41.59
#                  @200k      41.60
# highest ckpt:    @206k      41.96
#
# steps/sec = ~0.235
# examples/sec = ~900 with rpc2
@model_registry.RegisterSingleTaskModel
class WmtEnFrCascadedEncoderBNMTV25Dec(WmtEnFr):
  """Params for WMT'14 En->Fr using the Big transformer model."""

  def Task(self):
    p = model.HybridModel.Params()
    p.name = 'wmt14_en_fr'
    p.encoder = encoder.CascadedEncoder.Params()

    p.encoder.name = 'wmt14_en_fr_seq2seqenc_bnmtdec'
    p.encoder.pre_encoder_tpl = WmtEnFrSyncMultiHead().Task().encoder
    p.encoder.pre_encoder_tpl.name = 'rnn_pre_enc'
    p.encoder.freeze_pre_encoder = True
    p.encoder.ln_preencoder_output = True

    # Transformer layer hparams.
    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Xavier(1.0)

    p.encoder.num_transformer_layers = 4
    p.encoder.transformer_tpl.tr_atten_tpl.Set(
        num_attention_heads=16,
        params_init=default_params_init,
        vn=disable_vn,
        residual_dropout_prob=0.2,
        atten_dropout_prob=0.2)

    p.encoder.transformer_tpl.tr_atten_tpl.atten_tpl.Set(
        enable_ctx_pre_proj=True,
        enable_ctx_post_proj=True,
        context_dim=1024,
        vn=disable_vn)

    p.encoder.transformer_tpl.tr_fflayer_tpl.Set(
        hidden_dim=8192,
        residual_dropout_prob=0.2,
        relu_dropout_prob=0.2,
        params_init=default_params_init,
        vn=disable_vn)

    # Decoder
    p.decoder = WmtEnFrSyncMultiHead().Task().decoder

    p.train.Set(
        learning_rate=2.0,
        optimizer=optimizer.Adam.ParamsB(),
        clip_gradient_norm_to_value=0.0,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=schedule.TransformerSchedule.Params().Set(
            warmup_steps=12000, worker_replicas=1, model_dim=1024))
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    # Lint and pyformat seem to be disagreeing here.
    # pylint: disable=bad-continuation
    # TODO(ankurbpn): Switch to placer checkpoint path when available.
    p.train.init_from_checkpoint_rules = {
        ('/cns/vz-d/home/miachen/brain/rs=6.3/wmt14_en_fr_bnmtsync_multihead4_'
         'att2s_dropout2_bs20/train/ckpt-00088469'): ([
            'wmt14_en_fr_seq2seqenc_bnmtdec/seq2seq_enc/rnn_pre_enc/(.*)',
            'wmt14_en_fr/enc/%s'
        ], [])
    }
    # pylint: enable=bad-continuation
    return p


# Sync replica training with 16 towers (4 workers, 4 gpus each).
# http://mldash/compare?eidstrs=4469163094101595266,5255742622966437683
#
# Checkpoint size: 2.8G
# Checkpoint size (Adam): 4.2G
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAdafactor(WmtEnFrTransformerBig):
  """Params for WMT'14 En->Fr using Transformer big model with Adafactor."""

  def Task(self):
    p = super(WmtEnFrTransformerBigAdafactor, self).Task()
    p.train.optimizer = bf_optimizer.Adafactor.ParamsA()
    return p


# http://mldash/experiments/2681491516311678785#scalars
# Trained using tpu_topology=16x16
# 39.5 BLEU @ 200k
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerJF(WmtEnFrTransformer):
  """WMT'14 En->Fr (32k wpm) Transformer base model, adj. dropouts and lr."""

  # BATCH_SIZE
  #   56 Used 8.28G of 8.00G hbm
  #
  #   48 Total hbm usage >= 7.83G:
  #       reserved        528.00M
  #       program           6.27G
  #       arguments         1.05G
  BATCH_SIZE = 48
  MAX_LEN = 100

  AUX_DROPOUT_PROB = 0.1
  HIDDEN_DIM = 2048
  LR = 3.0
  MODEL_DIM = 512
  NUM_HEADS = 8
  NUM_LAYERS = 6
  VOCAB_SIZE = 32000
  WARMUP_STEPS = 40000

  def Train(self):
    p = super(WmtEnFrTransformerJF, self).Train()
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_LEN,
                                      self.MAX_LEN)
    return p

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=self.MODEL_DIM,
        hidden_dim=self.HIDDEN_DIM,
        num_heads=self.NUM_HEADS,
        num_layers=self.NUM_LAYERS,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=self.AUX_DROPOUT_PROB,
        atten_dropout_prob=self.AUX_DROPOUT_PROB,
        relu_dropout_prob=self.AUX_DROPOUT_PROB,
        learning_rate=self.LR,
        warmup_steps=self.WARMUP_STEPS)

    p.eval.samples_per_summary = 7500

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    return p


# Transformer Base, DF 4x4.
# https://mldash.corp.google.com/experiments/5946559580207331340#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerJF2(WmtEnFrTransformerJF):
  """WMT'14 En->Fr (32k wpm) Transformer base."""

  BATCH_SIZE = 64
  MAX_LEN = 100


# Pre-train Transformer base encoder with masked LM, DF 4x4.
# https://mldash.corp.google.com/experiments/2643988437080353123#scalars
# Test MLM accuracy: 57.5%
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerEncoderJF(WmtEnFrTransformerJF2):
  """WMT'14 En->Fr (32k wpm) Transformer base model, adj. dropouts and lr."""

  BATCH_SIZE = 256
  MAX_LEN = 100

  def Train(self):
    p = super(WmtEnFrTransformerEncoderJF, self).Train()
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_LEN,
                                      self.MAX_LEN)
    return p

  def Task(self):
    p = encoder_model.EncoderModel.Params()
    p.name = 'encoder_model'
    p.encoder = WmtEnFrTransformerJF2().Task().encoder.Copy()
    p.masked_lm_augmenter.mask_prob = 0.12
    p.masked_lm_augmenter.random_prob = 0.015
    p.masked_lm_augmenter.same_prob = 0.015
    p.model_dim = 512
    p.softmax = model_helper.ChangeToSimpleSoftmax(p.softmax)

    tp = p.train
    tp.clip_gradient_norm_to_value = 5.0
    tp.learning_rate = 2e-4
    tp.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(
            warmup_init=0.,
            warmup=20000,
            decay_start=20000,
            decay_end=600000,
            min=0.1,
            num_splits=1))
    return p


# Transformer Base initialized with pre-trained encoder, DF 4x4.
# https://mldash.corp.google.com/experiments/5845504803701531796#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerPreEncJF(WmtEnFrTransformerJF2):
  """Transformer base with pretrained encoder."""

  _pretrained_encoder_params = (
      '/cns/qo-d/home/miachen/brain/rs=6.3'
      '/wmt14_en_fr_xformer_encoder2_df4x4/ckpt-01185900')

  def Task(self):
    p = super(WmtEnFrTransformerPreEncJF, self).Task()

    # Try to not initialize the adam accumulators.
    p.train.init_from_checkpoint_rules = {
        self._pretrained_encoder_params: ([('wmt14_en_fr_transformer/enc/(.*)',
                                            'encoder_model/encoder/%s')], []),
    }
    return p


# BERT-version transformer base, DF 4x4.
# https://mldash.corp.google.com/experiments/2107209050069278383#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrBERTJF(WmtEnFrTransformerJF2):
  """WMT'14 En->Fr (32k wpm) Transformer base."""

  def Task(self):
    p = super(WmtEnFrBERTJF, self).Task()
    p.encoder.transformer_stack.transformer_tpl.tr_fflayer_tpl.activation = (
        'GELU')
    p.decoder.trans_tpl.tr_fflayer_tpl.activation = 'GELU'
    return p


# BERT base encoder, DF 4x4.
# https://mldash.corp.google.com/experiments/278866536272657635#scalars
# Test MLM accuracy: 55.8%
@model_registry.RegisterSingleTaskModel
class WmtEnFrBERTEncoderJF(WmtEnFrBERTJF):
  """WMT'14 En->Fr (32k wpm) Transformer base model, adj. dropouts and lr."""

  BATCH_SIZE = 256
  MAX_LEN = 100
  WARMUP_STEPS = 10000
  DECAY_END = 300000

  def Task(self):
    p = encoder_model.EncoderModel.Params()
    p.name = 'encoder_model'
    p.encoder = WmtEnFrBERTJF().Task().encoder.Copy()
    p.masked_lm_augmenter.mask_prob = 0.12
    p.masked_lm_augmenter.random_prob = 0.015
    p.masked_lm_augmenter.same_prob = 0.015
    p.model_dim = 512
    p.softmax = model_helper.ChangeToSimpleSoftmax(p.softmax)

    tp = p.train
    tp.optimizer = optimizer.Adam.Params().Set(beta1=0.9, beta2=0.999)
    tp.learning_rate = 1e-4
    tp.lr_schedule = schedule.PiecewiseSchedule.Params().Set(
        boundaries=[self.WARMUP_STEPS],
        schedules=[
            schedule.LinearSchedule.Params().Set(
                start=(0, 0.), limit=(self.WARMUP_STEPS, 1.)),
            schedule.LinearSchedule.Params().Set(
                start=(self.WARMUP_STEPS, 1.), limit=(self.DECAY_END, 0.))
        ])
    tp.clip_gradient_norm_to_value = 1.0
    tp.max_steps = self.DECAY_END
    return p


# Trained using tpu_topology=16x16, split=1
# 44.23 canonical bleu @250k steps
# (http://screenshot/2g6E1t22Rhn; unfortunately mldash was not recorded)
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformer4EnsembleJF(WmtEnFrTransformerJF):
  """Ensemble of 4 WmtEnFrTransformerJF models."""

  NUM_ENSEMBLE = 4
  BATCH_SIZE = 96  # Fits on Dragonfish but not Jellyfish.
  # Actual batch size received by each submodel is 96/4=24.

  def Task(self):
    p_base = super(WmtEnFrTransformer4EnsembleJF, self).Task()
    p = model.EnsembleModel.Params()
    p.name = 'ensemble_model'

    p.input = p_base.input.Copy() if p_base.input else None
    p.train = p_base.train.Copy() if p_base.train else None
    p.eval = p_base.eval.Copy() if p_base.eval else None
    p.online_encoder = None

    if not p_base.encoder.name:
      p_base.encoder.name = 'encoder'
    p.encoder = encoder.EnsembleEncoder.Params()
    p.encoder.name = 'ensemble_encoder'
    p.encoder.encoder_tpls = [
        p_base.encoder.Copy() for _ in range(self.NUM_ENSEMBLE)
    ]

    if not p_base.decoder.name:
      p_base.decoder.name = 'decoder'
    p.decoder = decoder.EnsembleDecoder.Params()
    p.decoder.name = 'ensemble_decoder'
    p.decoder.decoder_tpls = [
        p_base.decoder.Copy() for _ in range(self.NUM_ENSEMBLE)
    ]

    return p


# http://mldash/experiments/2440026616471570208#scalars
# Evolution of WmtEnFrTransformerBigAdafactor16, see
#   http://sheets/1SBDQYBybpaUN6YDpMnsCB58VwVGBsciXd2VqZWcuhiw
#
# --worker_split_size=2 is required.
#
# 16x16 is required until packed inputs are implemented.
#
# step:107500, corpus_bleu: 0.400017
# step:180600, corpus_bleu: 0.408551
# step:182700, corpus_bleu: 0.412494 (peak)
# step:349500, corpus_bleu: 0.408971
#
# Effective batch size is 8k:
# ~8k examples/sec,
# ~1  global_step/sec.
#
# Comparable with baseline with batch size 32 and
# --worker_split_size=1 on 8x8 DF:
#   http://mldash/experiments/2771276387358706962
#
#
# batch size 96 with --worker_split_size=1 on 8x8 DF:
#   http://mldash/experiments/8705534760668996448
#
# canonical_bleu: 0.450363 @192500
# canonical_bleu: 0.4484 @320k
# corpus_bleu: 0.4133 @320k
# test log_pplx: 2.161 @375k
# global_step/sec: 1.00
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAdafactorJF(WmtEnFrTransformerJF):
  """Params for WMT'14 En->Fr using the Big transformer model."""
  BATCH_SIZE = 24
  MAX_LEN = 96  # MAX_LEN = 100 is unnecessary

  AUX_DROPOUT_PROB = 0.1
  HIDDEN_DIM = 8192
  LR = 3
  MODEL_DIM = 1024
  NUM_HEADS = 16
  WARMUP_STEPS = 40000

  LR = 3.0

  def Task(self):
    p = super(WmtEnFrTransformerBigAdafactorJF, self).Task()
    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.train.lr_schedule.warmup_steps = 40000
    p.train.learning_rate = self.LR
    return p


# http://mldash/experiments/145023710179743034
# decoder_test:
#   corpus_bleu: ~39.0 @ 500k 1d10h
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerDeterministicJF(WmtEnFrTransformerJF):

  def Task(self):
    p = super(WmtEnFrTransformerDeterministicJF, self).Task()
    return base_config.SetupDeterministicTransformer(p)


# 8x8,split=1 comparison with 16x16,split=2 baseline:
# http://mldash/compare?eidstrs=6589557059864150850,2440026616471570208
#
# with --worker_split_size=1
# Total hbm usage >= 6.92G:
#     reserved        528.00M
#     program           3.60G
#     arguments         2.80G (100.0% utilization)
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAdafactorDeterministicJF(
    WmtEnFrTransformerBigAdafactorJF):
  """Functional WmtEnFrTransformerBigAdafactorJF with deterministic dropouts."""

  def Task(self):
    p = super(WmtEnFrTransformerBigAdafactorDeterministicJF, self).Task()
    p = base_config.SetupDeterministicTransformer(p)
    p.encoder.transformer_stack.cls = (
        mt_layers.FunctionalTransformerStack.Params().cls)
    p.decoder.cls = (decoder.FunctionalTransformerDecoder.Params().cls)
    return p


# Comparison with baseline trained with BATCH_SIZE = 32, split=1, DF 8x8
# http://mldash/compare?eidstrs=2771276387358706962,5402733192121954579
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAdafactorBfloat16JF(
    WmtEnFrTransformerBigAdafactorJF):
  """Bfloat16 half-precision activations TransformerBig for JF."""
  # Used 8.29G of 8.00G hbm.
  # BATCH_SIZE = 48

  # Total hbm usage >= 7.50G:
  #     reserved        528.00M
  #     program           4.18G
  #     arguments         2.80G (100.0% utilization)
  BATCH_SIZE = 40

  def Train(self):
    p = super(WmtEnFrTransformerBigAdafactorBfloat16JF, self).Train()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrTransformerBigAdafactorBfloat16JF, self).Task()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


# Comparison with baseline without bucketing
# (WmtEnFrTransformerBigAdafactorBfloat16JF) trained on DF 8x8
# http://mldash/compare?eidstrs=3255063725786974123,7582855659570537825
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrBucketingTransformerBigAdafactorBfloat16JF(
    WmtEnFrTransformerBigAdafactorBfloat16JF):
  """Bfloat16 activations bucketing TransformerBig."""

  def Task(self):
    p = super(WmtEnFrBucketingTransformerBigAdafactorBfloat16JF, self).Task()
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=self.MODEL_DIM,
        hidden_dim=self.HIDDEN_DIM,
        num_heads=self.NUM_HEADS,
        num_layers=self.NUM_LAYERS,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=self.AUX_DROPOUT_PROB,
        atten_dropout_prob=self.AUX_DROPOUT_PROB,
        relu_dropout_prob=self.AUX_DROPOUT_PROB,
        learning_rate=self.LR,
        warmup_steps=self.WARMUP_STEPS,
        do_bucketing=True)
    # The current implementation uses 6 buckets with threshold max sequence
    # length of 16, 32, 48, 64, 80, and 96. More experiments are needed for the
    # best configuration.
    p.bucket_lengths = [16, 32, 48, 64, 80, 96]

    p.eval.samples_per_summary = 7500

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.train.lr_schedule.warmup_steps = 40000
    p.train.learning_rate = self.LR

    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16

    return p


# Comparison with time-major baseline
# (WmtEnFrTransformerBigAdafactorBfloat16JF) trained on DF 8x8
# https://mldash.corp.google.com/compare?eidstrs=879539827302312406,1385618467391905837
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBatchMajorBigAdafactorBfloat16JF(
    WmtEnFrTransformerBigAdafactorBfloat16JF):
  """Bfloat16 activations batch-major TransformerBig for JF."""

  def Task(self):
    p = super(WmtEnFrTransformerBatchMajorBigAdafactorBfloat16JF, self).Task()
    p = base_config.SetupTransformerBatchMajorParams(
        name='wmt14_en_fr_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=self.MODEL_DIM,
        hidden_dim=self.HIDDEN_DIM,
        num_heads=self.NUM_HEADS,
        num_layers=self.NUM_LAYERS,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=self.AUX_DROPOUT_PROB,
        atten_dropout_prob=self.AUX_DROPOUT_PROB,
        relu_dropout_prob=self.AUX_DROPOUT_PROB,
        learning_rate=self.LR,
        warmup_steps=self.WARMUP_STEPS)

    p.eval.samples_per_summary = 7500

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.train.lr_schedule.warmup_steps = 40000
    p.train.learning_rate = self.LR

    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16

    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAdafactorBfloat16JF96(
    WmtEnFrTransformerBigAdafactorDeterministicJF):
  """Bfloat16 half-precision activations TransformerBig for JF."""
  BATCH_SIZE = 96
  MAX_LEN = 96
  WARMUP_STEPS = 40000

  def Train(self):
    p = super(WmtEnFrTransformerBigAdafactorBfloat16JF96, self).Train()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrTransformerBigAdafactorBfloat16JF96, self).Task()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigAdafactorJFTest(WmtEnFrTransformerBigAdafactorJF):
  """Model config for trainer_tpu_test."""

  def Train(self):
    p = super(WmtEnFrTransformerBigAdafactorJFTest, self).Train()
    p.is_nmt_example = True
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrTransformerBigAdafactorJFTest, self).Task()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrPackedTransformerJF(WmtEnFrTransformerBig):
  """Transformer training with packed inputs."""

  DATADIR = '/cns/vz-d/home/ankurbpn/data/wmt14_en_fr_old'
  BATCH_SIZE = 8
  MAX_LEN = 192
  VOCAB_SIZE = 32000

  def Train(self):
    params = input_generator.NmtPackedInput.Params()

    params.file_random_seed = 0

    params.file_parallelism = 16
    params.file_buffer_size = 10000000

    params.bucket_upper_bound = [self.MAX_LEN]
    params.bucket_batch_limit = [self.BATCH_SIZE]
    params.packed_len = self.MAX_LEN
    params.file_pattern = os.path.join(
        self.DATADIR, 'train-packed-shard-??-192-?????-of-00020')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                              'wordpiece-source-target.32k')
    params.tokenizer.vocab_size = self.VOCAB_SIZE
    params.num_samples = 5301594
    p = model_helper.FixateInputShape(params, self.BATCH_SIZE, self.MAX_LEN,
                                      self.MAX_LEN)
    return p

  def Dev(self):
    params = input_generator.NmtPackedInput.Params()
    params.file_random_seed = 27182818
    # How many threads to run in parallel.
    params.file_parallelism = 1
    params.file_buffer_size = 1
    params.bucket_upper_bound = [200]
    params.bucket_batch_limit = [self.BATCH_SIZE]
    params.packed_len = self.MAX_LEN
    params.force_shape = False
    params.file_pattern = os.path.join(self.DATADIR,
                                       'dev-packed-256-?????-of-00020')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                              'wordpiece-source-target.32k')
    params.tokenizer.vocab_size = 32000
    params.num_samples = 6002
    return params

  def Test(self):
    params = input_generator.NmtPackedInput.Params()
    params.file_random_seed = 27182818
    # How many threads to run in parallel.
    params.file_parallelism = 1
    params.file_buffer_size = 1
    params.bucket_upper_bound = [200]
    params.bucket_batch_limit = [self.BATCH_SIZE]
    params.packed_len = self.MAX_LEN
    params.force_shape = False
    params.file_pattern = os.path.join(self.DATADIR,
                                       'test-packed-256-?????-of-00020')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                              'wordpiece-source-target.32k')
    params.tokenizer.vocab_size = 32000
    params.num_samples = 3003
    return params

  def Task(self):
    p = super(WmtEnFrPackedTransformerJF, self).Task()

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)
      pp.packed_input = True

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)
    return p


# http://mldash/experiments/311448708023938320
@model_registry.RegisterSingleTaskModel
class WmtEnFrTextPackedTransformerJF(WmtEnFrPackedTransformerJF):
  """Transformer training with packed inputs."""

  DATADIR = '/cns/oz-d/home/nmt/rs=6.3/data/wmt14/en_fr'
  # 16 x 128 Total hbm usage >= 8.17G
  BATCH_SIZE = 8
  MAX_LEN = 192
  VOCAB_SIZE = 32000

  def Train(self):
    params = input_generator.TextPackedInput.Params()

    params.file_random_seed = 123456
    params.file_parallelism = 1
    params.file_buffer_size = 10000000

    params.tokenizer.vocab_size = self.VOCAB_SIZE
    params.bucket_upper_bound = [self.MAX_LEN]
    params.bucket_batch_limit = [self.BATCH_SIZE]

    params.file_pattern = 'text:' + os.path.join(self.DATADIR, 'train.txt')
    params.wpm = os.path.join(self.DATADIR, 'wordpiece-mixed')
    params = model_helper.FixateInputShape(params, self.BATCH_SIZE,
                                           self.MAX_LEN, self.MAX_LEN)

    params.tokenizer = tokenizers.WordPieceModel.Params()
    params.tokenizer.normalization = 'none'
    params.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    assert not params.use_per_host_infeed

    return params

  def Dev(self):
    params = input_generator.NmtPackedInput.Params()
    params.file_random_seed = 27182818
    # How many threads to run in parallel.
    params.file_parallelism = 1
    params.file_buffer_size = 1
    params.tokenizer.vocab_size = self.VOCAB_SIZE
    params.bucket_upper_bound = [200]
    params.bucket_batch_limit = [self.BATCH_SIZE]
    params.packed_len = 200
    params.force_shape = False
    params.file_pattern = os.path.join(
        self.DATADIR,
        'newstest2013-en_fr.wpm32k.packed_nmtexample-?????-of-00001')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    params.num_samples = 3000
    return params

  def Test(self):
    params = input_generator.NmtPackedInput.Params()
    params.file_random_seed = 27182818
    # How many threads to run in parallel.
    params.file_parallelism = 1
    params.file_buffer_size = 1
    params.tokenizer.vocab_size = self.VOCAB_SIZE
    params.bucket_upper_bound = [200]
    params.bucket_batch_limit = [self.BATCH_SIZE]
    params.packed_len = 200
    params.force_shape = False
    params.file_pattern = os.path.join(
        self.DATADIR,
        'newstest2014-en_fr.wpm32k.packed_nmtexample-?????-of-00001')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    params.num_samples = 3003
    return params


# http://mldash/compare?eidstrs=6168714465044162273,6704340600197970915
# Data prep:
#   http://cs/corpora/process-wmt14-de_en-en_de-fr_en-en_fr-spm.sh
#
# lepikhin_WmtEnFrTransformerSpmJF_nov7
#   LR = 3.0, WARMUP_STEPS = 40000 (aka TransformerBig learning rate schedule)
#   38.57 sacrebleu @ 86k
#   38.81 sacrebleu @ 142k
#   39.06 sacrebleu @ 193k
# lepikhin_WmtEnFrTransformerSpmJF2_nov7
#   LR = 1.0, WARMUP_STEPS = 8000
#   38.51 sacrebleu @ 84k
#
# 8x8 throughput:
#   num_packed_tokens:1879067 packed_records:29491 batch:8192
#   steps/sec: 0.35, examples/sec: 2885.05
#
# Using separate 32k SPMs for source and target.
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerSpmJF(base_model_params.SingleTaskModelParams):
  """Transformer (Base) training with dynamically packed inputs. SPM."""

  DATADIR = ('/placer/prod/home/translate-train/research/'
             'wmt14_en_fr_wmt18_en_de/en_fr/spm')

  BATCH_SIZE = 64
  MAX_LEN = 128
  VOCAB_SIZE = 2**15
  LR = 3.0
  WARMUP_STEPS = 40000

  def _SetSPMs(self, p, is_eval):
    del is_eval
    spm_en = os.path.join(self.DATADIR, 'spm.en.32k')
    spm_fr = os.path.join(self.DATADIR, 'spm.fr.32k')
    p.tokenizer_dict = {
        'src':
            tokenizers.SentencePieceModel.Params().Set(
                spm_model=spm_en + '.model', vocab_size=self.VOCAB_SIZE),
        'tgt':
            tokenizers.SentencePieceModel.Params().Set(
                spm_model=spm_fr + '.model', vocab_size=self.VOCAB_SIZE),
    }
    p.spm = '\t'.join([
        spm_en,
        spm_fr,
    ])
    return p

  def Train(self):
    p = input_generator.TextPackedInput.Params()
    p.natural_order_model = True
    p.force_shape = True
    p.append_eos = True
    p.flush_every_n = 0
    p.file_parallelism = 1
    p.file_buffer_size = 10000000  # 10M
    p.bucket_upper_bound = [self.MAX_LEN]
    p.bucket_batch_limit = [self.BATCH_SIZE]
    p.file_pattern = 'text:' + os.path.join(self.DATADIR, 'train-shuffled.txt')
    p = self._SetSPMs(p, False)
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_LEN,
                                      self.MAX_LEN)
    assert not p.use_per_host_infeed

    p.max_len = [98]
    p.max_len_schedule = [-1]

    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Dev(self):
    p = input_generator.TextPackedInput.Params()
    p.natural_order_model = True
    p.num_samples = 3000
    p.append_eos = True
    p.test = True
    p.force_shape = False
    p.file_random_seed = 123456
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p = self._SetSPMs(p, True)
    p.bucket_upper_bound = [300]
    p.bucket_batch_limit = [32]
    p.file_pattern = 'text:' + os.path.join(self.DATADIR, 'newstest2013.txt')
    assert not p.use_per_host_infeed
    return p

  def Test(self):
    p = self.Dev()
    p.num_samples = 3003
    p.file_pattern = 'text:' + os.path.join(self.DATADIR, 'newstest2014.txt')
    assert not p.use_per_host_infeed
    return p

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer',
        vocab_size=self.VOCAB_SIZE,
        model_dim=512,
        hidden_dim=2048,
        num_heads=8,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=self.LR,
        warmup_steps=self.WARMUP_STEPS)
    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(
          pp.token_emb, use_matmul=True)
      pp.packed_input = True
    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)

    p.eval.samples_per_summary = 7500
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


# http://mldash/compare?eidstrs=6284530702071788713,3874420515899967942,1198096503759712755
# 8x8:
#   num_packed_tokens is for _both_ src and tgt (so ~950k src tokens)
#   num_packed_tokens:1902402 packed_records:29059 batch:4096
#
# 40.48 sacrebleu @ 123k 4x4
# 40.71 sacrebleu @ 156k 4x8
# 41.04 sacrebleu @ 155k 8x8
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigSpmDF(WmtEnFrTransformerSpmJF):
  """TransformerBig SPM TPU model."""

  BATCH_SIZE = 32
  MAX_LEN = 256

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer_big',
        vocab_size=self.VOCAB_SIZE,
        model_dim=1024,
        hidden_dim=8192,
        num_heads=16,
        num_layers=6,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=40000)
    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(
          pp.token_emb, use_matmul=True)
      pp.packed_input = True
    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)
    # same as WmtEnFrPackedTransformerBigAdafactorBfloat16DF_New
    # http://mldash/experiments/4887080878177004237
    # (latter using 192 by 40 packing).
    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.eval.samples_per_summary = 7500
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


# sacrebleu @ steps:
# decoder_test
#   40.17      40.26      40.16
#   000139800  000207100  000211100
# http://mldash/compare?eidstrs=6816940598288351896,1198096503759712755
# sacrebleu is worse than WmtEnFrTransformerBigSpmDF, despite
# fraction_of_correct_next_step_preds (both dev and test) being higher
# for SharedSpm run.
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerBigSharedSpmDF(WmtEnFrTransformerBigSpmDF):
  """Transformer (Base) training with dynamically packed inputs. SPM."""

  DATADIR = ('/placer/prod/home/translate-train/research/'
             'wmt14_en_fr_wmt18_en_de/en_fr/spm')

  def _SetSPMs(self, p, is_eval):
    del is_eval
    spm_en = os.path.join(self.DATADIR, 'spm.shared.32k')
    spm_fr = os.path.join(self.DATADIR, 'spm.shared.32k')
    p.tokenizer_dict = {
        'src':
            tokenizers.SentencePieceModel.Params().Set(
                spm_model=spm_en + '.model', vocab_size=self.VOCAB_SIZE),
        'tgt':
            tokenizers.SentencePieceModel.Params().Set(
                spm_model=spm_fr + '.model', vocab_size=self.VOCAB_SIZE),
    }
    p.spm = '\t'.join([
        spm_en,
        spm_fr,
    ])
    return p


# https://mldash.corp.google.com/experiments/5927489247526945759
# num_packed_tokens:~7.7M (src+tgt) packed_records:~115k batch:16384
@model_registry.RegisterSingleTaskModel
class WmtEnFrTransformerSpmBfloat16DF(WmtEnFrTransformerSpmJF):
  """Bfloat16 half-precision activations TransformerBig for DF."""
  BATCH_SIZE = 128
  MAX_LEN = 256

  def Train(self):
    p = super(WmtEnFrTransformerSpmBfloat16DF, self).Train()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrTransformerSpmBfloat16DF, self).Task()
    # same as WmtEnFrPackedTransformerBigAdafactorBfloat16DF_New
    # http://mldash/experiments/4887080878177004237#scalars
    # (latter using 192 by 40 packing).
    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.train.lr_schedule.warmup_steps = 40000
    p.train.learning_rate = 3.0
    p = base_config.SetupDeterministicTransformer(p)
    p.encoder.transformer_stack.cls = (
        mt_layers.FunctionalTransformerStack.Params().cls)
    p.decoder.cls = (decoder.FunctionalTransformerDecoder.Params().cls)
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


# The same as WmtEnFrTransformerSpmJF except using TextPackedInputV2. The
# results are very similar to the baseline with varying degrees of speedup.
#
# Compare with baseline on Dragonfish 8x8:
# http://mldash/compare?eidstrs=4604456529521444342,1292264416866093374
# http://mldash/compare?eidstrs=8085305108736309488,817424815866022612
#
# Jellyfish 8x8: http://mldash/experiments/847955562893939712
# Dragonfish 8x8 but with p.packing_factor set to 3:
# http://mldash/experiments/7308116049121506688
@model_registry.RegisterSingleTaskModel
class WmtEnFrInputV2TransformerSpmDF(WmtEnFrTransformerSpmJF):
  """The same as WmtEnFrTransformerSpmJF but with TextPackedInputV2 on DF."""

  def _SetSPMs(self, p):
    spm_en = os.path.join(self.DATADIR, 'spm.en.32k')
    spm_fr = os.path.join(self.DATADIR, 'spm.fr.32k')
    p.tokenizer_dict = {
        'src':
            tokenizers.SentencePieceModel.Params().Set(
                spm_model=spm_en + '.model', vocab_size=self.VOCAB_SIZE),
        'tgt':
            tokenizers.SentencePieceModel.Params().Set(
                spm_model=spm_fr + '.model', vocab_size=self.VOCAB_SIZE),
    }

  def Train(self):
    p = input_generator.TextPackedInputV2.Params()
    p.flush_every_n = 0
    p.file_parallelism = 1
    p.file_buffer_size = 10000000  # 10M
    p.file_pattern = 'text:' + os.path.join(self.DATADIR, 'train-shuffled.txt')
    p.num_batcher_threads = 128
    p.tpu_infeed_parallelism = 8

    self._SetSPMs(p)
    p.bucket_upper_bound = [98]
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_LEN,
                                      self.MAX_LEN)
    p.packing_factor = 3.5

    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Dev(self):
    p = input_generator.TextPackedInputV2.Params()
    p.num_samples = 3000
    p.file_random_seed = 123456
    p.file_parallelism = 1
    p.file_buffer_size = 1
    self._SetSPMs(p)
    p.file_pattern = 'text:' + os.path.join(self.DATADIR, 'newstest2013.txt')

    p.bucket_upper_bound = [300]
    p.bucket_batch_limit = [32]
    p.source_max_length = 300
    p.target_max_length = 300
    return p

  def Test(self):
    p = self.Dev()
    p.num_samples = 3003
    p.file_pattern = 'text:' + os.path.join(self.DATADIR, 'newstest2014.txt')
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrPackedTransformerBigBfloat16DF(WmtEnFrPackedTransformerJF):
  """Bfloat16 half-precision activations TransformerBig for DF."""
  # Total hbm usage >= 15.49G:
  #     reserved        528.00M
  #     program           10.77G
  #     arguments         4.20G (100.0% utilization)
  # num_splits = 1
  BATCH_SIZE = 40

  def Train(self):
    p = super(WmtEnFrPackedTransformerBigBfloat16DF, self).Train()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p

  def Task(self):
    p = super(WmtEnFrPackedTransformerBigBfloat16DF, self).Task()
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


# http://mldash/experiments/2792970256690367429
# old data
# Adafactor18, TransformerBig(lr:3, wu:40k, md:1024)
# BFloat16
# Packed
# len=192, batch=40
# 8x8 DF, split=1
# ~984k packed source tokens
# decoder_test corpus_bleu: 42.18@400K
@model_registry.RegisterSingleTaskModel
class WmtEnFrPackedTransformerBigAdafactorBfloat16DF(
    WmtEnFrPackedTransformerBigBfloat16DF):
  """Bfloat16 half-precision activations TransformerBig for DF."""
  # Total hbm usage >= 15.11G:
  #     reserved        528.00M
  #     program           11.79G
  #     arguments         2.80G (100.0% utilization)
  BATCH_SIZE = 40

  def Task(self):
    p = super(WmtEnFrPackedTransformerBigAdafactorBfloat16DF, self).Task()

    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.train.lr_schedule.warmup_steps = 40000
    p.train.learning_rate = 3.0
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrV2Quantized32kLargeJFOldZeroAtten(WmtEnFrV2Quantized32kLargeJF):
  """Transformer training with packed inputs."""

  DATADIR = '/placer/prod/home/brain-speech-exp/babelfish/wmt14_en_fr_wpm32k'

  def Train(self):
    p = super(WmtEnFrV2Quantized32kLargeJFOldZeroAtten, self).Train()
    p.file_pattern = os.path.join(
        self.DATADIR, 'train-split-backward-maxlen200-?????-of-00036')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.is_nmt_example = False
    p.num_samples = 36302505
    p = model_helper.FixateInputShape(p, 64, 100, 100)
    return p

  def Dev(self):
    p = super(WmtEnFrV2Quantized32kLargeJFOldZeroAtten, self).Dev()
    p.file_pattern = os.path.join(
        self.DATADIR, 'dev-split-backward-maxlen200-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.is_nmt_example = False
    p.num_samples = 6002
    return p

  def Test(self):
    p = super(WmtEnFrV2Quantized32kLargeJFOldZeroAtten, self).Test()
    p.file_pattern = os.path.join(
        self.DATADIR, 'test-split-backward-maxlen200-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.is_nmt_example = False
    p.num_samples = 3003
    return p

  def Task(self):
    p = super(WmtEnFrV2Quantized32kLargeJFOldZeroAtten, self).Task()
    p.decoder.use_zero_atten_state = True
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrPackedV2Quantized32kLargeJFZeroAtten(WmtEnFrV2Quantized32kLargeJF):
  """Transformer training with packed inputs."""

  DATADIR = '/cns/vz-d/home/ankurbpn/data/wmt14_en_fr_old'
  BATCH_SIZE = 32
  MAX_LEN = 192
  VOCAB_SIZE = 32000

  def Train(self):
    p = input_generator.NmtPackedInput.Params()

    p.file_random_seed = 0

    p.file_parallelism = 16
    p.file_buffer_size = 10000000

    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.bucket_upper_bound = [self.MAX_LEN]
    p.bucket_batch_limit = [self.BATCH_SIZE]
    p.packed_len = self.MAX_LEN
    p.file_pattern = os.path.join(self.DATADIR,
                                  'train-packed-shard-??-192-?????-of-00020')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 5301594
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_LEN,
                                      self.MAX_LEN)
    return p

  def Dev(self):
    p = input_generator.NmtPackedInput.Params()
    p.file_random_seed = 27182818
    # How many threads to run in parallel.
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.bucket_upper_bound = [200]
    p.bucket_batch_limit = [self.BATCH_SIZE]
    p.packed_len = self.MAX_LEN
    p.force_shape = False
    p.file_pattern = os.path.join(self.DATADIR, 'dev-packed-256-?????-of-00020')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 6002
    return p

  def Test(self):
    p = input_generator.NmtPackedInput.Params()
    p.file_random_seed = 27182818
    # How many threads to run in parallel.
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.bucket_upper_bound = [200]
    p.bucket_batch_limit = [self.BATCH_SIZE]
    p.packed_len = self.MAX_LEN
    p.force_shape = False
    p.file_pattern = os.path.join(self.DATADIR,
                                  'test-packed-256-?????-of-00020')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    p.num_samples = 3003
    return p

  def Task(self):
    p = super(WmtEnFrPackedV2Quantized32kLargeJFZeroAtten, self).Task()
    p.encoder.packed_input = True
    p.decoder.packed_input = True
    p.decoder.use_zero_atten_state = True
    return p


# A BNMT model suitable for online inference. All LSTM layers in the encoder
# are uni-directional. The attention mechanism has been replaced with the
# Monotonic Chunkwise Attention (MoChA). The model is trained in natural order.
@model_registry.RegisterSingleTaskModel
class WmtEnFrOnline(WmtEnFr):
  """Params for WMT'14 En->Fr sync training for streaming using JF.
  """

  DATADIR = '/cns/lu-d/home/wmach/brain-speech-exp/babelfish/wmt14_en_fr_wpm32k'
  MODEL_DIM = 512

  def Train(self, params=None):
    p = super(WmtEnFrOnline, self).Train(params=params)
    p.is_nmt_example = True
    p.natural_order_model = True
    p.file_pattern = os.path.join(
        self.DATADIR,
        'train-split-backward-maxlen200-natural-order-?????-of-00036')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    if py_utils.use_tpu():
      p = model_helper.FixateInputShape(p, 128, 100, 100)
    return p

  def Dev(self):
    p = super(WmtEnFrOnline, self).Dev()
    p.is_nmt_example = True
    p.natural_order_model = True
    p.file_pattern = os.path.join(
        self.DATADIR,
        'dev-split-backward-maxlen200-natural-order-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    return p

  def Test(self):
    p = super(WmtEnFrOnline, self).Test()
    p.natural_order_model = True
    p.is_nmt_example = True
    p.file_pattern = os.path.join(
        self.DATADIR,
        'test-split-backward-maxlen200-natural-order-00000-of-00001')
    p.tokenizer.wpm_model = os.path.join(self.DATADIR,
                                         'wordpiece-source-target.32k')
    return p

  def CoreModel(self):
    p = model.MTModelV1.Params()
    p.encoder = encoder.MTEncoderUniRNN.Params()
    p.decoder = decoder.MTOnlineDecoder.Params()
    p.decoder.attention = bf_attention.MonotonicChunkwiseAttention.Params().Set(
        chunk_size=0,
        pre_sigmoid_noise=2.,
        hidden_bias_init=-2,
        residual_on_eos=True)
    return p

  def Task(self):

    p = self.CoreModel()
    p.name = 'wmt14_en_fr'
    p.eval.inference_source_language = 'en'
    p.eval.inference_target_language = 'fr'
    p.eval.samples_per_summary = 7500
    p.train.lr_schedule.start_step = 800000
    p.train.lr_schedule.half_life_steps = 100000
    p.train.lr_schedule.min = 0.1

    default_params_init = py_utils.WeightInit.Uniform(0.04)
    enc = p.encoder
    enc.num_lstm_layers = 6
    enc.lstm_cell_size = self.MODEL_DIM
    enc.lstm_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        output_nonlinearity=False, params_init=default_params_init)
    enc.dropout_prob = 0.2
    enc.emb = model_helper.ChangeToSimpleEmbedding(enc.emb)
    enc.emb.embedding_dim = self.MODEL_DIM

    dec = p.decoder
    dec.source_dim = self.MODEL_DIM
    rnn_cell_tpl = rnn_cell.LayerNormalizedLSTMCell.Params().Set(
        output_nonlinearity=False, params_init=default_params_init)
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.rnn_cell_dim = self.MODEL_DIM
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.dropout_prob = 0.2

    dec.label_smoothing = layers.UniformLabelSmoother.Params()
    dec.label_smoothing.num_classes = 32000
    dec.label_smoothing.uncertainty = 0.1
    dec.attention.source_dim = self.MODEL_DIM
    dec.attention.query_dim = self.MODEL_DIM
    dec.attention.hidden_dim = self.MODEL_DIM
    dec.atten_rnn_cls = rnn_layers.FRNNWithAttention
    dec.feed_attention_context_vec_to_softmax = True
    dec.beam_search.num_hyps_per_beam = 20
    dec.emb = model_helper.ChangeToSimpleEmbedding(dec.emb)
    dec.emb.embedding_dim = self.MODEL_DIM
    dec.softmax = model_helper.ChangeToSimpleSoftmax(dec.softmax)
    dec.use_zero_atten_state = True

    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.learning_rate = 1e-4
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=1200000, decay_end=3600000, min=0.5))
    p.train.grad_norm_to_clip_to_zero = 100000.0

    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransSM3(WmtEnFrTransformerBigAdafactorJF):
  """Params for WMT'14 En->Fr using SM3 as optimizer."""
  LR = 0.25

  # http://mldash/compare?eidstrs=5018134219077023914,8825857994064385840&tag=log_pplx

  def Task(self):
    p = super(WmtEnFrTransSM3, self).Task()
    # Settings that are simple: momentum, and adaptive gradient accumulation.
    p.train.optimizer = bf_optimizer.SM3.Params().Set(momentum=0.9)
    p.train.learning_rate = self.LR
    p.train.lr_schedule = schedule.PolynomialSchedule.Params().Set(
        power=2, start=(0, 0.0), limit=(40000, 1.0))
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrTransSM3ProjLayerNorm(WmtEnFrTransformerBigAdafactorJF):
  """Params for WMT'14 En->Fr with projected layer norm.

  http://mldash.corp.google.com/compare?eidstrs=161362799200611828,3075413584602855466

  """
  LR = 0.25

  def Task(self):
    p = super(WmtEnFrTransSM3ProjLayerNorm, self).Task()
    # Settings that are simple: momentum, and adaptive gradient accumulation.
    p.train.optimizer = bf_optimizer.SM3.Params().Set(momentum=0.9)
    p.train.learning_rate = self.LR
    p.train.lr_schedule = schedule.PolynomialSchedule.Params().Set(
        power=2, start=(0, 0.0), limit=(40000, 1.0))
    p.decoder.trans_tpl.tr_atten_tpl.ln_tpl = (
        bf_layers.ProjectedLayerNorm.Params())
    p.decoder.trans_tpl.tr_fflayer_tpl.ln_tpl = (
        bf_layers.ProjectedLayerNorm.Params())
    p.encoder.transformer_stack.ln_tpl = (bf_layers.ProjectedLayerNorm.Params())
    p.encoder.transformer_stack.transformer_tpl.tr_atten_tpl.ln_tpl = (
        bf_layers.ProjectedLayerNorm.Params())
    p.encoder.transformer_stack.transformer_tpl.tr_fflayer_tpl.ln_tpl = (
        bf_layers.ProjectedLayerNorm.Params())
    return p


# Sync replica training with 8x8 DragonFish.
# https://mldash.corp.google.com/experiments/4887080878177004237#scalars
#
# highest ckpt:    @241k      43.71
#
# examples/sec = ~4200
@model_registry.RegisterSingleTaskModel
class WmtEnFrPackedTransformerBigAdafactorBfloat16DFNew(
    WmtEnFrPackedTransformerBigAdafactorBfloat16DF):
  """Bfloat16 half-precision activations TransformerBig for DF."""
  # Total hbm usage >= 15.07G:
  #     reserved        528.00M
  #     program           11.76G
  #     arguments         2.80G (100.0% utilization)
  DATADIR = '/cns/vz-d/home/ankurbpn/data/wmt14_en_fr_new'
  BATCH_SIZE = 40
  MAX_LEN = 192
  VOCAB_SIZE = 32000

  def Train(self):
    p = super(WmtEnFrPackedTransformerBigAdafactorBfloat16DFNew, self).Train()
    p.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    p.natural_order_model = True
    return p

  def Dev(self):
    params = super(WmtEnFrPackedTransformerBigAdafactorBfloat16DFNew,
                   self).Dev()
    params.num_samples = 3000
    params.file_pattern = os.path.join(self.DATADIR,
                                       'dev-packed-192-?????-of-?????')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    params.natural_order_model = True
    return params

  def Test(self):
    params = super(WmtEnFrPackedTransformerBigAdafactorBfloat16DFNew,
                   self).Test()
    params.num_samples = 3003
    params.file_pattern = os.path.join(self.DATADIR,
                                       'test-packed-192-?????-of-?????')
    params.tokenizer.wpm_model = os.path.join(self.DATADIR, 'wordpiece-mixed')
    params.natural_order_model = True
    return params


# Sync replica training with 8x8 DragonFish.
# https://mldash.corp.google.com/experiments/3556557224628026940
# Total hbm usage >= 14.0G:
#     program           10.67G
#     arguments         2.81G (100.0% utilization)
# highest corpus bleu:  43.7
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrGPipeTransformer6LPackedNew(
    WmtEnFrPackedTransformerBigAdafactorBfloat16DFNew):
  """Bfloat16 activations Transformer6L with packed input new dataset for DF."""
  LAYERS = 6
  SPLITS = 1
  NUM_MICRO_BATCHES = 1
  LEARNING_RATE = 3.0
  WARMUP_STEPS = 40000
  IS_TRANSPARENT = False
  CLIPPING_THRESHOLD = None
  ADD_UNNORMALIZED_INPUT = False
  BATCH_MAJOR = False

  def Task(self):
    p = base_config.SetupGPipeTransformerParams(
        name='wmt14_en_fr_gpipe_transformer_packed_new',
        vocab_size=self.VOCAB_SIZE,
        model_dim=1024,
        hidden_dim=8192,
        num_heads=16,
        num_layers=self.LAYERS,
        splits=self.SPLITS,
        num_micro_batches=self.NUM_MICRO_BATCHES,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=self.LEARNING_RATE,
        warmup_steps=self.WARMUP_STEPS,
        is_transparent=self.IS_TRANSPARENT,
        add_unnormalized_input=self.ADD_UNNORMALIZED_INPUT,
        batch_major=self.BATCH_MAJOR,
        packed_input=True)
    p.eval.samples_per_summary = 7500
    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=self.CLIPPING_THRESHOLD,
        factored=True,
    )
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16
    return p


# http://mldash.corp.google.com/experiments/8685870156841034632
@model_registry.RegisterSingleTaskModel
class WmtEnFrGPipeTransformer6LPackedBatchMajor(
    WmtEnFrGPipeTransformer6LPackedNew):
  BATCH_MAJOR = True


# Sync replica training with 8x8 DragonFish, split=1.
# https://mldash.corp.google.com/experiments/2765041763701747437
#
@model_registry.RegisterSingleTaskModel
class WmtEnFrGPipeTransformer6LTransparentPackedNew(
    WmtEnFrGPipeTransformer6LPackedNew):
  LAYERS = 6
  SPLITS = 1
  NUM_MICRO_BATCHES = 1
  IS_TRANSPARENT = True
  LEARNING_RATE = 2.0
  WARMUP_STEPS = 12000
  CLIPPING_THRESHOLD = 1.


# Sync replica training with 8x8 DragonFish, split=1.
# https://mldash.corp.google.com/experiments/7415220880549657211
# Total hbm usage >= 11.25G:
# program           5.86G
# arguments         4.88G
#
# examples/sec = 11500
@model_registry.RegisterSingleTaskModel
class WmtEnFrGPipeTransformer12LTransparentPackedNew(
    WmtEnFrGPipeTransformer6LTransparentPackedNew):
  """12 Layers GPipeTransformer for DF."""
  LAYERS = 12
  SPLITS = 1
  BATCH_SIZE = 32
  NUM_MICRO_BATCHES = 32


# Sync replica training with 8x8 DragonFish, split=2.
# https://mldash.corp.google.com/experiments/4793271716160025672#scalars
#
# highest ckpt:    @219k      44.06
#
# examples/sec = ~1150
@model_registry.RegisterSingleTaskModel
class WmtEnFrPackedTransparentTransformerDeepAdafactorBfloat16DF(
    WmtEnFrPackedTransformerBigAdafactorBfloat16DFNew):
  """Bfloat16 half-precision activations TransformerDeep for DF."""
  BATCH_SIZE = 32

  def Task(self):
    p = base_config.SetupTransformerParams(
        name='wmt14_en_fr_transformer_big',
        vocab_size=32000,
        model_dim=1024,
        hidden_dim=8192,
        num_heads=16,
        num_layers=12,
        inference_source_language='en',
        inference_target_language='fr',
        input_dropout_prob=0.1,
        residual_dropout_prob=0.1,
        atten_dropout_prob=0.1,
        relu_dropout_prob=0.1,
        learning_rate=3.0,
        warmup_steps=40000,
        is_transparent=True)

    p.encoder.transformer_stack.num_transparent_outputs = 12

    for pp in [p.encoder, p.decoder]:
      pp.token_emb = model_helper.ChangeToSimpleEmbedding(pp.token_emb)
      pp.packed_input = True

    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)
    if py_utils.use_tpu():
      p.fprop_dtype = tf.bfloat16

    p.train.optimizer = bf_optimizer.Adafactor.Params().Set(
        beta1=0.9,
        beta2=0.98,
        multiply_by_parameter_scale=False,
        clipping_threshold=None,
        factored=True,
    )
    p.train.optimizer.clipping_threshold = 1.0
    p.train.learning_rate = 2.0
    p.train.lr_schedule.warmup_steps = 12000
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrOnlineWait5(WmtEnFrOnline):
  """Waits fixed number of steps before decoding.

  Progresses encoder and decoder based on fixed emission rate.
  """
  WAIT_K = 5
  EMISSION_RATE = 1.0
  MODEL_DIM = 512

  def Task(self):
    p = super(WmtEnFrOnlineWait5, self).Task()
    p.decoder.attention = bf_attention.WaitKAttention.Params().Set(
        wait_k=self.WAIT_K,
        emission_rate=self.EMISSION_RATE,
        source_dim=self.MODEL_DIM,
        query_dim=self.MODEL_DIM,
        hidden_dim=self.MODEL_DIM,
        inner_atten_params=attention.MultiHeadedAttention.Params().Set(
            num_attention_heads=4,
            inner_atten_params=attention.AdditiveAttention.Params(),
            use_source_vec_as_attention_value=True,
            enable_ctx_pre_proj=False,
            enable_query_proj=True,
            atten_dropout_prob=0.3))
    # p.decoder.beam_search.num_hyps_per_beam = 1  # Use greedy for streaming
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrOnlineBase(WmtEnFrOnline):
  """Baseline that shares almost all parameters directly with WmtEnFrOnline."""
  MODEL_DIM = 512

  def CoreModel(self):
    p = model.MTModelV1.Params()
    p.encoder = encoder.MTEncoderUniRNN.Params()
    # Mirroring what's being used in wait-k
    p.decoder.attention = attention.MultiHeadedAttention.Params().Set(
        num_attention_heads=4,
        inner_atten_params=attention.AdditiveAttention.Params(),
        use_source_vec_as_attention_value=True,
        enable_ctx_pre_proj=False,
        enable_query_proj=True,
        atten_dropout_prob=0.3)
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrOnlineGreedy(WmtEnFrOnline):
  """Greedy and latency-instrumented version of WmtEnFrOnline."""
  MODEL_DIM = 512

  def CoreModel(self):
    p = model.MTOnlineModel.Params()
    # Encoder and decoder come for free, but leaving them here for clarity
    p.encoder = encoder.MTEncoderUniRNN.Params()
    p.decoder = decoder.MTOnlineDecoder.Params()
    p.decoder.attention = bf_attention.MonotonicChunkwiseAttention.Params().Set(
        chunk_size=0,
        pre_sigmoid_noise=2.,
        hidden_bias_init=-2,
        residual_on_eos=True)
    p.decoder.emission_delay_loss.loss_type = 'average_finite_lagging'
    p.decoder.emission_delay_loss_weight = 0.0
    return p

  def Task(self):
    p = super(WmtEnFrOnlineGreedy, self).Task()
    p.decoder.beam_search.num_hyps_per_beam = 1  # Use greedy for streaming
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrOnlineWait5Greedy(WmtEnFrOnline):
  """Greedy and latency-instrumented version of WmtEnFrOnlineWait5."""

  MODEL_DIM = 512
  WAIT_K = 5
  EMISSION_RATE = 1.0

  def CoreModel(self):
    p = model.MTOnlineModel.Params()
    # Encoder and decoder come for free, but leaving them here for clarity
    p.encoder = encoder.MTEncoderUniRNN.Params()
    p.decoder = decoder.MTOnlineDecoder.Params()
    p.decoder.attention = bf_attention.WaitKAttention.Params().Set(
        wait_k=self.WAIT_K,
        emission_rate=self.EMISSION_RATE,
        inner_atten_params=attention.MultiHeadedAttention.Params().Set(
            num_attention_heads=4,
            inner_atten_params=attention.AdditiveAttention.Params(),
            use_source_vec_as_attention_value=True,
            enable_ctx_pre_proj=False,
            enable_query_proj=True,
            atten_dropout_prob=0.3))
    return p

  def Task(self):
    p = super(WmtEnFrOnlineWait5Greedy, self).Task()
    p.decoder.beam_search.num_hyps_per_beam = 1  # Use greedy for streaming
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrOnlineWait5GreedyAdditive(WmtEnFrOnline):
  """Greedy and latency-instrumented version of WmtEnFrOnlineWait5."""

  MODEL_DIM = 512
  WAIT_K = 5
  EMISSION_RATE = 1.0

  def CoreModel(self):
    p = model.MTOnlineModel.Params()
    # Encoder and decoder come for free, but leaving them here for clarity
    p.encoder = encoder.MTEncoderUniRNN.Params()
    p.decoder = decoder.MTOnlineDecoder.Params()
    p.decoder.attention = bf_attention.WaitKAttention.Params().Set(
        wait_k=self.WAIT_K,
        emission_rate=self.EMISSION_RATE,
        inner_atten_params=attention.AdditiveAttention.Params())
    return p

  def Task(self):
    p = super(WmtEnFrOnlineWait5GreedyAdditive, self).Task()
    p.decoder.beam_search.num_hyps_per_beam = 1  # Use greedy for streaming
    return p


# https://mldash.corp.google.com/experiments/1765188024932348550
@model_registry.RegisterSingleTaskModel
class WmtEnFrOfflineUniEnc(WmtEnFrOnline):
  """Unidirectional baseline."""

  def CoreModel(self):
    p = model.MTOnlineModel.Params()
    p.encoder = encoder.MTEncoderUniRNN.Params()
    p.decoder = decoder.MTOnlineDecoder.Params()
    p.decoder.attention = attention.AdditiveAttention.Params()
    return p


# https://mldash.corp.google.com/experiments/1597541047788759583#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrOfflineBiEnc(WmtEnFrOnline):
  """Bidirectional baseline."""

  def CoreModel(self):
    p = model.MTOnlineModel.Params()
    p.encoder = encoder.MTEncoderBiRNN.Params().Set(
        encoder_out_dim=self.MODEL_DIM)
    p.decoder = decoder.MTOnlineDecoder.Params()
    p.decoder.attention = attention.AdditiveAttention.Params()
    return p


# https://mldash.corp.google.com/experiments/9199483663973653387#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrPrefixProb05(WmtEnFrOfflineBiEnc):
  """Bidirectional baseline trained on prefix pairs."""
  SOURCE_EOS = True
  TARGET_EOS = True
  TARGET_PREFIX_LAG = 0
  PREFIX_PROB = 0.5

  def Train(self):
    p = super(WmtEnFrPrefixProb05,
              self).Train(params=input_generator.NmtPrefixInput.Params())
    p.source_eos = self.SOURCE_EOS
    p.target_eos = self.TARGET_EOS
    p.target_prefix_lag = self.TARGET_PREFIX_LAG
    p.source_prefix_distribution = 'uniform'
    p.prefix_prob = self.PREFIX_PROB
    return p


# https://mldash.corp.google.com/experiments/5589557159393828997#scalars
@model_registry.RegisterSingleTaskModel
class WmtEnFrUniPrefixProb05(WmtEnFrOfflineUniEnc):
  """Unidirectional baseline trained on prefix pairs."""
  SOURCE_EOS = True
  TARGET_EOS = True
  TARGET_PREFIX_LAG = 0
  PREFIX_PROB = 0.5

  def Train(self):
    p = super(WmtEnFrUniPrefixProb05,
              self).Train(params=input_generator.NmtPrefixInput.Params())
    p.source_eos = self.SOURCE_EOS
    p.target_eos = self.TARGET_EOS
    p.target_prefix_lag = self.TARGET_PREFIX_LAG
    p.source_prefix_distribution = 'uniform'
    p.prefix_prob = self.PREFIX_PROB
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrXfTgBiSpmDF(WmtEnFrTransformerBigSpmDF):
  """Trained on bitext where target-original pairs are tagged.

  Target-originality was determined by a classifier trained to distinguish
  monolingual French from monolingual English that was machine-translated into
  French.
  """

  def Train(self):
    p = super(WmtEnFrXfTgBiSpmDF, self).Train()
    p.file_pattern = 'text:/cns/qo-d/home/prkriley/wmt14/train/enfr/train.en_fr.orig_tagged.tsv-?????-of-00140'
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrXfTgBiTgBTSpmDFLF(WmtEnFrTransformerBigSpmDF):
  """Trained on bitext and BT data where target-original pairs are tagged.

  Data filtered by SAFT LangID: English source, French target.

  Target-originality was determined by a classifier trained to distinguish
  monolingual French from monolingual English that was machine-translated into
  French.
  """

  def Train(self):
    p = super(WmtEnFrXfTgBiTgBTSpmDFLF, self).Train()
    p.file_pattern = 'text:/cns/lu-d/home/prkriley/wmt14/train/enfr/train.en_fr.orig_tagged_bitext_and_bt.filtered_by_src_trg_langid.tsv-?????-of-00200'
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrXfTgBiTBTSpmDF(WmtEnFrTransformerBigSpmDF):
  """Trained on bitext where target-original pairs are tagged, and TBT data.

  Target-originality was determined by a classifier.
  TBT = Tagged Back-Translation
  """

  def Train(self):
    p = super(WmtEnFrXfTgBiTBTSpmDF, self).Train()
    p.file_pattern = 'text:/cns/qo-d/home/prkriley/wmt14/train/enfr/train.en_fr.orig_tagged_plus_tbt.shuf.tsv-?????-of-00150'
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrXfTgBiTBTSpmDFLF(WmtEnFrTransformerBigSpmDF):
  """Trained on bitext where target-original pairs are tagged, and TBT data.

  Data filtered by SAFT LangID: English source, French target.

  Target-originality was determined by a classifier.
  TBT = Tagged Back-Translation
  """

  def Train(self):
    p = super(WmtEnFrXfTgBiTBTSpmDFLF, self).Train()
    p.file_pattern = 'text:/cns/qo-d/home/prkriley/wmt14/train/enfr/train.en_fr.orig_tagged_plus_tbt.filtered_by_src_trg_langid.tsv-?????-of-00172'
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrXfTgBiTBTFTSpmDF(WmtEnFrTransformerBigSpmDF):
  """Trained on bitext where target-original pairs are tagged + TBT + FT data.

  Target-originality was determined by a classifier.
  TBT = Tagged Back-Translation
  FT = Forward-Translation
  """

  def Train(self):
    p = super(WmtEnFrXfTgBiTBTFTSpmDF, self).Train()
    p.file_pattern = 'text:/cns/qo-d/home/prkriley/wmt14/train/enfr/train.en_fr.orig_tagged_plus_tbt_plus_ft.shuf.tsv-?????-of-00200'
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrXfTBTFTSpmDF(WmtEnFrTransformerBigSpmDF):
  """Trained on bitext + TBT + FT data.

  TBT = Tagged Back-Translation
  FT = Forward-Translation
  """

  def Train(self):
    p = super(WmtEnFrXfTBTFTSpmDF, self).Train()
    p.file_pattern = 'text:/cns/qo-d/home/prkriley/wmt14/train/enfr/train.en_fr.raw_plus_tbt_plus_ft.shuf.tsv-?????-of-00200'
    return p


# 8x8 DF: http://mldash/2531889030779081085
#
# canonical_bleu: 0.443734 @534200
# test log_pplx: 2.241 @469k
# global_step/sec: 1.28
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybridJF(WmtEnFrTransformerBigAdafactorJF):
  """Transformer encoder + RNN decoder, similar to that used in production."""

  DROPOUT = 0.1
  ATTENTION_DROPOUT = 0.1
  MODEL_DIM = 1024
  HIDDEN_DIM = 8192
  ENCODER_NUM_LAYERS = 6
  DECODER_NUM_LAYERS = 8
  NUM_HEADS = 16

  BATCH_SIZE = 96

  def Task(self):
    p = base_config.HybridParams(
        name='hybrid',
        inference_source_language='en',
        inference_target_language='fr',
        encoder_vocab_size=self.VOCAB_SIZE,
        model_dim=self.MODEL_DIM,
        encoder_hidden_dim=self.HIDDEN_DIM,
        encoder_num_heads=self.NUM_HEADS,
        encoder_num_layers=self.ENCODER_NUM_LAYERS,
        encoder_residual_dropout_prob=self.DROPOUT,
        encoder_input_dropout_prob=self.DROPOUT,
        encoder_atten_dropout_prob=self.ATTENTION_DROPOUT,
        encoder_relu_dropout_prob=self.DROPOUT,
        add_unnormalized_residuals=False,
        decoder_vocab_size=self.VOCAB_SIZE,
        decoder_num_layers=self.DECODER_NUM_LAYERS,
        decoder_atten_hidden_dim=self.MODEL_DIM,
        decoder_dropout_prob=self.DROPOUT,
        decoder_atten_dropout_prob=0.0,
        label_smoothing_uncertainty=0.1,
        num_hyps_per_beam=4,
        learning_rate=3.0,
        warmup_steps=40000)
    # for tpu
    p.encoder.token_emb = model_helper.ChangeToSimpleEmbedding(
        p.encoder.token_emb)
    p.decoder.emb = model_helper.ChangeToSimpleEmbedding(p.decoder.emb)
    p.decoder.softmax = model_helper.ChangeToSimpleSoftmax(p.decoder.softmax)
    return p


# 8x8 DF: http://mldash/1961274792250804033
#
# canonical_bleu: 0.448978 @239500
# test log_pplx: 2.209 @634k
# global_step/sec: 1.25
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybrid2JF(WmtEnFrHybridJF):
  """Similar to WmtEnFrHybridJF, but implemented with MTDecoderV1Extended.

  Differences from WmtEnFrHybridJF:
  1) WmtEnFrHybridJF has use_prev_atten_ctx=True, while this implementation
     has it hardcoded to False.
  2) Turned on GradNormTracker.
  """

  def Task(self):
    p = super(WmtEnFrHybrid2JF, self).Task()
    p.decoder = decoder.MTDecoderV1Extended.Params()
    p.decoder.source_dim = self.MODEL_DIM
    p.decoder.emb.vocab_size = self.VOCAB_SIZE
    p.decoder.emb.embedding_dim = self.MODEL_DIM
    p.decoder.rnn_layers = self.DECODER_NUM_LAYERS
    p.decoder.rnn_cell_dim = self.MODEL_DIM
    rnn_cell_tpl = rnn_cell.LayerNormalizedLSTMCellLean.Params().Set(
        num_output_nodes=self.MODEL_DIM,
        output_nonlinearity=True,
        params_init=py_utils.WeightInit.Uniform(0.04))
    p.decoder.rnn_cell_tpl = rnn_cell_tpl.Copy()
    p.decoder.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    p.decoder.dropout_prob = self.DROPOUT
    p.decoder.attention = attention.MultiHeadedAttention.Params().Set(
        source_dim=self.MODEL_DIM,
        hidden_dim=self.MODEL_DIM,
        query_dim=self.MODEL_DIM,
        context_dim=self.MODEL_DIM,
        ctx_post_proj_dim=self.MODEL_DIM,
        num_attention_heads=self.NUM_HEADS,
        use_source_vec_as_attention_value=True,
        enable_ctx_pre_proj=False,
        enable_ctx_post_proj=False,
        enable_query_proj=True,
        enable_source_proj=True,
        atten_dropout_prob=self.ATTENTION_DROPOUT)
    p.decoder.attention_wiring = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
                                  (0, 6), (0, 7)]
    p.decoder.atten_rnn_cls = rnn_layers.FRNNWithAttention
    p.decoder.softmax.num_classes = self.VOCAB_SIZE
    p.decoder.beam_search.num_hyps_per_beam = 4
    p.decoder.label_smoothing = layers.UniformLabelSmoother.Params().Set(
        num_classes=self.VOCAB_SIZE, uncertainty=0.1)
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    return p


# 8x8 DF: http://mldash/7256613410795838234
#
# canonical_bleu: 0.452309 @201700
# test log_pplx: 2.193 @245k
# global_step/sec: 0.67
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybridAttenJF(WmtEnFrHybrid2JF):
  """Variant of WmtEnFrHybridJF: each decoder layer has its own attention."""

  def Task(self):
    p = super(WmtEnFrHybridAttenJF, self).Task()
    p.decoder.attention_wiring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
                                  (5, 6), (6, 7)]
    return p


# 8x8 DF: http://mldash/6983134178503785450 (dropout = 0.2)
#
# canonical_bleu: 0.458914 @201100
# test log_pplx: 2.203 @420k
# global_step/sec: 0.70
# effective batch size: 12288
#
# With dropout = 0.1 the BLEU starts dropping after 150k steps, although pplx
# continued to improve: http://mldash/3930423403283381720. Comment from @orhanf:
# beam search could be improved?
# canonical_bleu: 0.452774 @152100
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybridAttenAddCtxJF(WmtEnFrHybridAttenJF):
  """WmtEnFrHybridAttenJF with add_context_to_input = True and dropout = 0.2."""

  DROPOUT = 0.2
  ATTENTION_DROPOUT = 0.2

  def Task(self):
    p = super(WmtEnFrHybridAttenAddCtxJF, self).Task()
    p.decoder.add_context_to_input = True
    return p


# 8x8 DF: http://mldash/5823866489786959544
#
# canonical_bleu: 0.454441 @260800
# test log_pplx: 2.194 @272k
# global_step/sec: 0.72
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybridAttenSkip1JF(WmtEnFrHybrid2JF):
  """Variant of WmtEnFrHybrid2JF with improved parallelism.

  Skip 1 layer in attention wiring.
  """

  def Task(self):
    p = super(WmtEnFrHybridAttenSkip1JF, self).Task()
    p.decoder.attention_wiring = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5),
                                  (4, 6), (5, 7)]
    return p


# 8x8 DF: http://mldash/3983377472831869917
#
# canonical_bleu: 0.453675 @223100
# test log_pplx: 2.192 @334k
# global_step/sec: 0.79
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybridAttenSkip2JF(WmtEnFrHybrid2JF):
  """Variant of WmtEnFrHybrid2JF with improved parallelism.

  Skip 2 layers in attention wiring.
  """

  def Task(self):
    p = super(WmtEnFrHybridAttenSkip2JF, self).Task()
    p.decoder.attention_wiring = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5),
                                  (3, 6), (4, 7)]
    return p


# 8x8 DF: http://mldash/108336691810199146
#
# canonical_bleu: 0.450867 @346400
# test log_pplx: 2.196 @397k
# global_step/sec: 0.97
# effective batch size: 12288
@model_registry.RegisterSingleTaskModel
class WmtEnFrHybridAttenEvenJF(WmtEnFrHybrid2JF):
  """Variant of WmtEnFrHybrid2JF optimized for speed.

  Only 0,2,4-th layers generates attention, and attention links skip 1 layer.
  """

  def Task(self):
    p = super(WmtEnFrHybridAttenEvenJF, self).Task()
    p.decoder.attention_wiring = [(0, 1), (0, 2), (0, 3), (2, 4), (2, 5),
                                  (4, 6), (4, 7)]
    return p


################### Experiments for ESN: Harsh ###################
# Baseline. Adapted from WmtEnFrV2, with the following changes:
# - Disabling dropout.
# - Cell size 512.
# - Both encoder and decoder have 2 layers.
# - TPU friendly.
# - Learnable activation: 1. y=g(x) , 2. y=g(x, tanh(x))
# - Input topology: Random ??? <CHECK>
@model_registry.RegisterSingleTaskModel
class WmtEnFrV2ESNBase(WmtEnFr):
  """Params for WMT'14 En->Fr in sync training."""

  DIM = 512
  EMB_DIM = 512
  _NUM_LAYERS_RNN = 6  # number of NN layers

  def Train(self):
    p = super(WmtEnFrV2ESNBase, self).Train()
    p = model_helper.FixateInputShape(p, 64, 100, 100)
    return p

  def Task(self):
    p = super(WmtEnFrV2ESNBase, self).Task()
    emb_fprop_mode = 'matmul' if py_utils.tpu_compat() else 'gather'
    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    default_params_init = py_utils.WeightInit.Uniform(0.04)

    p.encoder = encoder.MTEncoderBiRNN.Params()
    enc = p.encoder
    enc.emb = layers.SimpleEmbeddingLayer.Params().Set(
        vocab_size=32000,
        embedding_dim=self.EMB_DIM,
        params_init=default_params_init,
        vn=disable_vn,
        fprop_mode=emb_fprop_mode)  # TPU compatibility
    enc.num_lstm_layers = self._NUM_LAYERS_RNN
    enc.lstm_cell_size = self.DIM
    enc.encoder_out_dim = 2 * self.DIM
    enc.lstm_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=self.DIM,
        params_init=default_params_init,
        init_distribution='none')
    enc.dropout_prob = 0.0

    dec = p.decoder
    dec.emb = layers.SimpleEmbeddingLayer.Params().Set(
        vocab_size=32000,
        embedding_dim=self.EMB_DIM,
        params_init=default_params_init,
        vn=disable_vn,
        fprop_mode=emb_fprop_mode)  # TPU compatibility
    rnn_cell_tpl = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=self.DIM,
        params_init=default_params_init,
        init_distribution='none')
    dec.rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.atten_rnn_cell_tpl = rnn_cell_tpl.Copy()
    dec.dropout_prob = 0.0
    dec.source_dim = 2 * self.DIM
    dec.attention.hidden_dim = self.DIM
    dec.rnn_cell_dim = self.DIM
    dec.rnn_layers = self._NUM_LAYERS_RNN
    dec.softmax.num_shards = 1  # TPU compatibility
    dec.label_smoothing = layers.UniformLabelSmoother.Params().Set(
        num_classes=32000, uncertainty=0.1)
    dec.beam_search.num_hyps_per_beam = 16

    p.train.max_steps = 300000
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.learning_rate = 1e-4
    # p.train.lr_schedule = (
    #     lr_schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params(
    #     ).Set(warmup=500, decay_start=1200000, decay_end=3600000, min=0.5))
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=512000, decay_end=3600000, min=0.5))
    p.train.grad_norm_to_clip_to_zero = 100000.0
    return p


@model_registry.RegisterSingleTaskModel
class WmtEnFrV2ESNTopology(WmtEnFrV2ESNBase):
  """Params for WMT'14 En->Fr in sync training."""

  DIM = 512
  EMB_DIM = 512
  ENC_PROJ_DIM = 1024
  ENC_HAVE_PROJ = False
  DEC_PROJ_DIM = 512
  DEC_HAVE_PROJ = False
  IS_AVERAGE_ESN = False
  IS_CONCAT_ESN = False

  ORDER = 2  # order of higher order ESN
  _NUM_LAYERS_RNN = 2  # 6
  SPARSITY_MULTIPLIER = 3
  SPARSITY = SPARSITY_MULTIPLIER * 1.0 / DIM  # 0.8
  IS_ESN = 'HOtopology'  # 'topology' # 'spectral'
  WT_TOPOLOGIES = ['lap_chain']  # ['spec_lap_chain']  # ['lap_chain']
  TRAIN_WTS = False  # only for is_esn = topology
  IS_ENC_LSTM = False  # True
  IS_DEC_LSTM = True  # False
  IS_ATTN_LSTM = True

  # SPARSITY = DIM/avgL * 1/DIM
  # avgL = 100

  def Task(self):
    p = super(WmtEnFrV2ESNTopology, self).Task()
    default_params_init = py_utils.WeightInit.Uniform(0.04)
    disable_vn = py_utils.VariationalNoiseParams(1.0, False, False)
    emb_fprop_mode = 'matmul' if py_utils.tpu_compat() else 'gather'

    p.encoder.emb = layers.SimpleEmbeddingLayer.Params().Set(
        vocab_size=32000,
        embedding_dim=self.EMB_DIM,
        params_init=default_params_init,
        vn=disable_vn,
        fprop_mode=emb_fprop_mode,
        use_basis=False,
        trainable=True)

    p.decoder.emb = layers.SimpleEmbeddingLayer.Params().Set(
        vocab_size=32000,
        embedding_dim=self.EMB_DIM,
        params_init=default_params_init,
        vn=disable_vn,
        fprop_mode=emb_fprop_mode,
        use_basis=False,
        trainable=True)

    lstm = rnn_cell.LSTMCellSimple.Params().Set(
        num_output_nodes=self.DIM, params_init=default_params_init)

    esn = self.layers_MT(
        is_esn=self.IS_ESN,
        wt_topologies=self.WT_TOPOLOGIES,
        train_wts=self.TRAIN_WTS,
        order=self.ORDER)
    # 'spectral')#'topology')#'ensemble'

    if self.IS_ENC_LSTM:
      encoder_cell = lstm.Copy().Set(init_distribution='none', trainable=True)
    else:
      encoder_cell = esn.Copy()

    if self.IS_DEC_LSTM:
      decoder_cell = lstm.Copy().Set(init_distribution='none', trainable=True)
    else:
      decoder_cell = esn.Copy()

    if self.IS_ATTN_LSTM:
      attention_cell = lstm.Copy().Set(init_distribution='none', trainable=True)
    else:
      attention_cell = esn.Copy()

    enc = p.encoder
    dec = p.decoder

    if self.ENC_HAVE_PROJ:
      enc.have_projection = self.ENC_HAVE_PROJ
      enc.encoder_out_dim = self.ENC_PROJ_DIM
      dec.source_dim = self.ENC_PROJ_DIM
    else:
      enc.encoder_out_dim = 2 * self.DIM

    if self.DEC_HAVE_PROJ:
      dec.have_projection = self.DEC_HAVE_PROJ
      dec.proj_dim = self.DEC_PROJ_DIM

    if self.IS_AVERAGE_ESN:
      enc.is_average_esn = True
      dec.is_average_esn = True

    if self.IS_CONCAT_ESN:
      enc.is_concat_esn = True
      dec.is_concat_esn = True

    p.encoder.input_scale = [5.0] * self._NUM_LAYERS_RNN
    p.encoder.spectral_radius = [0.9] * self._NUM_LAYERS_RNN
    p.encoder.input_sparsity = [self.SPARSITY] * self._NUM_LAYERS_RNN
    p.encoder.hidden_sparsity = [self.SPARSITY] * self._NUM_LAYERS_RNN

    p.decoder.input_scale = [10.0] * self._NUM_LAYERS_RNN
    p.decoder.spectral_radius = [0.9] * self._NUM_LAYERS_RNN
    p.decoder.input_sparsity = [self.SPARSITY] * self._NUM_LAYERS_RNN
    p.decoder.hidden_sparsity = [self.SPARSITY] * self._NUM_LAYERS_RNN

    # p.encoder.dropout_prob = 0.10
    # p.decoder.dropout_prob = 0.10

    p.encoder.num_lstm_layers = self._NUM_LAYERS_RNN
    p.decoder.rnn_layers = self._NUM_LAYERS_RNN

    p.train.learning_rate = 1e-4
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=512000, decay_end=3600000, min=0.5))

    p.encoder.lstm_tpl = encoder_cell.Copy()
    p.decoder.rnn_cell_tpl = decoder_cell.Copy()
    p.decoder.atten_rnn_cell_tpl = attention_cell.Copy()
    return p

  def Train(self):
    p = super(WmtEnFrV2ESNTopology, self).Train()
    p = model_helper.FixateInputShape(p, 64, 100, 100)
    return p

  def layers_MT(self, is_esn, wt_topologies, train_wts, order):
    # defining common arguments
    esn_init_dist = 'uniform'  # Keep init dist as Uniform for laplacian topo
    esn_output_mode = 'non_linear'
    esn_layer_norm = True  # False
    esn_train_scale = True
    esn_train_radius = True
    if is_esn == 'random':
      esn = rnn_cell.ESNCell.Params().Set(
          output_mode=esn_output_mode,
          leak_weight=0.9,
          init_distribution=esn_init_dist,
          layer_norm=esn_layer_norm,
          train_scale=esn_train_scale,
          train_radius=esn_train_radius,
          trainable=False)
    elif is_esn == 'topology':  # the esn topology cell
      # wt_topologies = 'lap_sw'  # 'scrl' #'lap_sw'
      # 'lap_grid' # 'lap_chain'  # 'rot'  # 'random'
      # wt_topologies = ['scrl', 'scru', 'eye']  # ['random']*10
      # wt_topologies = ['lap_chain', 'lap_chain_comp']
      # ['scrl', 'scrbd', 'rot']
      # wt_topologies = ['lap_chain', 'lap_grid', 'lap_sw']
      esn = rnn_cell.ESNcellTopology.Params().Set(
          output_mode=esn_output_mode,  # 'non_linear',#'actNN',
          leak_weight=0.9,
          init_distribution=esn_init_dist,
          layer_norm=esn_layer_norm,
          train_scale=esn_train_scale,
          train_radius=esn_train_radius,
          hidden_dim_actnn=5,
          nn_layer2=False,  # Ignore actNN
          weight_topology_res=wt_topologies,  # 'rot',
          weight_topology_in=wt_topologies,  # 'rot',
          trainable_res=train_wts,
          trainable_in=train_wts)
    elif is_esn == 'spectral':  # spectral learning
      # wt_topologies = #'spec_lap_sw' # 'spec_lap_grid' # 'spec_lap_chain'
      # wt_topologies = ['spec_lap_chain', 'spec_lap_grid', 'spec_lap_sw']
      esn = rnn_cell.ESNcellSpectral.Params().Set(
          output_mode=esn_output_mode,  # 'non_linear', 'actNN'
          leak_weight=0.9,
          init_distribution=esn_init_dist,
          layer_norm=esn_layer_norm,
          train_scale=esn_train_scale,
          train_radius=esn_train_radius,
          hidden_dim_actnn=5,
          nn_layer2=False,
          weight_topology_res=wt_topologies,
          weight_topology_in=wt_topologies,
          trainable_res=False,
          trainable_in=False)
    elif is_esn == 'HOtopology':  # the Higher order esn topology cell
      # wt_topologies = ['scrl', 'scru', 'eye']  # ['random']*10
      # wt_topologies = ['lap_chain', 'lap_chain_comp']
      # ['scrl', 'scrbd', 'rot']
      # wt_topologies = ['lap_chain', 'lap_grid', 'lap_sw']
      esn = rnn_cell.HigherOrderESNTopologyCell.Params().Set(
          output_mode=esn_output_mode,  # 'non_linear',#'actNN',
          leak_weight=0.9,
          init_distribution=esn_init_dist,
          layer_norm=esn_layer_norm,
          train_scale=esn_train_scale,
          train_radius=esn_train_radius,
          hidden_dim_actnn=5,
          nn_layer2=False,  # Ignore actNN
          weight_topology_res=wt_topologies,  # 'rot',
          weight_topology_in=wt_topologies,  # 'rot',
          trainable_res=train_wts,
          trainable_in=train_wts,
          order=order)
    return esn
