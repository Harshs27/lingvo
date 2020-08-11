# Lint as: python2, python3
"""Train WPM LMs on 1 Billion Words benchmark data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from lingvo.core import base_model_params
from lingvo.core import layers
from lingvo.core import optimizer
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.tasks.lm import layers as lm_layers
from lingvo.tasks.lm import model as lm_model
from google3.learning.brain.research.babelfish import burger_layers
from google3.learning.brain.research.babelfish import model_helper
from google3.learning.brain.research.babelfish import model_registry
from google3.learning.brain.research.babelfish.lm import input_generator as lm_inp
from google3.learning.brain.research.babelfish.lm import layers as bf_lm_layers
from google3.learning.brain.research.babelfish.lm import model2


class OneBWds(base_model_params.SingleTaskModelParams):
  """Params for training a WPM LM on One Billion Wds text corpus."""

  # One Billion Words benchmark corpus is available in iq and ok.
  CORPUS_DIR = os.path.join('/cns/ok-d/home/ciprianchelba/',
                            '1-billion-word-language-modeling-benchmark/')
  VOCAB_SIZE = -1  # This base model parameterization is not meant to be used.
  WPM_MODEL = None
  MAX_TOKENS = 1024

  def Train(self):
    p = lm_inp.LmInput.Params()
    p.name = '1bwds_train_set'
    p.tokenizer.wpm_model = self.WPM_MODEL
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.target_max_length = self.MAX_TOKENS
    # The WPM vocabulary sets the following ids for sentence start/end and
    # unknown word.
    p.tokenizer.target_sos_id = 1
    p.tokenizer.target_eos_id = 2
    p.tokenizer.target_unk_id = 0
    p.file_pattern = os.path.join(
        self.CORPUS_DIR, 'training-monolingual.tokenized.shuffled.sstable/',
        'news.en-000??-of-00100')
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 128, 128, 128, 64, 32, 16, 8, 4]
    p.num_batcher_threads = 16
    p.file_parallelism = 10
    p.file_buffer_size = 10000000
    p.tokenizer.normalization = ''  # No preprocessing of input text.
    return p

  def Dev(self):
    p = lm_inp.LmInput.Params()
    p.name = '1bwds_dev_set'
    p.tokenizer.wpm_model = self.WPM_MODEL
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.target_max_length = self.MAX_TOKENS
    # The WPM vocabulary sets the following ids for sentence start/end and
    # unknown word.
    p.tokenizer.target_sos_id = 1
    p.tokenizer.target_eos_id = 2
    p.tokenizer.target_unk_id = 0
    p.file_pattern = os.path.join(
        self.CORPUS_DIR, 'heldout-monolingual.tokenized.shuffled.sstable/',
        'news.en.heldout-00001-of-00050')
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 128, 128, 128, 64, 32, 16, 8, 4]
    p.num_batcher_threads = 1
    p.num_samples = 6206  # Number of sentences to evaluate on.
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p.tokenizer.normalization = ''  # No preprocessing of input text.
    return p

  def Test(self):
    p = lm_inp.LmInput.Params()
    p.name = '1bwds_test_set'
    p.tokenizer.wpm_model = self.WPM_MODEL
    p.tokenizer.vocab_size = self.VOCAB_SIZE
    p.target_max_length = self.MAX_TOKENS
    # The WPM vocabulary sets the following ids for sentence start/end and
    # unknown word.
    p.tokenizer.target_sos_id = 1
    p.tokenizer.target_eos_id = 2
    p.tokenizer.target_unk_id = 0
    p.file_pattern = os.path.join(
        self.CORPUS_DIR, 'heldout-monolingual.tokenized.shuffled.sstable/',
        'news.en.heldout-00000-of-00050')
    p.bucket_upper_bound = [10, 20, 30, 40, 50, 100, 256, 512, 1024]
    p.bucket_batch_limit = [128, 128, 128, 128, 64, 32, 16, 8, 4]
    p.num_batcher_threads = 1
    p.num_samples = 6075  # Number of sentences to evaluate on.
    p.file_parallelism = 1
    p.file_buffer_size = 1
    p.tokenizer.normalization = ''  # No preprocessing of input text.
    return p

  def Task(self):
    p = model2.LanguageModelV2.Params()
    p.name = '1bwds_lm'
    p.eval.samples_per_summary = 10000  # Force eval over entire dev/test sets.
    p.lm = lm_layers.RnnLm.CommonParams(
        vocab_size=self.VOCAB_SIZE, emb_dim=1024, num_layers=2, rnn_dims=2048)

    # If the vocab is very large, computes the softmax chunk-by-chunk.
    # TODO(zhifengc): Fine-tune and generalize this estimation.
    target_max_alloc = 128 * (2**20)  # 128M element matrix.
    chunk_size = max(1, int(target_max_alloc / self.VOCAB_SIZE))
    if chunk_size < 8192:
      p.lm.softmax.chunk_size = chunk_size

    # The same settings as the NMT model.
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    p.train.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=500, decay_start=1200000, decay_end=4800000, min=1.0))
    # TODO(ciprianchelba): Tune the following three params.
    p.train.learning_rate = 1e-4
    p.train.l2_regularizer_weight = 1e-6
    p.train.clip_gradient_norm_to_value = 1.0

    return p

  @staticmethod
  def WpmModel(key):
    return ('/cns/ok-d/home/ciprianchelba/'
            '1-billion-word-language-modeling-benchmark/wpm/model/'
            'one_billion_wds.%s.wpm' % key)


# https://mldash.corp.google.com/experiments/7123167683856186279
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM16k(OneBWds):
  """Train LM on 1Bwds text corpus after applying 16k WPM."""
  VOCAB_SIZE = 16000
  WPM_MODEL = OneBWds.WpmModel('16k')


# https://mldash.corp.google.com/experiments/5943976048786844669
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32k(OneBWds):
  """Train LM on 1Bwds text corpus after applying 32k WPM."""
  VOCAB_SIZE = 32000
  WPM_MODEL = OneBWds.WpmModel('32k')


# https://mldash.corp.google.com/experiments/7894849434395844018
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM64k(OneBWds):
  """Train LM on 1Bwds text corpus after applying 64k WPM."""
  VOCAB_SIZE = 64000
  WPM_MODEL = OneBWds.WpmModel('64k')


# https://mldash.corp.google.com/experiments/3406845182200077323
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM128k(OneBWds):
  """Train LM on 1Bwds text corpus after applying 128k WPM."""
  VOCAB_SIZE = 128000
  WPM_MODEL = OneBWds.WpmModel('128k')


# https://mldash.corp.google.com/experiments/3782799518447221099
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM256k(OneBWds):
  """Train LM on 1Bwds text corpus after applying 256k WPM."""
  VOCAB_SIZE = 256000
  WPM_MODEL = OneBWds.WpmModel('256k')


class OneBWdsJFBase(OneBWds):
  """One billion word LM trained on JellyFish (JF)."""

  BATCH_SIZE = 128
  MAX_SEQLEN = 100
  LAYERS = 2
  EMB_DIMS = 1024
  RNN_DIMS = 1024

  def Train(self):
    p = super(OneBWdsJFBase, self).Train()
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_SEQLEN)
    return p

  def Task(self):
    p = super(OneBWdsJFBase, self).Task()
    p.lm = lm_layers.RnnLm.CommonParams(
        vocab_size=self.VOCAB_SIZE,
        emb_dim=self.EMB_DIMS,
        num_layers=self.LAYERS,
        residual_start=1,  # Tune it?
        rnn_dims=self.RNN_DIMS,
        softmax_max_alloc=128 * (2**20))  # 128M element mtatrix.

    p.lm.emb = model_helper.ChangeToSimpleEmbedding(p.lm.emb)
    p.lm.softmax = model_helper.ChangeToSimpleSoftmax(p.lm.softmax)

    tp = p.train
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    tp.lr_schedule.decay_start = 2000000
    tp.lr_schedule.decay_end = 10000000
    return p


# https://mldash.corp.google.com/experiments/3690201184135139955
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM16kJF2L(OneBWdsJFBase):
  """Train LM on 1Bwds text corpus after applying 16k WPM."""
  VOCAB_SIZE = 16000
  WPM_MODEL = OneBWds.WpmModel('16k')
  LAYERS = 2


# https://mldash.corp.google.com/experiments/6143245372198576159
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM16kJF4L(OneBWdsWPM16kJF2L):
  """Train LM on 1Bwds text corpus after applying 16k WPM."""
  LAYERS = 4


# https://mldash.corp.google.com/experiments/2979878568030552969
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM16kJF6L(OneBWdsWPM16kJF2L):
  """Train LM on 1Bwds text corpus after applying 16k WPM."""
  LAYERS = 6


# https://mldash.corp.google.com/experiments/719028322763529913
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM16kJF8L(OneBWdsWPM16kJF2L):
  """Train LM on 1Bwds text corpus after applying 16k WPM."""
  LAYERS = 8


# https://mldash.corp.google.com/experiments/5096535877981300568
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kJF2L(OneBWdsJFBase):
  """Train LM on 1Bwds text corpus after applying 32k WPM."""
  BATCH_SIZE = 64
  VOCAB_SIZE = 32000
  WPM_MODEL = OneBWds.WpmModel('32k')
  LAYERS = 2


# https://mldash.corp.google.com/experiments/4864587065127878148
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kJF4L(OneBWdsWPM32kJF2L):
  """Train LM on 1Bwds text corpus after applying 32k WPM."""
  LAYERS = 4


# https://mldash.corp.google.com/experiments/1473565548098184597
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kJF6L(OneBWdsWPM32kJF2L):
  """Train LM on 1Bwds text corpus after applying 32k WPM."""
  LAYERS = 6


# https://mldash.corp.google.com/experiments/2015235164889032025
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kJF8L(OneBWdsWPM32kJF2L):
  """Train LM on 1Bwds text corpus after applying 32k WPM."""
  LAYERS = 8


# Add mldash link
@model_registry.RegisterSingleTaskModel
class OneBWdsJFESN(OneBWdsJFBase):
  """Train LM on 1Bwds text corpus with ESN."""
  BATCH_SIZE = 128  # 128, 64
  WPM_MODEL = OneBWds.WpmModel('16k')  # change vocab size below accordingly.
  VOCAB_SIZE = 16000
  MAX_SEQLEN = 100
  EMB_DIMS = 1024
  RNN_DIMS = 2048  # 512  # 2048
  RESIDUAL_START = 100  # dummy large number to skip residual connections
  # avgL = 100
  # SPARSITY = 0.8  # 1/RNN_DIMS  # 0.8
  # SPARSITY = RNN_DIMS/avgL * 1/RNN_DIMS
  LAYERS = 2  # 2, 4, 6
  ORDER = 2  # order of higher order ESN
  SPARSITY_MULTIPLIER = 1
  SPARSITY = SPARSITY_MULTIPLIER * 1.0 / RNN_DIMS  # 0.8
  IS_ESN = 'spectral'  # 'HOtopology'  # 'topology' # 'spectral'
  WT_TOPOLOGIES = ['spec_lap_chain', 'spec_lap_grid']
  # ['lap_chain']  # ['spec_lap_chain']  # ['lap_chain']
  TRAIN_WTS = False  # only for is_esn = topology

  def Train(self):
    p = super(OneBWdsJFESN, self).Train()
    p = model_helper.FixateInputShape(p, self.BATCH_SIZE, self.MAX_SEQLEN)
    return p

  def Task(self):
    p = super(OneBWdsJFESN, self).Task()
    print('sparsity!: ', self.SPARSITY)
    input_scale = [10.0] * self.LAYERS  # 1,5,10, 50
    spectral_radius = [0.9] * self.LAYERS  # 0.1, 0.5, 0.9
    input_sparsity = [self.SPARSITY] * self.LAYERS
    hidden_sparsity = [self.SPARSITY] * self.LAYERS
    p.lm = lm_layers.RnnLm.CommonParams(
        vocab_size=self.VOCAB_SIZE,
        emb_dim=self.EMB_DIMS,
        num_layers=self.LAYERS,
        residual_start=self.RESIDUAL_START,  # Tune it?
        rnn_dims=self.RNN_DIMS,
        softmax_max_alloc=128 * (2**20),
        is_esn=self.IS_ESN,  # 'spectral',  # 'basis', 'basis3D',
        wt_topologies=self.WT_TOPOLOGIES,
        train_wts=self.TRAIN_WTS,
        order=self.ORDER,
        esn_init_dist='uniform',  # 'uniform',  # gaussian
        esn_output_mode='non_linear',  # actNN
        esn_layer_norm=True,  # False,
        esn_input_scale=input_scale,
        esn_radius=spectral_radius,
        esn_train_scale=True,  # False, #True,
        esn_train_radius=True,  # False, #True,
        esn_input_sparsity=input_sparsity,
        esn_hidden_sparsity=hidden_sparsity)

    p.lm.emb = model_helper.ChangeToSimpleEmbedding(p.lm.emb)
    p.lm.softmax = model_helper.ChangeToSimpleSoftmax(p.lm.softmax)

    tp = p.train
    tp.lr_schedule = (
        schedule.LinearRampupExponentialDecayScaledByNumSplitSchedule.Params()
        .Set(warmup=512, decay_start=1200000, decay_end=4800000, min=1.0))
    # Tune the following four params.
    tp.learning_rate = 1e-4
    tp.lr_schedule.decay_start = 512000
    tp.lr_schedule.decay_end = 3600000
    tp.l2_regularizer_weight = 1e-6
    tp.grad_norm_to_clip_to_zero = 100.0

    return p


# https://mldash.corp.google.com/experiments/6638008489108295331
# Word-level log-pplx on eval_test: 3.380 @524.1k
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kTransformer6LGraph(OneBWds):
  """Transformer model, built with RnnLmGraph."""

  VOCAB_SIZE = 32000
  EMBEDDING_DIM = 2048
  WPM_MODEL = OneBWds.WpmModel('32k')
  TOPOLOGY = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]

  def TransformerLMParams(self):
    # Embedding layer.
    eb = burger_layers.EmbeddingBurgerLayer.Params().Copy()
    eb.vocab_size = self.VOCAB_SIZE
    eb.embedding_dim = self.EMBEDDING_DIM
    eb.max_num_shards = 16
    emb_params_init = py_utils.WeightInit.Gaussian(
        1.0 / math.sqrt(self.EMBEDDING_DIM))
    eb.params_init = emb_params_init
    eb.scale_sqrt_depth = True
    eb_b = burger_layers.BurgerLayer.Params().Copy().Set(ingredient=eb)

    # Positional embedding layer.
    pe = burger_layers.PositionalEmbeddingBurgerLayer.Params().Copy()
    pe.max_timescale = 10000
    pe.embedding_dim = self.EMBEDDING_DIM
    pe_b = burger_layers.BurgerLayer.Params().Copy().Set(ingredient=pe)

    # Transformer layer.
    tr = burger_layers.TransformerBurgerLayer.Params().Copy()
    tr.tr_atten_tpl.num_attention_heads = 4
    tr.tr_atten_tpl.atten_dropout_prob = 0.1
    tr.tr_atten_tpl.residual_dropout_prob = 0.1
    tr.tr_atten_tpl.is_masked = True
    tr.tr_fflayer_tpl.hidden_dim = self.EMBEDDING_DIM * 4
    tr.tr_atten_tpl.atten_tpl.enable_ctx_pre_proj = True
    tr.tr_atten_tpl.atten_tpl.enable_ctx_post_proj = True
    tr_b = burger_layers.BurgerLayer.Params().Copy().Set(ingredient=tr)

    # Build the model.
    p = bf_lm_layers.RnnLmGraph.Params()
    p.name = 'transformer_lm'
    p.auto_adjust = True
    p.nodes = [eb_b, pe_b, tr_b, tr_b, tr_b, tr_b, tr_b, tr_b]
    p.topology = self.TOPOLOGY

    p.embedding_dim = self.EMBEDDING_DIM
    p.softmax.num_classes = self.VOCAB_SIZE
    p.softmax.input_dim = self.EMBEDDING_DIM
    p.vocab_size = self.VOCAB_SIZE
    return p

  def Task(self):
    p = model2.LanguageModelV2.Params()
    p.name = '1bwds_wpm_level_lm'
    p.eval.samples_per_summary = 10000
    p.lm = self.TransformerLMParams()

    target_max_alloc = 128 * (2**20)  # 128M element matrix.
    chunk_size = max(1, int(target_max_alloc / self.VOCAB_SIZE))
    if chunk_size < 8192:
      p.lm.softmax.chunk_size = chunk_size

    p.lm.label_smoother = layers.UniformLabelSmoother.Params().Set(
        num_classes=self.VOCAB_SIZE, uncertainty=0.1)

    p.train.Set(
        learning_rate=2.0,
        optimizer=optimizer.Adam.ParamsA(),
        clip_gradient_norm_to_value=0.0,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=schedule.TransformerSchedule.Params().Set(
            warmup_steps=40000, worker_replicas=1, model_dim=1024))
    p.train.grad_norm_tracker = layers.GradNormTracker.Params().Set(
        name='gradient_norm_tracker')
    return p


# https://mldash.corp.google.com/experiments/8720372650304077561
# Word-level log-pplx on eval_test: 3.374 @411.6k
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kTransformer6LGraphDense(OneBWdsWPM32kTransformer6LGraph):
  """Transformer model, built with RnnLmGraph class.

  This model adds dense residual connections between layers.
  """
  TOPOLOGY = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (1, 3),
              (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 5), (2, 6), (2, 7),
              (3, 5), (3, 6), (3, 7), (4, 6), (4, 7), (5, 7)]


# Standard 6L transformer on 8x8
# https://mldash.corp.google.com/experiments/8892520739127593728
# ============================================
# Try the following params and deploy 8 accelerators to train a bigger model
# LAYERS = 94  #4.9B params
# SPLITS = [10, 22, 34, 46, 58, 70, 82, 94]  # On 8 accelerator, 16G mem each.
# MAX_SEQLEN = 1024
# BATCH_SIZE = 32
# NUM_MICRO_BATCHES = 32
# Total HBM: 15.39G + 14.59G * 6 + 14.21G
# Examples/sec on 4x4: 15
# Model size: 4.89 billion parameters
# ============================================
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kGPipeTransformer(OneBWds):
  """LM using gpipe transformer."""
  VOCAB_SIZE = 32000
  EMBEDDING_DIM = 2048
  MAX_TOKENS = 100
  BATCH_SIZE = 8
  LAYERS = 6
  SPLITS = 2
  NUM_MICRO_BATCHES = 8
  WPM_MODEL = OneBWds.WpmModel('32k')

  def Train(self):
    p = super(OneBWdsWPM32kGPipeTransformer, self).Train()
    p.bucket_upper_bound = [self.MAX_TOKENS]
    p.bucket_batch_limit = [self.BATCH_SIZE]
    p.fixed_input_shape = True
    return p

  def Task(self):
    """Language model on PTB dataset using gpipe transformer."""
    p = lm_model.FixedShapeInputLanguageModel.Params()
    p.eval.samples_per_summary = 0
    p.name = '1bwds_wpm_level_lm'
    p.lm = lm_layers.GPipeTransformerLm.CommonParams(
        model_dim=self.EMBEDDING_DIM,
        vocab_size=self.VOCAB_SIZE,
        hidden_dim=self.EMBEDDING_DIM * 4,
        num_layers=self.LAYERS,
        splits=self.SPLITS,
        num_micro_batches=self.NUM_MICRO_BATCHES,
        num_heads=4,
        softmax_max_alloc=128 * (2**20),
        atten_dropout_prob=0.1,
        residual_dropout_prob=0.1)
    p.train.Set(
        learning_rate=0.5,
        optimizer=optimizer.Adam.ParamsA(),
        clip_gradient_norm_to_value=0.0,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=schedule.TransformerSchedule.Params().Set(
            warmup_steps=40000, worker_replicas=1,
            model_dim=self.EMBEDDING_DIM))
    return p


# On a DF 8x8
# https://mldash.corp.google.com/experiments/1592084979808357749
# Total HBM >= 13.07G
# program      10.61G
# arguments     1.95G
# Word-level log-pplx on eval_test: 3.432 @124k
@model_registry.RegisterSingleTaskModel
class OneBWdsWPM32kLightWeightConv(OneBWds):
  """LM using light weight convolutions."""
  VOCAB_SIZE = 32000
  EMBEDDING_DIM = 1024
  MAX_TOKENS = 100
  BATCH_SIZE = 128
  KERNEL_SIZES = [15] * 2 + [31] * 4 + [63] * 11
  DROPOUT_PROB = 0.1
  WPM_MODEL = OneBWds.WpmModel('32k')

  def Train(self):
    p = super(OneBWdsWPM32kLightWeightConv, self).Train()
    p.bucket_upper_bound = [self.MAX_TOKENS]
    p.bucket_batch_limit = [self.BATCH_SIZE]
    p.fixed_input_shape = True
    return p

  def Task(self):
    """Language model on PTB dataset using lightweight conv stack."""
    p = lm_model.FixedShapeInputLanguageModel.Params()
    p.eval.samples_per_summary = 0
    p.name = '1bwds_wpm_pay_less_attention_lm'
    p.lm = bf_lm_layers.LightWeightConvLm.CommonParams(
        vocab_size=self.VOCAB_SIZE,
        kernel_sizes=self.KERNEL_SIZES,
        model_dim=self.EMBEDDING_DIM,
        hidden_dim=self.EMBEDDING_DIM * 4,
        num_heads=16,
        input_dropout_prob=self.DROPOUT_PROB,
        atten_dropout_prob=self.DROPOUT_PROB,
        relu_dropout_prob=self.DROPOUT_PROB,
        residual_dropout_prob=self.DROPOUT_PROB,
        num_softmax_shards=16,
        softmax_max_alloc=128 * (2**20))
    p.train.Set(
        learning_rate=1,
        l2_regularizer_weight=0,
        optimizer=optimizer.Momentum.Params().Set(
            use_nesterov=True, alpha=0.99),
        clip_gradient_norm_to_value=0.1,
        grad_norm_to_clip_to_zero=0.0,
        lr_schedule=schedule.LinearRampupCosineSchedule.Params().Set(
            warmup_steps=4000, initial_value=1.0, total_steps=250000))
    return p
