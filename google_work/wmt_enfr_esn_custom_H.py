# Lint as: python2, python3
"""WmtEnFrV2ESN_actNNv1 models for on-device En->Fr dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from google3.learning.brain.research.babelfish import model_registry
from google3.learning.brain.research.babelfish.mt.params.research.wmt14.en_fr import WmtEnFrV2ESNTopology

topology_cell = 'topology'  # 'HOtopology'  # 'topology'


@model_registry.RegisterSingleTaskModel
class V01(WmtEnFrV2ESNTopology):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['scrl']
  TRAIN_WTS = True  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V02(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['scrl']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V03(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['scrbd']
  TRAIN_WTS = True  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V04(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['scrbd']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V05(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['rot']
  TRAIN_WTS = True  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V06(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['rot']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V07(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['random']
  TRAIN_WTS = True  # only for is_esn = topology
  SPARSITY_MULTIPLIER = 1  # only when topo = random


@model_registry.RegisterSingleTaskModel
class V08(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['random']
  TRAIN_WTS = False  # only for is_esn = topology
  SPARSITY_MULTIPLIER = 1  # only when topo = random


@model_registry.RegisterSingleTaskModel
class V09(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['random']
  TRAIN_WTS = True  # only for is_esn = topology
  SPARSITY_MULTIPLIER = 3  # only when topo = random


@model_registry.RegisterSingleTaskModel
class V10(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['random']
  TRAIN_WTS = False  # only for is_esn = topology
  SPARSITY_MULTIPLIER = 3  # only when topo = random


@model_registry.RegisterSingleTaskModel
class V11(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['scrl', 'scru', 'eye']
  TRAIN_WTS = True  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V12(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['scrl', 'scru', 'eye']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V13(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['lap_chain']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V14(V01):
  IS_ESN = 'spectral'
  WT_TOPOLOGIES = ['spec_lap_chain']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V15(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['lap_grid']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V16(V01):
  IS_ESN = 'spectral'
  WT_TOPOLOGIES = ['spec_lap_grid']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V17(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['lap_sw']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V18(V01):
  IS_ESN = 'spectral'
  WT_TOPOLOGIES = ['spec_lap_sw']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V19(V01):
  IS_ESN = topology_cell  # 'topology'
  WT_TOPOLOGIES = ['lap_chain', 'lap_grid', 'lap_sw']
  TRAIN_WTS = False  # only for is_esn = topology


@model_registry.RegisterSingleTaskModel
class V20(V01):
  IS_ESN = 'spectral'
  WT_TOPOLOGIES = ['spec_lap_chain', 'spec_lap_grid', 'spec_lap_sw']
  TRAIN_WTS = False  # only for is_esn = topology
