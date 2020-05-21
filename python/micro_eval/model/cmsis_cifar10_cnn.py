import collections
import logging

import tvm.micro
from tvm.micro.device import MemConstraint

from micro_eval import util
from micro_eval.util import mock_c_mod

from . import CompiledModel, LoweredModule, TunableModel

_LOG = logging.getLogger(__name__)

# CMSIS config
HEADERS = [
    'cmsis_gcc.h',
    'arm_math.h',
    'arm_nnsupportfunctions.h'
]
INCLUDE_PATHS = [
    f'{util.CMSIS_NN_PATH}/CMSIS/Core/Include',
    f'{util.CMSIS_NN_PATH}/CMSIS/DSP/Include',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Include',
    f'{util.CMSIS_ST_PATH}',
]

CIFAR10_IMPLS = {
  'arm': {
    'src': f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn/cmsis_cifar10_cnn.c',
    'extra_srcs': [
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_RGB.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7_opt.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_with_offset.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c',
    ],
  },
  'tflite': {
    'src': f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn/cmsis_cifar10_cnn_tfl.c',
    'extra_srcs': [
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_s8_s16.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_s8.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/NNSupportFunctions/arm_nn_vec_mat_mult_t_s8.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_max_pool_s8.c',
      f'{util.CMSIS_NN_PATH}/CMSIS/NN/Source/PoolingFunctions/arm_avgpool_s8.c',
    ],
  },
}


CIFAR10_INCLUDE_PATH = f'{util.get_repo_root()}/cmsis_src/cmsis_cifar10_cnn'

class CmsisCifar10Cnn(TunableModel):

    @property
    def impl(self):
      return self.config.get('impl', 'arm')

    def get_config_str(self):
      return f'cmsis_cifar10_{self.impl}'

    def lower_model(self, compiled_model, dev_config):
      assert self.ctx_str == 'micro_dev', 'Only runnable on micro_dev'
      _LOG.info('Building C module and programming on device...')
      micro_mod = tvm.micro.create_micro_mod(
        compiled_model.ir_mod,
        dev_config,
        lib_src_paths=CIFAR10_IMPLS[self.impl]['extra_srcs'],
        lib_headers=HEADERS,
        lib_include_paths=INCLUDE_PATHS + [CIFAR10_INCLUDE_PATH])

      _LOG.info('graph %s',compiled_model.ir_mod.graph_str)
      return LoweredModule(compiled_model.ir_mod.graph_str, micro_mod, []) #micro_mod

    def build_model(self):
      mod = mock_c_mod.build([CIFAR10_IMPLS[self.impl]['src']], 'cifar10',
                             [('data', util.LabelledShape(N=1, H=32, W=32, C=3, dtype='uint8'))],
                             util.LabelledShape(N=1, X=10, dtype='int8'))
      return CompiledModel(self.target, mod, [], 'cifar10', {})

    def get_autotvm_measure_option(self):
      raise NotImplementedError("Can't use AutoTVM with CMSIS models")

    def extract_tunable_tasks(self):
      raise NotImplementedError("Can't use AutoTVM with CMSIS models")

    def dataset_generator_name(self):
      return 'cifar10'

    def adapt_model_outputs(self, outputs):
        return {'label': outputs['label'][0]}

    def section_constraints(self):
      return collections.OrderedDict([
        ('text', (45000, MemConstraint.ABSOLUTE_BYTES)),
        ('rodata', (4096, MemConstraint.ABSOLUTE_BYTES)),
        ('data', (100000, MemConstraint.ABSOLUTE_BYTES)),
        ('bss', (1320, MemConstraint.ABSOLUTE_BYTES)),
        ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
        ('heap', (100.0, MemConstraint.WEIGHT)),
        ('workspace', (130000, MemConstraint.ABSOLUTE_BYTES)),
        # NOTE we need a deeper stack: since CMSIS makes deep func calls
        ('stack', (1024, MemConstraint.ABSOLUTE_BYTES)),
      ])
