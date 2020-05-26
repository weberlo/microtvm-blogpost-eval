import collections
import json
import logging
import numpy as np

import tvm
import tvm.relay
import tvm.micro
from tvm.micro.device import MemConstraint

from micro_eval import util
from . import CompiledModel, LoweredModule, TunableModel


_LOG = logging.getLogger(__name__)


HEADERS = [
    'cmsis_gcc.h',
    'arm_math.h',
    'arm_nnsupportfunctions.h'
]
INCLUDE_PATHS = [
    f'{util.CMSIS_NN_PATH}/CMSIS/DSP/Include',
    f'{util.CMSIS_NN_PATH}/CMSIS/NN/Include',
    f'{util.CMSIS_ST_PATH}',
]

CIFAR10_CLASSES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']


# Generated random params, keyed by (data_layout, kernel_layouts).
GENERATED_RANDOM_PARAMS = {}


_RANDOM_BOUNDS = {
    'mean_data': (130, 140),
    'conv0_weight': (-30, 30),
    'conv0_bias': (-3, 3),
    'conv1_weight': (-30, 30),
    'conv1_bias': (-3, 3),
    'conv2_weight': (-30, 30),
    'conv2_bias': (-3, 3),
    'dense0_weight': (-30, 30),
    'dense0_bias': (-3, 3),
}


def _gen_random_params(mod, param_shapes):
    _cache_key = (tuple(k, v.shape) for k in sorted(param_shapes.keys()))
    if _cache_key not in GENERATED_RANDOM_PARAMS:
        # generate random params
        params = {}
        for param in mod['main'].params[1:]:
            name = param.name_hint
            low, high = _RANDOM_BOUNDS[name]
            rand_tensor = param_shapes[name].gen_rand_tensor(low, high)
            params[param.name_hint] = rand_tensor.data

        GENERATED_RANDOM_PARAMS[_cache_key] = params

    return GENERATED_RANDOM_PARAMS[_cache_key]


_CMSIS_PARAM_SHAPES = {
    'mean_data': util.LabelledShape(N=1, H=32, W=32, C=3, dtype='uint8'),
    'conv0_weight': util.LabelledShape(O=32, H=5, W=5, I=3, dtype='int8'),
    'conv0_bias': util.LabelledShape(B=32, dtype='int8'),
    'conv1_weight': util.LabelledShape(O=32, H=5, W=5, I=32, dtype='int8'),
    'conv1_bias': util.LabelledShape(B=32, dtype='int8'),
    'conv2_weight': util.LabelledShape(O=64, H=5, W=5, I=32, dtype='int8'),
    'conv2_bias': util.LabelledShape(B=64, dtype='int8'),
    'dense0_weight': util.LabelledShape(O=10, I=1024, dtype='int8'),
    'dense0_bias': util.LabelledShape(B=10, dtype='int8'),
}


def _load_cmsis_params(mod, param_file, param_shapes):
    with open(param_file) as f:
        cmsis_params = json.load(f)

    params = {}
    for formal_param in mod['main'].params[1:]:  # exclude data
        name = formal_param.name_hint
        _LOG.debug('name %r %r %r', name, _CMSIS_PARAM_SHAPES[name].dtype, len(cmsis_params[name]))
        cmsis_tensor = util.LabelledTensor(
            data=np.array(cmsis_params[name],
                          dtype=_CMSIS_PARAM_SHAPES[name].dtype, copy=True).reshape(
                              _CMSIS_PARAM_SHAPES[name].shape),
            shape=_CMSIS_PARAM_SHAPES[name])
        if name == 'conv0_bias':
            cmsis_tensor = _CMSIS_PARAM_SHAPES[name].gen_zero_tensor()

        param_shape = param_shapes[name]
        relay_shape = util.LabelledShape(
            zip(param_shape.layout, [x.value for x in formal_param.checked_type.shape]),
            dtype=param_shape.dtype)

        assert param_shape.dims == relay_shape.dims
        param = cmsis_tensor.resize(param_shape)
        data = param.data
        if name == 'mean_data':
            data = relay_shape.gen_zero_tensor().data
        params[name] = tvm.nd.array(data, tvm.cpu(0))

    return params


class Cifar10Cnn(TunableModel):

    # Kernel layouts, keyed by op implementation name. Valid for micro_dev target only, on x86 defaults
    # to HWIO due to broader support for data layouts.
    ARM_KERNEL_LAYOUTS = {
        'direct': 'HWIO',
        'direct_simd': 'HWOI',
    }

    DATA_LAYOUT = 'NHWC'

    @property
    def _kernel_layouts(self):
        conv_op_impl = self.config.get('conv_op_impl', ['direct_simd'] * 3)
        assert len(conv_op_impl) == 3, f'Expect 3 op impls in conv_op_impl, have {conv_op_impl!r}'

        kernel_layouts = []
        for strat in conv_op_impl:
            if self.ctx_str == 'cpu':
                kernel_layouts.append('HWIO')
            else:
                kernel_layouts.append(self.ARM_KERNEL_LAYOUTS[strat])
        return kernel_layouts

    @property
    def _is_simd(self):
        return any(l == 'HWOI' for l in self._kernel_layouts)

    # HACK: for now, models run under the interpreter need an attribute "interp_lower_config"
    # which is tvm.transform.PassContext() used to evaluate the IRModule
    interp_lower_config = tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"})

    def _lower_cpu(self, compiled_model):
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            return LoweredModule(*tvm.relay.build(compiled_model.ir_mod[compiled_model.entry_point],
                                                  target="llvm", params=compiled_model.params))

    def _lower_micro_dev(self, compiled_model, dev_config):
        with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
            graph, c_mod, params = tvm.relay.build(
                compiled_model.ir_mod[compiled_model.entry_point], target=self.target,
                params=compiled_model.params)

        _LOG.info('Building C module and programming on device...')
        micro_mod = tvm.micro.create_micro_mod(
            c_mod, dev_config, lib_headers=HEADERS, lib_include_paths=INCLUDE_PATHS)

        return LoweredModule(graph, micro_mod, params)

    def lower_model(self, compiled_model, dev_config=None):
        # Normally we would examine target to determine how to lower, but target does not currently
        # adequately describe the runtime environment.
        if self.ctx_str == 'cpu':
            return self._lower_cpu(compiled_model)
        elif self.ctx_str == 'micro_dev':
            assert dev_config is not None, 'dev_config required to lower for micro_dev'
            return self._lower_micro_dev(compiled_model, dev_config)
        else:
            assert False, f"don't know how to lower for context {self.ctx_str}"

    def build_model(self):
        kernel_layouts = self._kernel_layouts

        # TODO change relay/op/tensor/unary.cc _make.clip to accept exprs instead of doubles
        # TODO discrepancies between outputs might be a result of the bias_add op
        # not matching the semantics of the CMSIS bias add.
        data_shape = util.LabelledShape.from_dims_and_layout(
            dict(N=1, C=3, H=32, W=32), self.DATA_LAYOUT, dtype='uint8')
        conv0_weight_shape = util.LabelledShape.from_dims_and_layout(
            dict(H=5, W=5, I=3, O=32), kernel_layouts[0], dtype='int8')

        if self._is_simd:
            # to fit our SIMD intrinsic, we make the 'C' dimension a multiple of 4
            data_shape = util.LabelledShape.from_dims_and_layout(
                dict(N=1, C=4, H=32, W=32), self.DATA_LAYOUT, dtype='uint8')
            conv0_weight_shape = util.LabelledShape.from_dims_and_layout(
                dict(H=5, W=5, O=32, I=4), kernel_layouts[0], dtype='int8')

        _LOG.debug('data_shape %r', data_shape)
        _LOG.debug('conv0_weight_shape %r', conv0_weight_shape)

        param_shapes = collections.OrderedDict([
            ('data', data_shape),
            ('mean_data', data_shape),
            ('conv0_weight', conv0_weight_shape),
            ('conv0_bias', util.LabelledShape(B=32, dtype='int8')),
            ('conv1_weight', util.LabelledShape.from_dims_and_layout(
                dict(O=32, I=32, H=5, W=5), kernel_layouts[1], dtype='int8')),
            ('conv1_bias', util.LabelledShape(B=32, dtype='int8')),
            ('conv2_weight', util.LabelledShape.from_dims_and_layout(
                dict(O=64, I=32, H=5, W=5), kernel_layouts[2], dtype='int8')),
            ('conv2_bias', util.LabelledShape(B=64, dtype='int8')),
            ('dense0_weight', util.LabelledShape(O=10, I=1024, dtype='int8')),
            ('dense0_bias', util.LabelledShape(B=10, dtype='int8')),
        ])

        bias_add_axis = self.DATA_LAYOUT.index('C')
        params = []
        for p, s in param_shapes.items():
            joined_shape = ', '.join(str(x) for x in s.shape)
            params.append(f'        %{p}: Tensor[({joined_shape}), {s.dtype}]')
        param_args = ',\n'.join(params)
        _LOG.debug('params %s', param_args)

        mod = tvm.relay.fromtext(CIFAR10_RELAY_MODEL.format(
            bias_add_axis=bias_add_axis,
            data_layout=self.DATA_LAYOUT,
            kernel_layouts=kernel_layouts,
            param_args=param_args))
        if self.config.get('use_random_params', True):
            params = _gen_random_params(mod, param_shapes)
        else:
            params = _load_cmsis_params(mod, self.config.relpath('parameter_file'), param_shapes)

        return CompiledModel(
            self.target, mod, params, 'main',
            {'data_layout': self.DATA_LAYOUT, 'kernel_layouts': kernel_layouts})

    def extract_tunable_tasks(self, compiled_model):
        with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
            tasks = tvm.autotvm.task.extract_from_program(
                compiled_model.ir_mod[compiled_model.entry_point],
                compiled_model.params,
                self.target)

        print('tasks', tasks)
        assert len(tasks) == 3
        return tasks

    def get_autotvm_measure_option(self, num_runners : int, tracker_host : str, tracker_port : int,
                                   tracker_key : str, dev_config : dict, task_index : int,
                                   task : tvm.autotvm.task.Task):
        builder = tvm.autotvm.LocalBuilder(
            build_func=tvm.micro.cross_compiler(
                dev_config,
                tvm.micro.LibType.OPERATOR,
                lib_headers=HEADERS,
                lib_include_paths=INCLUDE_PATHS),
            n_parallel=num_runners)
        builder.build_kwargs.setdefault('build_option', {})['disable_vectorize'] = True
        runner = tvm.autotvm.RPCRunner(
            tracker_key, tracker_host, tracker_port, n_parallel=num_runners,
            number=1, repeat=1, timeout=0)

        return tvm.autotvm.measure_option(builder=builder, runner=runner)

    def get_config_str(self):
        return '-'.join(self._kernel_layouts)

    WORKSPACE_SIZE_BYTES_BY_TASK_INDEX = [132000, 132000, 10000]

    def dataset_generator_name(self):
        return 'cifar10'

    def section_constraints(self, task_index_and_task=None):
        if task_index_and_task is None:
            return collections.OrderedDict([
                ('text', (23000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (300, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (0x80, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (820, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4496, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (145000, MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ])

        task_index, task = task_index_and_task
        if task.name == 'conv2d_direct.arm_cpu':
            return collections.OrderedDict([
                ('text', (28000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (800, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (self.WORKSPACE_SIZE_BYTES_BY_TASK_INDEX[task_index],
                               MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ])
        elif task.name == 'conv2d_direct_simd.arm_cpu':
            return collections.OrderedDict([
                ('text', (23000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (800, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (self.WORKSPACE_SIZE_BYTES_BY_TASK_INDEX[task_index],
                               MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ])
        else:
            assert False, f"don't know how to generate section constraints for {task.task_name}"

    def adapt_sample_inputs(self, sample):
        data_nt = sample['data']
        if self._is_simd:
            data_nt = data_nt.resize(data_nt.shape.as_template_for(C=4))

        return {'data': data_nt}

    def adapt_model_outputs(self, outputs):
        return {'label': outputs['label'][0]}


CIFAR10_RELAY_MODEL = """
    v0.0.4
    def @main({param_args}) {{
        %0 = cast(cast(%data, "int16") - cast(%mean_data, "int16"), "int8");
        %1 = nn.conv2d(
             %0,
             %conv0_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[0]}",
             out_dtype="int32");
      %2 = nn.bias_add(%1, cast(%conv0_bias, "int32"), axis={bias_add_axis});
      %3 = right_shift(%2, 9);
      %4 = cast(%3, "int8");
      %5 = nn.max_pool2d(%4,
             pool_size=[3, 3],
             strides=[2, 2],
             layout="{data_layout}",
             ceil_mode=True);
      %6 = cast(nn.relu(cast(%5, "int16")), "int8");
      %7 = nn.conv2d(
             %6,
             %conv1_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[1]}",
             out_dtype="int32");
      %8 = nn.bias_add(cast(%7, "int16"), cast(%conv1_bias, "int16"), axis={bias_add_axis});
      %9 = right_shift(cast(%8, "int32"), 9);
      %10 = cast(%9, "int8");
      %11 = cast(nn.relu(cast(%10, "int16")), "int16");
      %12 = nn.avg_pool2d(cast(%11, "int32"),
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %13 = nn.conv2d(cast(%12, "int8"),
              %conv2_weight,
              padding=[2, 2],
              channels=64,
              kernel_size=[5, 5],
              data_layout="{data_layout}",
              kernel_layout="{kernel_layouts[2]}",
              out_dtype="int32");
      %14 = nn.bias_add(cast(%13, "int16"), cast(%conv2_bias, "int16"), axis={bias_add_axis});
      %15 = right_shift(cast(%14, "int32"), 9);
      %16 = cast(%15, "int8");
      %17 = cast(nn.relu(%16), "int16");
      %18 = nn.avg_pool2d(cast(%17, "int32"),
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %19 = nn.batch_flatten(cast(%18, "int16"));
      %20 = nn.dense(cast(%19, "int16"), %dense0_weight, units=10, out_dtype="int16");
      %21 = nn.bias_add(%20, cast(left_shift(cast(%dense0_bias, "int32"), 3), "int16"), axis=-1);
      %22 = right_shift(cast(%21, "int32"), 5);
      cast(%22, "int8")
    }}
"""
