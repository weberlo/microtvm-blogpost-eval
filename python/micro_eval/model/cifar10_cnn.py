import collections
import json
import logging
import numpy as np

import tvm
from tvm import relay

from micro_eval import util
from . import CompiledModel, TunableModel


_LOG = logging.getLogger(__name__)


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
            params[param.name_hint] = tvm.nd.array(rand_tensor)

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


def _load_cmsis_params(mod, param_shapes):
    with open(f'{util.get_repo_root()}/data/cifar10_cnn_params.json') as f:
        cmsis_params = json.load(f)

    params = {}
    for formal_param in mod['main'].params[1:]:  # exclude data
        name = formal_param.name_hint
        print('name', name, _CMSIS_PARAM_SHAPES[name].dtype, len(cmsis_params[name]))
        cmsis_tensor = util.LabelledTensor(
            data=np.array(cmsis_params[name], dtype=_CMSIS_PARAM_SHAPES[name].dtype, copy=True).reshape(_CMSIS_PARAM_SHAPES[name].shape),
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

    def _kernel_layouts(self, target):
        conv_op_impl = self.config.get('conv_op_impl', ['direct_simd'] * 3)
        assert len(conv_op_impl) == 3, f'Expect 3 op impls in conv_op_impl, have {conv_op_impl!r}'

        kernel_layouts = []
        for strat in conv_op_impl:
            if target == 'x86':
                kernel_layouts.append('HWIO')
            else:
                kernel_layouts.append(self.ARM_KERNEL_LAYOUTS[strat])
        return kernel_layouts

    def build_model(self, target):
        kernel_layouts = self._kernel_layouts(target)

        # TODO change relay/op/tensor/unary.cc _make.clip to accept exprs instead of doubles
        # TODO discrepancies between outputs might be a result of the bias_add op
        # not matching the semantics of the CMSIS bias add.
        data_shape = util.LabelledShape.from_dims_and_layout(
            dict(N=1, C=3, H=32, W=32), data_layout, dtype='uint8')
        conv0_weight_shape = util.LabelledShape.from_dims_and_layout(
            dict(H=5, W=5, I=3, O=32), kernel_layouts[0], dtype='int8')

        if any(l in ('direct_simd', 'partial_im2col') for l in kernel_layouts):
            # to fit our SIMD intrinsic, we make the 'C' dimension a multiple of 4
            data_shape = util.LabelledShape.from_dims_and_layout(
                dict(N=1, C=4, H=32, W=32), data_layout, dtype='uint8')
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

        bias_add_axis = data_layout.index('C')
        params = []
        for p, s in param_shapes.items():
            joined_shape = ', '.join(str(x) for x in s.shape)
            params.append(f'        %{p}: Tensor[({joined_shape}), {s.dtype}]')
        param_args = ',\n'.join(params)
        _LOG.debug('params %s', param_args)

        mod = relay.fromtext(CIFAR10_RELAY_MODEL)
        if use_random_params:
            params = _gen_random_params(mod, data_layout, kernel_layouts)
        else:
            params = _load_cmsis_params(mod, param_shapes)

        return CompiledModel(
            mod, params, 'main', {'data_layout': self.DATA_LAYOUT, 'kernel_layouts': kernel_layouts})

    def extract_tunable_tasks(self, target):

        with tvm.target.build_config(opt_level=3, disable_vectorize=True):
            tasks = autotvm.task.extract_from_program(mod.mod['main'], mod.params, TARGET)
            assert len(tasks) == 3
            return tasks

    def get_autotvm_measure_option(self, num_runners : int, tracker_host : str, tracker_port : int,
                                 tracker_key : str, task : tvm.autotvm.task.Task):
        builder = autotvm.LocalBuilder(build_func=model_inst.get_build_func(), n_parallel=num_servers)
        builder.build_kwargs.setdefault('build_option', {})['disable_vectorize'] = True
        runner = autotvm.RPCRunner(tracker_key, tracker_host, tracker_port, n_parallel=num_servers,
                                   number=1, repeat=1, timeout=TIMEOUT_SEC)

        return autotvm.measure_option(builder=builder, runner=runner)

    def get_config_str(self, target):
        return '-'.join(self._kernel_layouts(target))


CIFAR10_RELAY_MODEL = f"""
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
