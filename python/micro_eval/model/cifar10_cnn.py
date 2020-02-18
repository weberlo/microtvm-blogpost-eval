import numpy as np

import tvm
from tvm import relay

from micro_eval.util import NamedType

def gen_cifar10_cnn(data_layout, kernel_layouts, simd_friendly, use_random_params=False):
    # kernel layouts are specified per conv, but if only a single layout is
    # passed, that layout is used for all convs
    if isinstance(kernel_layouts, str):
        kernel_layouts = [kernel_layouts] * 3
    # TODO change relay/op/tensor/unary.cc _make.clip to accept exprs instead of doubles
    # TODO discrepancies between outputs might be a result of the bias_add op
    # not matching the semantics of the CMSIS bias add.
    if simd_friendly:
        # to fit our SIMD intrinsic, we make the 'C' dimension a multiple of 4
        data_shape_dict = dict(N=1, C=4, H=32, W=32)
        conv0_shape_dict = dict(O=32, I=4, H=5, W=5)
    else:
        data_shape_dict = dict(N=1, C=3, H=32, W=32)
        conv0_shape_dict = dict(O=32, I=3, H=5, W=5)
    data_shape = NamedType(data_shape_dict).with_layout(data_layout).shape
    conv0_kernel_shape = NamedType(conv0_shape_dict).with_layout(kernel_layouts[0]).shape
    conv1_kernel_shape = NamedType(dict(O=32, I=32, H=5, W=5)).with_layout(kernel_layouts[1]).shape
    conv2_kernel_shape = NamedType(dict(O=64, I=32, H=5, W=5)).with_layout(kernel_layouts[2]).shape
    bias_add_axis = data_layout.index('C')
    mod = relay.fromtext(f"""
    v0.0.4
    def @main(%data: Tensor[{data_shape}, uint8],
        %mean_data: Tensor[{data_shape}, uint8],
        %conv0_weight: Tensor[{conv0_kernel_shape}, int8],
        %conv0_bias: Tensor[(32), int8],
        %conv1_weight: Tensor[{conv1_kernel_shape}, int8],
        %conv1_bias: Tensor[(32), int8],
        %conv2_weight: Tensor[{conv2_kernel_shape}, int8],
        %conv2_bias: Tensor[(64), int8],
        %dense0_weight: Tensor[(10, 1024), int8],
        %dense0_bias: Tensor[(10), int8]) {{
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
      %6 = nn.relu(%5);
      %7 = nn.conv2d(
             %6,
             %conv1_weight,
             padding=[2, 2],
             channels=32,
             kernel_size=[5, 5],
             data_layout="{data_layout}",
             kernel_layout="{kernel_layouts[1]}",
             out_dtype="int32");
      %8 = nn.bias_add(%7, cast(%conv1_bias, "int32"), axis={bias_add_axis});
      %9 = right_shift(%8, 9);
      %10 = cast(%9, "int8");
      %11 = nn.relu(%10);
      %12 = nn.avg_pool2d(%11,
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %13 = nn.conv2d(%12,
              %conv2_weight,
              padding=[2, 2],
              channels=64,
              kernel_size=[5, 5],
              data_layout="{data_layout}",
              kernel_layout="{kernel_layouts[2]}",
              out_dtype="int32");
      %14 = nn.bias_add(%13, cast(%conv2_bias, "int32"), axis={bias_add_axis});
      %15 = right_shift(%14, 9);
      %16 = cast(%15, "int8");
      %17 = nn.relu(%16);
      %18 = nn.avg_pool2d(%17,
              pool_size=[3, 3],
              strides=[2, 2],
              count_include_pad=True,
              layout="{data_layout}",
              ceil_mode=True);
      %19 = nn.batch_flatten(%18);
      %20 = nn.dense(%19, %dense0_weight, units=10, out_dtype="int32");
      %21 = nn.bias_add(%20, left_shift(cast(%dense0_bias, "int32"), 3), axis=-1);
      %22 = right_shift(%21, 5);
      cast(%22, "int8")
    }}
    """)
    if use_random_params:
        # generate random params
        params = {}
        for param in mod['main'].params[1:]:
            shape = list(map(lambda x: x.value, param.checked_type.shape))
            dtype = param.checked_type.dtype
            if 'bias' in param.name_hint:
                result = tvm.nd.array(np.random.randint(-3, 3, size=shape, dtype=dtype), tvm.cpu(0))
            elif 'weight' in param.name_hint:
                result = tvm.nd.array(np.random.randint(-30, 30, size=shape, dtype=dtype), tvm.cpu(0))
            elif 'mean' in param.name_hint:
                result = tvm.nd.array(np.random.randint(130, 140, size=shape, dtype=dtype), tvm.cpu(0))
            else:
                assert False
            params[param.name_hint] = result
    else:
        with open('cifar10_cnn_params.json', 'r') as f:
            params = json.load(f)
        for formal_param in mod['main'].params[1:]:
            param_shape = list(map(lambda x: x.value, formal_param.checked_type.shape))
            dtype = formal_param.checked_type.dtype
            name = formal_param.name_hint

            orig_np = np.array(params[name]).astype(dtype)

            if name == 'mean_data':
                shape = NamedType(data_layout, param_shape)
                cmsis_data_layout = 'NHWC'
                cmsis_shape = shape.get_shape(cmsis_data_layout)
                cmsis_np = orig_np.reshape(cmsis_shape)
                relay_np = transform_data_layout(cmsis_np, cmsis_data_layout, data_layout)
            elif 'conv' in name and 'weight' in name:
                shape = NamedType(kernel_layout, param_shape)
                cmsis_kernel_layout = 'IHWO'
                cmsis_shape = shape.get_shape(cmsis_kernel_layout)
                cmsis_np = orig_np.reshape(cmsis_shape)
                relay_np = transform_data_layout(cmsis_np, cmsis_kernel_layout, kernel_layout)
            elif 'dense' in name and 'weight' in name:
                dense_layout = 'OI'
                shape = NamedType(dense_layout, param_shape)
                # TODO they might be doing matmul weight reordering (figure 6 in their paper)
                cmsis_dense_layout = 'IO'
                cmsis_shape = shape.get_shape(cmsis_dense_layout)
                cmsis_np = orig_np.reshape(cmsis_shape)
                relay_np = transform_data_layout(cmsis_np, cmsis_dense_layout, dense_layout)
            else:
                assert name in ['conv0_bias', 'conv1_bias', 'conv2_bias', 'dense0_bias']
                relay_np = orig_np
            params[name] = tvm.nd.array(relay_np, tvm.cpu(0))
    return mod, params

