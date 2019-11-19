# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import os

import numpy as np
import topi
import tvm
from tvm.contrib import graph_runtime, util
from tvm import relay
import tvm.micro as micro
from tvm.relay.testing import resnet
from tvm.relay import transform
from tvm.relay import create_executor

def conv2d_spatial_pack_nhwc(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Spatial pack compute for Conv2d NHWC"""
    out_dtype = out_dtype or data.dtype

    N, IH, IW, CI = get_const_tuple(data.shape)
    if len(kernel.shape) == 4:
        pre_packed = False
        KH, KW, _, CO = get_const_tuple(kernel.shape)
    else:  # kernel tensor is pre packed
        pre_packed = True
        CO, KH, KW, _, VC = get_const_tuple(kernel.shape)
        CO = CO * VC

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = \
            get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)


    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    data_pad = nn.pad(data, [0, pad_top, pad_left, 0], [0, pad_down, pad_right, 0])

    # ==================== define configuration space ====================
    n, co, oh, ow = cfg.axis(N), cfg.axis(CO), cfg.axis(OH), cfg.axis(OW)
    ci, kh, kw = cfg.reduce_axis(CI), cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    co, vc = cfg.define_split('tile_co', co, num_outputs=2)
    oh, vh = cfg.define_split('tile_oh', oh, num_outputs=2)
    ow, vw = cfg.define_split('tile_ow', ow, num_outputs=2)

    cfg.define_reorder("reorder_0",
                       [n, oh, ow, co, kh, kw, ci, vh, vw, vc],
                       policy='candidate', candidate=[
                           [n, oh, ow, co, kh, kw, ci, vh, vw, vc],
                           [n, oh, ow, co, kh, kw, ci, vc, vh, vw]])

    cfg.define_annotate("ann_reduce", [kh, kw], policy='try_unroll')
    cfg.define_annotate("ann_spatial", [vh, vw, vc], policy='try_unroll_vec')
    # ====================================================================

    VC = cfg["tile_co"].size[-1]
    VH = cfg["tile_oh"].size[-1]
    VW = cfg["tile_ow"].size[-1]

    kvshape = (CO // VC, KH, KW, CI, VC)
    ovshape = (N, OH // VH, OW // VW, CO // VC, VH, VW, VC)
    oshape = (N, OH, OW, CO)

    if dilation_h != 1 or dilation_w != 1:
        # undilate input data
        dvshape = (N, OH // VH, OW // VW, KH, KW, CI, VH, VW)
        data_vec = tvm.compute(dvshape, lambda n, h, w, kh, kw, ci, vh, vw:
                               data_pad[n][(h*VH+vh)*HSTR+kh*dilation_h]
                               [(w*VW+vw)*WSTR+kw*dilation_w][ci],
                               name='data_vec_undilated')
    else:
        dvshape = (N, OH // VH, OW // VW, KH + (VH-1)*HSTR, KW + (VW-1)*WSTR, CI)
        data_vec = tvm.compute(dvshape, lambda n, h, w, vh, vw, ci:
                               data_pad[n][h*VH*HSTR+vh][w*VW*WSTR+vw][ci],
                               name='data_vec')
    if pre_packed:
        kernel_vec = kernel
    else:
        kernel_vec = tvm.compute(kvshape, lambda co, kh, kw, ci, vc:
                                 kernel[kh][kw][ci][co*VC+vc],
                                 name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    if dilation_h != 1 or dilation_w != 1:
        conv = tvm.compute(ovshape, lambda n, h, w, co, vh, vw, vc: \
            tvm.sum(data_vec[n, h, w, kh, kw, vh, vw, ci].astype(out_dtype) *
                    kernel_vec[co, kh, kw, ci, vc].astype(out_dtype),
                    axis=[ci, kh, kw]), name='conv')
    else:
        conv = tvm.compute(ovshape, lambda n, h, w, co, vh, vw, vc: \
            tvm.sum(data_vec[n, h, w, vh*HSTR+kh, vw*WSTR+kw, ci].astype(out_dtype) *
                    kernel_vec[co, kh, kw, ci, vc].astype(out_dtype),
                    axis=[ci, kh, kw]), name='conv')

    output = tvm.compute(oshape, lambda n, h, w, co:
                         conv[n][h//VH][w//VW][co//VC][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='spatial_conv_output_NHWC')
    return output


def schedule_conv2d_spatial_pack_nhwc(cfg, s, conv, output, last):
    """Spatial Pack schedule for Conv2d NHWC"""
    data_vec = conv.op.input_tensors[0]
    kernel_vec = conv.op.input_tensors[1]
    n, oh, ow, co, vh, vw, vc = s[conv].op.axis
    ci, kh, kw = s[conv].op.reduce_axis

    # schedule conv
    cfg["reorder_0"].apply(s, conv, [n, oh, ow, co, kh, kw, vh, vw, ci, vc])
    cfg["ann_reduce"].apply(s, conv, [kh, kw],
                            axis_lens=[get_const_int(kh.dom.extent),
                                       get_const_int(kw.dom.extent)],
                            max_unroll=16,
                            cfg=cfg)
    cfg["ann_spatial"].apply(s, conv, [vh, vw, vc],
                             axis_lens=[cfg['tile_oh'].size[-1],
                                        cfg['tile_ow'].size[-1],
                                        cfg['tile_co'].size[-1]],
                             max_unroll=16,
                             cfg=cfg)

    # schedule fusion
    n, h, w, co = s[last].op.axis
    co, vc = cfg['tile_co'].apply(s, last, co)
    oh, vh = cfg['tile_oh'].apply(s, last, h)
    ow, vw = cfg['tile_ow'].apply(s, last, w)
    s[last].reorder(n, oh, ow, co, vh, vw, vc)
    if last != output:
        s[output].compute_inline()
        cfg["ann_spatial"].apply(s, last, [vh, vw, vc],
                                 axis_lens=[cfg['tile_oh'].size[-1],
                                            cfg['tile_ow'].size[-1],
                                            cfg['tile_co'].size[-1]],
                                 max_unroll=16,
                                 cfg=cfg)
    s[conv].compute_at(s[last], ow)

    # mark parallel
    s[last].parallel(ow)

    if data_vec.op.name == 'data_vec_undilated':
        _, h, _, _, _, _, _, _ = s[data_vec].op.axis
    else:
        _, h, _, _, _, _ = s[data_vec].op.axis

    s[data_vec].parallel(h)

    return s


def reset_gdbinit():
    with open('/home/lweber/gdb-conf/.gdbinit', 'w') as f:
        gdbinit_contents = (
"""
layout asm
target remote localhost:3333

print "utvm_task.num_args"
print utvm_task.num_args

""")
        for i in range(1):
            gdbinit_contents += (
f"""
print "[[TENSOR {i}]]"

print "utvm_task.arg_values[{i}]"
print utvm_task.arg_values[{i}]

print "*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)"
print (*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle))

print "data addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).data)

print "ctx addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).ctx)

print "ndim addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).ndim)

print "dtype addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).dtype)

print "shape addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).shape)

print "strides addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).strides)

print "byte_offset addr"
print &((*((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)).byte_offset)

print "((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)->data"
print ((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)->data

print "((int8_t*) ((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)->data)[0]"
print ((int8_t*) ((TVMArray*) (utvm_task.arg_values[{i}]).v_handle)->data)[0]

""")

        gdbinit_contents += (
f"""
set $pc = UTVMInit
break UTVMDone
""")
        f.write(gdbinit_contents)


class DummyCMod:
    def __init__(self):
        pass

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, '/home/lweber/tvm-micro/tests/python/unittest/cmsis_conv.c')


# Use the host emulated micro device.
#DEV_CONFIG_A = micro.device.host.default_config()
DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
#DEV_CONFIG_B = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6667)

# default
#N, H, W, CO, CI, KH, KW = 1, 32, 32, 4, 4, 3, 3
#STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
#LAYOUT = 'NCHW'
#OUT_DTYPE = 'float32'

# cmsis cifar10 dims
#N, H, W, CO, CI, KH, KW = 1, 16, 16, 32, 32, 5, 5
N, H, W, CO, CI, KH, KW = 1, 24, 24, 32, 32, 5, 5
STRIDES, PADDING, DILATION = (1, 1), (2, 2), (1, 1)
LAYOUT = 'NHWC'
DTYPE = 'int8'

def test_cmsis_conv():
    import time
    reset_gdbinit()

    def get_cmsis_tensors():
        with micro.Session(DEV_CONFIG) as sess:
            micro_mod = sess.create_micro_mod(DummyCMod())
            micro_func = micro_mod['arm_conv_wrapper']
            ctx = tvm.micro_dev(0)

            data_np = np.random.randint(-100, 100, size=(N, H, W, CI), dtype=DTYPE)
            kernel_np = np.random.randint(-5, 5, size=(CO, CI, KH, KW), dtype=DTYPE)

            data_tvm = tvm.nd.array(data_np, ctx=ctx)
            kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
            output_tvm = tvm.nd.empty([N, H, W, CO], ctx=ctx, dtype=DTYPE)
            start = time.time()
            task_cycles = micro_func(data_tvm, kernel_tvm, output_tvm)
            end = time.time()
            wall_clock_time = end - start
            print(f'task took {wall_clock_time} seconds')

            return data_np, kernel_np, output_tvm.asnumpy(), task_cycles, wall_clock_time

    def get_micro_output(data_np, kernel_np):
        func_name = 'conv2d'
        data = tvm.placeholder((N, H, W, CI), name='data')
        kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
        conv = topi.nn.conv2d_nhwc(data, kernel, STRIDES, PADDING, DILATION, DTYPE)
        sched = tvm.create_schedule([conv.op])
        with tvm.build_config(disable_vectorize=True):
            c_mod = tvm.build(sched, [data, kernel,conv], target='c', name=func_name)

        with micro.Session(DEV_CONFIG) as sess:
            micro_mod = sess.create_micro_mod(c_mod)
            micro_func = micro_mod[func_name]
            ctx = tvm.micro_dev(0)

            data_tvm = tvm.nd.array(data_np, ctx=ctx)
            kernel_tvm = tvm.nd.array(kernel_np, ctx=ctx)
            output_tvm = tvm.nd.empty([N, H, W, CO], ctx=ctx, dtype=DTYPE)
            start = time.time()
            task_cycles = micro_func(data_tvm, kernel_tvm, output_tvm)
            end = time.time()
            wall_clock_time = end - start

            return output_tvm.asnumpy(), task_cycles, wall_clock_time

    #def verify_result_relay(data_np, kernel_np, output_np):
    #    print('[Verifying]')
    #    # Construct Relay program.
    #    data_var = relay.var("data", shape=(N, H, W, CI), dtype=DTYPE)
    #    kernel_var = relay.var("kernel", dtype=DTYPE)
    #    conv_expr = relay.nn.conv2d(
    #            data_var, kernel_var,
    #            kernel_size=(KH, KW),
    #            strides=STRIDES,
    #            padding=PADDING,
    #            dilation=DILATION,
    #            channels=CO,
    #            data_layout=LAYOUT,
    #            out_layout=LAYOUT)
    #    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
    #    mod = relay.Module.from_expr(func)
    #    mod = transform.InferType()(mod)

    #    print(mod)

    #    data_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
    #    print(data_shape)
    #    kernel_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    #    print(kernel_shape)
    #    output_shape = list(map(lambda x: x.value, mod['main'].ret_type.shape))
    #    print(output_shape)

    #    intrp = create_executor('debug')
    #    data_tvm = tvm.nd.array(data_np, ctx=tvm.cpu(0))
    #    kernel_tvm = tvm.nd.array(kernel_np, ctx=tvm.cpu(0))
    #    expected_output_tvm = intrp.evaluate(mod['main'])(data_tvm, kernel_tvm).data

    #    tvm.testing.assert_allclose(output_np, expected_output_tvm.asnumpy())

    assert False, "use topi.testing.conv2d_nhwc_python to verify!"

    data_np, kernel_np, output_np, cmsis_cycles, cmsis_time = get_cmsis_tensors()
    assert np.sum(output_np) != 0
    expected_output_np, micro_cycles, micro_time = get_micro_output(data_np, kernel_np)

    #verify_result_relay(data_np, kernel_np, output_np)
    print('[CMSIS]')
    print(f'Cycles: {cmsis_cycles}')
    print(f'Time: {cmsis_time}')
    print('[MicroTVM]')
    print(f'Cycles: {micro_cycles}')
    print(f'Time: {micro_time}')
    print('[MicroTVM Speedup]')
    print(f'Cycles: {cmsis_cycles / micro_cycles}')
    print(f'Time: {cmsis_time / micro_time}')
    tvm.testing.assert_allclose(output_np, expected_output_np)

    #data_np = np.random.randint(-100, 100, size=(N, H, W, CI), dtype=DTYPE)
    #kernel_np = np.random.randint(-5, 5, size=(CO, CI, KH, KW), dtype=DTYPE)
    #expected_output_np, micro_cycles, micro_time = get_micro_output(data_np, kernel_np)
    #print('[MicroTVM]')
    #print(f'Cycles: {micro_cycles}')
    #print(f'Time: {micro_time}')

    #data_np = np.random.randint(-100, 100, size=(N, H, W, CI), dtype=DTYPE)
    #kernel_np = np.random.randint(-5, 5, size=(CO, CI, KH, KW), dtype=DTYPE)
    #output_np = np.zeros((N, H, W, CO), dtype=DTYPE)
    #verify_result_relay(data_np, kernel_np, output_np)

# N, H, W, CO, CI, KH, KW = 1, 16, 16, 32, 32, 5, 5
# [CMSIS]
# Cycles: 10918149.0
# Time: 0.23316740989685059
# [MicroTVM]
# Cycles: 95281.0
# Time: 0.16225314140319824
# [MicroTVM Speedup]
# Cycles: 114.58894218154721
# Time: 1.437059448466583

# N, H, W, CO, CI, KH, KW = 1, 18, 18, 32, 32, 5, 5
# [CMSIS]
# Cycles: 13792940.0
# Time: 0.24947428703308105
# [MicroTVM]
# Cycles: 109095.0
# Time: 0.16560769081115723
# [MicroTVM Speedup]
# Cycles: 126.43054218800128
# Time: 1.5064172793614825

# N, H, W, CO, CI, KH, KW = 1, 20, 20, 32, 32, 5, 5
# [CMSIS]
# Cycles: 245489.0
# Time: 0.26293063163757324
# [MicroTVM]
# Cycles: 131617.0
# Time: 0.16651177406311035
# [MicroTVM Speedup]
# Cycles: 1.8651769908142566
# Time: 1.5790512900181988

# N, H, W, CO, CI, KH, KW = 1, 24, 24, 32, 32, 5, 5
# [CMSIS]
# Cycles: 7707967.0
# Time: 0.2836883068084717
# [MicroTVM]
# Cycles: 185840.0
# Time: 0.17332792282104492
# [MicroTVM Speedup]
# Cycles: 41.47636138613861
# Time: 1.6367143977221146

# N, H, W, CO, CI, KH, KW = 1, 28, 28, 32, 32, 5, 5
# [CMSIS]
# Cycles: 16546962.0
# Time: 0.33393096923828125
# [MicroTVM]
# Cycles: 2710118.0
# Time: 0.18435430526733398
# [MicroTVM Speedup]
# Cycles: 6.105624183153648
# Time: 1.811354330749394

# N, H, W, CO, CI, KH, KW = 1, 30, 30, 32, 32, 5, 5
# [CMSIS]
# Cycles: 4694608.0
# Time: 0.3659372329711914
# [MicroTVM]
# Cycles: 4411234.0
# Time: 0.2863960266113281
# [MicroTVM Speedup]
# Cycles: 1.0642391675435943
# Time: 1.2777315289635973

# N, H, W, CO, CI, KH, KW = 1, 31, 31, 32, 32, 5, 5
# [CMSIS]
# Cycles: 7286871.0
# Time: 0.37795591354370117
# [MicroTVM]
# Cycles: 3704344.0
# Time: 0.3617746829986572
# [MicroTVM Speedup]
# Cycles: 1.9671150951423517
# Time: 1.0447273712215623

# N, H, W, CO, CI, KH, KW = 1, 32, 32, 32, 32, 5, 5
# [CMSIS]
# Cycles: 9957309.0
# Time: 0.3922536373138428
# [MicroTVM]
# Cycles: 13185781.0
# Time: 0.49291563034057617
# [MicroTVM Speedup]
# Cycles: 0.7551550416315879
# Time: 0.7957825095601416


def test_conv2d():
    from tvm.relay import create_executor
    from tvm.relay import transform

    dshape = (N, H, W, CI)
    dtype = 'int8'
    func_name = 'fused_nn_conv2d'

    reset_gdbinit()

    # Construct Relay program.
    data_var = relay.var("data", shape=(N, H, W, CI), dtype=DTYPE)
    kernel_var = relay.var("kernel", shape=(CO, CI, KH, KW), dtype=DTYPE)
    conv_expr = relay.nn.conv2d(
            data_var, kernel_var,
            kernel_size=(KH, KW),
            strides=STRIDES,
            padding=PADDING,
            dilation=DILATION,
            channels=CO,
            data_layout=LAYOUT,
            out_layout=LAYOUT)
    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
    mod = relay.Module.from_expr(func)
    mod = transform.InferType()(mod)

    x_shape = list(map(lambda x: x.value, mod['main'].params[0].checked_type.shape))
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    out_shape = list(map(lambda x: x.value, mod['main'].ret_type.shape))
    print(x_shape)
    print(w_shape)
    print(out_shape)
    input('ayy')

    with tvm.build_config(disable_vectorize=True):
        graph, c_mod, params = relay.build(mod, target="c")
    print(c_mod.get_source())
    input('ayy')

    with micro.Session(DEV_CONFIG) as sess:
        micro_mod = sess.create_micro_mod(c_mod)
        micro_func = micro_mod[func_name]
        ctx = tvm.micro_dev(0)

        x_data = tvm.nd.array(np.random.uniform(size=x_shape).astype(dtype), ctx)
        w_data = tvm.nd.array(np.random.uniform(size=w_shape).astype(dtype), ctx)
        result = tvm.nd.array(np.zeros(shape=out_shape, dtype=dtype), ctx)
        micro_func(x_data, w_data, result)

        out_data = np.zeros(out_shape, dtype=dtype)
        params = { 'x': x_data.asnumpy(), 'w': w_data.asnumpy() }
        intrp = create_executor('debug')
        expected_result = intrp.evaluate(mod['main'])(x_data, w_data).data

        tvm.testing.assert_allclose(result.asnumpy(), expected_result.asnumpy())


if __name__ == "__main__":
    test_cmsis_conv()
    #test_conv2d()
