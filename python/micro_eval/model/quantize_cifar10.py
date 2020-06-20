import argparse

import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import onnx
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.type_functor import TypeMutator, TypeVisitor
from tvm.relay import transform

from micro_eval import util
from micro_eval.util import model_util

# TODO don't hardcode so many things as int8

def with_dtype(ty, target_dtype):
    class DtypeReplacer(TypeMutator):
        def __init__(self, target_dtype):
            self.target_dtype = target_dtype

        def visit_tensor_type(self, tt):
            return relay.TensorType(tt.shape, self.target_dtype)

    return DtypeReplacer(target_dtype).visit(ty)


class QuantPrefixCutter(ExprMutator):
    def __init__(self, params):
        ExprMutator.__init__(self)
        self.params = set(params)
        self.subtree_params = set()
        self.new_func_params = []
        self.prefix_sb = relay.ScopeBuilder()

    def visit_var(self, var):
        if var in self.params:
            self.subtree_params.add(var)
        return var

    def visit_call(self, call):
        if call.op.name == 'cast' and call.attrs['dtype'] == 'int8':
            res = super().visit_call(call)
            if len(self.subtree_params) == 0:
                return res
            else:
                assert len(self.subtree_params) == 1
                param = next(iter(self.subtree_params))
                self.prefix_sb.let(param.name_hint, res)
                self.subtree_params.clear()
                # return new parameter, then we can use
                # relay.analysis.free_vars at the end of the pass to generate
                # new `mid_func` type signature
                return relay.Var(param.name_hint, with_dtype(param.type_annotation, 'int8'))
        else:
            return super().visit_call(call)

    def visit_function(self, func):
        # override to make sure we *don't* visit the function params
        return relay.Function(
            func.params,
            self.visit(func.body),
            func.ret_type,
            func.type_params,
            func.attrs)


def partition_prefix(mod):
    assert len(mod.functions) == 1
    func = mod['main']
    prefix_cutter = QuantPrefixCutter(func.params)
    mid_body = prefix_cutter.visit(func.body)
    mid_func = relay.Function(
        relay.analysis.free_vars(mid_body),
        mid_body,
        func.ret_type,
        # TODO figure out how to divvy type params and attrs as well
        func.type_params,
        # TODO what's an example of data that's stored in attrs?
        func.attrs)
    mid_mod = tvm.IRModule.from_expr(mid_func)

    scope_builder = prefix_cutter.prefix_sb
    ret_expr = relay.Tuple(list(map(lambda b: b[0], scope_builder._bindings[-1])))
    scope_builder.ret(ret_expr)
    pre_func_body = scope_builder.get()
    pre_func = relay.Function(relay.analysis.free_vars(pre_func_body), pre_func_body)
    pre_mod = tvm.IRModule.from_expr(pre_func)

    return pre_mod, mid_mod


class QuantSuffixCutter(ExprMutator):
    def __init__(self):
        ExprMutator.__init__(self)
        self.mid_body = None

    def visit(self, expr):
        if hasattr(expr, 'checked_type') and expr.checked_type.dtype == 'int8':
            self.mid_body = expr
            return relay.Var('input', expr.checked_type)
        else:
            return super().visit(expr)


class ForbiddenDtypeException(Exception):
    def __init__(self, invalid_dtype, allowed_dtypes):
        self.invalid_dtype = invalid_dtype
        self.allowed_dtypes = allowed_dtypes


class TyPostPartitionChecker(TypeVisitor):
    def __init__(self, allowed_dtypes):
        TypeVisitor.__init__(self)
        self.allowed_dtypes = allowed_dtypes

    def visit_tensor_type(self, tt):
        if tt.dtype not in self.allowed_dtypes:
            raise ForbiddenDtypeException(tt.dtype, self.allowed_dtypes)


class PostPartitionChecker(ExprVisitor):
    def __init__(self, allowed_dtypes):
        ExprVisitor.__init__(self)
        self.allowed_dtypes = allowed_dtypes
        self.ty_checker = TyPostPartitionChecker(allowed_dtypes)

    def visit(self, expr):
        if hasattr(expr, 'checked_type'):
            self.ty_checker.visit(expr.checked_type)
        return super().visit(expr)

def partition_suffix(mod):
    assert len(mod.functions) == 1
    func = mod['main']
    suffix_cutter = QuantSuffixCutter()
    post_body = suffix_cutter.visit(func.body)
    post_func = relay.Function(
        relay.analysis.free_vars(post_body),
        post_body,
        func.ret_type)
    post_mod = tvm.IRModule.from_expr(post_func)

    mid_body = suffix_cutter.mid_body
    mid_func = relay.Function(
        func.params,
        mid_body)
    mid_mod = tvm.IRModule.from_expr(mid_func)

    return mid_mod, post_mod


# class VarReplacer(ExprMutator):
#     def __init__(self, replace_map):
#         ExprMutator.__init__(self)
#         self.replace_map = replace_map
#
#     def visit_var(self, var):
#         if var in self.replace_map:
#             return self.replace_map[var]
#
#
# def with_int_signature(func):
#     sig_rewriter = SignatureRewriter()
#     new_params = [
#         relay.Var(
#             param.name_hint,
#             sig_rewriter.visit(param.type_annotation)
#         )
#         for param in func.params]
#     new_ret_type = sig_rewriter.visit(func.ret_type)
#     new_body = VarReplacer(dict(zip(func.params, new_params))).visit(func.body)
#     return relay.Function(
#         new_params,
#         new_body,
#         new_ret_type,
#         func.type_params,
#         func.attrs)


def partition_quantized(mod, allowed_dtypes):
    # TODO we have the prefix/suffix conversion code chopped off, but we want
    # it to be a *partition*, so we need to save the pieces we're cutting off.
    # Also, do we want any restrictions on how much gets cut off?
    # Right now, with the CIFAR-10 CNN, it cuts off some bias add and dense
    # operations (i.e., not just conversion ops like clip, cast, and round),
    # causing type inference to fail.
    #
    # should the user receive diagnostics about the results (e.g., letting them
    # know some operators were chopped off in the prefix/suffix?)
    #
    # keep in mind that we have an implicit assumption in the prefix cutter
    # that the first operator is quantizable, which is fairly safe, since a CNN
    # usually starts with a conv, but if we have plans on generalizing the
    # quantization pass to arbitrary models, this will need to change.
    #
    # TODO TODO you just finished generalizing the quant prefix cutter to work
    # with multiple input args (`test_multiple_arg_conversions`), but you still
    # need to construct the prefix function (probably using a let list that
    # builds bindings that convert each of the args and returns a tuple of all
    # of them at the end)
    #
    # once it's all mostly working, test that `x |> prefix |> mid_func |> suffix`
    # is the same as `x |> orig_func`
    assert len(mod.functions) == 1
    pre_mod, mid_mod = partition_prefix(mod)
    mid_mod, post_mod = partition_suffix(mid_mod)

    try:
        checker = PostPartitionChecker(allowed_dtypes)
        checker.visit(mid_mod['main'])
    except ForbiddenDtypeException as e:
        mod_str = '  ' + str(mod).replace('\n', '\n  ')
        raise RuntimeError(f'found dtype `{e.invalid_dtype}` in middle partition'
            f' when allowed dtypes were `{e.allowed_dtypes}`. module shown below:\n{mod_str}')

    return pre_mod, mid_mod, post_mod


def quantize(mod, params):
    with relay.quantize.qconfig(
      calibrate_mode='global_scale',
      global_scale=8.0,
      nbit_activation=8,
      dtype_activation="int8",
      skip_conv_layers=[],
      skip_dense_layers=False,
      dtype_input="int8",
      dtype_weight="int8"):
        print('input:', mod)
        print()
        quantized = relay.quantize.quantize(mod, params) #, dataset=numpy_samples)
        print('output:', quantized)
        print()
        allowed_dtypes = {'int8'}
        pre_mod, mid_mod, post_mod = partition_quantized(quantized, allowed_dtypes)
        print('partitioned prefix:', pre_mod)
        print()
        print('partitioned middle:', mid_mod)
        print()
        print('partitioned suffix:', post_mod)
        print()

    #with open(args.quantized_tvm_model, 'w') as f:
    #    f.write(tvm.ir.save_json(quantized))


def quantize_cifar10():
  parser = argparse.ArgumentParser()
  parser.add_argument('--onnx-model', default=f'{util.get_repo_root()}/data/cifar10.onnx',
                      help='path to unquantized cifar10 model')
  parser.add_argument('--num-samples', default=1024, type=int,
                      help='number of samples to use for data-aware quantization')
  parser.add_argument('--quantized-tvm-model',
                      default=f'{util.get_repo_root()}/data/quantized-cifar10.onnx',
                      help='path to write quantized cifar10 model')

  args = parser.parse_args()

  onnx_model = onnx.load(args.onnx_model)

  data_shape = util.LabelledShape(N=1, C=3, H=32, W=32, dtype='uint8')
  mod, params = relay.frontend.from_onnx(onnx_model, {"data": data_shape.shape})

  samples = model_util.get_sample_points(args.num_samples, data_shape.layout)
  numpy_samples = []
  for data in samples:
    numpy_samples.append({'data': data['data'].data})

  quantize(mod, params)


def get_param_type(func, name):
    for param in func.params:
        if param.name_hint == name:
            return param.checked_type


def test_conv_quant():
    dshape = (1, 4, 16, 16)
    # TODO quantization shouldn't crash if a pre-quantized graph (already int8) is handed to it.
    dtype = 'float32'
    func_name = 'main'
    x = relay.var("x", shape=dshape, dtype=dtype)
    conv_expr = relay.nn.conv2d(
            x, relay.var("w"),
            kernel_size=(3, 3),
            padding=(1, 1),
            channels=4)
    func = relay.Function(relay.analysis.free_vars(conv_expr), conv_expr)
    mod = tvm.IRModule.from_expr(func)

    # TODO crash is probs coming from non-const weights? add param for weight
    weight_ty = mod['main'].params[1].type_annotation
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    params = {
        'w': tvm.nd.array(np.random.uniform(0, 1, size=w_shape).astype(weight_ty.dtype), ctx=tvm.cpu(0))
    }
    quantize(mod, params)


def test_add_quant():
    # TODO with respect to partitioning, what should we do in cases where the
    # function is so small the quantizer doesn't quantize anything?
    func = relay.fromtext("""
    v0.0.4
    fn (%x : Tensor[(10, 10), float32],
        %y : Tensor[(10, 10), float32]) {
      add(%x, %y)
    }
    """)
    mod = tvm.IRModule.from_expr(func)
    params = {}
    quantize(mod, params)


def test_multiple_arg_conversions():
    dshape = (1, 4, 16, 16)
    # TODO quantization shouldn't crash if a pre-quantized graph (already int8) is handed to it.
    dtype = 'float32'
    func_name = 'main'
    conv1 = relay.nn.conv2d(
        relay.var("x1", shape=dshape, dtype=dtype),
        relay.var("w1"),
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=4)
    conv2 = relay.nn.conv2d(
        relay.var("x2", shape=dshape, dtype=dtype),
        relay.var("w2"),
        kernel_size=(3, 3),
        padding=(1, 1),
        channels=4)
    res = relay.add(conv1, conv2)
    func = relay.Function(relay.analysis.free_vars(res), res)
    mod = tvm.IRModule.from_expr(func)

    w1_ty = get_param_type(mod['main'], 'w1')
    w2_ty = get_param_type(mod['main'], 'w2')
    w1_shape = list(map(lambda x: x.value, w1_ty.shape))
    w2_shape = list(map(lambda x: x.value, w2_ty.shape))
    params = {
        'w1': tvm.nd.array(np.random.uniform(0, 1, size=w1_shape).astype(w1_ty.dtype), ctx=tvm.cpu(0)),
        'w2': tvm.nd.array(np.random.uniform(0, 1, size=w2_shape).astype(w2_ty.dtype), ctx=tvm.cpu(0))
    }
    quantize(mod, params)


if __name__ == '__main__':
    # quantize_cifar10()
    # test_conv_quant()
    # test_add_quant()
    test_multiple_arg_conversions()
