import argparse

from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import onnx
import tvm
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.relay.type_functor import TypeMutator
from tvm.relay import transform

from micro_eval import util
from micro_eval.util import model_util

class QuantPrefixCutter(ExprMutator):
    def __init__(self, params):
        ExprMutator.__init__(self)
        self.params = set(params)
        self.subtree_params = set()

    def visit_var(self, var):
        if var in self.params:
            self.subtree_params.add(var)
        return var

    def visit_call(self, call):
        print(call.checked_type.dtype)
        if call.op.name == 'cast' and call.checked_type.dtype == 'int8':
            res = super().visit_call(call)
            if len(self.subtree_params) == 0:
                return res
            else:
                assert len(self.subtree_params) == 1
                res = next(iter(self.subtree_params))
                self.subtree_params.clear()
                return res
        else:
            return super().visit_call(call)


class QuantSuffixCutter(ExprMutator):
    def __init__(self):
        ExprMutator.__init__(self)
        self.pre_cast_expr = None

    def visit_call(self, call):
        # print(call.checked_type.dtype)
        if call.op.name == 'annotation.stop_fusion':
            return call
        else:
            new_args = []
            for arg in call.args:
                arg_res = self.visit(arg)
                if type(arg_res) == relay.Call and arg_res.op.name == 'annotation.stop_fusion':
                    return arg_res
                else:
                    new_args.append(arg_res)
            return relay.Call(call.func, new_args, call.attrs)

#    def visit_function(self, func):
#        # import pdb; pdb.set_trace()
#        # new_params = [self.visit(x) for x in fn.params]
#        sig_rewriter = SignatureRewriter()
#        new_params = [
#            relay.Var(
#                param.name_hint,
#                sig_rewriter.visit(param.type_annotation)
#            )
#            for param in func.params]
#        new_body =
#        new_ret_type = sig_rewriter.visit(func.ret_type)
#        return relay.Function(
#            func.params,
#            self.visit(func.body),
#            func.ret_type,
#            func.type_params,
#            func.attrs)


class SignatureRewriter(TypeMutator):
    def visit_tensor_type(self, tt):
        return relay.TensorType(tt.shape, 'int8')


def with_int_signature(func):
    sig_rewriter = SignatureRewriter()
    new_params = [
        relay.Var(
            param.name_hint,
            sig_rewriter.visit(param.type_annotation)
        )
        for param in func.params]
    new_ret_type = sig_rewriter.visit(func.ret_type)
    return relay.Function(
        new_params,
        func.body,
        new_ret_type,
        func.type_params,
        func.attrs)


def partition_quantized(mod):
    # TODO
    # - change to expr mutator
    # - match cast ops then call a separate visitor that finds the param at the bottom of the chain
    # - return the result from the visitor
    # - change types in function signature (use type functor?)
    assert len(mod.functions) == 1
    func = mod['main']
    func = with_int_signature(func)

    prefix_cutter = QuantPrefixCutter(func.params)
    func = prefix_cutter.visit(func)
    print('[Without Conversion Prefix]')
    print(func)

    suffix_cutter = QuantSuffixCutter()
    func = suffix_cutter.visit(func)
    print('[Without Conversion Suffix]')
    print(func)

    mod['main'] = func
    mod = transform.InferType()(mod)
    import pdb; pdb.set_trace()
    return mod


def quantize(mod, params):
    with relay.quantize.qconfig(
      calibrate_mode='global_scale',
      global_scale=8.0,
      nbit_activation=8,
      dtype_activation="int8",
      skip_conv_layers=[],
      dtype_input="int8",
      dtype_weight="int8"):
        print('input:', mod)
        quantized = relay.quantize.quantize(mod, params) #, dataset=numpy_samples)
        print('output:', quantized)
        print('partitioned:', partition_quantized(quantized))

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


def quantize_test():
    # """Program:
    #    fn (%x : Tensor[(10, 10), float32]) {
    #      let %t1 = log(x);
    #      let %t2 = add(%t1, %x);
    #      %t1
    #    }
    # """
    # tp = relay.TensorType((10, 10), "float32")
    # x = relay.var("x", tp)
    # sb = relay.ScopeBuilder()
    # t1 = sb.let("t1", relay.log(x))
    # t2 = sb.let("t2", relay.add(t1, x))
    # sb.ret(t2)
    # f = relay.Function([x], sb.get())

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
    mod = transform.InferType()(mod)

    # TODO crash is probs coming from non-const weights? add param for weight
    import numpy as np
    weight_ty = mod['main'].params[1].type_annotation
    w_shape = list(map(lambda x: x.value, mod['main'].params[1].checked_type.shape))
    params = {
        'w': tvm.nd.array(np.random.uniform(0, 1, size=w_shape).astype(weight_ty.dtype), ctx=tvm.cpu(0))
    }
    quantize(mod, params)


if __name__ == '__main__':
#   quantize_cifar10()
  quantize_test()
