import functools
import inspect

import tvm

from micro_eval.util import NamedType, BakedType

def op_decl(in_tensors=None):
    def _inner_decorator(orig_func):
        assert in_tensors is not None, f'operator `{orig_func.__name__}` must have input tensors'
        # set up layout registry
        double_decl_err_str = f'`{orig_func.__name__}` has already been declared'
        assert not hasattr(orig_func, 'compute_funcs'), double_decl_err_str
        assert not hasattr(orig_func, 'schedule_funcs'), double_decl_err_str
        assert not hasattr(orig_func, 'input_tensors'), double_decl_err_str
        orig_func.compute_funcs = {}
        orig_func.schedule_funcs = {}

        arg_names = inspect.getfullargspec(orig_func).args
        # trim off `compute_func` and `schedule_func` from arg names, since the
        # user doesn't supply them (the decorator does so via dispatch)
        arg_names = arg_names[2:]
        for tensor in in_tensors:
            assert tensor in arg_names
        orig_func.input_tensors = in_tensors

        @functools.wraps(orig_func)
        def _wrapper(*args, **kwargs):
            new_args = []
            layout_dict = {}
            for arg, arg_name in zip(args, arg_names):
                # replace tensor types with TVM placeholders
                assert not isinstance(arg, NamedType), 'shape must have a set layout'
                if isinstance(arg, BakedType):
                    shape = arg
                    assert arg_name in orig_func.input_tensors, f'unexpected input tensor `{arg_name}`'
                    arg_typ, arg_shape, arg_dtype = shape.gen_spec()
                    assert arg_typ == 'TENSOR'
                    arg = tvm.placeholder(arg_shape, name=arg_name, dtype=arg_dtype)
                    layout_dict[arg_name] = shape.layout
                elif isinstance(arg, tuple) and arg[0] == 'TENSOR' and arg_name in in_tensors:
                    raise RuntimeError(
                        f'arg `{arg_name} := {arg}` is already expanded to a spec. Must pass a (NamedType, layout str) pair')
                new_args.append(arg)

            # dispatch to compute/schedule funcs that match the layout signature
            layout_signature = _gen_layout_signature(layout_dict, orig_func.input_tensors)
            assert layout_signature in orig_func.compute_funcs, f'no supported compute func for operator `{orig_func.__name__}` with layout configuration {layout_signature}'
            compute_func = orig_func.compute_funcs[layout_signature]
            assert layout_signature in orig_func.schedule_funcs, f'no supported schedule func for operator `{orig_func.__name__}` with layout configuration {layout_signature}'
            schedule_func = orig_func.schedule_funcs[layout_signature]
            # prepend compute/schedule funcs to original args
            new_args = [compute_func, schedule_func] + new_args

            return orig_func(*new_args, **kwargs)
        return _wrapper
    return _inner_decorator


def _gen_layout_signature(layout_dict, input_tensors):
    if len(layout_dict) != len(input_tensors):
        missing_tensors = []
        for input_tensor in input_tensors:
            if input_tensor not in layout_dict:
                missing_tensors.append(input_tensor)
        raise RuntimeError(f'missing the following input tensors: {missing_tensors}')

    layout_signature = []
    for name in input_tensors:
        assert name in layout_dict
        layout_signature.append(layout_dict[name])
    return tuple(layout_signature)


def _register_layout(func_type_str, base_op_func, **layout_dict):
    """Route layouts with the specified signature to the compute/schedule func being decorated."""
    assert func_type_str in ['compute', 'schedule']
    def _inner_decorator(func):
        layout_signature = _gen_layout_signature(layout_dict, base_op_func.input_tensors)
        dispatch_dict = getattr(base_op_func, f'{func_type_str}_funcs')
        assert layout_signature not in dispatch_dict
        dispatch_dict[layout_signature] = func
        return func
    return _inner_decorator


def register_compute(base_op_func, **layouts):
    return _register_layout('compute', base_op_func, **layouts)


# TODO automatically generate a `_callback` and call `traverse_inline` for scheds
def register_schedule(base_op_func, **layouts):
    return _register_layout('schedule', base_op_func, **layouts)
