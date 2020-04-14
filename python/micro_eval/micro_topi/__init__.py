import functools
import inspect

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import ExprVisitor
from tvm.autotvm.task.topi_integration import TaskExtractEnv
from tvm.autotvm.task.dispatcher import DispatchContext
from tvm.autotvm.task.space import ConfigSpace
from topi.util import get_const_tuple

from micro_eval.util import NamedType, BakedType, deep_tuple

# init autotvm env to register uTVM ops
TaskExtractEnv()

class ManualConfigContext(DispatchContext):
    """Apply a manually-generated config entity for each workload.

    Parameters
    ----------
    query_to_config : Dict[Tuple[str, str], ConfigSpace]
        Mapping from (target, workload) to the corresponding config.
    """
    def __init__(self, query_to_config):
        super(ManualConfigContext, self).__init__()
        if isinstance(query_to_config, dict):
            self._query_to_config = query_to_config
        else:
            # when a single config space is passed, it is assumed we are in a
            # single-op setting, where the target and workload are both set to
            # `None` on dispatch.
            self._query_to_config = {(None, None): query_to_config}

    def _query_inside(self, target, workload):
        key = (target, workload)
#        print('have configs', '\n'.join([repr(x) for x in self._query_to_config.keys()]))
        assert key in self._query_to_config, f'unknown query `{key}` encountered'
        return self._query_to_config[key]


class ManualConfigSpace(ConfigSpace):
    """Use as the argument to `with ApplyConfig(...)` to use a deterministic op config"""

    def __init__(self):
        super(ManualConfigSpace, self).__init__()
        self.is_fallback = False
        # NOTE most important part of this class: we don't want to be in
        # collection mode, because the config the user specifies would then be
        # overwritten by a fallback config.
        self._collect = False

    def __setitem__(self, name, entity):
        """set the entity(knob) of by name

        Parameters
        ----------
        name: str
            name of the entity
        entity: SplitEntity, ReorderEntity, AnnotateEntity, OtherOptionEntity
            value of the entity
        """
        self._entity_map[name] = entity

    def __repr__(self):
        return "(%s, %s, %s)" % (str(self._entity_map)[12:-1], self.template_key, self.code_hash)


def collect_conv_workloads(func):
    class ConvCollector(ExprVisitor):
        def __init__(self):
            super(ConvCollector, self).__init__()
            self.convs = []

        def visit_call(self, call):
            if isinstance(call.op, relay.op.op.Op) and 'conv2d' in call.op.name:
                data_type = call.args[0].checked_type
                data_ser = ['TENSOR', list(get_const_tuple(data_type.shape)), data_type.dtype]
                kernel_type = call.args[1].checked_type
                kernel_ser = ['TENSOR', list(get_const_tuple(kernel_type.shape)), kernel_type.dtype]
                serialized_attrs = [
                    call.attrs[key]
                    for key in ['strides', 'padding', 'dilation', 'out_dtype']]
                for i, attr in enumerate(serialized_attrs):
                    if isinstance(attr, tvm.container.Array):
                        serialized_attrs[i] = get_const_tuple(attr)
                workload = ['conv2d_nhwc_spatial_pack.arm_cpu'] + [data_ser] + [kernel_ser] + serialized_attrs
                # prepend the workload, because the traversal order is from the
                # end of the network to the beginning (it's a chain of calls
                # where the outermost call is the result of the net and each
                # nested call corresponds to the previous layer)
                self.convs = [deep_tuple(workload)] + self.convs
            for arg in call.args:
                self.visit(arg)

    collector = ConvCollector()
    collector.visit(func)
    return collector.convs


# TODO replace task extraction in mainline TVM with this function. it's stateless 'n' shit
def collect_conv_tasks(func, target, template_key):
    workloads = collect_conv_workloads(func)
    tasks = []
    for workload in workloads:
        name, data, kernel, strides, padding, dilation, data_layout, out_dtype = workload
        data_shape = data[0:-1]
        data_out_dtype = data[-1]
        data_arg = ('TENSOR', data_shape, data_out_dtype)
        kernel_shape = kernel[0:-1]
        kernel_out_dtype = kernel[-1]
        kernel_arg = ('TENSOR', kernel_shape, kernel_out_dtype)
        args = (data_arg, kernel_arg, strides, padding, dilation, data_layout, out_dtype)
        tasks.append(autotvm.task.create(
            'topi_nn_conv2d',
            args,
            target,
            None,
            template_key=template_key))
    return tasks
