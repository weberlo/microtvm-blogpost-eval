import functools
import inspect

import tvm
from tvm.autotvm.task.topi_integration import TaskExtractEnv

from micro_eval.util import NamedType, BakedType
from tvm.autotvm.task.dispatcher import DispatchContext
from tvm.autotvm.task.space import ConfigSpace

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
