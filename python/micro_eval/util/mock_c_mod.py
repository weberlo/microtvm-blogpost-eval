"""Defines functions to build a mock IRModule from C sources."""

import json

def build(src_path, entry_point_name, params, return_type):
  """Build a mock C module and graph."""
  nodes = []
  arg_nodes = []
  dltype = []
  shapes = []
  for name, shape in params:
    arg_nodes.append(len(nodes))
    dltype.append(shape.dtype)
    shapes.append(shape.shape)
    nodes.append({'op': 'null',
                  'name': name,
                  'inputs': [],
#                  'attrs': {'T': f'type: {shape.dtype}'},
#                  'shape': list(shape.shape),
                  })
  heads = [[len(nodes), 0, 0]]
  dltype.append(return_type.dtype)
  shapes.append(return_type.shape)
  nodes.append({'op': 'tvm_op',
                'name': entry_point_name,
                'attrs': {
                  'func_name': entry_point_name,
                  'flatten_data': '0',
                  'num_inputs': str(len(params)),
                  'num_outputs': '1',
                  'T': return_type.dtype,
                  },
                'inputs': [[i, 0, 0] for i, _ in enumerate(params)],
                #'shape': return_type.shape,
  })
  graph = {
    'nodes': nodes,
    'arg_nodes': arg_nodes,
    'heads': heads,
    'node_row_ptr': list(range(len(nodes) + 1)),
    'attrs': {
      'dltype': ['list_str', dltype],
      'storage_id': ['list_int', list(range(len(nodes)))],
      'shape': ['list_shape', shapes],
#      'device_index': ['list_int', [0x0c for _ in range(len(nodes))]],
    },
  }

  return MockCMod(src_path, json.dumps(graph), entry_point_name, params)


class MockParam:
  def __init__(self, name, shape):
    self.name_hint = name
    self.shape = shape


class MockFunc:
  def __init__(self, name, params):
    self.name = name
    self.params = params


class MockCMod:
    def __init__(self, src_path, graph_str, entry_point, params):
        self.src_path = src_path
        self.graph_str = graph_str
        self.entry_point = entry_point
        self.entry_point_func = MockFunc(entry_point, [MockParam(n, s) for n, s in params])

    def __getitem__(self, key):
      if key == self.entry_point:
        return self.entry_point_func

      raise KeyError(key)

    def export_library(self, out_obj_path, fcompile=None):
        assert fcompile is not None
        fcompile(out_obj_path, self.src_path)
