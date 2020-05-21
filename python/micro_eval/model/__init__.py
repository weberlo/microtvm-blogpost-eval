import abc
import argparse
import collections
import importlib
import typing

import numpy
import tvm

from micro_eval import dataset
from micro_eval import util
from micro_eval.util import config_util


# Represents a lowered module (the output of relay.build()).
LoweredModule = collections.namedtuple('LoweredModule', ['graph', 'mod', 'params'])


class CompiledModel:
  """Container class that holds the output of build_model."""

  def __init__(self, target : tvm.target.Target, ir_mod : tvm.ir.IRModule,
               params : typing.Dict[str, util.LabelledTensor], entry_point : str,
               config : dict):
    """Construct a new container.

    Params
    ------
    target : tvm.target.Target
        The target for which this model was compiled.
    ir_mod : tvm.ir.IRModule
        The Relay IRModule.
    params : dict[str, micro_eval.util.LabelledTensor]
        Dict containing the model parameters. Keys are the parameter names and values are
        LabelledTensor containing the parameter values.
    entry_point : str
        Name of the function in mod which is the model entry point.
    config : dict
        Arbitrary key-value config which may be used downstream.
    """
    self.target = target
    self.ir_mod = ir_mod
    self.params = params
    self.entry_point = entry_point
    self.config = config


class TunableModel(metaclass=abc.ABCMeta):

  def __init__(self, target, ctx_str, config=None):
    """Instantiate a new TunableModel.

    Params
    ------
    target : tvm.target.Target
        The target to be used with this model.

    ctx_str : str
        The intended context string.

    config : Any
        JSON config loaded from the file given by the --model-config command-line argument, passed
         to scripts in model_eval.bin.
    """
    self.target = target
    self.ctx_str = ctx_str
    self.config = config

  @abc.abstractmethod
  def get_config_str(self) -> str:
    """Compute a string describing the model configuration.

    This string is used to construct the autotuning log names.

    Returns
    -------
    str :
        A string describing the model config.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def build_model(self) -> CompiledModel:
    """Build a relay IRModule for the given target.

    Returns
    -------
    CompiledModel
        The compiled module.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def lower_model(self, compiled_model : CompiledModel,
                  dev_config : typing.Optional[typing.Dict] = None) -> LoweredModule:
    """Lower the Relay IRModule into the target instruction set.

    Params
    ------
    compiled_model : CompiledModel
        The return value of build_model.

    dev_config : dict
        If supplied, the return value from tvm.micro.device.abc.generate_config.
        Describes the memory layout of the target device.

    Returns
    -------
    LoweredModule :
        Loadable module, graph, and params for the target.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def extract_tunable_tasks(self) -> typing.List[tvm.autotvm.task.Task]:
    """Extract and return tasks from the model built by build_model.

    Returns
    -------
    List[tvm.autotvm.task.Task] :
        A list of tasks which should be autotuned.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_autotvm_measure_option(self, num_runners : int, tracker_host : str, tracker_port : int,
                                 tracker_key : str, task : tvm.autotvm.task.Task) -> typing.Dict:
    """Return autotvm.measure_option result to be used for autotuning this model task.

    Params
    ------
    num_runners : int
        Number of TVM RPC servers available; suggest to use this to limit # of concurrent builders.
    tracker_host : str
        Hostname of the machine running the TVM RPC tracker server.
    tracker_port : int
        Port number for the TVM RPC tracker server.
    tracker_key : str
        Key on the tracker under which the runners register.
    task : tvm.autotvm.task.Task
        The task being tuned.

    Returns
    -------
    dict:
       The return value from tvm.autotvm.measure_option.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def section_constraints(self, task_index=None) -> collections.OrderedDict:
    """Return the section_constraints= argument to tvm.micro.device.abc.generate_config.

    Params
    ------
    task_index : Optional[int]
        If given, the index of the task to be tuned. Needed for now due to split memory
        allocation between TVM RPC Server and device.

    Returns
    -------
    collections.OrderedDict :
        The section constraints. An example:
            OrderedDict([
                ('text', (28000, MemConstraint.ABSOLUTE_BYTES)),
                ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
                ('bss', (800, MemConstraint.ABSOLUTE_BYTES)),
                ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
                ('heap', (100.0, MemConstraint.WEIGHT)),
                ('workspace', (100000, MemConstraint.ABSOLUTE_BYTES)),
                ('stack', (128, MemConstraint.ABSOLUTE_BYTES)),
            ])

    """
    raise NotImplementedError()

  @abc.abstractmethod
  def dataset_generator_name(self) -> str:
    """Return the canonical name of the DatasetGenerator subclass for this model."""
    raise NotImplementedError()

  def adapt_sample_inputs(self, sample : dataset.ModelParamsResults) -> dataset.ModelParamsResults:
    """Adapt an input sample to fit this model.

    Some models may have several different implementations which are nevertheless expected to return
    identical or nearly-identical results from the same dataset. This function allows those models to
    do things like reshape, resize, or transpose input samples for use with the model.

    Params
    ------
    samples : List[DatasetSample]
        The list of samples to adapt.

    Returns
    -------
    List[ModelParamsResults]:
        The adapted samples.
    """
    return sample

  def adapt_model_outputs(self, result : dataset.ModelParamsResults) -> dataset.ModelParamsResults:
    """Adapt generated model outputs to fit its dataset samples.

    Some models may have several different implementations which are nevertheless expected to return
    identical or nearly-identical results from the same dataset. This function allows those models to
    do things like reshape, resize, or transpose model outputs to match the expected results from the
    DatasetSample.

    Params
    ------
    samples : DatasetSample
        The results to adapt.

    Returns
    -------
    ModelParamsResults:
        The adapted results.
    """
    return result


SETTING_TO_TARGET_AND_CONTEXT = {
  'micro_dev': ('c -device=micro_dev', 'micro_dev'),
  'interp': ('llvm', 'cpu'),
  'cpu': ('llvm', 'cpu'),
}



class ModelInstantiationError(Exception):
  """Raised when a model can't be instantiated."""


def instantiate_from_spec(spec : str) -> typing.Tuple[CompiledModel, str]:
  """Instantiate a TunableModel from argparse args.

  Params
  ------
  spec : str
      The model specification string.

  Returns
  -------
  TunableModel :
      The instantiated model.
  str :
      The setting for this model.

  Raises
  ------
  ModelInstantiationError:
      When a problem occurs instantiating the model due to e.g. unknown model name, bad
      configuration parameters.
  """
  parts = spec.split(':')
  if len(parts) > 3:
    raise ModelInstantiationError(f'model spec: want at most 3 colon-separated parts, got {spec}')
  elif len(parts) == 3:
    model_name, setting, config_path = parts
  elif len(parts) == 2:
    model_name, setting = parts
    config_path = None
  else:
    raise ModelInstantiationError(f'model spec: want at least 2 colon-separated parts, got {spec}')

  def _strip_label(index, arg, label):
    if '=' not in arg:
      return arg

    expected = f'{label}='
    if arg.startswith(expected):
      return setting[len(expected):]

    raise ModelInstantiationError(
      f'model spec: expected item #{index} to begin with label "{expected}" if one is '
      f'present; got {arg}')

  setting = _strip_label(1, setting, 'setting')
  if config_path is not None:
    config_path = _strip_label(2, config_path, 'config')

  try:
    mod = importlib.import_module(f'.{model_name}', __name__)
  except ImportError as err:
    raise ModelInstantiationError(f'Could not import {__name__}.{model_name}') from err

  model_config = {}
  if config_path:
    try:
      model_config = config_util.Config.load(config_path)
    except Exception as err:
      raise ModelInstantiationError(
        f'Could not load model config from {config_path}') from err

  for key in dir(mod):
    val = getattr(mod, key)
    try:
      is_subclass = issubclass(val, TunableModel)
    except TypeError:
      continue

    if is_subclass and val != TunableModel:
      target_name, ctx_str = SETTING_TO_TARGET_AND_CONTEXT[setting]
      target = tvm.target.create(target_name)
      try:
        return val(config=model_config, target=target, ctx_str=ctx_str), setting
      except Exception as err:
        raise ModelInstantiationError(
          f'Could not instantiate {__name__}.{model_name}.{key}') from err

  raise ModelInstantiationError(
    f'Did not find a subclass of TunableModel in {__name__}.{model_name}')
