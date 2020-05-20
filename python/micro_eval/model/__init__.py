import abc
import argparse
import typing


class CompiledModel:
  """Container class that holds the output of build_model."""

  def __init__(self, mod : tvm.ir.IRModule, params : typing.Dict[str, util.LabelledTensor],
               entry_point : str, config : dict):
    """Construct a new container.

    Params
    ------
    mod : tvm.ir.IRModule
        The compiled module.
    params : dict[str, micro_eval.util.LabelledTensor]
        Dict containing the model parameters. Keys are the parameter names and values are
        LabelledTensor containing the parameter values.
    entry_pint : str
        Name of the function in mod which is the model entry point.
    config : dict
        Arbitrary key-value config which may be used downstream.
    """
    self.mod = mod
    self.params = params
    self.entry_point = entry_point
    self.config = config


class TunableModel(metaclass=abc.ABCMeta):

  def __init__(self, target, config=None):
    """Instantiate a new TunableModel.

    Params
    ------
    target : tvm.target.Target
        The target to be used with this model.

    config : Any
        JSON config loaded from the file given by the --model-config command-line argument, passed
         to scripts in model_eval.bin.
    """
    self.target = target
    self.config = config

  @abc.abstractmethod
  def get_config_str(self, target : tvm.target.Target) -> str:
    """Compute a string describing the model configuration.

    This string is used to construct the autotuning log names.

    Params
    ------

    Returns
    -------
    str :
        A string describing the model config.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def build_model(self, target : tvm.target.Target):
    """Build a relay IRModule for the given target.

    Params
    ------
    target : tvm.target.Target
        The build target

    Returns
    -------
    tvm.runtime.IRModule :
        The compiled module.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def extract_tunable_tasks(self, target : tvm.target.Target) -> typing.List[tvm.autotvm.task.Task]:
    """Extract and return tasks from the model built by build_model.

    Params
    ------
    target : tvm.target.Target
        The build target.

    Returns
    -------
    List[tvm.autotvm.task.Task] :
        A list of tasks which should be autotuned.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_autotvm_measure_option(self, num_runners : int, tracker_host : str, tracker_port : int,
                                 tracker_key : str, task : tvm.autotvm.task.Task):
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
  def section_constraints(self):
    """Return the section_constraints= argument to tvm.micro.device.abc.generate_config.

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


def instantiate_from_args(args : argparse.Namespace):
  """Instantiate a TunableModel from argparse args.

  Params
  ------
  args : argparse.Namespace
      Parsed command-line arguments

  Returns
  -------
  TunableModel :
      The instantiated model.

  Raises
  ------
  ModelInstantiationError:
      When a problem occurs instantiating the model due to e.g. unknown model name, bad
      configuration parameters.
  """
  try:
    mod = importlib.import_module(f'.{args.model_name}', __name__)
  except ImportError as err:
    raise ModelInstantiationError(f'Could not import {__name__}.{args.model_name}') from err

  model_config = None
  if args.model_config:
    try:
      with open(args.model_config) as config_f:
        model_config = json.load(config_f)
    except Exception as err:
      raise ModelInstantiationError(
        f'Could not load model config from {args.model_config}') from err

  for key in dir(mod):
    val = getattr(mod, key)
    if issubclass(val, TunableModel) and val != TunableModel:
      try:
        return val(config=config)
      except Exception as err:
        raise ModelInstantiationError(
          f'Could not instantiate {__name__}.{args.model_name}.{key}') from err

  raise ModelInstantiationError(
    f'Did not find a subclass of TunableModel in {__name__}.{args.model_name}')
