import abc
import collections
import importlib
import typing

import numpy

from micro_eval import util


# Describes an input parameter set or output result set.
ModelParamsResults = typing.Dict[str, typing.Union[numpy.ndarray, util.LabelledTensor]]


class DatasetSample(collections.namedtuple('_DatasetSample', ['inputs', 'outputs'])):
  """Represents one sample in a dataset.

  Attributes
  ----------
  inputs : typing.Dict[str, typing.Union[numpy.ndarray, util.LabelledTensor]]
      The model inputs for this sample. Keys are model parameter names and values are parameter
      value.
  outputs : typing.Dict[str, typing.Union[numpy.ndarray, util.LabelledTensor]]
      The model outputs for this sample. Keys are model output names and values are the expected
      value or label.
  """

  def __new__(cls,
              inputs : ModelParamsResults,
              outputs : ModelParamsResults):
    return super(cls, cls).__new__(cls, inputs, outputs)


class DatasetGenerator(metaclass=abc.ABCMeta):

  @classmethod
  def instantiate(cls, name : str, config : typing.Dict):
    """Return the dataset generator for this model given config.

    Params
    ------
    name: str
        Name of the module (in this package) which contains the dataset generator.

    config: Dict
        Arbitrary configuration for the generator.

    Returns
    -------
    DatasetGenerator :
        A subclass of this one.
    """
    mod = importlib.import_module(f'.{name}', __name__)
    for key in dir(mod):
      val = getattr(mod, key)
      try:
        is_subclass = issubclass(val, DatasetGenerator)
      except TypeError:
        continue

      if is_subclass and val != DatasetGenerator:
        return val(config)

    assert False, 'cannot find dataset generator'

  def __init__(self, config):
    self.config = config

  @abc.abstractmethod
  def generate(self, num_samples):
    """Generate samples from this dataset.

    Params
    ------
    num_samples : int
        The number of samples to generate.

    Returns
    -------
    List[DatasetSample]:
        The generated samples.
    """
    raise NotImplementedError()
