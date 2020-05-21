import json
import os
import typing

from . import get_repo_root


class Config(dict):
  """Extends dict to add path lookup capabilities."""

  @classmethod
  def load(cls, path : str):
    """Load JSON dict from path.

    Params
    ------
    path : str
        The path to the JSON file. The JSON file should contain a dict at the
        top-level.

    Returns
    -------
    Config :
        The loaded Config instance.
    """
    with open(path) as f:
      obj = json.load(f)

    if not isinstance(obj, dict):
      raise ValueError(f'Expected JSON file to contain a dict, got {obj!r}')

    c = cls(obj)
    c._path = path
    return c

  def relpath(self, key : typing.Union[str, int, bool]) -> str:
    return os.path.join(os.path.dirname(self._path), self[key])
