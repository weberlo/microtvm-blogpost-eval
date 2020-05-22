"""Defines utility functions for logging."""

import datetime
import logging
import logging.handlers
import sys
import typing

import colorlog

from micro_eval import util


def gen_log_file_name(labels : typing.List[str]):
  """Generate a new log file name for this process.

  Params
  ------
  labels : List[str]
      A list of strings that describe the program being logged, ordered from least specific to
      most specific. These influence the log file name and/or directory hierarchy (if one becomes
      used later on).

  Returns
  -------
  str :
      The path to the new log file.
  """
  now = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
  safe_labels = (".".join(labels)).replace('/', '_').replace(':', '_')
  return f'{util.get_repo_root()}/logs/{safe_labels}.{now}.log'


def config(labels : typing.List[str], level : int = logging.INFO, console_only : bool=False):
  """Configure logging subsystem and create a new logfile; our logging.basicConfig().

  Params
  ------
  level : int
      Root log level, one of logging.{ERROR,WARN,INFO,DEBUG}.
  labels : List[str]
      A list of strings that describe the program being logged, ordered from least specific to
      most specific. These influence the log file name and/or directory hierarchy (if one becomes
      used later on).
  console_only : bool
      If True, don't log to disk. labels is ignored in this case.
  """
  root_logger = logging.getLogger()
  was_configured = root_logger.hasHandlers()

  shared_kw = dict(datefmt='%Y-%m-%d %H:%M:%S')

  # want basicConfig(force=True), but only in 3.8, so reproduced here.
  for h in root_logger.handlers:
    root_logger.removeHandler(h)
    h.close()

  stream_handler = logging.StreamHandler(sys.stderr)
  stream_handler.setFormatter(colorlog.ColoredFormatter(
    fmt='%(asctime)s.%(msecs)03d %(log_color)s%(levelname)s%(reset)s %(filename)s:%(lineno)d %(message)s',
    log_colors={
      'DEBUG': 'cyan',
      'INFO': 'green',
      'WARNING': 'yellow',
      'ERROR': 'red',
      'CRITICAL': 'red,bg_white',
    },
    reset=True,
    style='%',
    secondary_log_colors={
      'message': {
        'fail': 'red',
        'pass': 'green',
      },
    },
    **shared_kw))
  root_logger.addHandler(stream_handler)

  if not console_only:
    file_handler = logging.FileHandler(gen_log_file_name(labels))
    file_handler.setFormatter(logging.Formatter(
      fmt='%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d %(message)s',
      **shared_kw))
    root_logger.addHandler(file_handler)

  root_logger.setLevel(level)

  if was_configured:
    root_logger.warn('Log handlers were configured before log_util.config() was invoked; messages '
                     'prior to this line may be missing from this log, and previously-configured '
                     'logs may not contain messages beginning with this one.')
  root_logger.info('Logging started for process: %r', sys.argv)
