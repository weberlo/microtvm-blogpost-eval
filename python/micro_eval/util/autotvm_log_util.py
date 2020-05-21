# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Defines functions for accessing autotvm logs."""

from micro_eval import util


def gen_cifar10_job_name(kernel_layouts):
  return f'arm.stm32f746xx.cifar10.{"-".join(kernel_layouts)}'


def get_tuning_log_root():
  """Return path to a directory under which all autotvm logs should live."""
  return f'{util.get_repo_root()}/autotune-logs'


def get_eval_tuning_log_root():
  """Return path to a directory under which eval-worthy autotvm logs should live.

  Logs in this directory can be used for evaluating model performance. Everything in this directory
  is expected to be checked in to git (unlike things in the jobs directory). An eval-worthy autotune
  log is one which has been processed after autotunign finishes and reduced to just the
  best-performing entry for each workload.
  """
  return f'{get_tuning_log_root()}/eval'


def get_default_log_path(job_name):
  """Return the default path for AutoTVM logs with the given job name.

  Params
  ------
  job_name : str
       A name which forms the prefix of the log file. This should identify
       the model and target on which it ran.

  Returns
  -------
  str
      The default path to this log. This is expected to be a symlink to the real
      log produced in tuning.
  """
  return f'{get_tuning_log_root()}/promoted/{job_name}.log'


def gen_tuning_log_path(job_name):
  """Generate a new path to an AutoTVM tuning log with the given job name.

  The AutoTVM tuning log contains an entry describing the best-found configuration
  for each tuned workload that has been tuned in the model.

  Params
  ------
  job_name : str
       A name which forms the prefix of the log file. This should identify
       the model and target on which it ran.

  Returns
  -------
  str
      The default path to this log. This is expected to be a symlink to the real
      log produced in tuning.
  """
  timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
  return f'{get_tuning_log_root()}/jobs/{job_name}.log.{timestamp}'


class EvalLogExistsError(Exception):
  """Raised when an eval tuning log already exists."""

  def __init__(self, eval_log_path):
    super(EvalLogExistsError, self).__init__(
      f"Refusing to overwrite already-existing eval log {eval_log_path}")
    self.eval_log_path = eval_log_path


def promote(job_name, tuning_log_path):
  """Promote the given tuning log to the default one used for that job.

  Updates the symlink at get_default_log_path().

  Params
  ------
  job_name : str
      A name which forms the prefix of the log file. This should identify
      the model and target on which it ran.

  tuning_log_path : str
      Path to the AutoTVM tuning log to become the new default log.
  """
  symlink_path = get_default_log_path(job_name)
  if os.path.lexists(symlink_path):
    assert os.path.islink(symlink_path), (
      f'Expected to find a symlink at {symlink_path}, found a regular file. Remove it and try '
      f'again')

    os.unlink(symlink_path)

  eval_log_path = os.path.join(get_eval_tuning_log_path(), os.path.basename(tuning_log_path))
  if os.path.exists(eval_log_path):
    raise EvalLogExistsError(eval_log_path)

  shutil.copy2(tuning_log_path, eval_log_path)
  os.symlink(os.path.relpath(tuning_log_path, os.path.dirname(symlink_path)), symlink_path)
