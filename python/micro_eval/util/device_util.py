"""Defines utility functions for connecting to attached devices."""

import atexit
import contextlib
import enum
import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import typing

from micro_eval import util
from . import config_util


# When True, print RPC server output to stdout.
DEBUG_RPC_SERVER_OUTPUT = True


class MonitorExited(Exception):
  """Raised when the monitor thread exits."""


class ProcessNeverHealthy(Exception):
  """Raised when the process never emits a log line marking it healthy."""


class ManagedSubprocess:
  """Wraps a managed subprocess."""

  LIVE_INSTANCES = []
  LIVE_INSTANCES_GUARD = threading.RLock()

  @classmethod
  def kill_all_at_exit(cls):
    while True:
      with cls.LIVE_INSTANCES_GUARD:
        if not cls.LIVE_INSTANCES:
          break

        to_kill = cls.LIVE_INSTANCES.pop()

      to_kill._internal_stop(True)

  class LogEventType(enum.Enum):
    ADJUST_LOGLEVEL = 'adjust_loglevel'
    HEALTHY = 'healthy'
    UNHEALTHY = 'unhealthy'

  class _HealthyQueueEntry(enum.Enum):
    HEALTHY = 'healthy'
    MONITOR_EXIT = 'monitor_exit'

  class Stream(enum.Enum):
    STDOUT = 'stdout'
    STDERR = 'stderr'

  def __init__(self, name : str, args : typing.List[str], cwd : str,
               log_file_path : typing.Optional[str] = None,
               log_event_emitter : typing.Callable[[str], typing.Iterator[typing.Tuple[LogEventType, dict]]] = None,
               healthy_timeout_sec : typing.Optional[float] = None,
               stop_signal=signal.SIGINT,
               stop_timeout_sec : float = 5.0):
    """Configure, but don't start, a new managed subprocess.

    Params
    ------
    name : str
        A short name that describes the process in logs.

    args : List[str]
        List of string arguments to the subprocess.

    cwd : str
        Working directory for the subprocess.

    log_file_path : str or None
        If specified, path to a file to which stdout/stderr are written.

    log_event_emitter : Callable[[Stream, str], Iterator[(LogEventType, dict)]
        If specified, a function called for every log line read. The function receives the stream identifier
        and log line and should yield a tuple of (LogEventType, data) for each event that occurred due to the
        content of the log line. See LogEventType for a description of the events and associated data.

    healthy_timeout_sec : float or None
        If specified, block_until_healthy waits by default for this amount of time.

    stop_signal : signal.*
        A signal sent to the process to cleanly stop it. The process is expected to exit within
        stop_timeout_sec. If it does not, it will be sent the OS kill signal.

    stop_timeout_sec : float
        Timeout in seconds between when the stop signal is sent and when a follow-on kill signal may
        be sent. If the process exits before this timeout, no follow-on kill signal is sent.
    """
    self._log = logging.getLogger(f'{__name__}.{name}')
    self._name = name
    self._args = args
    self._cwd = cwd
    self._log_file_path = log_file_path
    self._stop_signal = stop_signal
    self._stop_timeout_sec = stop_timeout_sec
    self._log_event_emitter = log_event_emitter
    self._healthy_timeout_sec = healthy_timeout_sec
    self._healthy_queue = queue.Queue(maxsize=4)  # 2 events from each stderr/stdout.
    self._proc = None
    self._log_workers = []

  def _process_log_events(self, line, log_level, this_log_level, tags):
    state_transition = None
    gen = self._log_event_emitter(line)
    while True:
      try:
        event_type, event_data = next(gen)
      except StopIteration:
        break
      except Exception:
        self._log.error('%s: log event emitter raised exception', self._name, exc_info=True)

      if event_type == self.LogEventType.ADJUST_LOGLEVEL:
        this_log_level = event_data
      elif event_type in (self.LogEventType.HEALTHY,
                          self.LogEventType.UNHEALTHY):
        assert state_transition is None
        state_transition = event_type
        tags.append(event_type.value)
        log_level = (logging.DEBUG
                     if event_type == self.LogEventType.HEALTHY
                     else logging.WARN)
        if event_type == self.LogEventType.UNHEALTHY:
          this_log_level = log_level

    return log_level, this_log_level, state_transition


  def _log_worker(self, in_from_proc, out_to_logfile=None):
    has_ever_been_healthy = False
    log_level = logging.INFO  # Start at INFO until HEALTHY received.
    try:
      for line in in_from_proc:
        this_log_level = log_level
        tags = ['stdio']
        state_transition = None

        line = str(line, 'utf-8', errors='replace').rstrip('\r\n')
        if self._log_event_emitter:
          log_level, this_log_level, state_transition = self._process_log_events(
            line, log_level, this_log_level, tags)

        self._log.log(this_log_level, '%s: %s %s',
                      self._name, ' '.join('[{}]'.format(t) for t in tags), line)
        if out_to_logfile is not None:
          out_to_logfile.write(line)

        if state_transition == self.LogEventType.HEALTHY:
          self._log.info('%s: preceding log line marked the process healthy; log level is now DEBUG',
                         self._name)
          if not has_ever_been_healthy:
            self._healthy_queue.put(self._HealthyQueueEntry.HEALTHY)
            has_ever_been_healthy = True

        elif state_transition == self.LogEventType.UNHEALTHY:
          self._log.info('%s: preceding log line marked the process unhealthy; log level is now WARN')

    except Exception as e:
      self._log.error('%s: monitor thread caught exception', self._name, exc_info=True)

    finally:
      try:
        in_from_proc.close()
      except:
        pass

      if out_to_logfile is not None:
        try:
          out_to_logfile.close()
        except:
          pass

      invoke_stop = False
      with self.LIVE_INSTANCES_GUARD:
        if self in self.LIVE_INSTANCES:
          invoke_stop = True
          self.LIVE_INSTANCES.remove(self)

      if invoke_stop:
        self._log.error('%s: monitor thread exiting while process still alive; killing process',
                        self._name)
        self._internal_stop(True)

      try:
        self._healthy_queue.put_nowait(self._HealthyQueueEntry.MONITOR_EXIT)
      except queue.Full:
        items = []
        try:
          while True:
            items.append(self._healthy_queue.get_nowait())
        except queue.Empty:
          pass
        logging.error('%s: healthy_queue unexpectedly full: %r', self._name, items)

  def start(self, block_until_healthy : typing.Optional[bool]=None):
    """Start the managed subprocess.

    Params
    ------
    block_until_healthy : bool
        If specified, controls whether this call blocks until the subprocess is either
        healthy or dies. If not specified, defaults to True if log_event_emitter is given,
        or False if it is not.
    """
    if block_until_healthy is None:
      block_until_healthy = self._log_event_emitter is not None

    out_to_logfile = None
    if self._log_file_path is not None:
      out_to_logfile = open(self._log_file_path, 'w')

    self._proc = subprocess.Popen(
      self._args, cwd=self._cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    with self.LIVE_INSTANCES_GUARD:
      self.LIVE_INSTANCES.append(self)

    self._log_workers.append(
      threading.Thread(target=self._log_worker, args=(self._proc.stdout, out_to_logfile),
                       name=f'{self._name}.stdout', daemon=True))
    self._log_workers.append(
      threading.Thread(target=self._log_worker, args=(self._proc.stderr, out_to_logfile),
                       name=f'{self._name}.stderr', daemon=True))
    for w in self._log_workers:
      w.start()

    if block_until_healthy:
      self.block_until_healthy(self._healthy_timeout_sec)

  def block_until_healthy(self, timeout_sec=None):
    if timeout_sec is None:
      timeout_sec = self._healthy_timeout_sec

    try:
      event = self._healthy_queue.get(timeout=timeout_sec)
    except queue.Empty:
      raise ProcessNeverHealthy(
        f'{self._name}: did not become healthy within {timeout_sec} sec') from None

    if event == self._HealthyQueueEntry.HEALTHY:
      return
    elif event == self._HealthyQueueEntry.MONITOR_EXIT:
      raise MonitorExited(f'{self._name}: monitor exited, likely due to process death')

  def stop(self, block_until_stopped=True):
    """Stop the managed subprocess."""
    with self.LIVE_INSTANCES_GUARD:
      if self not in self.LIVE_INSTANCES:
        return

      self.LIVE_INSTANCES.remove(self)

    self._internal_stop(block_until_stopped)

  def _internal_stop(self, block_until_stopped):
    self._log.info('%s: stopping with signal: %s', self._name, self._stop_signal)
    try:
      self._proc.send_signal(self._stop_signal)
    except subprocess.ProcessLookupError:
      pass
    if block_until_stopped:
      self.block_until_stopped()

  def block_until_stopped(self):
    try:
      self._proc.wait(self._stop_timeout_sec)
    except subprocess.TimeoutExpired:
      self._log.error('%s: did not stop after %d seconds; killing',
                      self._name, self._stop_timeout_sec)
      self._proc.kill()
      self._proc.wait(1.0)

    if threading.current_thread() not in self._log_workers:
      for w in self._log_workers:
        w.join()


atexit.register(ManagedSubprocess.kill_all_at_exit)


class DeviceTransportLauncher:
  """Launches and manages subprocesses that form the transport between TVM and μC.

  Specifically, these subprocesses currently include:
   - OpenOCD
   - TVM tvm.exec.rpc_server
   - TVM tracker

  Each of these pieces requires configuration tailored to the target and/or workload
  being executed. The environment_config and runtime_options parameterize that config.

  Template OpenOCD Config
  -----------------------
  The user needs to provide a template openocd.cfg, which is interpolated using Python
  str.format() against the environment_config['hardware'] sub-dict for that target.
  Additionally, these special keys can be used:
    - openocd_tcl_port: Port for the TCL server to listen on.
  """

  DEFAULT_ENVIRONMENT_CONFIG_PATH = f'{util.get_repo_root()}/env-config.json'

  def __init__(self, runtime_options, environment_config_path=DEFAULT_ENVIRONMENT_CONFIG_PATH):
    """Begin managing devices.

    Params
    ------
    runtime_options : dict
        Describes the subprocesses needed for this specific script. During autotuning,
        a full RPC tracker plus per-device TVM RPC server and OpenOCD instance may be
        needed. During evaluation, only OpenOCD may be needed.

        The format is as follows:
            {"num_instances": int,  # Number of device instances to launch; 0 for max.
             "hla_serials": ["serial1", "serial2", ...],  # Specific serial numbers to use.
             "use_tracker": bool  # True if tracker and RPC server should be started.
            }

    environment_config_path : str
        Path to a JSON file describing the attached hardware and other runtime related things like
        the location of openocd binaries and config templates. The file format is as follows:
            {"hardware": [
                 {"openocd_cfg_template": "relative/path/to/openocd_cfg_template",
                  "hla_serial": "<serial from lsusb>",  # specific to STM32 boards.
                  "additional_template_param": "value"},
                 {"openocd_cfg_template": "relative/path/to/openocd_cfg_template",
                  ...
                 },
             ],
             "openocd_bin_path": "relative/path/to/openocd",
             "openocd_base_port": 6666,   # Base port for launched OpenOCDs
             "rpc_server_key": "rpc_server_key",
             "tracker_port": 9190,
             "work_dirtree_root": "relative/path/to/workdir/root"
            }
    """
    self.environment_config = config_util.Config.load(environment_config_path)
    self.runtime_options = runtime_options

    self._environment_config_dir = os.path.dirname(environment_config_path)
    self._is_active = False
    self._openocds = None
    self._tvm_rpc_servers = None
    self._tvm_tracker = None

  _TRACKER_LOG_LINE_RE = re.compile('^((ERROR)|(WARN)(ING)?):.*')

  @property
  def work_dirtree_root(self):
    return self.environment_config.relpath('work_dirtree_root')

  @classmethod
  def _tracker_log_event_emitter(cls, line):
    if line.startswith('INFO:bind to '):
      yield ManagedSubprocess.LogEventType.HEALTHY, None

    m = cls._TRACKER_LOG_LINE_RE.match(line)
    if m:
      yield ManagedSubprocess.LogEventType.ADJUST_LOGLEVEL, getattr(logging, m.group(1))

  @property
  def _instance_configs(self):
    instance_configs = self.environment_config['hardware']
    if self.runtime_options.get('hla_serials'):
      configs_by_hla_serial = {h['hla_serial']: h for h in instance_configs}
      instance_configs = [configs_by_hla_serial[s] for s in self.runtime_options['hla_serials']]

    num_instances = self.runtime_options.get('num_instances', len(instance_configs))
    assert len(instance_configs) >= num_instances, (
      f'Not enough hardware or hla_serials provided: want {num_instances}, '
      f'have {len(instance_configs)}')

    return instance_configs[:num_instances]

  def generate_openocd_configs(self):
    with open(self.environment_config['openocd_cfg_template']) as template_f:
      config_template = template_f.read()

    base_port = int(self.environment_config['openocd_tcl_base_port'])
    for n, hardware in enumerate(self._instance_configs):
      config_path = f'{self.work_dirtree_root}/{n}/openocd.cfg'
      config = config_template.format(**hardware, openocd_tcl_port=base_port + n)
      config_dir = os.path.dirname(config_path)
      if not os.path.exists(config_dir):
        os.makedirs(config_dir)

      with open(config_path, 'w') as config_f:
        config_f.write(config)

  def generate_rpc_server_configs(self,
                                  generate_config_func : typing.Callable,
                                  generate_config_kw : dict = None):
    for n in range(len(self._instance_configs)):
      config_path = f'{self.work_dirtree_root}/{n}/utvm-dev-config.json'
      openocd_host, openocd_port = self.openocd_host_port_tuple(n)
      config = generate_config_func(openocd_host, openocd_port, **generate_config_kw)
      config_dir = os.path.dirname(config_path)
      if not os.path.exists(config_dir):
        os.makedirs(config_dir)

      with open(config_path, 'w') as config_f:
        json.dump(config, config_f)

  @property
  def tracker_host_port_tuple(self):
    return '127.0.0.1', int(self.environment_config.get('tracker_port', 9190))

  def openocd_host_port_tuple(self, index):
    return '127.0.0.1', int(self.environment_config.get('openocd_base_port', 6666)) + index

  def _launch_tracker_and_rpc_servers(self):
    tracker_host, tracker_port = self.tracker_host_port_tuple()
    self._tvm_tracker = ManagedSubprocess(
      'tracker',
      [sys.executable, '-m', 'tvm.exec.rpc_tracker',
       '--host', tracker_host, '--port', str(tracker_port)],
      cwd=self.work_dirtree_root,
      log_event_emitter=self._tracker_log_event_emitter,
      healthy_timeout_sec=5.0)
    self._tvm_tracker.start(block_until_healthy=True)

    self._tvm_rpc_servers = []
    for n in range(len(self._instance_configs)):
      rpc_server = ManagedSubprocess(
        f'rpc_server.{n}',
        [sys.executable, '-m', 'tvm.exec.rpc_server',
         f'--tracker={tracker_host}:{tracker_port}',
         f'--key={self.environment_config["rpc_server_key"]}',
         f'--utvm-dev-config={config_path}'],
        cwd=f'{config_dir}',
        log_event_emitter=self._tracker_log_event_emitter,
        healthy_timeout=5.0)
      rpc_server.start(block_until_healthy=False)
      self._tvm_rpc_servers.append(rpc_server)

    for r in self._tvm_rpc_servers:
      r.block_until_healthy()

  LISTEN_RE = re.compile('Info : Listening on port [0-9]+ for tcl connections')

  @classmethod
  def _openocd_event_emitter(cls, line):
    if cls.LISTEN_RE.match(line):
      yield ManagedSubprocess.LogEventType.HEALTHY, None

    if ' : ' in line:
      level = line.split(' : ', 1)[0]
      if level in ('Error', 'Warn'):
        yield ManagedSubprocess.LogEventType.ADJUST_LOGLEVEL, getattr(logging, level.upper())

  @contextlib.contextmanager
  def launch(self, generate_config_func : typing.Callable, generate_config_kw : dict = None,
             generate_config : bool = True):
    """Returns a context manager that launches the transport and kills it on exit.

    Params
    ------
    generate_config_func : Callable[dict]
        Called to generate the uTVM device config. Called with two arguments
        (openocd_host, openocd_port) plus generate_config_kw. Expected to be roughly
        tvm.micro.device.stm32f746xx.generate_config.

    generate_config_kw : dict
        Additional kwargs passed to generate_config_func. Useful for passing section_constraints to
        device config generators.

    generate_config : bool
        If True, write configuration before launching subprocessings. If False, skip this (the user
        is assumed to have already generated the configuration in the expected locatin.
    """
    openocd_host = '127.0.0.1'
    if generate_config:
      self.generate_openocd_configs()

    self._openocds = []
    for n, hardware in enumerate(self._instance_configs):
      openocd = ManagedSubprocess(
        f'openocd.{n}',
        [self.environment_config.relpath('openocd_bin_path'), '-f', 'openocd.cfg'],
        cwd=f'{self.work_dirtree_root}/{n}',
        log_event_emitter=self._openocd_event_emitter,
        healthy_timeout_sec=hardware['connect_timeout_sec'])
      openocd.start(block_until_healthy=False)
      self._openocds.append(openocd)

    for o in self._openocds:
      o.block_until_healthy(hardware['connect_timeout_sec'])

    if self.runtime_options['use_tracker']:
      if generate_config:
        self.generate_rpc_server_configs(generate_config_func, generate_config_kw)

      self._launch_tracker_and_rpc_servers()

    # Yield to contextlib.
    exc = yield

    for o in self._openocds:
      o.stop(block_until_stopped=False)

    if self.runtime_options['use_tracker']:
      for r in self._tvm_rpc_servers:
        r.stop(block_until_stopped=False)

      for r in self._tvm_rpc_servers:
        r.block_until_stopped()

      self.tracker.stop()

    for o in self._openocds:
      o.block_until_stopped()
