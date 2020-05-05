<!-- - TODO add RPC auto-restart script to this repo and explain this snippet in `tune_relay_microtvm.py`
```python
RPC_SERVER_DEV_CONFIG_BASE = f'/home/lweber/micro-rpc-tempdirs'
for i in range(10):
    DEV_CONFIG['server_port'] = 6666 + i
    with open(f'{RPC_SERVER_DEV_CONFIG_BASE}/{i}/utvm_dev_config.json', 'w') as f:
        json.dump(DEV_CONFIG, f, indent=4)
``` -->

# Setup
- Currently, must use Linux for host OS.
- Must have TVM compiled and on your `PYTHONPATH`.
- Add `micro_eval` to your `PYTHONPATH` environment variable.

## Arm STM32
- Download [CMSIS](https://github.com/ARM-software/CMSIS_5).
  - Check out the hash `b5ef1c9be72f4263ca56e9cdd457e0bf4cb29775`, corresponding to version ???
  - Export an environment variable `CMSIS_NN_PATH=/path/to/CMSIS_5`.
- Download the [Arm embedded toolchain](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/gnu-rm/downloads) for your host machine.
- Install OpenOCD.
  - Easiest route is via your distro's package manager (e.g., `apt-get install openocd`).
  - Connect the board to your machine via USB-JTAG.
  - Create an OpenOCD config file `board.cfg` with the following contents:
  ```
  source [find interface/stlink-v2-1.cfg]
  source [find target/stm32f7x.cfg]
  ```
- Run `openocd -f /path/to/board.cfg`.
- You should first try to run the µTVM tests.
  - Navigate to your local TVM repo and find the file `tests/python/unittest/test_runtime_micro.py`.
  - Near the top, you will see an assignment `DEV_CONFIG = micro.device.host.generate_config()`.
  - Change the right-hand side to `micro.device.arm.stm32f746xx.generate_config('127.0.0.1', 6666)`.
    - The arguments here are the OpenOCD server address and the server port (which defaults to 6666).
  - Try running the tests with `python tests/python/unittest/test_runtime_micro.py`
  - It should fail at `test_multiple_sessions`, if you only have one device connected, because each session requires a separate physical device.

### (Optional) Multiple Boards
If you're tuning, it helps to have more than one board to test kernels on.
In order to have multiple boads connected and being driven by µTVM, you need to find each of their serial numbers, so they can be uniquely identified.
With only one board plugged in at a time, run the following bash command to find the current board's serial number:
```bash
  openocd -d3 \
    -f /usr/share/openocd/scripts/interface/stlink-v2-1.cfg \
    -f /usr/share/openocd/scripts/target/stm32f7x.cfg \
    -c 'hla_serial wrong_serial' \
    2>&1 \
    | grep 'Device serial number'
```
and you should see a line like the following:
```
Debug: 264 19 libusb1_common.c:65 string_descriptor_equal(): Device serial number '066EFF545057717867150931' doesn't match requested serial 'wrong_serial
```
In the case above, `066EFF545057717867150931` is the serial number.
You then need to create an OpenOCD config file that makes use of this serial number.
The reason why we need separate configs is so you can bind each openocd instance to different gdb, tcl, and telnet ports.
Make a config for each board.
Here's an example of configs for two different boards:
board0.cfg:
```
source [find interface/stlink-v2-1.cfg]
source [find target/stm32f7x.cfg]
hla_serial 066EFF545057717867150931

gdb_port 3333
tcl_port 6666
telnet_port 4444
```
board1.cfg:
```
source [find interface/stlink-v2-1.cfg]
source [find target/stm32f7x.cfg]
hla_serial 066BFF485157717867193328

gdb_port 3334
tcl_port 6667
telnet_port 4445
```
- You will need to launch an OpenOCD instance for each board.
  - For board 0, run `openocd -f /path/to/board0.cfg`, for board 1, run `openocd -f /path/to/board1.cfg` in a *separate* terminal, etc.

## RISC-V Spike
- During installation of each of the projects below, you may want to `git checkout` to a stable commit in them (e.g., the latest release in the repo's "releases" tab).  In my experience, the `master` branch doesn't have much vetting, in terms of bugs.

- Follow the instructions [here](https://github.com/riscv/riscv-gnu-toolchain) to set up and install the RISC-V GNU toolchain.
  - This repo's `master` branch doesn't seem particularly stable, so we'd recommend commit hash `8520fc0baeb6b5345349fbab2e3bc560e4e7f351` (tag: rvv-0.8), as earlier versions have caused us problems.
- Add the toolchain's install path to your shell's `PATH` environment variable.
- We need a special version of OpenOCD that has RISC-V support.
  - Follow the instructions [here](https://github.com/riscv/riscv-openocd) to set up and install a RISC-V-compatible version of OpenOCD.
  - **NOTE**: you need to pass the `--enable-remote-bitbang` flag to `./configure`, as this is the communication protocol we will use.
  - This repo's `master` branch doesn't seem particularly stable, so we'd recommend commit hash `1599853032c03d4cbd6eac1e6869ef750ec`, as earlier and later versions have caused us problems.
- Make sure the resulting OpenOCD binary is on your `PATH`.
- Export the `RISCV` environment variable as the directory where you would like Spike installed (e.g., `~/bin/risc-v/spike`).
- Follow the instructions [here](https://github.com/riscv/riscv-isa-sim) to set up and install Spike.
- Add `$RISCV/bin` to your `PATH`.
- Now follow the instructions in the ["Debugging with GDB" section](https://github.com/riscv/riscv-isa-sim#debugging-with-gdb) to ensure you have everything set up correctly.
  - If the text says `"Instruction sets want to be free!" at the end of the GDB session, then everything is working!
- TODO add spike.cfg file to repo
- TODO redo install
