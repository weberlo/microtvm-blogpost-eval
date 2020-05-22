![microTVM logo](logo.png "MicroTVM CIFAR10-CNN Demo")

microTVM is an effort to run TVM on bare-metal microcontrollers. You can read more about the current
design in the original [microTVM RFC](https://github.com/apache/incubator-tvm/issues/2563). This repo
shows you how to run CIFAR10-CNN on the host machine and on an [STM Nucleo-F746ZG development board](
https://www.st.com/en/evaluation-tools/nucleo-f746zg.html).

## Hardware you will need

* Linux machine (OS X is also unofficially supported, and will be officially supported in the future)
* [STM Nucleo-F746ZG development board](https://www.st.com/en/evaluation-tools/nucleo-f746zg.html)
** Autotuning can be sped up by adding more of these development boards.
* micro USB cable

## Getting Started

1. Clone this repository (use `git clone --recursive` to clone submodules).
2. [Install TVM](https://docs.tvm.ai/install/from_source.html).
 * __NOTE__: Ensure you enable microTVM by setting `set(USE_MICRO ON)` in `build/config.cmake`.
 * __NOTE__: Ensure you have `0884659eb8c5fe51cc4cac9f2f8b6400f47fdee6` plus
   [PR 5648](https://github.com/apache/incubator-tvm/pull/5648).

3. Build OpenOCD. Here we have chosen a specific commit that works with the Nucleo board, but

    $ tools/patch-openocd.sh  # If using clang, fix a compiler error.
    $ cd 3rdparty/openocd
    # First, install dependencies listed in the README for your platform.
    $ ./bootstrap
    $ ./configure --prefix=$(pwd)/prefix
    $ make && make install

4. Install prerequisites:
    $ apt-get install gcc-arm-none-eabi

5. Configure hardware and external binaries.

    Copy `env-config.json.template` to `env-config.json` and modify the values called out in
    that file.

6. Setup virtualenv. Use `requirements.txt` and `constraints.txt`.

    $ python -mvenv _venv
    $ _venv/bin/activate
    $ pip install -r requirements.txt -c constraints.txt
    $ export PYTHONPATH=$(pwd)/python:$PYTHONPATH

## Run untuned models

You can use `python -m micro_eval.bin.eval` to evaluate models on the host or device. The simplest
invocation is as follows:

    $ python -m micro_eval.bin.eval cifar10_cnn:interp

Try it now. You should see the results of running 10 iterations of model evaluation through the TVM
host interpreter:

    INFO eval.py:108 got result: {'label': array([-33,   1, -38, -36, -43, -66, -38, -58, -44,  -6], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([ -70,  -46,   -4,  -89,  -37,  -63, -110, -108,   17,   -6], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([  9,   7,   4, -41, -46, -51, -49, -19, -45, -26], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([  1,  -6,   8, -21, -25,  -5, -20,  -3,  -4,   1], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([ -2,   5,   5,  -5, -21, -11, -10,  -5,  -9, -10], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([  1,  -4,   3, -14, -14,   1, -19,  -9,   0, -13], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([ 14, -23,  30, -28, -33,  -1, -46, -36, -33, -45], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([-10, -19,  -4,   5,  -5,  -5, -19, -29, -18, -21], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([-49, -28,  28,  -2, -18, -58, -48, -73, -32,   5], dtype=int8)}
    INFO eval.py:108 got result: {'label': array([ 15,  10,   0, -17, -42, -25, -23, -39, -44,   0], dtype=int8)}

You can use the TVM interpreter as a source of correctness for checking other invocations of the
same model. Try checking against the x86 runtime:

    $ python -m micro_eval.bin.eval cifar10_cnn:cpu --validate-against=cifar10_cnn:interp
    INFO eval.py:370 model_name  setting config     1    2    3    4    5    6    7    8    9
    ERROR eval.py:376 cifar10_cnn cpu             +005 -016 -027 -011 +059 -033 -032 -033 -015
    ERROR eval.py:376 cifar10_cnn interp          -030 +023 -027 -010 -003 +010 +013 -023 +023
    ...

This didn't work, because the cifar10_cnn model uses random parameters by default. Model configuration
can be passed as JSON. Now try using the configuration for validating output, which loads specific
parameters:

    $ python -m micro_eval.bin.eval cifar10_cnn:cpu:data/ --validate-against=cifar10_cnn:interp
    INFO eval.py:370 model_name  setting config                                1    2    3    4    5    6    7    8    9
    INFO eval.py:376 cifar10_cnn interp  data/cifar10-config-validate.json  -021 +017 +022 +018 +004 +007 -013 -008 -013
    INFO eval.py:376 cifar10_cnn cpu     data/cifar10-config-validate.json  -021 +017 +022 +018 +004 +007 -013 -008 -013

Now try running on the attached STM-Nucleo board:

    $ python -m micro_eval.bin.eval cifar10_cnn:micro_dev
    INFO eval.py:184 got prediction after 293.912 ms: {'label': array([  1, -16,  -2, -13,  -8,  17, -10, -15, -17, -14], dtype=int8)}
    ...

See if you can verify the output against the TVM interpreter runtime.

## Using tuned models

The previous result didn't run very quickly--__294__ ms vs __100__ ms in the equivalent CMSIS model.
TVM can improve the runtime by automatically reorganizing and optimizing the computation, a process
known as _autotuning_. First, let's see the result:

    $ python -m micro_eval.bin.eval cifar10_cnn:micro_dev --use-tuned-schedule
    INFO eval.py:184 got prediction after 157.354 ms: {'label': array([ -32,  -58,  -50,  -48,  103,   36, -110,   54,   66,   94], dtype=int8)}

This gave us about a 47% improvement in runtime.

## Running a CMSIS model

You can run and time CMSIS-NN's version of this CIFAR10-CNN network:

    $ python -m micro_eval.bin.eval cmsis_cifar10_cnn:micro_dev
    INFO eval.py:189 got prediction after 105.403 ms: {'label': array([-27,  11, -14, -16, -35, -35, -35, -11,  27, 127], dtype=int8)}
    ...

There are a couple of points to note about this runtime:
1. It uses ARM's quantization scheme, which isn't TFLite-compatible.
2. The output from this model will differ from the TVM-generated model because they aren't quite
   identical. See our blog post for more on this.

You can also run and time a CMSIS-NN model using TFLite-compatible quantization. While this model
doesn't output accurate results, it runs the same operations and differs only in weights, so the
runtime should roughly match what you would expect running CMSIS-CNN as a TFLite model:

    $ python -m micro_eval.bin.eval cmsis_cifar10_cnn:micro_dev:data/cmsis-config-tflite.json
    INFO eval.py:189 got prediction after 135.856 ms: {'label': array([ -44, -128, -128,  127, -128, -128, -128, -128,  113,  127], dtype=int8)}
    ...

## Running autotuning

__NOTE__: Autotuning takes a while on hardware--the above result was achieved by running autotuning for
hundreds of iterations. However, you can still try it and see some speedup without waiting as long.
Autotuning is a parallel process and can be sped up by adding more development boards.

    $ python -m micro_eval.bin.autotune cifar10_cnn:micro_dev tune --num-iterations 20

When the script exits successfully, you'll notice that
`autotune-logs/eval/cifar10_cnn-micro_dev-HWOI-HWOI-HWOI.log` has been changed. This is a symlink
that points to the default tuned schedule when `--use-tuned-schedule` is passed. A new file now
also exists in its directory--this is the _autotuning log_. It contains the fastest configurations
found during autotuning for each tuned task in the graph.

## Debugging

There are a lot of moving pieces here and it's easy for the system to fail. Here I've tried to document
some of the problems you can run into, and how to solve them.


### OpenOCD doesn't connect

1. Make sure the board is plugged in with a data cable :)
2. Double-check `hla_serial` in `env-config.json` (an empty string matches all boards).
3. Try running openocd separately:
    1. First, generate the _transport config_:

        $ python -m micro_eval.bin.autotune cifar10_cnn:micro_dev rpc_dev_config

    2. Now, try launching openocd:

        $ 3rdparty/openocd/prefix/bin/openocd -f microrpc-dev-config/dev-0/openocd.cfg

    Troubleshoot this until it connects to the device succesfully.

### Autotuing: cannot connect to tracker or other openocd issues

1. Double check a tracker/rpc server is not still running:

    $ ps ax | grep tvm.exec.rpc_tracker
    $ ps ax | grep tvm.exec.rpc_server

    Kill them if so.

## Running the tracker/server/openocd separately:

Sometimes openocd, the tracker, or the RPC server need to be launched separately. A script
`microrpc-dev-config/launch-openocds.sh` shows you how to do this, but you can also launch
them yourself. Here is how to use that script with autotuing and eval:

    # adjust task-index or omit for evaluating the whole model
    $ python -m micro_eval.bin.autotune cifar10_cnn:micro_dev rpc_dev_config --task-index=2

    # In another terminal:
    $ cd microrpc-dev-config
    $ ./launch-openocds.sh

    # In original terminal
    $ python -m micro_eval.bin.autotune  --pre-launched-tracker-hostport 127.0.0.1:9190 \
        --single-task-index=2...
    $ python -m micro_eval.bin.eval --openocd-server-hostport 127.0.0.1:6666 ...

## Stuck/misbehaving board

While rare with this demo, there's currently no protection included if the STM CPU enters
an exception handler. In this case, it might not be possible for the debugger to reset the
pending exception, and execution might start to produce obviously-wrong or unpredictable
results. You're more likely to encounter this if you experiment with the model. Hard-reset
the board or power it off in this case. We're working to address this issue in a future
PR.
