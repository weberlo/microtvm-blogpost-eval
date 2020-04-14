#!/bin/bash

cd ~/ws/tvm
pipenv shell 'source tools/py-shell.sh; cd ~/ws/stm-nucleo/microtvm-blogpost-eval; export PYTHONPATH=$(pwd)/python:$PYTHONPATH; export CMSIS_PATH=$(pwd)/../CMSIS_5; export MICRO_GDB_INIT_DIR=$(pwd)/debug/micro'
