#!/bin/bash

cd ~/ws/tvm
pipenv shell 'source tools/py-shell.sh; cd ~/ws/stm-nucleo/microtvm-blogpost-eval; export PYTHONPATH=$(pwd)/python:$PYTHONPATH; export CMSIS_NN_PATH=$(pwd)/../CMSIS_5; export CMSIS_ST_PATH=$(pwd)/../STM32Cube_FW_F7_V1.16.0/Drivers/CMSIS; export MICRO_GDB_INIT_DIR=$(pwd)/debug/micro'
