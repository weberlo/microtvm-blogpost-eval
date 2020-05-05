#!/bin/bash

cd ~/ws/tvm
pipenv shell 'source tools/py-shell.sh; cd ~/ws/stm-nucleo/microtvm-blogpost-eval; export PYTHONPATH=$(pwd)/python:$PYTHONPATH'
