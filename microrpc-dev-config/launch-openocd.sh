#!/bin/bash -e

cd $(dirname $0)

OPENOCD=$HOME/ws/openocd/prefix/bin/openocd
OPENOCD_PIDS=( )

function kill_child_processes() {
    PGID=$(ps -o pgid= $$ | grep -o [0-9]*)
    kill -- -$PGID
}

trap 'kill_child_processes' EXIT

python -m tvm.exec.rpc_tracker --host=127.0.0.1 --port=9190 &
OPENOCD_PIDS=( "${OPENOCD_PIDS[@]}" "$!" )

export CMSIS_ST_PATH=$(git rev-parse --show-toplevel)/3rdparty/STM32CubeF7/Drivers/CMSIS
export CMSIS_NN_PATH=$(git rev-parse --show-toplevel)/CMSIS_5

for d in `ls -1d *`; do
    if [ ! -d "$d" ]; then
        continue
    fi
    cd "$d"
    "$OPENOCD" -f openocd.cfg 2>&1 | grep -E '[ ]*((Info)|(Warn(ing)?)|(Error)|(Debug))[ ]*:' &
    OPENOCD_PIDS=( "${OPENOCD_PIDS[@]}" "$!" )

    port=$(expr 9091 + ${d#dev-} )

    python -m tvm.exec.rpc_server \
           --port $port \
           --tracker=127.0.0.1:9190 \
           --key=arm.stf32f746 \
           --utvm-dev-config=utvm-dev-config.json &
    OPENOCD_PIDS=( "${OPENOCD_PIDS[@]}" "$!" )
    cd ..
done

echo "Launched ${#OPENOCD_PIDS[@]} processes..."
while [ true ]; do
    read foo
done
