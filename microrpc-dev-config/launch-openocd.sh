#!/bin/bash -e

cd $(dirname $0)

OPENOCD=$HOME/ws/openocd/prefix/bin/openocd
OPENOCD_PIDS=( )

function kill_openocds() {
    for p in "${OPENOCD_PIDS[@]}"; do
        kill $p || echo -n
    done
}

trap 'kill_openocds' EXIT

for d in `ls -1d *`; do
    if [ ! -d "$d" ]; then
        continue
    fi
    cd "$d"
    "$OPENOCD" -f openocd.cfg 2>&1 | grep -E '[ ]*((Info)|(Warn(ing)?)|(Error)|(Debug))[ ]*:' &
    OPENOCD_PIDS=( "${OPENOCD_PIDS[@]}" "$!" )
    cd ..
done

echo "Launched ${#OPENOCD_PIDS} processes..."
while [ true ]; do
    read foo
done
