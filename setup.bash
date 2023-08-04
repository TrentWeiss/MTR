#!/bin/bash
scriptpath=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
export PYTHONPATH=${scriptpath}:${PYTHONPATH}
pyversion=$(python3 --version | grep -E "3.([0-9])*" -o | tr -d ".")
extraldpaths=${scriptpath}/build/lib.linux-x86_64-cpython-${pyversion}/mtr/ops/attention:${scriptpath}/build/lib.linux-x86_64-cpython-${pyversion}/mtr/ops/knn
if [ -z ${LD_LIBRARY_PATH} ]
then
    export LD_LIBRARY_PATH=${extraldpaths}
else
    export LD_LIBRARY_PATH=${extraldpaths}:${LD_LIBRARY_PATH}
fi