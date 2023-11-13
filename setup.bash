#!/bin/bash
source ~/cuda11_init
scriptpath=$(cd $(dirname "${BASH_SOURCE:-$0}") && pwd)
export PYTHONPATH=${scriptpath}:${PYTHONPATH}
pyversion=$(python3 -c "import sys; print(str(sys.version_info.major)+str(sys.version_info.minor))")
extraldpaths=${scriptpath}/build/lib.linux-x86_64-cpython-${pyversion}/mtr/ops/attention:${scriptpath}/build/lib.linux-x86_64-cpython-${pyversion}/mtr/ops/knn
if [ -z ${LD_LIBRARY_PATH} ]
then
    export LD_LIBRARY_PATH=${extraldpaths}
else
    export LD_LIBRARY_PATH=${extraldpaths}:${LD_LIBRARY_PATH}
fi
export TF_CPP_MIN_LOG_LEVEL=2