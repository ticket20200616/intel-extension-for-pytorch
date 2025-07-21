#! /bin/bash
set -ux

# parent of intel-extension-for-pytorch directory
RD=`realpath ../..`

# change directory to /root
cd /root

# activate the conda environment
. ./miniforge3/bin/activate
. activate compile_py310

cd $RD

# repository dubious ownership warning workaround
chown -hR root .

python intel-extension-for-pytorch/scripts/compile_bundle.py \
    --incremental \
    --max-jobs 1 \
    --with-vision --with-audio --with-torch-ccl /opt/intel/oneapi
