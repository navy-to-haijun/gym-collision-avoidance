#!/bin/bash
set -e

MAKE_VENV=${1:-true}
SOURCE_VENV=${2:-true}

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install this pkg and its requirements
python -m pip install -e $DIR

# Install RVO and its requirements
cd $DIR/gym_collision_avoidance/envs/policies/Python-RVO2
python -m pip install Cython
if [[ "$OSTYPE" == "darwin"* ]]; then
    export MACOSX_DEPLOYMENT_TARGET=10.15
    brew install cmake || true
fi
python setup.py build
python setup.py install

echo "Finished installing gym_collision_avoidance!"
