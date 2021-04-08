#!/bin/bash
PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${PROJ_DIR}

if [[ ! -e '.git' ]]
then
    echo "No git repository found. Initializing file structure and checking out git submodules."
    git init .
    mkdir -p opt/
    cd opt/
    git submodule add https://github.com/JohnBergago/ConFormSim.git
fi

if [[ ! -e 'opt/ConFormSim' ]]
then
    echo "ConFormSim submodule missing. Adding it to the current git repo."
    cd opt/
    git submodule add https://github.com/JohnBergago/ConFormSim.git
    cd ${PROJ_DIR}
fi

git submodule update --init --recursive

# set up the ConFormSim folder
$PROJ_DIR/opt/ConFormSim/setup.sh

# intitialize the conda environment
echo "Remove old conform conda env."
conda env remove -n conform --yes
conda env create -f $PROJ_DIR/etc/configs/conda_conform_env.yaml
conda activate conform

# pip install -e $PROJ_DIR/

echo -e "# ConFormAgent is set up."
echo -e "# Now run "
echo -e "# \t$ source env.sh"
echo -e "# to intialize the environment."


