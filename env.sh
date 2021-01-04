#!/bin/bash
# Parse arguments given to the script
update_conform=false
while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Initializes the ConFormAgent environment."
      echo " "
      echo "Usage: source env.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show this help message"
      echo "-u, --update              update the anaconda 'conform' environment "
      return 0
      ;;
    -u|--update)
      shift
      update_conform=true
      ;;
    *)
      break
      ;;
  esac
done

export CONFORM_PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export GIT_REPO_ROOT=${CONFORM_PROJ_DIR}

echo "CONFORM_PROJ_DIR is "
echo ${CONFORM_PROJ_DIR}

# activate conform environment
conda activate conform

if [ "$update_conform" = true ]; then
    echo "Updating conda 'conform' environment"
    conda env update -f ${CONFORM_PROJ_DIR}/etc/configs/conda_conform_env.yaml --prune
fi

