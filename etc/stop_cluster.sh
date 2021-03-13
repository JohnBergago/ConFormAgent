#!/bin/bash

# Parse arguments given to the script
yaml_file=
while test $# -gt 0; do
  echo $1
  case "$1" in
    -h|--help)
      echo "Stops a ray cluster configured in a YAML file."
      echo "Make sure you have sourced the env.sh file."
      echo "This will only work if the cluster was created using the start script."
      echo " "
      echo "Usage: ./stop_cluster.sh [options] config.yaml"
      echo " "
      echo "options:"
      echo "-h, --help                show this help message"
      echo " "
      echo "Arguments:"
      echo "config.yaml               cluster config file in YAML format"
      exit 0
      ;;
    *)
      yaml_file=$1
      shift
      ;;
  esac
done

python $CONFORM_PROJ_DIR/etc/kill_xvfb.py
ray exec $yaml_file.running "source ${CONFORM_PROJ_DIR}/env.sh &&
                     ray stop &&  
                     ray teardown ~/ray_bootstrap_config.yaml --yes"
rm -r $yaml_file.running