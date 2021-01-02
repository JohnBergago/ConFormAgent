#!/bin/bash


yaml_file=
export RAY_USER=$LOGNAME
# Parse arguments given to the script
while test $# -gt 0; do
  echo $1
  case "$1" in
    -h|--help)
      echo "Starts a ray cluster configured in a YAML file."
      echo "Make sure you have sourced the env.sh file."
      echo " "
      echo "Usage: ./start_cluster.sh [options] config.yaml"
      echo " "
      echo "options:"
      echo "-h, --help                show this help message"
      echo "--user                    the user for ssh access. Default will use LOGNAME from env" 
      echo " "
      echo "Arguments:"
      echo "config.yaml               cluster config file in YAML format"
      exit 0
      ;;
    --user)
      export RAY_USER=$2
      shift
      shift
      ;;
    *)
      yaml_file=$1
      shift
      ;;
  esac
done
echo "Ray user is: $RAY_USER"
cp $yaml_file $yaml_file.running
echo ${CONFORM_PROJ_DIR}
sed -i "s/ssh_user: user/ssh_user: ${RAY_USER}/" $yaml_file.running
sed -i "s+CONFORM_PROJ_DIR+${CONFORM_PROJ_DIR}+g" $yaml_file.running
ray up  --yes $yaml_file.running

