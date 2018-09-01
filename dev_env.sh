#!/bin/bash -ex
# The purpose of this script is simplify running scripts inside of our
# dev_env docker container.  It mounts the workspace and the
# workspace/../build directory inside of the container, and executes
# any arguments passed to the dev_env.sh
script_link="$( readlink "$BASH_SOURCE" )" || script_link="$BASH_SOURCE"
apparent_sdk_dir="${script_link%/*}"
if [ "$apparent_sdk_dir" == "$script_link" ]; then
  apparent_sdk_dir=.
fi
sdk_dir="$( command cd -P "$apparent_sdk_dir" > /dev/null && pwd -P )"

bash_args=$@
if [[ -z "$bash_args" ]] ; then
    bash_args=bash
fi

docker_data_volume="-v ${DATA_DIR}:${DATA_DIR}"
if [[ -z "$DATA_DIR" ]] ; then
    docker_data_volume="-v $(pwd):$(pwd)"
fi

docker_command=nvidia-docker

repo=${ARRAIY_REPO:=arraiy/torchgeometry}

$docker_command \
    run \
    -it \
    $docker_data_volume \
    -v $sdk_dir:/code/torchgeometry \
    -v $HOME/.Xauthority:/root/.Xauthority \
    --net=host \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --privileged \
    --device /dev/dri \
    $repo \
    bash -c "cd /code/torchgeometry && $bash_args"

