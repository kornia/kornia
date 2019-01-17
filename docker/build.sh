#!/bin/bash -ex
# This script is useful to build the dev_env container.
script_link="$( readlink "$BASH_SOURCE" )" || script_link="$BASH_SOURCE"
apparent_sdk_dir="${script_link%/*}"
if [ "$apparent_sdk_dir" == "$script_link" ]; then
  apparent_sdk_dir=.
fi

sdk_dir="$( command cd -P "$apparent_sdk_dir/../" > /dev/null && pwd -P )"

# This builds the docker file, from the root of the workspace
#docker build -t arraiy/torchgeometry -f Dockerfile $sdk_dir
docker build -t arraiy/torchgeometry -f Dockerfile.test $sdk_dir
