# The purpose of this script is simplify running scripts inside of our
# dev_env docker container.  It mounts the workspace and the
# workspace/../build directory inside of the container, and executes
# any arguments passed to the dev_env.sh
script_link="$( readlink "$BASH_SOURCE" )" || script_link="$BASH_SOURCE"
apparent_sdk_dir="${script_link%/*}"
if [ "$apparent_sdk_dir" = "$script_link" ]; then
  apparent_sdk_dir=.
fi
sdk_dir="$(cd -P "$apparent_sdk_dir" > /dev/null && pwd -P )"
if [ ! -e $sdk_dir/.dev_env/bin/conda ]; then
    $sdk_dir/setup_dev_env.sh	      
fi	

cmd="source $sdk_dir/.dev_env/bin/activate $sdk_dir/.dev_env/envs/venv"
if [ -z $CI ]; then
    eval $cmd
else
    echo $cmd >> "$BASH_ENV"
fi

export PYTHONPATH=$PYTHONPATH:$sdk_dir
