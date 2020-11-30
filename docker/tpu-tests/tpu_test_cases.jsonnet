local base = import 'templates/base.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import "templates/utils.libsonnet";

local tputests = base.BaseTest {
  frameworkPrefix: 'kornia',
  modelName: 'tpu-tests',
  mode: 'postsubmit',
  configMaps: [],

  timeout: 900, # 15 minutes, in seconds.

  image: std.extVar('image'),
  imageTag: std.extVar('image-tag'),

  tpuSettings+: {
    softwareVersion: 'pytorch-nightly',
  },
  accelerator: tpus.v3_8,

  command: utils.scriptCommand(
    |||
      python -c "import torch; print(torch.__version__)"
      python -c "import torch_xla; print(torch_xla.__version__)"
      python -c "import kornia; print(kornia.__version__)"
      pytest -v kornia/test/color kornia/test/enhance --device tpu --dtype float32 -k "not grad"
      test_exit_code=$?
      echo "\nFinished running commands.\n"
      test $test_exit_code -eq 0
    |||
  ),
};

tputests.oneshotJob
