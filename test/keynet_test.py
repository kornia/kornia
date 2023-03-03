from pytest import raises

from kornia.feature.keynet import KeyNetConf, from_keynet_conf_dict, keynet_default_config, to_keynet_conf_dict


class TestKeynetConf:
    # To compare the actual and expected tensors use `self.assert_close(...)`

    def test_smoke(self, device, dtype):
        # test the function with different parameters arguments, to check if the function at least runs with all the
        # arguments allowed.
        keynet_conf = keynet_default_config
        assert keynet_conf.num_filters is not None
        keynet_conf_dict = to_keynet_conf_dict(keynet_default_config)
        assert type(keynet_conf_dict) == dict
        assert type(from_keynet_conf_dict(keynet_conf_dict)) == KeyNetConf

    def test_exception(self, device, dtype):
        # tests the exceptions which can occur on your function
        with raises(Exception) as errinfo:
            from_keynet_conf_dict(8)
        assert 'Input conf must be dict' in str(errinfo.value)

        with raises(Exception) as errinfo:
            to_keynet_conf_dict(8)
        assert 'Input conf must be KeyNetConf' in str(errinfo.value)

    def test_params(self, device, dtype):
        keynet_conf = keynet_default_config
        assert keynet_conf.num_filters == 8
        assert keynet_conf.num_levels == 3
        assert keynet_conf.kernel_size == 5
        assert keynet_conf.Detector_conf == {
            'nms_size': 15,
            'pyramid_levels': 4,
            'up_levels': 1,
            'scale_factor_levels': 1.4142135623730951,
            's_mult': 22.0,
        }
