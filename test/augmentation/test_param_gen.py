import torch

from kornia.augmentation.param_gen import _random_prob_gen, _random_color_jitter_gen


class TestRandomProbGen:

    def test_random_prob_gen(self):
        torch.manual_seed(42)
        batch_size = 8

        halfs = _random_prob_gen(batch_size=batch_size, p=.5)
        expected_halfs = [False, False,  True, False,  True,  False,  True, False]
        zeros = _random_prob_gen(batch_size=batch_size, p=0.)['batch_prob']
        ones = _random_prob_gen(batch_size=batch_size, p=1.)['batch_prob']
        
        assert list(halfs.keys()) == ['batch_prob'], "Redundant keys found apart from `batch_prob`"
        assert (halfs['batch_prob'] == torch.tensor(expected_halfs)).long().sum() == batch_size
        assert (zeros == torch.tensor([False] * batch_size)).long().sum() == batch_size
        assert (ones == torch.tensor([True] * batch_size)).long().sum() == batch_size

    def test_color_jitter_gen(self):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = _random_color_jitter_gen(batch_size, brightness=0.2, contrast=0.3, saturation=0.4, hue=0.1)
        jitter_params_tuple = _random_color_jitter_gen(batch_size, brightness=(-0.2, 0.2),
            contrast=(0.7, 1.3), saturation=(0.6, 1.4), hue=(-0.1, 0.1))
        expected_jitter_params = {
            'brightness_factor': torch.tensor([
                0.15290771424770355, 0.1660015732049942, -0.046854496002197266, 0.18372227251529694,
                -0.043820708990097046, 0.04035814106464386, -0.09737100452184677, 0.11745654046535492]),
            'contrast_factor': torch.tensor([
                1.2644628286361694, 0.7799115180969238, 1.260758876800537, 1.056147813796997,
                1.2216426134109497, 1.0406291484832764, 1.1446564197540283, 0.957642674446106]),
            'hue_factor': torch.tensor([
                0.07708858698606491, 0.014780893921852112, -0.04668399319052696, 0.02548982948064804,
                -0.04607366397976875, -0.011727288365364075, -0.040615834295749664, 0.06633710116147995]),
            'saturation_factor': torch.tensor([
                0.6842519640922546, 0.8155958652496338, 0.8870501518249512, 0.7594910264015198,
                1.0377532243728638, 0.6049283742904663, 1.3612436056137085, 0.6602127552032471])
        }
        expected_jitter_params_tuple = {
            'brightness_factor': torch.tensor([
                0.15440548956394196, 0.03328382968902588, -0.06494089961051941, 0.12359000742435455,
                0.031170159578323364, 0.16159267723560333, 0.021863937377929688, -0.06307463347911835]),
            'contrast_factor': torch.tensor([
                1.0806050300598145, 0.9186461567878723, 1.1262571811676025, 1.2678465843200684,
                1.1734178066253662, 0.8688482046127319, 1.1731793880462646, 1.0536777973175049]),
            'hue_factor': torch.tensor([
                0.05078350752592087, -0.06095050647854805, -0.09899085015058517, -0.03863605484366417,
                -0.07670228183269501, 0.08205389231443405, 0.028803132474422455, 0.04142136126756668]),
            'saturation_factor': torch.tensor([
                1.1265044212341309, 0.9930416345596313, 1.3130433559417725, 0.715794563293457,
                1.025185465812683, 0.7269839644432068, 1.1233408451080322, 0.862247109413147])
        }

        assert set(list(jitter_params.keys())) == set(['brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor']), \
            "Redundant keys found apart from 'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor'"
        assert (jitter_params['brightness_factor'] == expected_jitter_params['brightness_factor']).long().sum() == batch_size
        assert (jitter_params['contrast_factor'] == expected_jitter_params['contrast_factor']).long().sum() == batch_size
        assert (jitter_params['hue_factor'] == expected_jitter_params['hue_factor']).long().sum() == batch_size
        assert (jitter_params['saturation_factor'] == expected_jitter_params['saturation_factor']).long().sum() == batch_size
        assert (jitter_params_tuple['brightness_factor'] == expected_jitter_params_tuple['brightness_factor']).long().sum() == batch_size
        assert (jitter_params_tuple['contrast_factor'] == expected_jitter_params_tuple['contrast_factor']).long().sum() == batch_size
        assert (jitter_params_tuple['hue_factor'] == expected_jitter_params_tuple['hue_factor']).long().sum() == batch_size
        assert (jitter_params_tuple['saturation_factor'] == expected_jitter_params_tuple['saturation_factor']).long().sum() == batch_size
