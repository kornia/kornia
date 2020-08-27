import torch

from kornia.augmentation.random_generator import random_prob_generator, random_color_jitter_generator


class TestRandomProbGen:

    def test_random_prob_gen(self):
        torch.manual_seed(42)
        batch_size = 8

        halfs = random_prob_generator(batch_size=batch_size, p=.5)
        expected_halfs = [False, False, True, False, True, False, True, False]
        zeros = random_prob_generator(batch_size=batch_size, p=0.)['batch_prob']
        ones = random_prob_generator(batch_size=batch_size, p=1.)['batch_prob']

        assert list(halfs.keys()) == ['batch_prob'], "Redundant keys found apart from `batch_prob`"
        assert (halfs['batch_prob'] == torch.tensor(expected_halfs)).long().sum() == batch_size
        assert (zeros == torch.tensor([False] * batch_size)).long().sum() == batch_size
        assert (ones == torch.tensor([True] * batch_size)).long().sum() == batch_size

    def test_random_prob_gen_same_on_batch(self):
        batch_size = 8

        torch.manual_seed(42)
        falses = random_prob_generator(batch_size=batch_size, p=.5, same_on_batch=True)['batch_prob']
        assert (falses == torch.tensor([False] * batch_size)).long().sum() == batch_size

        torch.manual_seed(0)
        trues = random_prob_generator(batch_size=batch_size, p=.5, same_on_batch=True)['batch_prob']
        assert (trues == torch.tensor([True] * batch_size)).long().sum() == batch_size


class TestColorJitterGen:

    def test_color_jitter_gen(self):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2]), contrast=torch.tensor([0.7, 1.3]),
            saturation=torch.tensor([0.6, 1.4]), hue=torch.tensor([-0.1, 0.1]))
        expected_jitter_params = {
            'brightness_factor': torch.tensor([
                1.15290772914886474609375, 1.16600155830383300781250, 0.95314550399780273437500,
                1.18372225761413574218750, 0.956179320812225341796875, 1.04035818576812744140625,
                0.90262901782989501953125, 1.11745655536651611328125]),
            'contrast_factor': torch.tensor([
                1.2644628286361694, 0.7799115180969238, 1.260758876800537, 1.056147813796997,
                1.2216426134109497, 1.0406291484832764, 1.1446564197540283, 0.957642674446106]),
            'hue_factor': torch.tensor([
                0.07708858698606491, 0.014780893921852112, -0.04668399319052696, 0.02548982948064804,
                -0.04607366397976875, -0.011727288365364075, -0.040615834295749664, 0.06633710116147995]),
            'saturation_factor': torch.tensor([
                0.6842519640922546, 0.8155958652496338, 0.8870501518249512, 0.7594910264015198,
                1.0377532243728638, 0.6049283742904663, 1.3612436056137085, 0.6602127552032471]),
            'order': torch.tensor([3, 2, 0, 1])
        }

        assert set(list(jitter_params.keys())) == set([
            'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order']), \
            "Redundant keys found apart from \
                'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order"
        assert (jitter_params['brightness_factor'] == expected_jitter_params['brightness_factor']) \
            .long().sum() == batch_size
        assert (jitter_params['contrast_factor'] == expected_jitter_params['contrast_factor']) \
            .long().sum() == batch_size
        assert (jitter_params['hue_factor'] == expected_jitter_params['hue_factor']) \
            .long().sum() == batch_size
        assert (jitter_params['saturation_factor'] == expected_jitter_params['saturation_factor']) \
            .long().sum() == batch_size
        assert (jitter_params['order'] == expected_jitter_params['order']).long().sum() == 4

    def test_color_jitter_tuple_gen(self):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params_tuple = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2]), contrast=torch.tensor([0.7, 1.3]),
            saturation=torch.tensor([0.6, 1.4]), hue=torch.tensor([-0.1, 0.1]))

        expected_jitter_params_tuple = {
            'brightness_factor': torch.tensor([
                1.15290772914886474609375, 1.16600155830383300781250, 0.95314550399780273437500,
                1.18372225761413574218750, 0.956179320812225341796875, 1.04035818576812744140625,
                0.90262901782989501953125, 1.11745655536651611328125]),
            'contrast_factor': torch.tensor([
                1.2644628286361694, 0.7799115180969238, 1.260758876800537, 1.056147813796997,
                1.2216426134109497, 1.0406291484832764, 1.1446564197540283, 0.957642674446106]),
            'hue_factor': torch.tensor([
                0.07708858698606491, 0.014780893921852112, -0.04668399319052696, 0.02548982948064804,
                -0.04607366397976875, -0.011727288365364075, -0.040615834295749664, 0.06633710116147995]),
            'saturation_factor': torch.tensor([
                0.6842519640922546, 0.8155958652496338, 0.8870501518249512, 0.7594910264015198,
                1.0377532243728638, 0.6049283742904663, 1.3612436056137085, 0.6602127552032471]),
            'order': torch.tensor([3, 2, 0, 1])
        }
        assert set(list(expected_jitter_params_tuple.keys())) == set([
            'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order']), \
            "Redundant keys found apart from \
                'brightness_factor', 'contrast_factor', 'hue_factor', 'saturation_factor', 'order"
        assert (jitter_params_tuple['brightness_factor'] == expected_jitter_params_tuple['brightness_factor']) \
            .long().sum() == batch_size
        assert (jitter_params_tuple['contrast_factor'] == expected_jitter_params_tuple['contrast_factor']) \
            .long().sum() == batch_size
        assert (jitter_params_tuple['hue_factor'] == expected_jitter_params_tuple['hue_factor']) \
            .long().sum() == batch_size
        assert (jitter_params_tuple['saturation_factor'] == expected_jitter_params_tuple['saturation_factor']) \
            .long().sum() == batch_size
        assert (jitter_params_tuple['order'] == expected_jitter_params_tuple['order']).long().sum() == 4

    def test_random_prob_gen_same_on_batch(self):
        torch.manual_seed(42)
        batch_size = 8
        jitter_params = random_color_jitter_generator(
            batch_size, brightness=torch.tensor([0.8, 1.2]), contrast=torch.tensor([0.7, 1.3]),
            saturation=torch.tensor([0.6, 1.4]), hue=torch.tensor([-0.1, 0.1]), same_on_batch=True)

        expected_res = {
            'brightness_factor': torch.tensor([1.15290772914886474609375] * batch_size),
            'contrast_factor': torch.tensor([1.24900233745574951171875] * batch_size),
            'hue_factor': torch.tensor([-0.0234272480010986328125] * batch_size),
            'saturation_factor': torch.tensor([1.367444515228271484375] * batch_size),
            'order': torch.tensor([2, 3, 0, 1])
        }
        assert (jitter_params['brightness_factor'] == expected_res['brightness_factor']).long().sum() == batch_size
        assert (jitter_params['contrast_factor'] == expected_res['contrast_factor']).long().sum() == batch_size
        assert (jitter_params['saturation_factor'] == expected_res['saturation_factor']).long().sum() == batch_size
        assert (jitter_params['hue_factor'] == expected_res['hue_factor']).long().sum() == batch_size
        assert (jitter_params['order'] == expected_res['order']).long().sum() == 4
