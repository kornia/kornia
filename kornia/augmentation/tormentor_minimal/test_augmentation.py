import tormentor
import torch
import pytest

base_cls = tormentor.DeterministicImageAugmentation

intermediary_cls = base_cls.__subclasses__()
all_augmentations = []
for cls in intermediary_cls:
    all_augmentations += cls.__subclasses__()


epsilon = .00000001
def all_similar(t1, t2):
    """Test the maximum square error is lesser than epsilon.
    """
    return ((t1 - t2) ** 2 > epsilon).view(-1).sum() == 0

def minimum_requirement_test(augmentations_cls_list):
    for augmentation_cls in augmentations_cls_list:

        # Every augmentation must define at least one of forward_batch, forward_sample
        assert base_cls.forward_batch is not augmentation_cls.forward_batch or base_cls.forward_sample is not augmentation_cls.forward_sample

        # Assert determinism per sample
        aug = augmentation_cls()
        img = torch.rand(3, 224, 224)
        assert all_similar(aug(img), aug(img))

        # Assert determinism per batch
        aug = augmentation_cls()
        img = torch.rand(10, 3, 224, 224)
        assert all_similar(aug(img), aug(img))

        # Augmentation states must be available after augmentation has been run once.
        aug = augmentation_cls()
        img1 = torch.rand(3, 224, 224)
        aug(img1)
        for state_name in augmentation_cls._state_names:
            assert getattr(aug, state_name) is not None

        # And they should be the same state after every execution on the same data
        aug = augmentation_cls()
        img1 = torch.rand(3, 224, 224)
        aug(img1)
        img1_states = {}
        for state_name in augmentation_cls._state_names:
            img1_states[state_name] = getattr(aug, state_name)
        img2 = torch.rand(3, 224, 224)
        aug(img2)
        img2_states = {}
        for state_name in augmentation_cls._state_names:
            img2_states[state_name] = getattr(aug, state_name)
        augmentation_cls._state_names


def hard_requirement_test(augmentations_cls_list):

    # these tests should be perceived as warnings and don't make sense for all augmentations
    for augmentation_cls in augmentations_cls_list:
        # Augmentation is defining both of forward_batch and forward_sample
        assert base_cls.forward_batch is not augmentation_cls.forward_batch
        assert base_cls.forward_sample is not augmentation_cls.forward_sample

        # Was the aug_statenames decorator used?
        assert len(augmentation_cls._state_names) > 0

        # Are two different augmentations really different?
        img = torch.rand(3, 224, 224)
        aug1 = augmentation_cls()
        aug2 = augmentation_cls()
        assert not all_similar(aug1(img), aug2(img))
