from unittest.mock import patch
from typing import Union, Tuple

import pytest
import torch
import torch.nn as nn

from torch.testing import assert_allclose
from torch.autograd import gradcheck

import kornia
import kornia.testing as utils  # test utils
from kornia.constants import pi
from kornia.augmentation import (
    ColorJitter,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomErasing,
    RandomEqualize,
    RandomGrayscale,
    RandomRotation,
    RandomCrop,
    RandomResizedCrop
)


from kornia.testing import BaseTester
from kornia.augmentation.base import AugmentationBase2D

# TOTEST
# _validate_input_dtype
# _transform_input
# identity_matrix

# Common
# Given arbitrary input shapes the output dims are consistent
# shape changing not shape changing
#  return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5,
                #  p_batch: float = 1.
# Smoke tests, lehet példányosítani, meg lehet hívni az abstract függvényeit stb
# generate_parameters annyi paramot ad vissza amennyi a batch size
# compute_transformation implementálva van és batchsiye stimmel
# apply_transform implementálva van és batchsiye stimmel

# class CommonAugmentation2DTests(BaseTester):
#     def test_smoke(self):
#         raise NotImplementedError("Implement a stupid routine.")
#     def test_exception(self):
#         raise NotImplementedError("Implement a stupid routine.")
#     def test_batch(self):
#         raise NotImplementedError("Implement a stupid routine.")
#     def test_jit(self):
#         raise NotImplementedError("Implement a stupid routine.")
#     def test_gradcheck(self):
#         raise NotImplementedError("Implement a stupid routine.")

#     def __init__(self, return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5,
#                  p_batch: float = 1.) -> None:
#         super(_AugmentationBase, self).__init__(p, p_batch=p_batch, same_on_batch=same_on_batch)
#         self.p = p
#         self.p_batch = p_batch
#         self.return_transform = return_transform

#     def __repr__(self) -> str:
#         return super().__repr__() + f", return_transform={self.return_transform}"
#     def forward(self, input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
#                 params: Optional[Dict[str, torch.Tensor]] = None,  # type: ignore
#                 return_transform: Optional[bool] = None
#                 ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

class CommonTests:

    # Override setter
    _augmentation_cls = None

    @property
    def augmentation_cls(self):
        return self._augmentation_cls

    @augmentation_cls.setter
    def augmentation_cls(self, val):
        raise ValueError("augmentation_cls should not be modified")

    ############################################################################################################
    # Fixtures 
    ############################################################################################################
    @pytest.fixture(params=[(4,5),(3,4,5),(2,3,4,5)],scope="class")
    def input_shape(self,request):
        return request.param

    ############################################################################################################
    # Test cases to implement 
    ############################################################################################################

    def test_smoke(self):
        raise NotImplementedError("test_smoke must be implemented. Check the function _test_smoke_implementation")

    def test_consistent_output_shape(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_random_p_0(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_random_p_0_return_transform(self):
        raise NotImplementedError("Implement a stupid routine.")

    def test_random_p_1(self):
        raise NotImplementedError("Implement a stupid routine.")
    
    def test_random_p_1_return_transform(self):
        raise NotImplementedError("Implement a stupid routine.")
    
    def test_sequential(self):
        raise NotImplementedError("Implement a stupid routine.")
    
    def test_random_batch_transform(self):
        raise NotImplementedError("Implement a stupid routine.")

# TODO Implement
# test_random_p_half_return_transform (manual force [True, False] for batchsize 2 inputs)
# test_batch_return_transform
# test_sequential_return_transform
# test_coordinate check
# test_jit
# test_gradcheck

# TODO WTF selective param gen


    def _create_augmentation_from_params(self,**params):
        return self.augmentation_cls(**params)
    
    ############################################################################################################
    # Test case implementations 
    ############################################################################################################

    # TODO Split up into multiple test cases
    def _test_smoke_implementation(self, device, dtype, **params):
        assert issubclass(self.augmentation_cls,AugmentationBase2D), f"{self.augmentation_cls} is not a subclass of AugmentationBase2D"
        
        # Can be instatiated
        augmentation = self._create_augmentation_from_params(**params,return_transform=False)
        assert issubclass(type(augmentation),AugmentationBase2D), f"{type(augmentation)} is not a subclass of AugmentationBase2D"
        
        # generate_parameters can be called and returns the correct amount of parameters
        batch_shape = (4,3,5,6)
        generated_params = augmentation.generate_parameters(batch_shape)
        assert isinstance(generated_params,dict)
        generated_params["batch_prob"] = torch.tensor([True,True,True,True])
        for key,value in generated_params.items():
            assert value.shape[0] == batch_shape[0], f"Value for {key} must have {batch_shape[0]} at position 0 insted of {value.shape[0]}"
        
        # compute_transformation can be called and returns the correct shaped transformation matrix
        expected_transformation_shape = torch.Size((batch_shape[0],3,3))
        test_input = torch.ones(batch_shape,device=device,dtype=dtype)
        transformation = augmentation.compute_transformation(test_input,generated_params)
        assert transformation.shape == expected_transformation_shape

        # apply_transform can be called and returns the correct batch sized output
        output = augmentation.apply_transform(test_input,generated_params)
        assert output.shape[0] == batch_shape[0]

        # augmentation can be called and returns the correct batch sized output
        output = augmentation(test_input,params=generated_params)
        assert output.shape[0] == batch_shape[0]

        # augmentation can be called and returns the correct batch sized output
        output, transformation = augmentation(test_input,params=generated_params,return_transform=True)
        assert output.shape[0] == batch_shape[0]
        assert transformation.shape == expected_transformation_shape

        test_transform = torch.rand(expected_transformation_shape,device=device,dtype=dtype)
        output, final_transformation = augmentation((test_input,test_transform),params=generated_params,return_transform=True)
        assert output.shape[0] == batch_shape[0]
        assert final_transformation.shape == expected_transformation_shape
        assert_allclose(final_transformation, transformation @ test_transform)

        output, transformation = augmentation((test_input,test_transform),params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert transformation.shape == expected_transformation_shape
        assert (transformation==test_transform).all()

        # Return transform 
        augmentation = self._create_augmentation_from_params(**params,return_transform=True)

        output, transformation = augmentation(test_input,params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert transformation.shape == expected_transformation_shape

        output, final_transformation = augmentation((test_input,test_transform),params=generated_params)
        assert output.shape[0] == batch_shape[0]
        assert final_transformation.shape == expected_transformation_shape
        assert_allclose(final_transformation, transformation @ test_transform)

    def _test_consistent_output_shape_implementation(self, device, dtype, input_shape, **params):

        # p==0.0
        augmentation = self._create_augmentation_from_params(**params,p=0.0)
        expected_output_shape = torch.Size((1,)*(4-len(input_shape))+tuple(input_shape))
        test_input = torch.rand(input_shape,device=device,dtype=dtype)
        output = augmentation(test_input)
        assert len(output.shape) == 4
        assert output.shape == expected_output_shape

        # p==1.0
        augmentation = self._create_augmentation_from_params(**params,p=1.0)
        expected_output_shape = torch.Size((1,)*(4-len(input_shape))+tuple(input_shape))
        test_input = torch.rand(input_shape,device=device,dtype=dtype)
        output = augmentation(test_input)
        assert len(output.shape) == 4
        assert output.shape == expected_output_shape

    def _test_random_p_0_implementation(self, device, dtype, **params):
        augmentation = self._create_augmentation_from_params(**params,p=0.0,return_transform=False)
        expected_output_shape = torch.Size((2,3,4,5))
        test_input = torch.rand((2,3,4,5),device=device,dtype=dtype)
        output = augmentation(test_input)
        assert (output==test_input).all()

    def _test_random_p_0_return_transform_implementation(self, device, dtype, **params):
        augmentation = self._create_augmentation_from_params(**params,p=0.0,return_transform=True)
        expected_output_shape = torch.Size((2,3,4,5))
        expected_transformation_shape = torch.Size((2,3,3))
        test_input = torch.rand((2,3,4,5),device=device,dtype=dtype)
        output, transformation = augmentation(test_input)

        assert (output==test_input).all()
        assert transformation.shape == expected_transformation_shape
        assert (transformation== kornia.eye_like(3,transformation)).all()

    def _test_random_p_1_implementation(self, device, dtype, input_tensor, expected_output, **params):
        augmentation = self._create_augmentation_from_params(**params,p=1.0,return_transform=False)
        output = augmentation(input_tensor.to(device).to(dtype))

        # Output should match
        assert output.shape == expected_output.shape
        assert_allclose(output, expected_output.to(device).to(dtype), atol=1e-4, rtol=1e-4), output
        
    def _test_random_p_1_return_transform_implementation(self, device, dtype, input_tensor, expected_output, expected_transformation, **params):
        augmentation = self._create_augmentation_from_params(**params,p=1.0,return_transform=True)
        output, transformation = augmentation(input_tensor.to(device).to(dtype))

        # Output should match
        assert output.shape == expected_output.shape
        assert_allclose(output, expected_output.to(device).to(dtype), atol=1e-4, rtol=1e-4), output

        # Transformation should match
        assert transformation.shape == expected_transformation.shape
        assert_allclose(transformation, expected_transformation.to(device).to(dtype), atol=1e-4, rtol=1e-4), transformation


    # TODO finish it
    def _test_random_batch_transform_implementation(self, device, dtype, input_tensor, expected_output, expected_transformation, batch_prob, **params):
        augmentation = self._create_augmentation_from_params(**params,p=1.0,return_transform=True)
        forward_params = {**augmentation.generate_parameters(input_tensor.shape),"batch_prob":batch_prob}
        output, transformation = augmentation(input_tensor.to(device).to(dtype),params = forward_params)

        # Output should match
        assert output.shape == expected_output.shape
        assert_allclose(output, expected_output.to(device).to(dtype), atol=1e-4, rtol=1e-4), output

        # Transformation should match
        assert transformation.shape == expected_transformation.shape
        assert_allclose(transformation, expected_transformation.to(device).to(dtype), atol=1e-4, rtol=1e-4), transformation



    # TODO finish it
    def _test_sequential(self, device, dtype,**params):

        f = nn.Sequential(
            self._create_augmentation_from_params(**params,return_transform=True)
            self._create_augmentation_from_params(**params,return_transform=True),
        )

        input = torch.rand(3, 5, 5).to(device)  # 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3).unsqueeze(0)  # 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform)
    
    # Can be instantiated
    # Can be called
    # Returns the proper number and shaped tensors
    # Returns the the proper transformations eye
    # Returns the proper transformation back transform
    # return_transform: bool = False, same_on_batch: bool = False, p: float = 0.5,
    #              p_batch: float = 1.


# ps = [0.0,1.0]
# return_transforms = [False,True]
# params = product(ps,return_transforms)
# class TestSubclass(CommonTests):
#     @pytest.fixture(params=params,scope="class")
#     def augmentation(self,request):
#         return RandomHorizontalFlip3D(request.param[0], return_transform=request.param[1],same_on_batch=True)
# def dummy_generator(n):
#     for i in range(n):
#         yield i
# def cartesian(default,*,**kwargs):
#     keys = kwargs.keys()
#     combinations = ({} for )

# def one_by_one(default,*,**kwargs):
#     pass


class TestColorJitter2(CommonTests):

    brightness = (1., (0.,2.), [0.,2.],torch.tensor(0.),torch.tensor((0.,2.)))
    contrast = (1., (0.,2.), [0.,2.],torch.tensor(0.),torch.tensor((0.,2.)))
    saturation = (1., (0.,2.), [0.,2.],torch.tensor(0.),torch.tensor((0.,2.)))
    hue = (0., (-0.5,0.5), [-0.5,0.5],torch.tensor(0.),torch.tensor((-0.5,0.5)))
    return_transform = (False,True)
    same_on_batch = (False,True)
    p = (0.,1.)

    params = []
    params.extend({"brightness":v} for v in brightness)
    params.extend({"contrast":v} for v in contrast)
    params.extend({"saturation":v} for v in saturation)
    params.extend({"hue":v} for v in hue)

    _augmentation_cls = ColorJitter

    @pytest.mark.parametrize("params", params)
    def test_smoke(self,device,dtype,params):
        super()._test_smoke_implementation(device=device, dtype=dtype,**params)

    def test_random_p_0(self,dtype,device):
        super()._test_random_p_0_implementation(device=device, dtype=dtype,
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5
                )

    @pytest.mark.parametrize("input_shape", [(3,4,5),(2,3,4,5)])
    def test_consistent_output_shape(self,device,dtype,input_shape):
        super()._test_consistent_output_shape_implementation(device=device, dtype=dtype, input_shape=input_shape)


    def test_random_p_0_return_transform(self,device,dtype):
        super()._test_random_p_0_return_transform_implementation(device=device, dtype=dtype,
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.5
                )

    def test_random_p_1(self,device,dtype):
        torch.manual_seed(42)
        input_tensor = torch.tensor([[.1, .2, .3, .4],
                              [.5, .6, .7, .8],
                              [.9, .0, .1, .2]], device=device, dtype=dtype).repeat(3, 1, 1)
        expected_output = torch.tensor([[[[0.6024, 0.7273, 0.8522, 0.9771],
                                          [1.0000, 1.0000, 1.0000, 1.0000],
                                          [1.0000, 0.4775, 0.6024, 0.7273]]]]).repeat(1, 3, 1, 1)
        super()._test_random_p_1_implementation(device=device, dtype=dtype,input_tensor=input_tensor,expected_output=expected_output,
                    brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1
                )
    def test_random_p_1_return_transform(self, device, dtype):
        torch.manual_seed(42)
        input_tensor = torch.tensor([[.1, .2, .3, .4],
                              [.5, .6, .7, .8],
                              [.9, .0, .1, .2]], device=device, dtype=dtype).repeat(3, 1, 1)
        expected_output = torch.tensor([[[[0.6024, 0.7273, 0.8522, 0.9771],
                                          [1.0000, 1.0000, 1.0000, 1.0000],
                                          [1.0000, 0.4775, 0.6024, 0.7273]]]], device=device, dtype=dtype).repeat(1, 3, 1, 1)
        expected_transformation = torch.tensor([[[1., 0., 0.],
                                                [0., 1., 0.],
                                                [0., 0., 1.]]], device=device, dtype=dtype)
        super()._test_random_p_1_return_transform_implementation(device=device, dtype=dtype,input_tensor=input_tensor,expected_output=expected_output,expected_transformation=expected_transformation,
                    brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1
                )

class TestRandomHorizontalFlip:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomHorizontalFlip(p=0.5)
        repr = "RandomHorizontalFlip(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_hflip(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip(p=0., return_transform=True)
        f2 = RandomHorizontalFlip(p=1.)
        f3 = RandomHorizontalFlip(p=0.)

        input = torch.tensor([[0., 0., 0., 0.],
                              [0., 0., 0., 0.],
                              [0., 0., 1., 2.]])  # 3 x 4

        input = input.to(device)

        expected = torch.tensor([[0., 0., 0., 0.],
                                 [0., 0., 0., 0.],
                                 [2., 1., 0., 0.]])  # 3 x 4

        expected = expected.to(device)

        expected_transform = torch.tensor([[-1., 0., 3.],
                                           [0., 1., 0.],
                                           [0., 0., 1.]])  # 3 x 3

        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3
        identity = identity.to(device)

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()
        assert (f2(input) == expected).all()
        assert (f3(input) == input).all()

    def test_batch_random_hflip(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)
        f1 = RandomHorizontalFlip(p=0.0, return_transform=True)

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[[0., 0., 0.],
                                   [0., 0., 0.],
                                   [1., 1., 0.]]]])  # 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[[-1., 0., 2.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert (f(input)[0] == expected).all()
        assert (f(input)[1] == expected_transform).all()
        assert (f1(input)[0] == input).all()
        assert (f1(input)[1] == identity).all()

    def test_same_on_batch(self, device, dtype):
        f = RandomHorizontalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        f = nn.Sequential(
            RandomHorizontalFlip(p=1.0, return_transform=True),
            RandomHorizontalFlip(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomHorizontalFlip(p=1.0, return_transform=True),
            RandomHorizontalFlip(p=1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected_transform = torch.tensor([[[-1., 0., 2.],
                                            [0., 1., 0.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform
        expected_transform_1 = expected_transform_1.to(device)

        assert(f(input)[0] == input).all()
        assert(f(input)[1] == expected_transform_1).all()
        assert(f1(input)[0] == input).all()
        assert(f1(input)[1] == expected_transform).all()

    def test_random_hflip_coord_check(self, device, dtype):

        f = RandomHorizontalFlip(p=1.0, return_transform=True)

        input = torch.tensor([[[[1., 2., 3., 4.],
                                [5., 6., 7., 8.],
                                [9., 10., 11., 12.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        input_coordinates = torch.tensor([[
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],  # x coord
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # y coord
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_output = torch.tensor([[[[4., 3., 2., 1.],
                                          [8., 7., 6., 5.],
                                          [12., 11., 10., 9.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        output, transform = f(input)
        result_coordinates = transform @ input_coordinates
        # NOTE: without rounding it might produce unexpected results
        input_coordinates = input_coordinates.round().long()
        result_coordinates = result_coordinates.round().long()

        # Tensors must have the same shapes and values
        assert output.shape == expected_output.shape
        assert (output == expected_output).all()
        # Transformed indices must not be out of bound
        assert (torch.torch.logical_and(result_coordinates[0, 0, :] >= 0,
                                        result_coordinates[0, 0, :] < input.shape[-1])).all()
        assert (torch.torch.logical_and(result_coordinates[0, 1, :] >= 0,
                                        result_coordinates[0, 1, :] < input.shape[-2])).all()
        # Values in the output tensor at the places of transformed indices must
        # have the same value as the input tensor has at the corresponding
        # positions
        assert (output[..., result_coordinates[0, 1, :], result_coordinates[0, 0, :]] ==
                input[..., input_coordinates[0, 1, :], input_coordinates[0, 0, :]]).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

            return kornia.random_hflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [0., 5., 5.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomHorizontalFlip(p=1.), (input, ), raise_exception=True)


class TestRandomVerticalFlip:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomVerticalFlip(p=0.5)
        repr = "RandomVerticalFlip(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_vflip(self, device, dtype):

        f = RandomVerticalFlip(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip(p=0., return_transform=True)
        f2 = RandomVerticalFlip(p=1.)
        f3 = RandomVerticalFlip(p=0.)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3
        input = input.to(device)

        expected = torch.tensor([[0., 1., 1.],
                                 [0., 0., 0.],
                                 [0., 0., 0.]])  # 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[1., 0., 0.],
                                           [0., -1., 2.],
                                           [0., 0., 1.]])  # 3 x 3
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[1., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 1.]])  # 3 x 3
        identity = identity.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)
        assert_allclose(f2(input), expected)
        assert_allclose(f3(input), input)

    def test_batch_random_vflip(self, device, dtype):

        f = RandomVerticalFlip(p=1.0, return_transform=True)
        f1 = RandomVerticalFlip(p=0.0, return_transform=True)

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected = torch.tensor([[[[0., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]])  # 1 x 1 x 3 x 3
        expected = expected.to(device)

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 2.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        identity = torch.tensor([[[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]])  # 1 x 3 x 3
        identity = identity.to(device)

        input = input.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected = expected.repeat(5, 3, 1, 1)  # 5 x 3 x 3 x 3
        expected_transform = expected_transform.repeat(5, 1, 1)  # 5 x 3 x 3
        identity = identity.repeat(5, 1, 1)  # 5 x 3 x 3

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)
        assert_allclose(f1(input)[0], input)
        assert_allclose(f1(input)[1], identity)

    def test_same_on_batch(self, device, dtype):
        f = RandomVerticalFlip(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        f = nn.Sequential(
            RandomVerticalFlip(p=1.0, return_transform=True),
            RandomVerticalFlip(p=1.0, return_transform=True),
        )
        f1 = nn.Sequential(
            RandomVerticalFlip(p=1.0, return_transform=True),
            RandomVerticalFlip(p=1.0),
        )

        input = torch.tensor([[[[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 1., 1.]]]])  # 1 x 1 x 3 x 3
        input = input.to(device)

        expected_transform = torch.tensor([[[1., 0., 0.],
                                            [0., -1., 2.],
                                            [0., 0., 1.]]])  # 1 x 3 x 3
        expected_transform = expected_transform.to(device)

        expected_transform_1 = expected_transform @ expected_transform

        assert_allclose(f(input)[0], input.squeeze())
        assert_allclose(f(input)[1], expected_transform_1)
        assert_allclose(f1(input)[0], input.squeeze())
        assert_allclose(f1(input)[1], expected_transform)

    def test_random_vflip_coord_check(self, device, dtype):

        f = RandomVerticalFlip(p=1.0, return_transform=True)

        input = torch.tensor([[[[1., 2., 3., 4.],
                                [5., 6., 7., 8.],
                                [9., 10., 11., 12.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        input_coordinates = torch.tensor([[
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],  # x coord
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # y coord
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_output = torch.tensor([[[[9., 10., 11., 12.],
                                          [5., 6., 7., 8.],
                                          [1., 2., 3., 4.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 4

        output, transform = f(input)
        result_coordinates = transform @ input_coordinates
        # NOTE: without rounding it might produce unexpected results
        input_coordinates = input_coordinates.round().long()
        result_coordinates = result_coordinates.round().long()

        # Tensors must have the same shapes and values
        assert output.shape == expected_output.shape
        assert (output == expected_output).all()
        # Transformed indices must not be out of bound
        assert (torch.torch.logical_and(result_coordinates[0, 0, :] >= 0,
                                        result_coordinates[0, 0, :] < input.shape[-1])).all()
        assert (torch.torch.logical_and(result_coordinates[0, 1, :] >= 0,
                                        result_coordinates[0, 1, :] < input.shape[-2])).all()
        # Values in the output tensor at the places of transformed indices must
        # have the same value as the input tensor has at the corresponding
        # positions
        assert (output[..., result_coordinates[0, 1, :], result_coordinates[0, 0, :]] ==
                input[..., input_coordinates[0, 1, :], input_coordinates[0, 0, :]]).all()

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return kornia.random_vflip(data)

        input = torch.tensor([[0., 0., 0.],
                              [0., 0., 0.],
                              [0., 1., 1.]])  # 3 x 3

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        input = input.repeat(2, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[0., 0., 0.],
                                  [5., 5., 0.],
                                  [0., 0., 0.]]])  # 3 x 3

        expected = expected.repeat(2, 1, 1)

        actual = op_trace(input)

        assert_allclose(actual, expected)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomVerticalFlip(p=1.), (input, ), raise_exception=True)


class TestColorJitter:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = ColorJitter(brightness=0.5, contrast=0.3, saturation=[0.2, 1.2], hue=0.1)
        repr = "ColorJitter(brightness=tensor([0.5000, 1.5000]), contrast=tensor([0.7000, 1.3000]), "\
               "saturation=tensor([0.2000, 1.2000]), hue=tensor([-0.1000,  0.1000]), "\
               "p=1.0, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_color_jitter(self, device, dtype):

        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_color_jitter_batch(self, device, dtype):
        f = ColorJitter()
        f1 = ColorJitter(return_transform=True)

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f1(input)[1], expected_transform)

    def test_same_on_batch(self, device, dtype):
        f = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, same_on_batch=True)
        input = torch.eye(3).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_random_brightness(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(brightness=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0000, 0.0233, 0.1233],
                                   [0.4233, 0.3233, 0.2233],
                                   [0.5233, 0.6233, 0.8233]],
                                  [[0.0000, 0.0233, 0.1233],
                                   [0.4233, 0.3233, 0.2233],
                                   [0.5233, 0.6233, 0.8233]],
                                  [[0.0000, 0.0233, 0.1233],
                                   [0.4233, 0.3233, 0.2233],
                                   [0.5233, 0.6233, 0.8233]]],
                                 [[[0.0000, 0.0252, 0.1252],
                                   [0.4252, 0.3252, 0.2252],
                                   [0.5252, 0.6252, 0.8252]],
                                  [[0.0000, 0.0252, 0.1252],
                                   [0.4252, 0.3252, 0.2252],
                                   [0.5252, 0.6252, 0.8252]],
                                  [[0.0000, 0.0252, 0.1252],
                                   [0.4252, 0.3252, 0.2252],
                                   [0.5252, 0.6252, 0.8252]]]], device=device, dtype=dtype)   # 1 x 1 x 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_brightness_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(brightness=(0.8, 1.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0000, 0.0233, 0.1233],
                                   [0.4233, 0.3233, 0.2233],
                                   [0.5233, 0.6233, 0.8233]],
                                  [[0.0000, 0.0233, 0.1233],
                                   [0.4233, 0.3233, 0.2233],
                                   [0.5233, 0.6233, 0.8233]],
                                  [[0.0000, 0.0233, 0.1233],
                                   [0.4233, 0.3233, 0.2233],
                                   [0.5233, 0.6233, 0.8233]]],
                                 [[[0.0000, 0.0252, 0.1252],
                                   [0.4252, 0.3252, 0.2252],
                                   [0.5252, 0.6252, 0.8252]],
                                  [[0.0000, 0.0252, 0.1252],
                                   [0.4252, 0.3252, 0.2252],
                                   [0.5252, 0.6252, 0.8252]],
                                  [[0.0000, 0.0252, 0.1252],
                                   [0.4252, 0.3252, 0.2252],
                                   [0.5252, 0.6252, 0.8252]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_contrast(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(contrast=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0849, 0.1699, 0.2548],
                                   [0.5097, 0.4247, 0.3398],
                                   [0.5946, 0.6795, 0.8494]],
                                  [[0.0849, 0.1699, 0.2548],
                                   [0.5097, 0.4247, 0.3398],
                                   [0.5946, 0.6795, 0.8494]],
                                  [[0.0849, 0.1699, 0.2548],
                                   [0.5097, 0.4247, 0.3398],
                                   [0.5946, 0.6795, 0.8494]]],
                                 [[[0.0821, 0.1642, 0.2463],
                                   [0.4926, 0.4105, 0.3284],
                                   [0.5747, 0.6568, 0.8210]],
                                  [[0.0821, 0.1642, 0.2463],
                                   [0.4926, 0.4105, 0.3284],
                                   [0.5747, 0.6568, 0.8210]],
                                  [[0.0821, 0.1642, 0.2463],
                                   [0.4926, 0.4105, 0.3284],
                                   [0.5747, 0.6568, 0.8210]]]], device=device, dtype=dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_contrast_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(contrast=[0.8, 1.2])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 3, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0849, 0.1699, 0.2548],
                                   [0.5097, 0.4247, 0.3398],
                                   [0.5946, 0.6795, 0.8494]],
                                  [[0.0849, 0.1699, 0.2548],
                                   [0.5097, 0.4247, 0.3398],
                                   [0.5946, 0.6795, 0.8494]],
                                  [[0.0849, 0.1699, 0.2548],
                                   [0.5097, 0.4247, 0.3398],
                                   [0.5946, 0.6795, 0.8494]]],
                                 [[[0.0821, 0.1642, 0.2463],
                                   [0.4926, 0.4105, 0.3284],
                                   [0.5747, 0.6568, 0.8210]],
                                  [[0.0821, 0.1642, 0.2463],
                                   [0.4926, 0.4105, 0.3284],
                                   [0.5747, 0.6568, 0.8210]],
                                  [[0.0821, 0.1642, 0.2463],
                                   [0.4926, 0.4105, 0.3284],
                                   [0.5747, 0.6568, 0.8210]]]], device=device, dtype=dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-5)

    def test_random_saturation(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(saturation=0.2)

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0000, 0.0907, 0.2272],
                                   [0.5454, 0.5000, 0.4000],
                                   [0.6818, 0.8000, 1.0000]],
                                  [[1.0000, 0.4454, 0.5818],
                                   [0.5454, 0.2636, 0.1636],
                                   [0.8000, 0.0000, 0.0543]],
                                  [[0.5556, 0.8000, 0.7000],
                                   [0.9000, 0.2636, 0.1636],
                                   [0.8000, 0.3429, 0.4090]]],
                                 [[[0.0000, 0.0971, 0.2314],
                                   [0.5485, 0.5000, 0.4000],
                                   [0.6828, 0.8000, 1.0000]],
                                  [[1.0000, 0.4485, 0.5828],
                                   [0.5485, 0.2657, 0.1657],
                                   [0.8000, 0.0000, 0.0628]],
                                  [[0.5556, 0.8000, 0.7000],
                                   [0.9000, 0.2657, 0.1657],
                                   [0.8000, 0.3429, 0.4142]]]], device=device, dtype=dtype)
        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_saturation_tensor(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(saturation=torch.tensor(0.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0000, 0.0907, 0.2272],
                                   [0.5454, 0.5000, 0.4000],
                                   [0.6818, 0.8000, 1.0000]],
                                  [[1.0000, 0.4454, 0.5818],
                                   [0.5454, 0.2636, 0.1636],
                                   [0.8000, 0.0000, 0.0543]],
                                  [[0.5556, 0.8000, 0.7000],
                                   [0.9000, 0.2636, 0.1636],
                                   [0.8000, 0.3429, 0.4090]]],
                                 [[[0.0000, 0.0971, 0.2314],
                                   [0.5485, 0.5000, 0.4000],
                                   [0.6828, 0.8000, 1.0000]],
                                  [[1.0000, 0.4485, 0.5828],
                                   [0.5485, 0.2657, 0.1657],
                                   [0.8000, 0.0000, 0.0628]],
                                  [[0.5556, 0.8000, 0.7000],
                                   [0.9000, 0.2657, 0.1657],
                                   [0.8000, 0.3429, 0.4142]]]], device=device, dtype=dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_saturation_tuple(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(saturation=(0.8, 1.2))

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.0000, 0.0907, 0.2272],
                                   [0.5454, 0.5000, 0.4000],
                                   [0.6818, 0.8000, 1.0000]],
                                  [[1.0000, 0.4454, 0.5818],
                                   [0.5454, 0.2636, 0.1636],
                                   [0.8000, 0.0000, 0.0543]],
                                  [[0.5556, 0.8000, 0.7000],
                                   [0.9000, 0.2636, 0.1636],
                                   [0.8000, 0.3429, 0.4090]]],
                                 [[[0.0000, 0.0971, 0.2314],
                                   [0.5485, 0.5000, 0.4000],
                                   [0.6828, 0.8000, 1.0000]],
                                  [[1.0000, 0.4485, 0.5828],
                                   [0.5485, 0.2657, 0.1657],
                                   [0.8000, 0.0000, 0.0628]],
                                  [[0.5556, 0.8000, 0.7000],
                                   [0.9000, 0.2657, 0.1657],
                                   [0.8000, 0.3429, 0.4142]]]], device=device, dtype=dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_hue(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(hue=0.1 / pi.item())

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6030, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],
                                  [[1.0000, 0.4940, 0.5960],
                                   [0.6000, 0.3020, 0.2020],
                                   [0.7990, 0.1000, 0.2000]],
                                  [[0.6090, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3930, 0.4920]]],
                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],
                                  [[1.0000, 0.5053, 0.6035],
                                   [0.6027, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],
                                  [[0.5920, 0.8000, 0.7000],
                                   [0.9000, 0.3018, 0.2018],
                                   [0.7991, 0.4062, 0.5071]]]], device=device, dtype=dtype)
        expected = expected.to(device)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_hue_list(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(hue=[-0.1 / pi, 0.1 / pi])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6030, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],
                                  [[1.0000, 0.4940, 0.5960],
                                   [0.6000, 0.3020, 0.2020],
                                   [0.7990, 0.1000, 0.2000]],
                                  [[0.6090, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3930, 0.4920]]],
                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],
                                  [[1.0000, 0.5053, 0.6035],
                                   [0.6027, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],
                                  [[0.5920, 0.8000, 0.7000],
                                   [0.9000, 0.3018, 0.2018],
                                   [0.7991, 0.4062, 0.5071]]]], device=device, dtype=dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_random_hue_list_batch(self, device, dtype):
        torch.manual_seed(42)
        f = ColorJitter(hue=[-0.1 / pi.item(), 0.1 / pi.item()])

        input = torch.tensor([[[[0.1, 0.2, 0.3],
                                [0.6, 0.5, 0.4],
                                [0.7, 0.8, 1.]],

                               [[1.0, 0.5, 0.6],
                                [0.6, 0.3, 0.2],
                                [0.8, 0.1, 0.2]],

                               [[0.6, 0.8, 0.7],
                                [0.9, 0.3, 0.2],
                                [0.8, 0.4, .5]]]], device=device, dtype=dtype)  # 1 x 1 x 3 x 3
        input = input.repeat(2, 1, 1, 1)  # 2 x 3 x 3

        expected = torch.tensor([[[[0.1000, 0.2000, 0.3000],
                                   [0.6030, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],
                                  [[1.0000, 0.4940, 0.5960],
                                   [0.6000, 0.3020, 0.2020],
                                   [0.7990, 0.1000, 0.2000]],
                                  [[0.6090, 0.8000, 0.7000],
                                   [0.9000, 0.3000, 0.2000],
                                   [0.8000, 0.3930, 0.4920]]],
                                 [[[0.1000, 0.2000, 0.3000],
                                   [0.6000, 0.5000, 0.4000],
                                   [0.7000, 0.8000, 1.0000]],
                                  [[1.0000, 0.5053, 0.6035],
                                   [0.6027, 0.3000, 0.2000],
                                   [0.8000, 0.1000, 0.2000]],
                                  [[0.5920, 0.8000, 0.7000],
                                   [0.9000, 0.3018, 0.2018],
                                   [0.7991, 0.4062, 0.5071]]]], device=device, dtype=dtype)

        assert_allclose(f(input), expected, atol=1e-4, rtol=1e-4)

    def test_sequential(self, device, dtype):

        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5

        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform, atol=1e-4, rtol=1e-5)

    def test_color_jitter_batch_sequential(self, device, dtype):
        f = nn.Sequential(
            ColorJitter(return_transform=True),
            ColorJitter(return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3

        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[0], expected, atol=1e-4, rtol=1e-5)
        assert_allclose(f(input)[1], expected_transform, atol=1e-4, rtol=1e-5)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.ColorJitter(p=1.), (input, ), raise_exception=True)


class TestRectangleRandomErasing:
    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_random_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=1.)
        assert rand_rec(input).shape == batch_shape

    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("batch_shape", [(1, 4, 8, 15), (2, 3, 11, 7)])
    def test_no_rectangle_erasing_shape(
            self, batch_shape, erase_scale_range, aspect_ratio_range):
        input = torch.rand(batch_shape)
        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=0.)
        assert rand_rec(input).equal(input)

    @pytest.mark.parametrize("erase_scale_range", [(.001, .001), (1., 1.)])
    @pytest.mark.parametrize("aspect_ratio_range", [(.1, .1), (10., 10.)])
    @pytest.mark.parametrize("shape", [(3, 11, 7)])
    def test_same_on_batch(self, shape, erase_scale_range, aspect_ratio_range):
        f = RandomErasing(erase_scale_range, aspect_ratio_range, same_on_batch=True, p=0.5)
        input = torch.rand(shape).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device, dtype):
        # test parameters
        batch_shape = (2, 3, 11, 7)
        erase_scale_range = (.2, .4)
        aspect_ratio_range = (.3, .5)

        rand_rec = RandomErasing(erase_scale_range, aspect_ratio_range, p=1.0)
        rect_params = rand_rec.__forward_parameters__(batch_shape, p=1.0, p_batch=1., same_on_batch=False)

        # evaluate function gradient
        input = torch.rand(batch_shape, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(
            rand_rec,
            (input, rect_params),
            raise_exception=True,
        )

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):
        @torch.jit.script
        def op_script(img):
            return kornia.augmentation.random_rectangle_erase(img, (.2, .4), (.3, .5))

        batch_size, channels, height, width = 2, 3, 64, 64
        img = torch.ones(batch_size, channels, height, width)
        expected = RandomErasing(
            1.0, (.2, .4), (.3, .5)
        )(img)
        actual = op_script(img)
        assert_allclose(actual, expected)


class TestRandomGrayscale:

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomGrayscale()
        repr = "RandomGrayscale(p=0.1, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_grayscale(self, device, dtype):

        f = RandomGrayscale(return_transform=True)

        input = torch.rand(3, 5, 5, device=device, dtype=dtype)  # 3 x 5 x 5

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[1], expected_transform)

    def test_same_on_batch(self, device, dtype):
        f = RandomGrayscale(p=0.5, same_on_batch=True)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_opencv_true(self, device, dtype):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor([[[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]], device=device, dtype=dtype)

        img_gray = kornia.augmentation.RandomGrayscale(p=1.)(data)
        assert_allclose(img_gray, expected)

    def test_opencv_false(self, device, dtype):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)

        expected = data

        img_gray = kornia.augmentation.RandomGrayscale(p=0.)(data)
        assert_allclose(img_gray, expected)

    def test_opencv_true_batch(self, device, dtype):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        # Output data generated with OpenCV 4.1.1: cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        expected = torch.tensor([[[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]],

                                 [[0.4684734, 0.8954562, 0.6064363, 0.5236061, 0.6106016],
                                  [0.1709944, 0.5133104, 0.7915002, 0.5745703, 0.1680204],
                                  [0.5279005, 0.6092287, 0.3034387, 0.5333768, 0.6064113],
                                  [0.3503858, 0.5720159, 0.7052018, 0.4558409, 0.3261529],
                                  [0.6988886, 0.5897652, 0.6532392, 0.7234108, 0.7218805]]], device=device, dtype=dtype)
        expected = expected.unsqueeze(0).repeat(4, 1, 1, 1)

        img_gray = kornia.augmentation.RandomGrayscale(p=1.)(data)
        assert_allclose(img_gray, expected)

    def test_opencv_false_batch(self, device, dtype):
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912],
                              [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008],
                              [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481],
                              [0.0086008, 0.8288748, 0.9647092, 0.8922020, 0.7614344],
                              [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]],

                             [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274],
                              [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.0632100],
                              [0.6171775, 0.8624780, 0.4126036, 0.7600935, 0.7279997],
                              [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165],
                              [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]],

                             [[0.2869580, 0.4700376, 0.2743714, 0.8135023, 0.2229074],
                              [0.9306560, 0.3734594, 0.4566821, 0.7599275, 0.7557513],
                              [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.1315770],
                              [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012],
                              [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)
        data = data.unsqueeze(0).repeat(4, 1, 1, 1)

        expected = data

        img_gray = kornia.augmentation.RandomGrayscale(p=0.)(data)
        assert_allclose(img_gray, expected)

    def test_random_grayscale_sequential_batch(self, device, dtype):
        f = nn.Sequential(
            RandomGrayscale(p=0., return_transform=True),
            RandomGrayscale(p=0., return_transform=True),
        )

        input = torch.rand(2, 3, 5, 5, device=device, dtype=dtype)  # 2 x 3 x 5 x 5
        expected = input

        expected_transform = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand((2, 3, 3))  # 2 x 3 x 3
        expected_transform = expected_transform.to(device)

        assert_allclose(f(input)[0], expected)
        assert_allclose(f(input)[1], expected_transform)

    def test_gradcheck(self, device, dtype):
        input = torch.rand((3, 5, 5), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.RandomGrayscale(p=1.), (input,), raise_exception=True)
        assert gradcheck(kornia.augmentation.RandomGrayscale(p=0.), (input,), raise_exception=True)


class TestCenterCrop:

    def test_no_transform(self, device, dtype):
        inp = torch.rand(1, 2, 4, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop(2)(inp)
        assert out.shape == (1, 2, 2, 2)

    def test_transform(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop(2, return_transform=True)(inp)
        assert len(out) == 2
        assert out[0].shape == (1, 2, 2, 2)
        assert out[1].shape == (1, 3, 3)

    def test_no_transform_tuple(self, device, dtype):
        inp = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        out = kornia.augmentation.CenterCrop((3, 4))(inp)
        assert out.shape == (1, 2, 3, 4)

    def test_gradcheck(self, device, dtype):
        input = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(kornia.augmentation.CenterCrop(3), (input,), raise_exception=True)


class TestRandomRotation:

    torch.manual_seed(0)  # for random reproductibility

    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomRotation(degrees=45.5)
        repr = "RandomRotation(degrees=tensor([-45.5000,  45.5000]), interpolation=BILINEAR, p=0.5, "\
               "p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_rotation(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, return_transform=True, p=1.)
        f1 = RandomRotation(degrees=45.0, p=1.)

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]], device=device, dtype=dtype)  # 4 x 4

        expected = torch.tensor([[[[0.0000, 0.2645, 0.4798, 0.0000],
                                   [0.2399, 0.0000, 0.7409, 0.3652],
                                   [0.1323, 0.4232, 1.3269, 1.2947],
                                   [0.0000, 0.0877, 0.0465, 0.2285]]]], device=device, dtype=dtype)  # 1 x 4 x 4

        expected_transform = torch.tensor([[[0.7396, 0.6731, -0.6190],
                                            [-0.6731, 0.7396, 1.4002],
                                            [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_2 = torch.tensor([[[[0.2631, 0.0000, 0.6914, 0.5261],
                                     [0.3457, 0.0000, 0.3235, 0.0000],
                                     [0.0000, 0.7043, 1.6793, 1.0616],
                                     [0.0000, 0.1307, 0.4526, 0.8613]]]], device=device, dtype=dtype)  # 1 x 4 x 4

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-6, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-6, atol=1e-4)
        assert_allclose(f1(input), expected_2, rtol=1e-6, atol=1e-4)

    def test_batch_random_rotation(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = RandomRotation(degrees=45.0, return_transform=True, p=1.)

        input = torch.tensor([[[[1., 0., 0., 2.],
                                [0., 0., 0., 0.],
                                [0., 1., 2., 0.],
                                [0., 0., 1., 2.]]]], device=device, dtype=dtype)  # 1 x 1 x 4 x 4

        expected = torch.tensor([[[[0.0000, 0.2645, 0.4798, 0.0000],
                                   [0.2399, 0.0000, 0.7409, 0.3652],
                                   [0.1323, 0.4232, 1.3269, 1.2947],
                                   [0.0000, 0.0877, 0.0465, 0.2285]]],
                                 [[[0.2631, 0.0000, 0.6914, 0.5261],
                                   [0.3457, 0.0000, 0.3235, 0.0000],
                                   [0.0000, 0.7043, 1.6793, 1.0616],
                                   [0.0000, 0.1307, 0.4526, 0.8613]]]], device=device, dtype=dtype)  # 2 x 1 x 4 x 4

        expected_transform = torch.tensor([[[0.7396, 0.6731, -0.6190],
                                            [-0.6731, 0.7396, 1.4002],
                                            [0.0000, 0.0000, 1.0000]],
                                           [[0.9472, 0.3207, -0.4018],
                                            [-0.3207, 0.9472, 0.5602],
                                            [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 2 x 3 x 3

        input = input.repeat(2, 1, 1, 1)  # 5 x 3 x 3 x 3

        out, mat = f(input)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomRotation(degrees=40, same_on_batch=True)
        input = torch.eye(6, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_sequential(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        f = nn.Sequential(
            RandomRotation(torch.tensor([-45.0, 90]), return_transform=True, p=1.),
            RandomRotation(10.4, return_transform=True, p=1.),
        )
        f1 = nn.Sequential(
            RandomRotation(torch.tensor([-45.0, 90]), return_transform=True, p=1.),
            RandomRotation(10.4, p=1.),
        )

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]], device=device, dtype=dtype)  # 4 x 4

        expected = torch.tensor([[[[1.2791, 0.1719, 0.2457, 1.3764],
                                   [0.1720, 0.0772, 1.8012, 0.9797],
                                   [0.0860, 0.0361, 0.9309, 0.1257],
                                   [0.6396, 0.0873, 0.0299, 0.0037]]]], device=device, dtype=dtype)  # 1 x 4 x 4

        expected_transform = torch.tensor([[[-0.0049, 1.0000, 0.0073],
                                            [-1.0000, -0.0049, 3.0073],
                                            [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 1 x 3 x 3

        expected_transform_2 = torch.tensor([[[0.9562, 0.2927, -0.3733],
                                              [-0.2927, 0.9562, 0.5046],
                                              [0.0000, 0.0000, 1.0000]]], device=device, dtype=dtype)  # 1 x 3 x 3

        out, mat = f(input)
        _, mat_2 = f1(input)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
        assert_allclose(mat, expected_transform, rtol=1e-4, atol=1e-4)
        assert_allclose(mat_2, expected_transform_2, rtol=1e-4, atol=1e-4)

    @pytest.mark.skip(reason="turn off all jit for a while")
    def test_jit(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        @torch.jit.script
        def op_script(data: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return kornia.random_rotation(data, degrees=45.0)

        input = torch.tensor([[1., 0., 0., 2.],
                              [0., 0., 0., 0.],
                              [0., 1., 2., 0.],
                              [0., 0., 1., 2.]])  # 4 x 4

        # Build jit trace
        op_trace = torch.jit.trace(op_script, (input, ))

        # Create new inputs
        input = torch.tensor([[0., 0., 0.],
                              [5., 5., 0.],
                              [0., 0., 0.]])  # 3 x 3

        expected = torch.tensor([[[0.0000, 0.2584, 0.0000],
                                  [2.9552, 5.0000, 0.2584],
                                  [1.6841, 0.4373, 0.0000]]])

        actual = op_trace(input)

        assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3), device=device, dtype=dtype)  # 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomRotation(degrees=(15.0, 15.0), p=1.), (input, ), raise_exception=True)


class TestRandomCrop:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False, p=1.)
        repr = "RandomCrop(crop_size=(2, 3), padding=(0, 1), fill=10, pad_if_needed=False, padding_mode=constant, "\
               "resample=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_no_padding(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]]], device=device, dtype=dtype)
        expected = torch.tensor([[[
            [3., 4., 5.],
            [6., 7., 8.]
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_no_padding_batch(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([
            [[[0., 1., 2.],
              [3., 4., 5.]]],
            [[[3., 4., 5.],
              [6., 7., 8.]]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=None, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_same_on_batch(self, device, dtype):
        f = RandomCrop(size=(2, 3), padding=1, same_on_batch=True, align_corners=True, p=1.)
        input = torch.eye(3, device=device, dtype=dtype).unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 3, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_padding_batch_1(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([[[
            [1., 2., 0.],
            [4., 5., 0.]
        ]], [[
            [7., 8., 0.],
            [0., 0., 0.]
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=1, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_padding_batch_2(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([[[
            [1., 2., 10.],
            [4., 5., 10.]
        ]], [[
            [4., 5., 10.],
            [7., 8., 10.],
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=(0, 1), fill=10, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_padding_batch_3(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([[[
            [2., 8., 8.],
            [5., 8., 8.]
        ]], [[
            [4., 5., 8.],
            [7., 8., 8.]
        ]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), padding=(0, 1, 2, 3), fill=8, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected, atol=1e-4, rtol=1e-4)

    def test_pad_if_needed(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)
        expected = torch.tensor([
            [[[9., 9., 9.],
              [0., 1., 2.]]],
            [[[0., 1., 2.],
              [9., 9., 9.]]]], device=device, dtype=dtype)
        rc = RandomCrop(size=(2, 3), pad_if_needed=True, fill=9, align_corners=True, p=1.)
        out = rc(inp)

        assert_allclose(out, expected)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomCrop(size=(3, 3), p=1.), (inp, ), raise_exception=True)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.).forward
        op_script = torch.jit.script(op)
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        actual = op_script(img)
        expected = kornia.center_crop3d(img)
        assert_allclose(actual, expected)

    @pytest.mark.skip("Need to fix Union type")
    def test_jit_trace(self, device, dtype):
        # Define script
        op = RandomCrop(size=(3, 3), p=1.).forward
        op_script = torch.jit.script(op)
        # 1. Trace op
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        op_trace = torch.jit.trace(op_script, (img,))

        # 2. Generate new input
        img = torch.ones(1, 1, 5, 6, device=device, dtype=dtype)

        # 3. Evaluate
        actual = op_trace(img)
        expected = op(img)
        assert_allclose(actual, expected)


class TestRandomResizedCrop:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self):
        f = RandomResizedCrop(size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0))
        repr = "RandomResizedCrop(size=(2, 3), scale=tensor([1., 1.]), ratio=tensor([1., 1.]), "\
               "interpolation=BILINEAR, p=1.0, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_no_resize(self, device, dtype):
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[3.0937, 4.3750, 4.8750],
               [3.9375, 5.4688, 5.9062]]]], device=device, dtype=dtype)
        rrc = RandomResizedCrop(
            size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomResizedCrop(
            size=(2, 3), scale=(1., 1.), ratio=(1.0, 1.0), same_on_batch=True)
        input = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_crop_scale_ratio(self, device, dtype):
        # This is included in doctest
        torch.manual_seed(0)
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype)

        expected = torch.tensor(
            [[[[0.0000, 0.2500, 0.7500],
               [2.2500, 3.2500, 3.7500],
               [4.5000, 6.2500, 6.7500]]]], device=device, dtype=dtype)
        rrc = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected)

    def test_crop_scale_ratio_batch(self, device, dtype):
        torch.manual_seed(0)
        batch_size = 2
        inp = torch.tensor([[
            [0., 1., 2.],
            [3., 4., 5.],
            [6., 7., 8.]
        ]], device=device, dtype=dtype).repeat(batch_size, 1, 1, 1)

        expected = torch. tensor(
            [[[[0.0000, 0.2500, 0.7500],
               [2.2500, 3.2500, 3.7500],
               [4.5000, 6.2500, 6.7500]]],
             [[[0.0000, 0.2500, 0.7500],
               [2.2500, 3.2500, 3.7500],
               [4.5000, 6.2500, 6.7500]]]], device=device, dtype=dtype)
        rrc = RandomResizedCrop(size=(3, 3), scale=(3., 3.), ratio=(2., 2.))
        # It will crop a size of (2, 2) from the aspect ratio implementation of torch
        out = rrc(inp)
        assert_allclose(out, expected, rtol=1e-4, atol=1e-4)

    def test_gradcheck(self, device, dtype):
        torch.manual_seed(0)  # for random reproductibility
        inp = torch.rand((1, 3, 3), device=device, dtype=dtype)  # 3 x 3
        inp = utils.tensor_to_gradcheck_var(inp)  # to var
        assert gradcheck(RandomResizedCrop(
            size=(3, 3), scale=(1., 1.), ratio=(1., 1.)), (inp, ), raise_exception=True)


class TestRandomEqualize:
    # TODO: improve and implement more meaningful smoke tests e.g check for a consistent
    # return values such a torch.Tensor variable.
    @pytest.mark.xfail(reason="might fail under windows OS due to printing preicision.")
    def test_smoke(self, device, dtype):
        f = RandomEqualize(p=0.5)
        repr = "RandomEqualize(p=0.5, p_batch=1.0, same_on_batch=False, return_transform=False)"
        assert str(f) == repr

    def test_random_equalize(self, device, dtype):
        f = RandomEqualize(p=1.0, return_transform=True)
        f1 = RandomEqualize(p=0., return_transform=True)
        f2 = RandomEqualize(p=1.)
        f3 = RandomEqualize(p=0.)

        bs, channels, height, width = 1, 3, 20, 20

        inputs = self.build_input(channels, height, width, device=device, dtype=dtype).squeeze(dim=0)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ])
        expected = self.build_input(channels, height, width, bs=1, row=row_expected,
                                    device=device, dtype=dtype)

        identity = kornia.eye_like(3, expected)  # 3 x 3

        assert_allclose(f(inputs)[0], expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[0], inputs, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f2(inputs), expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f3(inputs), inputs, rtol=1e-4, atol=1e-4)

    def test_batch_random_equalize(self, device, dtype):
        f = RandomEqualize(p=1.0, return_transform=True)
        f1 = RandomEqualize(p=0., return_transform=True)
        f2 = RandomEqualize(p=1.)
        f3 = RandomEqualize(p=0.)

        bs, channels, height, width = 2, 3, 20, 20

        inputs = self.build_input(channels, height, width, bs, device=device, dtype=dtype)

        row_expected = torch.tensor([
            0.0000, 0.07843, 0.15686, 0.2353, 0.3137, 0.3922, 0.4706, 0.5490, 0.6275,
            0.7059, 0.7843, 0.8627, 0.9412, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000, 1.0000
        ])
        expected = self.build_input(channels, height, width, bs, row=row_expected,
                                    device=device, dtype=dtype)

        identity = kornia.eye_like(3, expected)  # 2 x 3 x 3

        assert_allclose(f(inputs)[0], expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[0], inputs, rtol=1e-4, atol=1e-4)
        assert_allclose(f1(inputs)[1], identity, rtol=1e-4, atol=1e-4)
        assert_allclose(f2(inputs), expected, rtol=1e-4, atol=1e-4)
        assert_allclose(f3(inputs), inputs, rtol=1e-4, atol=1e-4)

    def test_same_on_batch(self, device, dtype):
        f = RandomEqualize(p=0.5, same_on_batch=True)
        input = torch.eye(4, device=device, dtype=dtype)
        input = input.unsqueeze(dim=0).unsqueeze(dim=0).repeat(2, 1, 1, 1)
        res = f(input)
        assert (res[0] == res[1]).all()

    def test_gradcheck(self, device, dtype):

        torch.manual_seed(0)  # for random reproductibility

        input = torch.rand((3, 3, 3), device=device, dtype=dtype)  # 3 x 3 x 3
        input = utils.tensor_to_gradcheck_var(input)  # to var
        assert gradcheck(RandomEqualize(p=0.5), (input,), raise_exception=True)

    @staticmethod
    def build_input(channels, height, width, bs=1, row=None, device='cpu', dtype=torch.float32):
        if row is None:
            row = torch.arange(width, device=device, dtype=dtype) / float(width)

        channel = torch.stack([row] * height)
        image = torch.stack([channel] * channels)
        batch = torch.stack([image] * bs)

        return batch.to(device, dtype)
