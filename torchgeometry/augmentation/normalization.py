import torch
from torch.jit import ScriptModule, script_method


class Normalise(ScriptModule):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(Normalise, self).__init__()
        self.mean: torch.Tensor = mean
        self.std: torch.Tensor = std

    @script_method
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '()'

# TODO: implement me
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L20
def normalise(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data_norm = None
    return data_norm

# - denormalise
