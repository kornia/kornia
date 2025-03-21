import torch
from testing.base import assert_close

def reproducibility_test(input, seq):
    """Any tests failed here indicate the output cannot be reproduced by the same params."""
    if isinstance(input, (tuple, list)):
        output_1 = seq(*input)
        output_2 = seq(*input, params=seq._params)
    else:
        output_1 = seq(input)
        output_2 = seq(input, params=seq._params)

    if isinstance(output_1, (tuple, list)) and isinstance(output_2, (tuple, list)):
        [
            assert_close(o1, o2)
            for o1, o2 in zip(output_1, output_2)
            if isinstance(o1, (torch.Tensor,)) and isinstance(o2, (torch.Tensor,))
        ]
    elif isinstance(output_1, (tuple, list)) and isinstance(output_2, (torch.Tensor,)):
        assert_close(output_1[0], output_2)
    elif isinstance(output_2, (tuple, list)) and isinstance(output_1, (torch.Tensor,)):
        assert_close(output_1, output_2[0])
    elif isinstance(output_2, (torch.Tensor,)) and isinstance(output_1, (torch.Tensor,)):
        assert_close(output_1, output_2, msg=f"{seq._params}")
    else:
        raise AssertionError(f"cannot compare {type(output_1)} and {type(output_2)}")