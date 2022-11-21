# Temporary shape check functions until KORNIA_CHECK_SHAPE is ready
def check_se2_t_shape(t):
    check_so2_t_shape(t)


def check_so2_t_shape(t):
    t_shape = t.shape
    len_t_shape = len(t_shape)
    if (
        ((len_t_shape == 3) and ((t_shape[1] != 2) or (t_shape[2] != 1)))
        or ((len_t_shape == 2) and (t_shape[1] != 2))
        or ((len_t_shape == 1) and (t_shape[0] != 2))
    ):
        raise ValueError(f"Invalid translation shape, we expect [B, 2], [B, 2, 1] or [2] Got: {t_shape}")


def check_v_shape(v):
    v_shape = v.shape
    len_v_shape = len(v_shape)
    if ((len_v_shape == 2) and (v_shape[1] != 3)) or ((len_v_shape == 1) and (v_shape[0] != 3)) or (len_v_shape > 3):
        raise ValueError(f"Invalid input shape, we expect [B, 3], [3] Got: {v_shape}")
