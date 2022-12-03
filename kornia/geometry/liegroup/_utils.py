# Temporary shape check functions until KORNIA_CHECK_SHAPE is ready
def check_so2_z_shape(z):
    z_shape = z.shape
    len_z_shape = len(z_shape)
    if (len_z_shape == 2 and z_shape[1] != 1) or (len_z_shape == 0 and not z.numel()) or (len_z_shape > 2):
        raise ValueError(f"Invalid input size, we expect [B, 1], [B] or []. Got: {z.shape}")


def check_so2_t_shape(t):
    t_shape = t.shape
    len_t_shape = len(t_shape)
    if ((len_t_shape == 2) and (t_shape[1] > 2)) or ((len_t_shape == 1) and (t_shape[0] != 2)) or (len_t_shape > 2):
        raise ValueError(f"Invalid translation shape, we expect [B, 2] or [2] Got: {t_shape}")


def check_v_shape(v):
    v_shape = v.shape
    len_v_shape = len(v_shape)
    if ((len_v_shape == 2) and (v_shape[1] != 3)) or ((len_v_shape == 1) and (v_shape[0] != 3)) or (len_v_shape > 3):
        raise ValueError(f"Invalid input shape, we expect [B, 3], [3] Got: {v_shape}")


def check_so2_theta_shape(theta):
    theta_shape = theta.shape
    len_theta_shape = len(theta_shape)
    if (
        (len_theta_shape == 2 and theta_shape[1] != 1)
        or (len_theta_shape == 0 and not theta.numel())
        or (len_theta_shape > 2)
    ):
        raise ValueError(f"Invalid input size, we expect [B, 1] or [B]. Got: {theta_shape}")


def check_so2_matrix_shape(matrix):
    matrix_shape = matrix.shape
    len_matrix_shape = len(matrix_shape)
    if (
        (len_matrix_shape == 3 and (matrix_shape[1] != 2 or matrix_shape[2] != 2))
        or (len_matrix_shape == 2 and (matrix_shape[0] != 2 or matrix_shape[1] != 2))
        or (len_matrix_shape > 3 or len_matrix_shape < 2)
    ):
        raise ValueError(f"Invalid input size, we expect [B, 2, 2] or [2, 2]. Got: {matrix_shape}")


def check_se2_t_shape(t):
    check_so2_t_shape(t)


def check_se2_r_t_shape(r, t):
    if ((len(r.z.shape) == 1) and (len(t.shape) == 2)) or ((len(r.z.shape) == 0) and len(t.shape) == 1):
        check_se2_t_shape(t)
    else:
        raise ValueError(
            f"Invalid input, both the inputs should be either batched or unbatched. Got: {r.z.shape} and {t.shape}"
        )
