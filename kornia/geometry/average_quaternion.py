from __future__ import annotations  

import numpy as np
import numpy.matlib as npm

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

def average_quaternions(quaternions: list[Quaternion] | Quaternion) -> Quaternion:
    # Check if the input is a list of quaternions or a single quaternion
    if isinstance(quaternions, list):
        Q = np.array([[quat.w, quat.x, quat.y, quat.z] for quat in quaternions])

    elif isinstance(quaternions, Quaternion):
        Q = np.array([[quaternions.w, quaternions.x, quaternions.y, quaternions.z]])
    else:
        raise ValueError("Input must be a list of quaternions or a single quaternion.")

    result = averageQuaternions(Q)

    result_quaternion = Quaternion(result[0], result[1], result[2], result[3])

    return result_quaternion


def averageQuaternions(Q: np.ndarray) -> np.ndarray:
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))

    for i in range(0, M):
        q = Q[i, :]
        A = np.outer(q, q) + A
    A = (1.0 / M) * A
    eigenValues, eigenVectors = np.linalg.eig(A)
    eigenVectors = eigenVectors[:, eigenValues.argsort()[::-1]]
    return np.real(eigenVectors[:, 0].A1)

# Example usage:
# quaternions_list = [Quaternion(1, 2, 3, 4), Quaternion(2, 3, 4, 5), ...]
# result = average_quaternions(quaternions_list)

# Example usage for a batched quaternion:
# quaternion_batch = Quaternion(1, 2, 3, 4)
# result = average_quaternions(quaternion_batch)
