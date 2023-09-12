import pytest
import keras_core as keras

from kornia.core import IntegratedTensor
from kornia.testing import BaseTester

class TestIntegratedTensor(BaseTester):
    def test_smoke(self, dtype):
        data = keras.random.uniform(shape=(1, 2, 3, 4), dtype=dtype)
        tensor = IntegratedTensor(data)
        assert isinstance(tensor, IntegratedTensor)
        assert tensor.shape == (1, 2, 3, 4)
        assert tensor.dtype == dtype
        self.assert_close(data, tensor.tensor)

    def test_serialization(self, dtype, tmp_path):
        data = keras.random.uniform(shape=(1, 2, 3, 4), dtype=dtype)
        tensor = IntegratedTensor(data)

        file_path = tmp_path / "tensor.keras"
        keras.saving.save_model(tensor, file_path)

        loaded_tensor = keras.saving.load_model(file_path, compile=False, safe_mode=True)
        assert isinstance(loaded_tensor, IntegratedTensor)

        self.assert_close(loaded_tensor.tensor, tensor.tensor)
        