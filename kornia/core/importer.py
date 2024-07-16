"""Singleton importer classes to import libs only if in need.
"""

class PILImporter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.Image = cls._load_pil()
        return cls._instance

    @staticmethod
    def _load_pil():
        from PIL import Image
        return Image


class NumpyImporter:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.np = cls._load_numpy()
        return cls._instance

    @staticmethod
    def _load_numpy():
        import numpy as np
        return np
