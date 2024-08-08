import unittest
import importlib
from types import ModuleType
from typing import Optional, List

from kornia.core.external import LazyLoader


class TestLazyLoader(unittest.TestCase):

    def test_lazy_load_success(self):
        lazy_loader = LazyLoader('math')
        self.assertIsNone(lazy_loader.module)
        result = lazy_loader.sqrt(16)
        self.assertIsNotNone(lazy_loader.module)
        self.assertEqual(result, 4)

    def test_lazy_load_failure(self):
        lazy_loader = LazyLoader('non_existent_module')
        with self.assertRaises(ImportError):
            _ = lazy_loader.some_attribute

    def test_getattr(self):
        lazy_loader = LazyLoader('math')
        self.assertEqual(lazy_loader.pi, 3.141592653589793)

    def test_dir(self):
        lazy_loader = LazyLoader('math')
        attributes = dir(lazy_loader)
        self.assertIn('sqrt', attributes)
        self.assertIn('pi', attributes)