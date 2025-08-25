#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

from os.path import abspath, dirname, join as path_join
from unittest import TestLoader, TextTestRunner

directory = dirname(__file__)
path = path_join(abspath(path_join(directory, '..', '..', 'src')))
sys.path.insert(1, path)


# Provide light-weight stubs for scikit-image only when the real package is
# not available. If scikit-image can be imported, use the actual implementation.
try:
    import skimage  # type: ignore  # noqa: F401
except Exception:
    import types
    import sys as _sys
    if 'skimage' not in _sys.modules:
        skimage_stub = types.ModuleType('skimage')
        _sys.modules['skimage'] = skimage_stub
        for sub in ['feature', 'color', 'io', 'filters', 'transform']:
            _sys.modules[f'skimage.{sub}'] = types.ModuleType(f'skimage.{sub}')

if len(sys.argv) > 1 and 'verbosity=' in sys.argv[1]:
    verbosity = int(sys.argv[1].split('=')[1])
else:
    verbosity = 1

sys.exit(not TextTestRunner(verbosity=verbosity).run(TestLoader().discover(directory)).wasSuccessful())
