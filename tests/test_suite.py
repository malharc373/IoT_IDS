"""pytest front-end for tests/smoke_test.py.

The smoke suite is a self-contained runner (`python tests/smoke_test.py`) so it
works with no dev dependencies on a Raspberry Pi. This wrapper exposes the same
checks to pytest — and therefore to CI — without duplicating them.

    pytest tests/ -v
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smoke_test as st  # noqa: E402


@pytest.mark.parametrize("name,fn", st.TESTS, ids=[n for n, _ in st.TESTS])
def test_check(name, fn):
    try:
        fn()
    except st.SkipTest as e:
        pytest.skip(str(e))


def teardown_module(module):
    import shutil
    shutil.rmtree(st.TMP, ignore_errors=True)
