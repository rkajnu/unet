# test_example.py
import pytest


def test_add():
    assert 3 == 3


def test_subtract():
    assert 4 == 4


# this will fail
def test_mult():
    assert 2 * 2 == 5
