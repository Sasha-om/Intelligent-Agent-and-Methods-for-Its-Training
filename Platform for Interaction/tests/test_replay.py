"""Юніт-тести буфера досвіду qlearn.replay.ReplayBuffer (етап 3, без torch)."""
import random

import pytest

from qlearn.replay import ReplayBuffer


def fill(buf, n):
    for k in range(n):
        buf.push(k, k % 4, -1.0, k + 1, False)


def test_len_and_capacity_fifo():
    buf = ReplayBuffer(capacity=5)
    fill(buf, 8)
    assert len(buf) == 5
    assert [t[0] for t in buf.items()] == [3, 4, 5, 6, 7]     # найстаріші витіснено


def test_transition_format():
    buf = ReplayBuffer(capacity=3)
    buf.push(7, 2, -0.3, 8, True)
    assert buf.items()[0] == (7, 2, -0.3, 8, True)


def test_sample_size_and_membership():
    buf = ReplayBuffer(capacity=100)
    fill(buf, 50)
    random.seed(0)
    batch = buf.sample(16)
    assert len(batch) == 16
    assert len(set(batch)) == 16                              # без повторів у батчі
    assert all(t in buf.items() for t in batch)


def test_sample_reproducible_with_seed():
    buf = ReplayBuffer(capacity=100)
    fill(buf, 50)
    random.seed(42)
    a = buf.sample(10)
    random.seed(42)
    b = buf.sample(10)
    assert a == b


def test_sample_too_many_raises():
    buf = ReplayBuffer(capacity=10)
    fill(buf, 3)
    with pytest.raises(ValueError):
        buf.sample(4)


def test_clear():
    buf = ReplayBuffer(capacity=10)
    fill(buf, 3)
    buf.clear()
    assert len(buf) == 0
