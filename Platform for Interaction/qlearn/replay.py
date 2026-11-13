"""Буфер досвіду (experience replay) для DQN — без torch.

Агент складає сюди переходи (s, slot, r, s2, done) і навчається на випадкових вибірках з них.
Це розриває кореляцію між сусідніми кроками епізоду й робить навчання мережі стабільнішим.
"""
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self._data = deque(maxlen=capacity)   # найстаріші переходи витісняються автоматично

    def push(self, s, slot, r, s2, done):
        """Зберегти перехід: стан, номер дії (0..3), винагорода, наступний стан, чи це фініш."""
        self._data.append((s, slot, r, s2, done))

    def sample(self, n):
        """n різних випадкових переходів (генератор random — той самий, що в усьому агенті: відтворюваність через seed)."""
        if n > len(self._data):
            raise ValueError(f"у буфері {len(self._data)} переходів, а потрібно {n}")
        return random.sample(list(self._data), n)

    def items(self):
        return list(self._data)

    def clear(self):
        self._data.clear()

    def __len__(self):
        return len(self._data)
