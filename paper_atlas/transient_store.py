"""Bounded process-memory state for active pages; never reads or writes files."""
from collections import OrderedDict
from copy import deepcopy
from threading import RLock


class TransientStore:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self._items = OrderedDict()
        self._lock = RLock()

    def put(self, key, value):
        with self._lock:
            self._items[key] = deepcopy(value)
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)

    def get(self, key):
        with self._lock:
            if key not in self._items:
                raise KeyError(key)
            return deepcopy(self._items[key])
