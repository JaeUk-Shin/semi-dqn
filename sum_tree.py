import numpy as np
from typing import Sequence


class SumTree:
    def __init__(self):
        self.capacity = 1
        self.leftmost = 1   # starting position of leaf nodes
        self.tree = np.zeros(2 * self.capacity, dtype=np.float)
        self.ptr = 1    # track a position to write a new data
        self.size = 1

    def query(self, random_nums: Sequence[float]) -> Sequence[int]:
        return [self.single_query(rand_num) for rand_num in random_nums]

    def add(self, values: Sequence[float]) -> None:
        for value in values:
            self.single_add(value)
        return

    def single_query(self, rand_num: float) -> int:
        assert 0. <= rand_num < self.root()
        idx = 1
        while idx < self.leftmost:
            value = self.tree[idx]
            left, right = 2 * idx, 2 * idx + 1
            if value < self.tree[left]:
                idx = left
            else:
                idx = right
                value -= self.tree[left]
        return idx

    def single_add(self, value: float) -> None:
        self.tree[self.ptr] = value
        parent = self.ptr // 2
        while parent > 0:
            # update keys of intermediate nodes
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2
        self.ptr += 1
        self.size += 1
        return

    def root(self):
        return self.tree[1]

    def expand_temp(self):
        # add extra level to the tree
        # for efficiency, we expand the original tree by copying the original data to even indices of new leaf nodes
        # this keeps sum tree structure of the original tree without key updates
        # values of unused leaf nodes are initialized to 0
        # if the current capacity is not reached, it is not recommended to resize the tree, since it is waste of memory
        self.capacity *= 2
        expanded_tree = np.zeros(2*self.capacity, dtype=np.float)
        expanded_tree[:self.capacity] = self.tree
        expanded_tree[self.capacity:: 2] = self.tree[self.leftmost:]
        self.tree = expanded_tree
        self.leftmost *= 2
        self.ptr = self.leftmost + 1    # data registration must be done at leaves
        return

    def expand(self):
        leaves = self.tree[self.leftmost:]
        self.capacity *= 2
        self.leftmost *= 2
        self.ptr = self.leftmost
        self.tree = np.zeros(2*self.capacity, dtype=np.float)
        self.add(leaves)
        return