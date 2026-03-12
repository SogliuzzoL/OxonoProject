import numpy as np


class Node:
    def __init__(self, prior=0):
        self.prior = prior
        self.value = 0
        self.visit_count = 0
        self.children = {}

    def compute_Q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value / self.visit_count

    def compute_U_value(self, parent_visit_count, c_puct=1.5):
        return c_puct * self.prior * np.sqrt(parent_visit_count) / (1 + self.visit_count)

    def compute_score(self, parent_visit_count, c_puct=1.5):
        return self.compute_Q_value() + self.compute_U_value(parent_visit_count, c_puct)

    def generate_children(self):
        pass


class MCTS:
    def __init__(self):
        self.root = Node()

    def run(self, state, num_simulations, c_puct=1.5):
        pass
