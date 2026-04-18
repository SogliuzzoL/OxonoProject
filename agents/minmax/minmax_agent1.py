# Minmax v1 — base: alpha-beta pruning + iterative deepening only.
# No move ordering, no killer moves, no transposition table, no quiescence.
import time

from agent import Agent
from files_99942500_26482510.heuristic import evaluate
from oxono import Game

MINUS_INFINITY = float("-inf")
PLUS_INFINITY  = float("inf")


class SearchTimeout(Exception):
    pass


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        t0 = time.time()
        for move in Game.actions(state):
            test_state = state.copy()
            Game.apply(test_state, move)
            if Game.is_terminal(test_state):
                return move

        remaining_time -= time.time() - t0
        pieces_remaining = state.pieces_x[self.player] + state.pieces_o[self.player]
        time_per_move = remaining_time / max(pieces_remaining, 1)
        deadline = time.time() + max(0.0, time_per_move)

        best_move = list(Game.actions(state))[0]
        depth = 1

        while True:
            if time.time() >= deadline:
                break
            try:
                current_best = _search(state, depth, self.player, deadline)
                if current_best is not None:
                    best_move = current_best
                depth += 1
            except SearchTimeout:
                break

        return best_move


def _search(state, depth, player, deadline):
    actions = list(Game.actions(state))
    if not actions:
        return None

    alpha = MINUS_INFINITY
    best_move = None
    best_score = MINUS_INFINITY

    for move in actions:
        _check_timeout(deadline)
        new_state = state.copy()
        Game.apply(new_state, move)
        score = _alpha_beta(new_state, depth - 1, player, alpha, PLUS_INFINITY, deadline)
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move


def _alpha_beta(state, depth, player, alpha, beta, deadline):
    _check_timeout(deadline)

    if Game.is_terminal(state):
        return Game.utility(state, player) * 10000
    if depth == 0:
        return evaluate(state, player)

    if state.current_player == player:
        best_score = MINUS_INFINITY
        for move in Game.actions(state):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)
            score = _alpha_beta(new_state, depth - 1, player, alpha, beta, deadline)
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score
    else:
        worst_score = PLUS_INFINITY
        for move in Game.actions(state):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)
            score = _alpha_beta(new_state, depth - 1, player, alpha, beta, deadline)
            worst_score = min(score, worst_score)
            beta = min(beta, worst_score)
            if beta <= alpha:
                break
        return worst_score


def _check_timeout(deadline):
    if time.time() >= deadline:
        raise SearchTimeout()
