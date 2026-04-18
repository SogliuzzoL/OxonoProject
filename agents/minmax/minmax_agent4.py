# Minmax v4 — v3 + transposition table with exact/lower/upper bounds.
# Positions already evaluated are cached by Zobrist hash. The bound type
# (exact, lower, upper) determines how the cached value is reused.
import time

from agent import Agent
from files_99942500_26482510.heuristic import evaluate
from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from oxono import Game

MINUS_INFINITY = float("-inf")
PLUS_INFINITY  = float("inf")
ZOBRIST_TABLE  = generate_zobrist_table()


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
        killer_moves = {}
        transposition_table = {}
        depth = 1

        while True:
            if time.time() >= deadline:
                break
            try:
                current_best = _search(state, depth, self.player, deadline,
                                       killer_moves, transposition_table)
                if current_best is not None:
                    best_move = current_best
                depth += 1
            except SearchTimeout:
                break

        return best_move


def _order_moves(state, player, depth, killer_moves):
    all_moves = list(Game.actions(state))
    killer = killer_moves.get(depth)

    ordered = []
    if killer is not None and killer in all_moves:
        ordered.append(killer)
        all_moves.remove(killer)

    scored_moves = []
    for move in all_moves:
        new_state = state.copy()
        Game.apply(new_state, move)
        move_score = evaluate(new_state, player)
        if Game.is_terminal(new_state):
            move_score += 100000
        scored_moves.append((move_score, move))
    scored_moves.sort(key=lambda x: x[0], reverse=True)

    ordered.extend(move for _, move in scored_moves)
    return ordered


def _search(state, depth, player, deadline, killer_moves, transposition_table):
    if not list(Game.actions(state)):
        return None

    alpha = MINUS_INFINITY
    best_move = None
    best_score = MINUS_INFINITY

    for move in _order_moves(state, player, depth, killer_moves):
        _check_timeout(deadline)
        new_state = state.copy()
        Game.apply(new_state, move)
        score = _alpha_beta(new_state, depth - 1, player, alpha, PLUS_INFINITY,
                            deadline, killer_moves, transposition_table)
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move


def _alpha_beta(state, depth, player, alpha, beta, deadline, killer_moves, transposition_table):
    _check_timeout(deadline)

    if Game.is_terminal(state):
        return Game.utility(state, player) * 10000
    if depth == 0:
        return evaluate(state, player)

    state_hash = compute_zobrist_hash(ZOBRIST_TABLE, state)
    if state_hash in transposition_table:
        cached_score, cached_depth, flag = transposition_table[state_hash]
        if cached_depth >= depth:
            if flag == "exact":
                return cached_score
            elif flag == "lower":
                alpha = max(alpha, cached_score)
            elif flag == "upper":
                beta = min(beta, cached_score)
            if alpha >= beta:
                return cached_score

    if state.current_player == player:
        best_score = MINUS_INFINITY
        cutoff = False
        for move in _order_moves(state, player, depth, killer_moves):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)
            score = _alpha_beta(new_state, depth - 1, player, alpha, beta,
                                deadline, killer_moves, transposition_table)
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                killer_moves[depth] = move
                cutoff = True
                break
        flag = "lower" if cutoff else "exact"
        transposition_table[state_hash] = (best_score, depth, flag)
        return best_score
    else:
        worst_score = PLUS_INFINITY
        cutoff = False
        for move in _order_moves(state, player, depth, killer_moves):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)
            score = _alpha_beta(new_state, depth - 1, player, alpha, beta,
                                deadline, killer_moves, transposition_table)
            worst_score = min(score, worst_score)
            beta = min(beta, worst_score)
            if beta <= alpha:
                killer_moves[depth] = move
                cutoff = True
                break
        flag = "upper" if cutoff else "exact"
        transposition_table[state_hash] = (worst_score, depth, flag)
        return worst_score


def _check_timeout(deadline):
    if time.time() >= deadline:
        raise SearchTimeout()
