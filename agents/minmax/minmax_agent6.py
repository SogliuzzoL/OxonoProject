# Minmax v6 — v5 + Late Move Reduction (LMR).
# Moves ranked below LATE_MOVE_THRESHOLD in the ordered list are searched at
# depth-2 instead of depth-1. If the reduced search beats alpha, we re-search
# at full depth to confirm. Based on AIMA 4ed chapter 6.
import time

from agent import Agent
from files_99942500_26482510.heuristic import evaluate
from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from oxono import Game

MINUS_INFINITY       = float("-inf")
PLUS_INFINITY        = float("inf")
ZOBRIST_TABLE        = generate_zobrist_table()
QUIESCENCE_DEPTH     = 2
LATE_MOVE_THRESHOLD  = 3   # first N moves get full depth, rest are reduced


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


def _alpha_beta(state, depth, player, alpha, beta, deadline, killer_moves,
                transposition_table, move_index=0):
    _check_timeout(deadline)

    if Game.is_terminal(state):
        return Game.utility(state, player) * 10000
    if depth == 0:
        return _quiescence(state, player, alpha, beta, deadline, QUIESCENCE_DEPTH)

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
        for index, move in enumerate(_order_moves(state, player, depth, killer_moves)):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)

            # Late Move Reduction: reduce depth for moves ranked beyond the threshold
            search_depth = depth - 1
            is_late_move = index >= LATE_MOVE_THRESHOLD and depth >= 3
            if is_late_move:
                search_depth = max(0, depth - 2)

            score = _alpha_beta(new_state, search_depth, player, alpha, beta,
                                deadline, killer_moves, transposition_table, index)

            # Re-search at full depth if reduced search beats alpha
            if is_late_move and score > alpha:
                score = _alpha_beta(new_state, depth - 1, player, alpha, beta,
                                    deadline, killer_moves, transposition_table, index)

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
        for index, move in enumerate(_order_moves(state, player, depth, killer_moves)):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)

            search_depth = depth - 1
            is_late_move = index >= LATE_MOVE_THRESHOLD and depth >= 3
            if is_late_move:
                search_depth = max(0, depth - 2)

            score = _alpha_beta(new_state, search_depth, player, alpha, beta,
                                deadline, killer_moves, transposition_table, index)

            if is_late_move and score < beta:
                score = _alpha_beta(new_state, depth - 1, player, alpha, beta,
                                    deadline, killer_moves, transposition_table, index)

            worst_score = min(score, worst_score)
            beta = min(beta, worst_score)
            if beta <= alpha:
                killer_moves[depth] = move
                cutoff = True
                break
        flag = "upper" if cutoff else "exact"
        transposition_table[state_hash] = (worst_score, depth, flag)
        return worst_score


def _quiescence(state, player, alpha, beta, deadline, q_depth):
    _check_timeout(deadline)

    stand_pat = evaluate(state, player)
    if q_depth <= 0:
        return stand_pat

    tactical_moves = [
        move for move in Game.actions(state)
        if Game.is_terminal(_apply_copy(state, move))
    ]
    if not tactical_moves:
        return stand_pat

    if state.current_player == player:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        for move in tactical_moves:
            score = _quiescence(_apply_copy(state, move), player, alpha, beta, deadline, q_depth - 1)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha
    else:
        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)
        for move in tactical_moves:
            score = _quiescence(_apply_copy(state, move), player, alpha, beta, deadline, q_depth - 1)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return beta


def _apply_copy(state, move):
    new_state = state.copy()
    Game.apply(new_state, move)
    return new_state


def _check_timeout(deadline):
    if time.time() >= deadline:
        raise SearchTimeout()
