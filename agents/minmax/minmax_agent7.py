# Minmax v7 — v6 + Singular Extensions.
# If one move's heuristic score exceeds the second-best by SINGULAR_MARGIN,
# that move is "singular": it is searched one extra ply (depth instead of depth-1).
# This focuses effort on positions where one move clearly dominates.
# Based on AIMA 4ed chapter 6.
import time

from agent import Agent
from files_99942500_26482510.heuristic import evaluate
from files_99942500_26482510.zobrist import compute_zobrist_hash, generate_zobrist_table
from oxono import Game

MINUS_INFINITY       = float("-inf")
PLUS_INFINITY        = float("inf")
ZOBRIST_TABLE        = generate_zobrist_table()
QUIESCENCE_DEPTH     = 2
LATE_MOVE_THRESHOLD  = 3
SINGULAR_MARGIN      = 500   # heuristic units above second-best to qualify as singular


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


def _score_moves(state, player):
    """Return list of (score, move) sorted best-first, with terminal moves first."""
    scored = []
    for move in Game.actions(state):
        new_state = state.copy()
        Game.apply(new_state, move)
        move_score = evaluate(new_state, player)
        if Game.is_terminal(new_state):
            move_score += 100000
        scored.append((move_score, move))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _order_moves(state, player, depth, killer_moves):
    killer = killer_moves.get(depth)
    scored = _score_moves(state, player)

    ordered = []
    remaining = []
    for score, move in scored:
        if killer is not None and move == killer:
            ordered.insert(0, (score, move))
        else:
            remaining.append((score, move))

    return ordered + remaining


def _singular_move(scored_moves):
    """
    Return the singular move if the top move's score exceeds the second-best
    by SINGULAR_MARGIN, else None.
    """
    if len(scored_moves) < 2:
        return None
    best_score  = scored_moves[0][0]
    second_score = scored_moves[1][0]
    if best_score - second_score >= SINGULAR_MARGIN:
        return scored_moves[0][1]
    return None


def _search(state, depth, player, deadline, killer_moves, transposition_table):
    scored = _score_moves(state, player)
    if not scored:
        return None

    alpha = MINUS_INFINITY
    best_move = None
    best_score = MINUS_INFINITY
    singular = _singular_move(scored)

    killer = killer_moves.get(depth)
    ordered = _order_moves(state, player, depth, killer_moves)

    for score, move in ordered:
        _check_timeout(deadline)
        new_state = state.copy()
        Game.apply(new_state, move)
        # Extend singular move by 1 extra ply
        search_depth = depth if move == singular else depth - 1
        node_score = _alpha_beta(new_state, search_depth, player, alpha, PLUS_INFINITY,
                                 deadline, killer_moves, transposition_table)
        if node_score > best_score:
            best_score = node_score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move


def _alpha_beta(state, depth, player, alpha, beta, deadline, killer_moves,
                transposition_table):
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

    scored = _score_moves(state, player)
    singular = _singular_move(scored)
    ordered = _order_moves(state, player, depth, killer_moves)

    if state.current_player == player:
        best_score = MINUS_INFINITY
        cutoff = False
        for index, (move_score, move) in enumerate(ordered):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)

            # LMR: reduce late moves
            search_depth = depth - 1
            is_late = index >= LATE_MOVE_THRESHOLD and depth >= 3
            if is_late and move != singular:
                search_depth = max(0, depth - 2)

            # Singular extension: extend moves that clearly dominate
            if move == singular:
                search_depth = depth

            score = _alpha_beta(new_state, search_depth, player, alpha, beta,
                                deadline, killer_moves, transposition_table)

            if is_late and move != singular and score > alpha:
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
        for index, (move_score, move) in enumerate(ordered):
            _check_timeout(deadline)
            new_state = state.copy()
            Game.apply(new_state, move)

            search_depth = depth - 1
            is_late = index >= LATE_MOVE_THRESHOLD and depth >= 3
            if is_late and move != singular:
                search_depth = max(0, depth - 2)

            if move == singular:
                search_depth = depth

            score = _alpha_beta(new_state, search_depth, player, alpha, beta,
                                deadline, killer_moves, transposition_table)

            if is_late and move != singular and score < beta:
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


def _quiescence(state, player, alpha, beta, deadline, q_depth):
    _check_timeout(deadline)
    stand_pat = evaluate(state, player)
    if q_depth <= 0:
        return stand_pat

    tactical_moves = [m for m in Game.actions(state) if Game.is_terminal(_apply(state, m))]
    if not tactical_moves:
        return stand_pat

    if state.current_player == player:
        if stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)
        for move in tactical_moves:
            score = _quiescence(_apply(state, move), player, alpha, beta, deadline, q_depth - 1)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha
    else:
        if stand_pat <= alpha:
            return stand_pat
        beta = min(beta, stand_pat)
        for move in tactical_moves:
            score = _quiescence(_apply(state, move), player, alpha, beta, deadline, q_depth - 1)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return beta


def _apply(state, move):
    new_state = state.copy()
    Game.apply(new_state, move)
    return new_state


def _check_timeout(deadline):
    if time.time() >= deadline:
        raise SearchTimeout()
