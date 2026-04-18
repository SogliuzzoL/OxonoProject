# Heuristic v2 — v1 + piece count advantage.
# Having more pieces remaining gives more placement flexibility.
from files_99942500_26482510.heuristic_v1 import (
    evaluate as _evaluate_v1,
    get_opponent,
    MAX_HEURISTIC_VALUE,
)
from oxono import Game

PIECE_ADVANTAGE_WEIGHT = 5


def evaluate(state, player):
    if Game.is_terminal(state):
        return Game.utility(state, player) * MAX_HEURISTIC_VALUE
    score = _evaluate_v1(state, player)
    score += score_remaining_pieces(state, player)
    return score


def score_remaining_pieces(state, player):
    opponent = get_opponent(player)
    player_pieces = state.pieces_x[player] + state.pieces_o[player]
    opponent_pieces = state.pieces_x[opponent] + state.pieces_o[opponent]
    return (player_pieces - opponent_pieces) * PIECE_ADVANTAGE_WEIGHT
