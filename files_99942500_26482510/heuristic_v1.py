# Heuristic v1 — base: line scoring + immediate threat detection only.
from oxono import Game

X_SYMBOL = "x"
O_SYMBOL = "o"
MAX_HEURISTIC_VALUE = 10000
BOARD_SIZE = 6


def evaluate(state, player):
    if Game.is_terminal(state):
        return Game.utility(state, player) * MAX_HEURISTIC_VALUE
    score = 0
    score += score_all_lines(state, player)
    score += score_immediate_threats(state, player)
    return score


def score_all_lines(state, player):
    board = state.board
    total = 0
    for row in board:
        total += score_line(row, player)
    for col in range(BOARD_SIZE):
        column = [board[row][col] for row in range(BOARD_SIZE)]
        total += score_line(column, player)
    return total


def score_line(line, player):
    score = 0
    consecutive_player = 0
    consecutive_opponent = 0
    consecutive_x = 0
    consecutive_o = 0

    for cell in line:
        if cell is not None and cell[1] == player:
            consecutive_player += 1
            consecutive_opponent = 0
        elif cell is not None and cell[1] != player:
            consecutive_opponent += 1
            consecutive_player = 0
        else:
            consecutive_player = 0
            consecutive_opponent = 0

        if cell is not None and cell[0] == X_SYMBOL:
            consecutive_x += 1
            consecutive_o = 0
        elif cell is not None and cell[0] == O_SYMBOL:
            consecutive_o += 1
            consecutive_x = 0
        else:
            consecutive_x = 0
            consecutive_o = 0

        if consecutive_player == 2:
            score += 10
        elif consecutive_player == 3:
            score += 100
        if consecutive_opponent == 2:
            score -= 10
        elif consecutive_opponent == 3:
            score -= 120

        if consecutive_x == 2:
            score += 5
        elif consecutive_x == 3:
            score += 30
        if consecutive_o == 2:
            score += 5
        elif consecutive_o == 3:
            score += 30

    return score


def score_immediate_threats(state, player):
    board = state.board
    opponent = get_opponent(player)
    score = 0
    player_threat_count = 0
    opponent_threat_count = 0

    for row in board:
        line_score, p_threats, o_threats = score_threats_in_line(row, player, opponent)
        score += line_score
        player_threat_count += p_threats
        opponent_threat_count += o_threats

    for col in range(BOARD_SIZE):
        column = [board[row][col] for row in range(BOARD_SIZE)]
        line_score, p_threats, o_threats = score_threats_in_line(column, player, opponent)
        score += line_score
        player_threat_count += p_threats
        opponent_threat_count += o_threats

    if player_threat_count >= 2:
        score += 5000
    if opponent_threat_count >= 2:
        score -= 6000

    return score


def score_threats_in_line(line, player, opponent):
    score = 0
    player_threat_count = 0
    opponent_threat_count = 0

    for start in range(len(line) - 3):
        window = line[start: start + 4]
        empty_count = sum(1 for cell in window if cell is None)
        if empty_count != 1:
            continue

        non_empty = [cell for cell in window if cell is not None]
        player_count = sum(1 for cell in non_empty if cell[1] == player)
        opponent_count = sum(1 for cell in non_empty if cell[1] == opponent)

        if player_count == 3:
            score += 800
            player_threat_count += 1
        if opponent_count == 3:
            score -= 900
            opponent_threat_count += 1

        x_count = sum(1 for cell in non_empty if cell[0] == X_SYMBOL)
        o_count = sum(1 for cell in non_empty if cell[0] == O_SYMBOL)

        if x_count == 3:
            score += 120
            player_threat_count += 1
        if o_count == 3:
            score += 120
            player_threat_count += 1

    return score, player_threat_count, opponent_threat_count


def get_opponent(player):
    return 1 - player
