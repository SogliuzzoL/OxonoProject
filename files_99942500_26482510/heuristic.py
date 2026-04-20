from oxono import Game


X_SYMBOL = "x"
O_SYMBOL = "o"
MAX_HEURISTIC_VALUE = 10000
BOARD_SIZE = 6


def evaluate(state, player):
    if Game.is_terminal(state):
        return Game.utility(state, player) * MAX_HEURISTIC_VALUE
    player_score = 0
    player_score += score_all_lines(state, player)
    player_score += score_immediate_threats(state, player)
    player_score += score_remaining_pieces(state, player)
    player_score += score_totem_control(state, player)

    return player_score


def score_all_lines(state, player):
    board = state.board
    total = 0

    for row in board:
        total += score_line(row, player)

    for col in range(len(board[0])):
        column = [board[row][col] for row in range(len(board))]
        total += score_line(column, player)

    return total


def score_line(line, player):
    score = 0

    number_consecutive_player_pieces = 0
    number_consecutive_opponent_pieces = 0
    number_consecutive_x_pieces = 0
    number_consecutive_y_pieces = 0

    for cell in line:
        # Update based on color
        if cell is not None and cell[1] == player:
            number_consecutive_player_pieces += 1
            number_consecutive_opponent_pieces = 0
        elif cell is not None and cell[1] != player:
            number_consecutive_opponent_pieces += 1
            number_consecutive_player_pieces = 0
        else:
            number_consecutive_player_pieces = 0
            number_consecutive_opponent_pieces = 0

        # Update based on symbol
        if cell is not None and cell[0] == X_SYMBOL:
            number_consecutive_x_pieces += 1
            number_consecutive_y_pieces = 0
        elif cell is not None and cell[0] == O_SYMBOL:
            number_consecutive_y_pieces += 1
            number_consecutive_x_pieces = 0
        else:
            number_consecutive_x_pieces = 0
            number_consecutive_y_pieces = 0

        # Score for consecutive color pieces
        if number_consecutive_player_pieces == 2:
            score += 10
        elif number_consecutive_player_pieces == 3:
            score += 100
        if number_consecutive_opponent_pieces == 2:
            score -= 10
        elif number_consecutive_opponent_pieces == 3:
            score -= 120

        # Score for consecutive symbol pieces
        if number_consecutive_x_pieces == 2:
            score += 5
        elif number_consecutive_x_pieces == 3:
            score += 30
        if number_consecutive_y_pieces == 2:
            score += 5
        elif number_consecutive_y_pieces == 3:
            score += 30

    return score


def score_immediate_threats(state, player):
    board = state.board
    opponent = get_opponent(player)
    score = 0
    player_threat_count = 0
    opponent_threat_count = 0

    for row in board:
        line_score, player_threats, opponent_threats = score_threats_in_line(
            row, player, opponent
        )
        score += line_score
        player_threat_count += player_threats
        opponent_threat_count += opponent_threats

    for col in range(len(board[0])):
        column = [board[row][col] for row in range(len(board))]
        line_score, player_threats, opponent_threats = score_threats_in_line(
            column, player, opponent
        )
        score += line_score
        player_threat_count += player_threats
        opponent_threat_count += opponent_threats

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
        window = line[start : start + 4]

        empty_cell_count = sum(1 for cell in window if cell is None)
        if empty_cell_count != 1:
            continue

        non_empty_cells_window = [cell for cell in window if cell is not None]

        # color threats
        player_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[1] == player
        )
        opponent_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[1] == opponent
        )

        if player_pieces_window_count == 3:
            score += 800
            player_threat_count += 1
        if opponent_pieces_window_count == 3:
            score -= 900
            opponent_threat_count += 1

        # symbol threats: shared threat, not attributed to a specific player
        x_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[0] == X_SYMBOL
        )
        o_pieces_window_count = sum(
            1 for cell in non_empty_cells_window if cell[0] == O_SYMBOL
        )

        if x_pieces_window_count == 3:
            score += 120
            player_threat_count += 1
        if o_pieces_window_count == 3:
            score += 120
            player_threat_count += 1

    return score, player_threat_count, opponent_threat_count


def score_remaining_pieces(state, player):
    """
    Score based on piece count advantage.
    Having more pieces remaining than the opponent means more placement flexibility.
    """
    opponent = get_opponent(player)
    player_pieces_remaining = state.pieces_x[player] + state.pieces_o[player]
    opponent_pieces_remaining = state.pieces_x[opponent] + state.pieces_o[opponent]
    return (player_pieces_remaining - opponent_pieces_remaining) * 5


def score_totem_control(state, player):
    """
    Score based on how many of the player's pieces surround each totem.
    More friendly pieces around a totem = more placement options on future turns.
    """
    score = 0
    opponent = get_opponent(player)
    board = state.board

    for totem_position in [state.totem_O, state.totem_X]:
        totem_row, totem_col = totem_position
        for delta_row in [-1, 0, 1]:
            for delta_col in [-1, 0, 1]:
                if delta_row == 0 and delta_col == 0:
                    continue
                adjacent_row = totem_row + delta_row
                adjacent_col = totem_col + delta_col
                if 0 <= adjacent_row < BOARD_SIZE and 0 <= adjacent_col < BOARD_SIZE:
                    cell = board[adjacent_row][adjacent_col]
                    if cell is not None and cell[1] == player:
                        score += 3
                    elif cell is not None and cell[1] == opponent:
                        score -= 3

    return score


def get_opponent(player):
    return 1 - player
