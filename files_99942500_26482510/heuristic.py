from oxono import Game


X_SYMBOL = "x"
O_SYMBOL = "o"


def evaluate(state, player):
    """
    Estimate the value of a state for the player
    """
    if Game.is_terminal(state):
        return Game.utility(state, player) * 10000
    player_score = 0
    player_score += score_all_lines(state, player)
    player_score += score_immediate_threats(state, player)

    return player_score



def score_all_lines(state, player):
    """
    Score for rows and columns checking color/symbol alignments
    """
    board = state.board
    total = 0

    for row in board:
        total += score_line(row, player)

    for col in range(len(board[0])):
        column = [board[row][col] for row in range(len(board))]
        total += score_line(column, player)

    return total


def score_line(line, player):
    """
    Create a score of a line (row or col) for a player
    """
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
    """
    Detect windows that are one move away from a winning pattern
    """
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

    # if one player has multiple threats possibilites
    if player_threat_count >= 2:
        score += 5000
    if opponent_threat_count >= 2:
        score -= 6000

    return score


def score_threats_in_line(line, player, opponent):

    score = 0
    player_threat_count = 0
    opponent_threat_count = 0
    # on check fenetre de 4 cases donc len-3
    # soit 0 à 3, soit 1 À 4, soit 2 À 5
    for start in range(len(line) - 3):
        window = line[start : start + 4]

        empty_cell_count = sum(1 for cell in window if cell is None)
        # on check uniquement les fenetres de cases avec exactement 1 case vide
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

        # symbol threats
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


def get_opponent(player):
    return 1 - player
