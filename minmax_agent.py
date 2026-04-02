from agent import Agent
from oxono import Game


class MinimaxAgent(Agent):
    def __init__(self, player):
        super().__init__(player)

    def act(self, state, remaining_time):
        pass


def count_sequence(line, player):
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
        #-->to develop further but it means that the color of the piece in the cell is the color off the player
            number_consecutive_player_pieces += 1
            number_consecutive_opponent_pieces = 0
        elif cell is not None and cell[1] != player:
            number_consecutive_opponent_pieces += 1
            number_consecutive_player_pieces = 0
        else:
            number_consecutive_player_pieces = 0
            number_consecutive_opponent_pieces = 0

        # Update based on symbol
        if cell is not None and cell[0] == 'x':
            number_consecutive_x_pieces += 1
            number_consecutive_y_pieces = 0
        elif cell is not None and cell[0] == 'o':
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
            score -= 100

        # Score for consecutive symbol pieces
        # --> not focus on player here but 
        if number_consecutive_x_pieces == 2:
            score += 10
        elif number_consecutive_x_pieces == 3:
            score += 50
        if number_consecutive_y_pieces == 2:
            score += 10
        elif number_consecutive_y_pieces == 3:
            score += 50

    return score


def evaluate(state, player):
    """
    Estimate the value of a state for the player
    """
    player_score = 0

    # Calculate score for all rows and columns
    for index in range(6):
        row = [state.board[index][col] for col in range(6)]
        player_score += count_sequence(row, player)

        column = [state.board[row_index][index] for row_index in range(6)]
        player_score += count_sequence(column, player)

    return player_score
