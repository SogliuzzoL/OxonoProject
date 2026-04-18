try:
    import torch
except ImportError:
    torch = None

from oxono import State


def all_possible_actions() -> list[tuple[str, tuple[int, int], tuple[int, int]]]:
    actions = []

    for totem in ["O", "X"]:
        for totem_row in range(6):
            for totem_col in range(6):
                for piece_row in range(6):
                    for piece_col in range(6):
                        if (totem_row, totem_col) != (piece_row, piece_col):
                            move = (
                                totem,
                                (totem_row, totem_col),
                                (piece_row, piece_col),
                            )
                            actions.append(move)

    return actions


def state_to_tensor(state: State):
    tensor = torch.zeros((7, 6, 6), dtype=torch.float32)
    for row in range(6):
        for col in range(6):
            square = state.board[row][col]
            if square is None:
                continue

            symbol = square[0]
            color = square[1]
            if symbol == "o":
                if color:
                    tensor[0, row, col] = 1.0
                else:
                    tensor[1, row, col] = 1.0
            elif symbol == "x":
                if color:
                    tensor[2, row, col] = 1.0
                else:
                    tensor[3, row, col] = 1.0

    totem_O = state.totem_O
    totem_X = state.totem_X

    tensor[4, totem_O[0], totem_O[1]] = 1.0
    tensor[5, totem_X[0], totem_X[1]] = 1.0

    current_player = state.current_player
    tensor[6, :, :] = 1.0 if current_player else 0.0

    return tensor
