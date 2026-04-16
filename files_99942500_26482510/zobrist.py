import random


def generate_zobrist_table():
    random.seed(42)
    table = {
        "pieces": [
            [[random.getrandbits(64) for _ in range(4)] for _ in range(6)]
            for _ in range(6)
        ],
        "totem_O": [[random.getrandbits(64) for _ in range(6)] for _ in range(6)],
        "totem_X": [[random.getrandbits(64) for _ in range(6)] for _ in range(6)],
        "player": [random.getrandbits(64) for _ in range(2)],
    }
    return table


def compute_zobrist_hash(table, state):
    h = 0
    for row in range(6):
        for col in range(6):
            piece = state.board[row][col]
            if piece is not None:
                symbol, player = piece
                type_idx = (0 if symbol == "o" else 2) + player
                h ^= table["pieces"][row][col][type_idx]

    row, col = state.totem_O
    h ^= table["totem_O"][row][col]

    row, col = state.totem_X
    h ^= table["totem_X"][row][col]

    h ^= table["player"][state.current_player]

    return h
