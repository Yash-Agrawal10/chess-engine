import numpy as np

# capital is my pieces, lowercase is their pieces
piecemap = {
    'K': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'P': 5,
    'p': 6, 'q': 7, 'r': 8, 'b': 9, 'n': 10, 'k': 11
}

def fen_to_pieceboards(fen: str):
    # convert FEN string to [piecemaps, white_to_move]
    # ordered by piecemap global variable

    # initialize variables
    pieceboards = np.zeros((12, 64))
    curr_row = 0
    curr_col = 0
    curr_square = 0
    white_to_move = False

    # interpret FEN string
    for i, char in enumerate(fen):
        if char == '/':
            curr_row += 1
            curr_col = 0
        elif char.isdigit():
            curr_col += int(char)
        elif char == ' ':
            i += 1
            if fen[i] == 'w':
                white_to_move = True
            break
        else:
            piece_index = piecemap[char]
            curr_square = curr_row * 8 + curr_col
            pieceboards[piece_index][curr_square] = 1
            curr_col += 1

    # flip worldview based on to_move (currently ignored, white is my_worldview)
    if not white_to_move:
        pieceboards = np.flip(pieceboards)

    # return
    return [pieceboards, white_to_move]

def normalize_eval(eval: int, white_to_move: bool) -> float:
    # normalize eval
    if not white_to_move:
        eval *= -1
    eval = 1 / (1 + np.exp(-eval))
    return eval

def denormalize_eval(normalized_eval: float, white_to_move: bool) -> int:
    # denormalize eval
    eval = np.log(normalized_eval / (1 - normalized_eval))
    if not white_to_move:
        eval *= -1
    return eval

def process_row(row):
    # convert FEN string to pieceboards
    fen = row["fen"]
    pieceboards, white_to_move = fen_to_pieceboards(fen)
    pieceboards = pieceboards.flatten()
    # normalize eval
    eval = row["eval"]
    normalized_eval = normalize_eval(eval, white_to_move)
    # return
    return {"pieceboards": pieceboards, "normalized_eval": normalized_eval}