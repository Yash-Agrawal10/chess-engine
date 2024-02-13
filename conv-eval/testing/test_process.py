import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import process

# True Board Layouts
start = np.zeros((12, 64))
start[0][60] = 1
start[1][62] = start[1][57] = 1
start[2][58] = start[2][61] = 1
start[3][56] = start[3][63] = 1
start[4][59] = 1
start[5][48:56] = 1
start[6][8:16] = 1
start[7][3] = 1
start[8][0] = start[8][7] = 1
start[9][2] = start[9][5] = 1
start[10][1] = start[10][6] = 1
start[11][4] = 1

e4 = np.zeros((12, 64))
e4[0][59] = 1
e4[1][62] = e4[1][57] = 1
e4[2][58] = e4[2][61] = 1
e4[3][56] = e4[3][63] = 1
e4[4][60] = 1
e4[5][48:56] = 1
e4[6][8:11] = e4[6][12:16] = e4[6][27] = 1
e4[7][4] = 1
e4[8][0] = e4[8][7] = 1
e4[9][2] = e4[9][5] = 1
e4[10][1] = e4[10][6] = 1
e4[11][3] = 1

# Helper Functions

def compare_pieceboards(expected, actual, print_errors=True) -> list[str]:
    errors = []
    for i in range(len(actual)):
        if not np.array_equal(actual[i], expected[i]):
            errors.append("error in pieceboard {}".format(i))
            if print_errors:
                print("error in pieceboard {}".format(i))
                print("expected:")
                print(expected[i].reshape(8, 8))
                print("actual:")
                print(actual[i].reshape(8, 8))
    return errors

# Tests

def test_starting_position():
    # expected output
    expected = start
    # actual output
    actual = process.fen_to_piecemaps("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") 
    # compare
    errors = compare_pieceboards(expected, actual)
    assert not errors, "errors occured:\n{}".format("\n".join(errors))

def test_e4():
    # expected output
    expected = e4
    # actual output
    actual = process.fen_to_piecemaps("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    # compare
    errors = compare_pieceboards(expected, actual)
    assert not errors, "errors occured:\n{}".format("\n".join(errors))