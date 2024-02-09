import numpy as np
import processing

def test_fen_to_worldviews():
        # test starting position function
        starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

        my_starting_wk = np.zeros((64))
        my_starting_wk[60] = 1
        my_starting_wn = np.zeros((64))
        my_starting_wn[62] = 1
        my_starting_wn[57] = 1
        my_starting_wb = np.zeros((64))
        my_starting_wb[58] = 1
        my_starting_wb[61] = 1
        my_starting_wr = np.zeros((64))
        my_starting_wr[56] = 1
        my_starting_wr[63] = 1
        my_starting_wq = np.zeros((64))
        my_starting_wq[59] = 1
        my_starting_wp = np.zeros((64))
        my_starting_wp[48:56] = 1

        my_starting_bp = np.zeros((64))
        my_starting_bp[8:16] = 1
        my_starting_bq = np.zeros((64))
        my_starting_bq[3] = 1
        my_starting_br = np.zeros((64))
        my_starting_br[0] = 1
        my_starting_br[7] = 1
        my_starting_bb = np.zeros((64))
        my_starting_bb[2] = 1
        my_starting_bb[5] = 1
        my_starting_bn = np.zeros((64))
        my_starting_bn[1] = 1
        my_starting_bn[6] = 1
        my_starting_bk = np.zeros((64))
        my_starting_bk[4] = 1

        their_starting_wk = np.zeros((64))
        their_starting_wk[3] = 1
        their_starting_wn = np.zeros((64))
        their_starting_wn[1] = 1
        their_starting_wn[6] = 1
        their_starting_wb = np.zeros((64))
        their_starting_wb[2] = 1
        their_starting_wb[5] = 1
        their_starting_wr = np.zeros((64))
        their_starting_wr[0] = 1
        their_starting_wr[7] = 1
        their_starting_wq = np.zeros((64))
        their_starting_wq[4] = 1
        their_starting_wk = np.zeros((64))
        their_starting_wk[3] = 1
        their_starting_wp = np.zeros((64))
        their_starting_wp[8:16] = 1
        their_starting_bp = np.zeros((64))
        their_starting_bp[48:56] = 1
        their_starting_bq = np.zeros((64))
        their_starting_bq[60] = 1
        their_starting_br = np.zeros((64))
        their_starting_br[56] = 1
        their_starting_br[63] = 1
        their_starting_bb = np.zeros((64))
        their_starting_bb[58] = 1
        their_starting_bb[61] = 1
        their_starting_bn = np.zeros((64))
        their_starting_bn[57] = 1
        their_starting_bn[62] = 1
        their_starting_bk = np.zeros((64))
        their_starting_bk[59] = 1

        my_starting_worldview = np.array([my_starting_wk, my_starting_wn, my_starting_wb, my_starting_wr, my_starting_wq, my_starting_wp, my_starting_bp, my_starting_bq, my_starting_br, my_starting_bb, my_starting_bn, my_starting_bk])
        their_starting_worldview = np.array([their_starting_bk, their_starting_bn, their_starting_bb, their_starting_br, their_starting_bq, their_starting_bp, their_starting_wp, their_starting_wq, their_starting_wr, their_starting_wb, their_starting_wn, their_starting_wk])
        white_to_move = True

        output = processing.fen_to_worldviews(starting_fen)
        errors = []
        for i in range(12):
            if not np.array_equal(output[0][i], my_starting_worldview[i]):
                    errors.append("error in my starting pieces {}".format(i))
            if not np.array_equal(output[1][i], their_starting_worldview[i]):
                    errors.append("error in their starting pieces {}".format(i))
        if not np.array_equal(output[0], my_starting_worldview):
            errors.append("error in my starting worldview")
        if not np.array_equal(output[1], their_starting_worldview):
            errors.append("error in their starting worldview")
        if output[2] != white_to_move:
            errors.append("error in starting side to move")

        assert not errors, "errors occured:\n{}".format("\n".join(errors))