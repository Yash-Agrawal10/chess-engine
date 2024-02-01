#include "eval.h"

using namespace chess;

int PAWN_VAL = 100;
int KNIGHT_VAL = 300;
int BISHOP_VAL = 320;
int ROOK_VAL = 500;
int QUEEN_VAL = 900;
int KING_VAL = 100000;

int PAWN_VALS[8][8] = {
    {00, 00, 00, 00, 00, 00, 00, 00}, 
    {50, 50, 50, 50, 50, 50, 50, 50}, 
    {20, 22, 25, 35, 35, 25, 22, 20}, 
    {05, 07, 10, 25, 25, 10, 07, 05}, 
    {00, 00, 05, 20, 20, 05, 00, 00}, 
    {07, 05, 05, 00, 00, 05, 05, 07}, 
    {05, 10, 10, -30, -30, 10, 10, 05},
    {00, 00, 00, 00, 00, 00, 00, 00}
    };

int KNIGHT_VALS[8][8] = {
    {-50, -40, -30, -30, -30, -30, -40, -50}, 
    {-40, -20, 00, 00, 00, 00, -20, -40}, 
    {-30, 00, 05, 10, 10, 05, 00, -30}, 
    {-20, 00, 10, 25, 25, 10, 00, -20}, 
    {-20, 00, 10, 25, 25, 10, 00, -20}, 
    {-30, 00, 05, 10, 10, 05, 00, -30}, 
    {-40, -20, 00, 00, 00, 00, -20, -40},
    {-50, -40, -30, -30, -30, -30, -40, -50}
    };

int BISHOP_VALS[8][8] = {
    {0, -5, -10, -20, -20, -10, -5, 0}, 
    {-05, -10, 10, 00, 00, 10, -10, -05}, 
    {-05, -10, 10, 00, 00, 10, -10, -05}, 
    {-10, -5, 15, 5, 5, 15, -5, -10}, 
    {-05, 00, 15, 5, 5, 15, 00, -05}, 
    {-05, 05, 05, 05, 05, 05, 05, -05}, 
    {-05, 10, 05, 05, 05, 05, 10, -05},
    {0, -5, -10, -20, -20, -10, -5, 0}
    };

int ROOK_VALS[8][8] = {
    {05, 05, 05, 05, 05, 05, 05, 05},
    {25, 25, 25, 25, 25, 25, 25, 25},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {05, 05, 05, 05, 05, 05, 05, 05},
    };

int QUEEN_VALS[8][8] = {
    {00, 05, 05, 05, 05, 05, 05, 00},
    {00, 00, 00, 00, 00, 00, 00, 00},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, 00, 00, 00, 00, 00, 00, -10},
    {-10, -10, -10, -10, -10, -10, -10, -10}
    };

int KING_VALS[8][8] = {
    {-45, -50, -50, -50, -50, -50, -50, -45},
    {-45, -50, -50, -50, -50, -50, -50, -45},
    {-35, -40, -40, -40, -40, -40, -40, -35},
    {-25, -30, -30, -30, -30, -30, -30, -25},
    {-15, -20, -20, -20, -20, -20, -20, -15},
    {-05, -10, -10, -10, -10, -10, -10, -05},
    {10, 00, 00, 00, 00, 00, 00, 10},
    {30, 20, 10, 00, 00, 10, 20, 30}
    };


int get_piece_value(Piece p, Square sq){
    Rank rank = sq.rank();
    int rank_int = int(rank);
    File file = sq.file();
    int file_int = int(file);
    switch (p){
        case 0: //WHITE PAWN
            return PAWN_VALS[7-rank][file] + PAWN_VAL;
        case 1: //WHITE KNIGHT
            return KNIGHT_VALS[7-rank][file] + KNIGHT_VAL;
        case 2: //WHITE BISHOP
            return BISHOP_VALS[7-rank][file] + BISHOP_VAL;
        case 3: //WHITE ROOK
            return ROOK_VALS[7-rank][file] + ROOK_VAL;
        case 4: //WHITE QUEEN
            return QUEEN_VALS[7-rank][file] + QUEEN_VAL;
        case 5: //WHITE KING
            return KING_VALS[7-rank][file] + KING_VAL;
        case 6: //BLACK PAWN
            return -PAWN_VALS[rank][file] - PAWN_VAL;
        case 7: //BLACK KNIGHT
            return -KNIGHT_VALS[rank][file] - KNIGHT_VAL;
        case 8: //BLACK BISHOP
            return -BISHOP_VALS[rank][file] - BISHOP_VAL;
        case 9: //BLACK ROOK
            return -ROOK_VALS[rank][file] - ROOK_VAL;
        case 10: //BLACK QUEEN
            return -QUEEN_VALS[rank][file] - QUEEN_VAL;
        case 11: //BLACK KING
            return -KING_VALS[rank][file] - KING_VAL;
        default: //NOTHING
            return 0;
    }
}

int evaluate_universal(Board& board){
    int eval = 0;
    for(Square sq = Square::underlying::SQ_A1; sq <= Square::underlying::SQ_H8; sq++){
        Piece p = board.at(sq);
        int value = get_piece_value(p, sq);
        eval += value;
    }
    return eval;
}

int evaluate(Board& board){
    int eval = evaluate_universal(board);
    Color side = board.sideToMove();
    if (side == Color::WHITE) return eval;
    else if (side == Color::BLACK) return -eval;
    else return 0;
}