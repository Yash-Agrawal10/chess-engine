#include "search.h"

using namespace chess;

const int maxEval = 1000000;

int negaMax(Board& board, int depth){
    if (depth==0) return evaluate(board);
    Movelist moves;
    movegen::legalmoves(moves, board);
    int max = -maxEval;
    for (Move move : moves){
        board.makeMove(move);
        int eval = -negaMax(board, depth-1);
        board.unmakeMove(move);
        if (eval > max) max = eval;
    }
    return max;
}

Move bestMove(Board& board, int depth){
    Movelist moves;
    movegen::legalmoves(moves, board);
    int max = -maxEval;
    Move bestMove = NULL;
    for (Move move : moves){
        board.makeMove(move);
        int eval = -negaMax(board, depth-1);
        board.unmakeMove(move);
        if (eval > max){
            max = eval;
            bestMove = move;
        }
    }
    return bestMove;
}

int alphaBeta(Board& board, int depth, int alpha, int beta){
    // should figure out sorting moves
    if (depth==0) return evaluate(board); // should replace with quiescence search
    Movelist moves;
    movegen::legalmoves(moves, board);
    for (Move move : moves){
        board.makeMove(move);
        int eval = -alphaBeta(board, depth-1, -beta, -alpha);
        board.unmakeMove(move);
        if (eval >= beta) return beta;
        if (eval > alpha) alpha = eval;
    }
    return alpha;
}

Move bestMoveAB(Board& board, int depth){
    Movelist moves;
    movegen::legalmoves(moves, board);
    int max = -maxEval-1;
    Move bestMove = NULL;
    for (Move move : moves){
        board.makeMove(move);
        int eval = -alphaBeta(board, depth-1, -maxEval, maxEval);
        board.unmakeMove(move);
        if (eval > max){
            max = eval;
            bestMove = move;
        }
    }
    return bestMove;
}