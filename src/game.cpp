#include "game.h"

using namespace chess;

int playSelf(int maxMoves, int depth){
    Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    int moves = 0;
    bool gameIsOver = false;
    Move move;
    while(!gameIsOver && moves < maxMoves){
        move = bestMoveAB(board, depth);
        std::cout << move << "\n";
        board.makeMove(move);

        std::pair<GameResultReason, GameResult> result = board.isGameOver();
        if (result.second != GameResult::NONE){
            gameIsOver = true;
            std::cout << "Game has ended!";
            return 0;
        }
    }
    std::cout << "Game did not complete after " << maxMoves << " moves.";
    return 0;
}

int playPlayer(int depth, bool computerTurn){
    Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    bool gameIsOver = false;
    Move move;
    std::string uci;
    while(!gameIsOver){
        if (computerTurn){
            move = bestMoveAB(board, depth);
            std::cout << move << "\n";
        }
        else{
            std::cin >> uci;
            if (uci == "cancel"){
                return 0;
            }
            move = uci::uciToMove(board, uci);
        }
        board.makeMove(move);
        computerTurn = !computerTurn;
        
        std::pair<GameResultReason, GameResult> result = board.isGameOver();
        if (result.second != GameResult::NONE){
            gameIsOver = true;
            std::cout << "Game has ended!";
            return 0;
        }
    }
    return 0;
}