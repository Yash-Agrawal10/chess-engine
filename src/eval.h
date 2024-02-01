#pragma once

#include "chess.hpp"

using namespace chess;

int get_piece_value(Piece, Square);

int evaluate_universal(Board&);

int evaluate(Board&);