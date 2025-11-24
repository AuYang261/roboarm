from chessboard import chessboard
import cv2
import numpy as np
import os


class ChineseChess(chessboard.Chessboard):
    # 定义棋子种类
    King = 1
    Advisor = 2
    Elephant = 3
    Horse = 4
    Rook = 5
    Cannon = 6
    Pawn = 7
    Empty = 0
    PieceNames = {
        1: "King",
        2: "Advisor",
        3: "Elephant",
        4: "Horse",
        5: "Rook",
        6: "Cannon",
        7: "Pawn",
        0: ".Empty",
    }

    def __init__(self):
        super().__init__(10, 9)

    def nearest_grid_point(self, row_ratio: float, col_ratio: float):
        return (round(row_ratio * (self.rows - 1)), round(col_ratio * (self.cols - 1)))
