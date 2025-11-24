# Description: Chessboard baseclass
import numpy as np


class Chessboard:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols

        self.points = np.zeros((rows * cols, 3), np.float32)
