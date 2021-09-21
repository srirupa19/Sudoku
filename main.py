from solver import SudokuSolver
import cv2

img = cv2.imread('board.png', 0)
player = SudokuSolver(img)
player.solve()

