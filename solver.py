import cv2
import math
import numpy as np
from evaluate import Solver
import keras

class SudokuSolver():
    def __init__(self, img):
        self.image = img 
        self.dim = 9

    def getWarpedGrid(self):
        # Apply Gaussian Blur and adaptive threshhold(to manage effect of shadows)
        blur = cv2.GaussianBlur(self.image.copy(), (9, 9), 0)
        threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 1)
        inverse = cv2.bitwise_not(threshold, threshold)

        # Get the contours
        contours, hierarchy = cv2.findContours(inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # Find the contour which has the maximum area(Sudoku board)
        final = contours[0]
        max_area = cv2.contourArea(final)

        for contour in contours:
            if cv2.contourArea(contour) > max_area:
                final = contour
                max_area = cv2.contourArea(contour)

        # perimeter = cv2.arcLength(final, True)
        epsilon = 0.01*cv2.arcLength(final, True)
        approx = cv2.approxPolyDP(final, epsilon, True)
        print(approx)
        points = sorted([approx[i][0].tolist() for i in range(4)])
        print(points)
        topLeft = points[0]
        bottomLeft = points[1]
        bottomRight = points[3]
        topRight = points[2]
        print(topLeft, bottomLeft, topRight, bottomRight)

        height = int(max(math.hypot(topLeft[0] - bottomLeft[0], topLeft[1] - bottomLeft[1]), 
                math.hypot(topRight[0] - bottomRight[0], topRight[1] - bottomRight[1])))

        width = int(max(math.hypot(topLeft[0] - topRight[0], topLeft[1] - topRight[1]), 
                math.hypot(bottomLeft[0] - bottomRight[0], bottomLeft[1] - bottomRight[1])))

        dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        # TODO: find coroners properly 
        corners = np.array([topLeft, topRight, bottomRight, bottomLeft], dtype="float32")

        # Apply warp transform to get an approximately square board
        grid = cv2.getPerspectiveTransform(corners, dimensions)
        warpGrid =  cv2.warpPerspective(inverse, grid, (width, height))
        return warpGrid

    def getCells(self):
        board = self.getWarpedGrid()

        # Segment the board into 9*9 cells 
        boardHeight = np.shape(board)[0]
        boardWidth = np.shape(board)[1]

        cellHeight = boardHeight // self.dim
        cellWidth = boardWidth // self.dim

        cells = []
        count = 1
        for y in range(0,boardHeight,cellHeight):
            for x in range(0, boardWidth, cellWidth):
                if y + cellHeight <= boardHeight and x + cellWidth <= boardWidth:
                    tiles = board[ y : y + cellHeight, x : x + cellWidth]
                    count += 1

                    cells.append(tiles)
        return cells
       
    def createBoard(self):
        cells = self.getCells()
        predictedDigits = [0 for _ in range(self.dim ** 2)]
        model = keras.models.load_model('/home/sriru/Srirupa/Sudoku/model_1')

        for count,cell in enumerate(cells):
            gray = cv2.threshold(cell, 128, 255, cv2.THRESH_BINARY)[1]
            contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                border_width = 5
                min_width_digit = 30
                min_height_digit = 30
                padding = 30

                if (x < border_width or y < border_width or h < min_height_digit or w < min_width_digit):
                    continue
                digit = cell[y:y + h ,x:x + w]

                digit= cv2.copyMakeBorder(digit,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=(0, 0, 0))
                digit = cv2.resize(digit, (28, 28))
                digit = digit.astype('float32')
                digit = digit.reshape(1, 28, 28, 1)
                digit /= 255
                
                predictedDigit = model.predict(digit, batch_size = 1)
                predictedDigits[count] = (np.argmax(predictedDigit, axis = 1)[0])

        grid = [predictedDigits[self.dim * i:self.dim * (i + 1)] for i in range(self.dim)]
        print("Unsolved board")
        for i in range(self.dim):
            print(*grid[i])
        return grid
            
    def solve(self):
        boardSolver = Solver(self.createBoard())
        solved = boardSolver.getSolvedBoard()

        # Print the solved board
        for i in range(self.dim):
            print(*solved[i])
        