class Solver():

    def __init__(self, board) -> None:
        self.board = board
        self.dim = 9

    def findNext(self, curr):
        for i in range(len(self.board[0])):
            for j in range(len(self.board[1])):
                if (self.board[i][j] == 0):
                    curr[0] = i
                    curr[1] = j
                    return True
        return False

    def rowPossible(self, row, number):
        for i in range(self.dim):
            if (self.board[row][i] == number):
                return False 
        return True 
    
    def colPossible(self, col, number):
        for i in range(self.dim):
            if (self.board[i][col] == number):
                return False 
        return True 

    def subGridPossible(self, row, col, number):
        for i in range(self.dim // 3):
            for j in range(self.dim // 3):
                if self.board[i + row][j + col] == number:
                    return False
        return True

    def isPossible(self, row, col, number):
        return self.rowPossible(row, number) and self.colPossible(col, number) and self.subGridPossible(row - row % 3, col - col % 3, number)


    def solve(self):
        curr = [0, 0]
        if not self.findNext(curr):
            return True 
        row = curr[0]
        col = curr[1]

        for i in range(1, 10):
            if self.isPossible(row, col, i):
                self.board[row][col] = i 
                if self.solve():
                    return True 
                self.board[row][col] = 0
        return False

    def getSolvedBoard(self):
        output = self.solve()
        if output:
            print("Succesfully Solved!")
        else:
            print("Incorrect Board!")
        return self.board

