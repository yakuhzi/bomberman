from collections import deque

import numpy as np
from typing import Tuple, List


class BFS:
    def __init__(self, field: np.array, coins: List[Tuple[int, int]]):
        self.field = field

        self.row = [-1, 0, 0, 1]
        self.col = [0, -1, 1, 0]

        self.N = field.shape[1]
        self.M = field.shape[0]

        self.visited = [[False for x in range(self.N)] for y in range(self.M)]
        self.queue = deque()

        if coins is not None:
            for coin in coins:
                x, y = coin
                self.field[x, y] = 2
        else:
            field[field == 1] = 2

    def is_valid(self, row: int, col: int) -> bool:
        return (row >= 0) and (row < self.M) and (col >= 0) and (col < self.N) and self.field[row][col] != -1 \
               and self.field[row][col] != 1 and not self.visited[row][col]

    def get_distance(self, position: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
        x, y = position

        # Mark the source cell as visited
        self.visited[x][y] = True

        # Enqueue the source node
        self.queue.append((x, y, 0))

        # Store min distance and coordinates of nearest coin
        min_dist = 0
        min_coin = (0, 0)

        # Loop until queue is empty
        while self.queue:
            # Dequeue first element
            (x, y, dist) = self.queue.popleft()

            # Check if current field is a coin
            if self.field[x, y] == 2:
                min_dist = dist
                min_coin = (x, y)
                break

            # Iterate over all possible movements (up, down, left, right)
            for k in range(4):
                # Check if field is valid field
                if self.is_valid(x + self.row[k], y + self.col[k]):
                    # Mark field as visited
                    self.visited[x + self.row[k]][y + self.col[k]] = True
                    # Enqueue field
                    self.queue.append((x + self.row[k], y + self.col[k], dist + 1))

        # Return min distance and coordinates of the coin
        return min_dist, min_coin
