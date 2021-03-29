from collections import deque
from typing import Tuple, List, Optional

import numpy as np


class BFS:
    """Class that calculates the minimum distance to reach an target using breadth-first search"""

    def __init__(self, field: np.array, targets: Optional[List[Tuple[int, int]]]):
        """
        :param field: Field of the game
        :param targets: Target (end positions) of the search. If None, target will be crates.
        """
        self.field = field

        # Helper variables to iterate over possible movements
        self.row = [-1, 0, 0, 1]
        self.col = [0, -1, 1, 0]

        # Size of the field
        self.M = field.shape[0]
        self.N = field.shape[1]

        # Matrix that stores the visited fields
        self.visited = [[False for x in range(self.N)] for y in range(self.M)]
        # Queue for fields to visit
        self.queue = deque()

        # Set the target positions to 2 in the field array. If None, the crates are set as targets.
        if targets is not None:
            for target in targets:
                x, y = target
                self.field[x, y] = 2
        else:
            field[field == 1] = 2

    def is_valid(self, row: int, col: int) -> bool:
        """
        Helper function that check if a position is a valid field.

        :param row: Row of the field.
        :param col: Column of the field.
        :return: Boolean indicating if the position is a valid field.
        """
        return (row >= 0) and (row < self.M) and (col >= 0) and (col < self.N) and self.field[row][col] != -1 \
               and self.field[row][col] != 1 and not self.visited[row][col]

    def get_distance(self, position: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
        """
        Calculates the minimum distance from the current position to the target.

        :param position: Start position of the search.
        :return: Minimum distance and target position.
        """
        x, y = position

        # Mark the source cell as visited
        self.visited[x][y] = True

        # Enqueue the source node
        self.queue.append((x, y, 0))

        # Store min distance and coordinates of nearest coin
        min_dist = float("inf")
        min_target = None

        # Loop until queue is empty
        while self.queue:
            # Dequeue first element
            (x, y, dist) = self.queue.popleft()

            # Check if current field is a coin
            if self.field[x, y] == 2:
                min_dist = dist
                min_target = (x, y)
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
        return min_dist, min_target
