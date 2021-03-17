import os
import pickle
import random
import torch
from typing import List

import numpy as np

from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder



ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']#, 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")

    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])

    self.logger.debug("Querying model for action.")

    state = state_to_features(game_state)
    state0 = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state0)
    move = torch.argmax(prediction).item()
    return ACTIONS[move]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    features = []
    walls_around_position = wall_around(game_state["field"], game_state["self"][3])
    features += walls_around_position

    min_coin_distance = 100
    coin_features = []
    field = field_to_obstacle_matrix(game_state["field"])
    for coin in game_state['coins']:
        #coin = game_state['coins'][index]

        min_coin_features = coin_information(coin, game_state["self"][3], min_coin_distance, field)

        if min_coin_features is not None:
            coin_features = min_coin_features
            min_coin_distance = coin_features[0]
    features += coin_features
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
    return features


def field_to_obstacle_matrix(field) -> List[List[int]]:
    field[field == 1] = -2
    field[field == 0] = 1
    return field


def coin_information(coin_position, player_position, min_coin_distance, field):
    grid = Grid(matrix=field)
    start = grid.node(player_position[0], player_position[1])
    end = grid.node(coin_position[0], coin_position[1])
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)
    if len(path) < min_coin_distance:
        coin_left = 1 if coin_position[0] < player_position[0] else 0
        coin_right = 1 if coin_position[0] > player_position[0] else 0
        coin_up = 1 if coin_position[1] < player_position[1] else 0
        coin_down = 1 if coin_position[1] > player_position[1] else 0
        return [len(path), coin_left, coin_right, coin_up, coin_down]


def wall_around(field, position) -> List[int]:
    wall_left = 1 if field[position[0]-1, position[1]] == -1 else 0
    wall_right = 1 if field[position[0]+1, position[1]] == -1 else 0
    wall_up = 1 if field[position[0], position[1]-1] == -1 else 0
    wall_down = 1 if field[position[0], position[1]+1] == -1 else 0
    return [wall_left, wall_right, wall_up, wall_down]
