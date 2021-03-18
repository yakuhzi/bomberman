import numpy as np
import os
import pickle
import random
import torch
from typing import List, Tuple

from agent_code.agent_007.coin_bfs import CoinBFS

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']  # , 'BOMB']


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
    random_prob = 0.1

    if random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])

    self.logger.debug("Querying model for action.")

    state = state_to_features(game_state)
    state0 = torch.tensor(state, dtype=torch.float)
    prediction = self.model(state0)
    action_index = torch.argmax(prediction).item()
    return ACTIONS[action_index]


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

    coin_features = coin_information(game_state["field"], game_state['coins'], game_state["self"][3])
    features += coin_features

    return features


def wall_around(field: np.array, position: Tuple[int, int]) -> List[int]:
    wall_left = 1 if field[position[0] - 1, position[1]] == -1 else 0
    wall_right = 1 if field[position[0] + 1, position[1]] == -1 else 0
    wall_up = 1 if field[position[0], position[1] - 1] == -1 else 0
    wall_down = 1 if field[position[0], position[1] + 1] == -1 else 0
    return [wall_left, wall_right, wall_up, wall_down]


def coin_information(field: np.array, coins: List[Tuple[int, int]], player_position: Tuple[int, int]) -> List[int]:
    coin_bfs = CoinBFS(field, coins)
    distance, coin_position = coin_bfs.get_distance(player_position)

    coin_left = 1 if coin_position[0] < player_position[0] else 0
    coin_right = 1 if coin_position[0] > player_position[0] else 0
    coin_up = 1 if coin_position[1] < player_position[1] else 0
    coin_down = 1 if coin_position[1] > player_position[1] else 0

    return [distance, coin_left, coin_right, coin_up, coin_down]


def field_to_obstacle_matrix(field: np.array) -> List[List[int]]:
    field[field == 1] = -2
    field[field == 0] = 1
    return field
