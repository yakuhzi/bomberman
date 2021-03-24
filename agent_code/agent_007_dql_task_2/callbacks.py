import numpy as np
import os
import pickle
import random
import torch
from typing import List, Tuple

from .bfs import BFS
from .callbacks_rule_based import act_rule_based, setup_rule_based

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
        setup_rule_based(self)
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
    random_prob = max(0.5 - game_state["round"] / 100, 0) + 0.1

    # input("Press Enter to continue...")
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        return act_rule_based(self, game_state)

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

    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    position = game_state["self"][3]
    bombs = game_state["bombs"]
    coins = game_state["coins"]

    features = []

    # 0-3
    obstacle_features = obstacle_information(field, explosion_map, bombs, position)
    features += obstacle_features

    # 4-8
    bomb_features = bomb_information(field, bombs, position)
    features += bomb_features

    # 9-13
    coin_features = coin_information(field, coins, position)
    features += coin_features

    # 14-18
    crate_features = crate_information(field, position)
    features += crate_features

    # 19
    features.append(1 if game_state["self"][2] else 0)  # Player has bomb
    # 20
    features.append(bomb_destroys_crate(field, position))  # Bomb would destroy crate
    return features


def obstacle_information(
    field: np.array,
    explosion_map: np.array,
    bombs: List[Tuple[Tuple[int, int], bool]],
    position: Tuple[int, int]
) -> List[int]:
    # Add explosion map to field (ignore explosions with lifetime of 1)
    field = field.copy()
    field[explosion_map > 1] = 2

    # Add bombs to field
    for bomb in bombs:
        field[bomb[0][0], bomb[0][1]] = 3

    # Check if fields are free
    obstacle_left = 1 if field[position[0] - 1, position[1]] != 0 else 0
    obstacle_right = 1 if field[position[0] + 1, position[1]] != 0 else 0
    obstacle_up = 1 if field[position[0], position[1] - 1] != 0 else 0
    obstacle_down = 1 if field[position[0], position[1] + 1] != 0 else 0

    return [obstacle_left, obstacle_right, obstacle_up, obstacle_down]


def bomb_information(field: np.array, bombs: List[Tuple[int, int]], player_position: Tuple[int, int]) -> List[int]:
    if len(bombs) == 0:
        return [0, 0, 0, 0, 0]

    bombs = list(map(lambda bomb: bomb[0], bombs))

    bomb_bfs = BFS(field.copy(), bombs)
    distance, coin_position = bomb_bfs.get_distance(player_position)

    if distance == 0:
        return [0, 1, 1, 1, 1]

    bomb_left = 1 if coin_position[0] < player_position[0] else 0
    bomb_right = 1 if coin_position[0] > player_position[0] else 0
    bomb_up = 1 if coin_position[1] < player_position[1] else 0
    bomb_down = 1 if coin_position[1] > player_position[1] else 0

    return [distance / 20, bomb_left, bomb_right, bomb_up, bomb_down]


def coin_information(field: np.array, coins: List[Tuple[int, int]], player_position: Tuple[int, int]) -> List[int]:
    if len(coins) == 0:
        return [0, 0, 0, 0, 0]

    coin_bfs = BFS(field, coins)
    distance, coin_position = coin_bfs.get_distance(player_position)

    coin_left = 1 if coin_position[0] < player_position[0] else 0
    coin_right = 1 if coin_position[0] > player_position[0] else 0
    coin_up = 1 if coin_position[1] < player_position[1] else 0
    coin_down = 1 if coin_position[1] > player_position[1] else 0

    return [distance / 20, coin_left, coin_right, coin_up, coin_down]


def crate_information(field: np.array, player_position: Tuple[int, int]) -> List[int]:
    crate_bfs = BFS(field.copy(), None)
    distance, crate_position = crate_bfs.get_distance(player_position)

    crate_left = 1 if crate_position[0] < player_position[0] else 0
    crate_right = 1 if crate_position[0] > player_position[0] else 0
    crate_up = 1 if crate_position[1] < player_position[1] else 0
    crate_down = 1 if crate_position[1] > player_position[1] else 0

    return [distance / 20, crate_left, crate_right, crate_up, crate_down]


def bomb_destroys_crate(field: np.array, position: Tuple[int, int]) -> int:
    left_free = True
    right_free = True
    up_free = True
    down_free = True

    for i in range(1, 2):
        if position[0] - i >= 0 and field[position[0] - i, position[1]] == -1:
            left_free = False
        if position[0] + i < field.shape[0] and field[position[0] + i, position[1]] == -1:
            right_free = False
        if position[1] - i >= 0 and field[position[0], position[1] - i] == -1:
            up_free = False
        if position[1] + i < field.shape[1] and field[position[0], position[1] + i] == -1:
            down_free = False

        if left_free and position[0] - i >= 0 and field[position[0] - i, position[1]] == 1:
            return 1
        if right_free and position[0] + i < field.shape[0] and field[position[0] + i, position[1]] == 1:
            return 1
        if up_free and position[1] - i >= 0 and field[position[0], position[1] - i] == 1:
            return 1
        if down_free and position[1] + i < field.shape[1] and field[position[0], position[1] + i] == 1:
            return 1

    return 0

