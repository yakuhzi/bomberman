import numpy as np
import os
import pickle
import random
from typing import List, Tuple

from .bfs import BFS

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
    random_prob = 0.3

    # input()

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        if action == 'BOMB':
            print("RANDOM BOMB")
        return action

    self.logger.debug("Querying model for action.")
    return get_best_action(self.model, game_state)


def state_to_features(game_state: dict, action: str) -> np.array:
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

    features = {}

    field = game_state["field"]
    explosion_map = game_state["explosion_map"]
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    position = game_state["self"][3]
    has_bomb = game_state["self"][2]

    is_action_valid = action_information(field, bombs, position, has_bomb, action)
    features["invalid_action"] = 1 if not is_action_valid else 0

    features["useless_bomb"] = 0
    # features["has_bomb"] = 1 if has_bomb else 0

    if has_bomb:
        if position[0] - 1 >= 0 and field[position[0] - 1, position[1]] == 1:
            features["useless_bomb"] = 1
        elif position[0] + 1 < field.shape[0] and field[position[0] + 1, position[1]] == 1:
            features["useless_bomb"] = 1
        elif position[1] - 1 >= 0 and field[position[0], position[1] - 1] == 1:
            features["useless_bomb"] = 1
        elif position[1] + 1 < field.shape[1] and field[position[0], position[1] + 1] == 1:
            features["useless_bomb"] = 1

    if is_action_valid:
        if action == 'LEFT':
            position = position[0] - 1, position[1]
        elif action == 'RIGHT':
            position = position[0] + 1, position[1]
        elif action == 'UP':
            position = position[0], position[1] - 1
        elif action == 'DOWN':
            position = position[0], position[1] + 1
        elif action == 'BOMB':
            # bombs.append((position, 3))
            features["useless_bomb"] = 0
            # features["has_bomb"] = 0

    danger = danger_ahead(explosion_map, position)
    features["danger_ahead"] = 1 if danger else 0

    coin_feature = coin_distance(field, coins, position)
    features["coin_distance"] = coin_feature

    crate_feature = crate_distance(field, position)
    features["crate_distance"] = crate_feature

    bomb_feature = bomb_distance(field, bombs, position)
    features["bomb_distance"] = bomb_feature

    return features


def get_q_value(model: dict, state: dict, action: str) -> float:
    q_value = 0.0
    features = state_to_features(state, action)

    for feature in features.keys():
        q_value += features[feature] * model[feature]

    return q_value


def get_best_action(model: dict, state: dict) -> str:
    possible_q_values = [get_q_value(model, state, action) for action in ACTIONS]
    best_index = int(np.argmax(possible_q_values))
    return ACTIONS[best_index]


def action_information(field: np.array, bombs: List[Tuple[Tuple[int, int], int]], position: Tuple[int, int], has_bomb: bool, action: str) -> bool:
    bombs = list(map(lambda bomb: bomb[0], bombs))

    if action == 'LEFT' and field[position[0] - 1, position[1]] != -1 and field[position[0] - 1, position[1]] != 1 and (position[0] - 1, position[1]) not in bombs:
        return True
    elif action == 'RIGHT' and field[position[0] + 1, position[1]] != -1 and field[position[0] + 1, position[1]] != 1 and (position[0] + 1, position[1]) not in bombs:
        return True
    elif action == 'UP' and field[position[0], position[1] - 1] != -1 and field[position[0], position[1] - 1] != 1 and (position[0], position[1] - 1) not in bombs:
        return True
    elif action == 'DOWN' and field[position[0], position[1] + 1] != -1 and field[position[0], position[1] + 1] != 1 and (position[0], position[1] + 1) not in bombs:
        return True
    elif action == 'BOMB' and has_bomb:
        return True
    elif action == 'WAIT':
        return True

    return False


def danger_ahead(explosion_map: np.array, position: Tuple[int, int]) -> bool:
    return explosion_map[position[0], position[1]] > 0


def coin_distance(field: np.array, coins: List[Tuple[int, int]], player_position: Tuple[int, int]) -> float:
    if len(coins) == 0:
        return 0

    coin_bfs = BFS(field.copy(), coins)
    distance, coin_position = coin_bfs.get_distance(player_position)
    return (distance + 1) / 15


def bomb_distance(field: np.array, bombs: List[Tuple[int, int]], player_position: Tuple[int, int]) -> float:
    if len(bombs) == 0:
        return 0

    bombs = list(map(lambda bomb: bomb[0], bombs))

    bomb_bfs = BFS(field.copy(), bombs)
    distance, bomb_position = bomb_bfs.get_distance(player_position)

    if distance > 3:
        return 0

    return (4 - distance) / 4


def crate_distance(field: np.array, player_position: Tuple[int, int]) -> float:
    if 1 not in field:
        return 0

    crate_bfs = BFS(field.copy(), None)
    distance, coin_position = crate_bfs.get_distance(player_position)
    return (distance + 1) / 15

