import numpy as np
import os
import pickle
import random
from typing import List, Tuple

from .coin_bfs import CoinBFS

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
    random_prob = 0
    random_prob_train = 0.3

    if random.random() < (random_prob_train if self.train else random_prob):
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2])

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
    coins = game_state['coins']
    position = game_state["self"][3]

    if action == 'LEFT' and field[position[0] - 1, position[1]] != -1:
        position = position[0] - 1, position[1]
    elif action == 'RIGHT' and field[position[0] + 1, position[1]] != -1:
        position = position[0] + 1, position[1]
    elif action == 'UP' and field[position[0], position[1] - 1] != -1:
        position = position[0], position[1] - 1
    elif action == 'DOWN' and field[position[0], position[1] + 1] != -1:
        position = position[0], position[1] + 1

    coin_feature = coin_distance(field, coins, position)
    features["coin_distance"] = coin_feature

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


def coin_distance(field: np.array, coins: List[Tuple[int, int]], player_position: Tuple[int, int]) -> float:
    coin_bfs = CoinBFS(field, coins)
    distance, coin_position = coin_bfs.get_distance(player_position)
    return distance

