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

    if "n_rounds" in game_state:
        random_prob = max(0.3 - (game_state["round"] / game_state["n_rounds"]) * 0.3, 0) + 0.05

    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

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
    others = game_state["others"]
    position = game_state["self"][3]
    has_bomb = game_state["self"][2]

    bomb_positons = list(map(lambda bomb: bomb[0], bombs))
    agent_positons = list(map(lambda agent: agent[3], others))
    bomb_dist_before_move = bomb_distance(field, bomb_positons, position)

    is_action_valid = action_information(field, bomb_positons, agent_positons, position, has_bomb, action)
    features["invalid_action"] = 1 if not is_action_valid else 0

    features["drop_bomb"] = 0

    if has_bomb:
        if position[0] - 1 >= 0 and field[position[0] - 1, position[1]] == 1 or \
            (position[0] - 1, position[1]) in agent_positons or (position[0] - 2, position[1]) in agent_positons:
            features["drop_bomb"] = 1
        elif position[0] + 1 < field.shape[0] and field[position[0] + 1, position[1]] == 1 or \
            (position[0] + 1, position[1]) in agent_positons or (position[0] + 2, position[1]) in agent_positons:
            features["drop_bomb"] = 1
        elif position[1] - 1 >= 0 and field[position[0], position[1] - 1] == 1 or \
            (position[0], position[1] - 1) in agent_positons or (position[0], position[1] - 2) in agent_positons:
            features["drop_bomb"] = 1
        elif position[1] + 1 < field.shape[1] and field[position[0], position[1] + 1] == 1 or \
            (position[0], position[1] + 1) in agent_positons or (position[0], position[1] + 2) in agent_positons:
            features["drop_bomb"] = 1

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
            features["drop_bomb"] = 0

    danger = danger_ahead(field, explosion_map, bombs, position)
    features["danger_ahead"] = 1 if danger else 0

    coin_feature = coin_distance(field, coins, position)
    features["coin_distance"] = coin_feature

    crate_feature = crate_distance(field, position)
    features["crate_distance"] = crate_feature

    bomb_feature = bomb_distance(field, bomb_positons, position)
    features["bomb_distance"] = bomb_feature

    agent_feature = agent_distance(field, agent_positons, position)
    features["agent_distance"] = agent_feature

    features["dead_end"] = 0

    if bomb_dist_before_move == 1:
        if is_action_valid:
            features["dead_end"] = dead_end(field, agent_positons, position, action)
        else:
            features["dead_end"] = 1

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


def action_information(
    field: np.array,
    bomb_positions: List[Tuple[int, int]],
    agent_positions: List[Tuple[int, int]],
    position: Tuple[int, int],
    has_bomb: bool,
    action: str
) -> bool:
    if action == 'LEFT' and field[position[0] - 1, position[1]] != -1 and field[position[0] - 1, position[1]] != 1 \
        and (position[0] - 1, position[1]) not in bomb_positions and (
        position[0] - 1, position[1]) not in agent_positions:
        return True
    elif action == 'RIGHT' and field[position[0] + 1, position[1]] != -1 and field[position[0] + 1, position[1]] != 1 \
        and (position[0] + 1, position[1]) not in bomb_positions and (
        position[0] + 1, position[1]) not in agent_positions:
        return True
    elif action == 'UP' and field[position[0], position[1] - 1] != -1 and field[position[0], position[1] - 1] != 1 \
        and (position[0], position[1] - 1) not in bomb_positions and (
        position[0], position[1] - 1) not in agent_positions:
        return True
    elif action == 'DOWN' and field[position[0], position[1] + 1] != -1 and field[position[0], position[1] + 1] != 1 \
        and (position[0], position[1] + 1) not in bomb_positions and (
        position[0], position[1] + 1) not in agent_positions:
        return True
    elif action == 'BOMB' and has_bomb:
        return True
    elif action == 'WAIT':
        return True

    return False


def danger_ahead(
    field: np.array, explosion_map: np.array, bombs: List[Tuple[int, int]], position: Tuple[int, int]
) -> bool:
    for bomb in [bomb for bomb in bombs if bomb[1] == 0]:
        x, y = bomb[0]
        explosion_map[x, y] = 3

        for i in range(1, 4):
            if 0 < x + i < 16:
                if field[x - i, y] == -1:
                    break
                explosion_map[x - i, y] = 3

        for i in range(1, 4):
            if 0 < y + i < 16:
                if field[x, y - i] == -1:
                    break
                explosion_map[x, y - i] = 3

        for i in range(1, 4):
            if 0 < x + i < 16:
                if field[x + i, y] == -1:
                    break
                explosion_map[x + i, y] = 3

        for i in range(1, 4):
            if 0 < y + i < 16:
                if field[x, y + i] == -1:
                    break
                explosion_map[x, y + i] = 3

    return explosion_map[position[0], position[1]] > 0


def coin_distance(field: np.array, coins: List[Tuple[int, int]], position: Tuple[int, int]) -> float:
    if len(coins) == 0:
        return 0

    coin_bfs = BFS(field.copy(), coins)
    distance, coin_position = coin_bfs.get_distance(position)

    if distance == float("inf"):
        return 0

    return (distance + 1) / 15


def bomb_distance(field: np.array, bomb_positions: List[Tuple[int, int]], position: Tuple[int, int]) -> float:
    if len(bomb_positions) == 0:
        return 0

    bomb_bfs = BFS(field.copy(), bomb_positions)
    distance, bomb_position = bomb_bfs.get_distance(position)

    if distance > 3 or distance == float("inf"):
        return 0

    return (4 - distance) / 4


def crate_distance(field: np.array, position: Tuple[int, int]) -> float:
    if 1 not in field:
        return 0

    crate_bfs = BFS(field.copy(), None)
    distance, crate_position = crate_bfs.get_distance(position)
    return (distance + 1) / 15


def agent_distance(field: np.array, agent_positions: List[Tuple[int, int]], position: Tuple[int, int]) -> float:
    if len(agent_positions) == 0:
        return 0

    agent_bfs = BFS(field.copy(), agent_positions)
    distance, agent_position = agent_bfs.get_distance(position)

    number_of_crates = (field == 1).sum()
    search_distance = 7

    if number_of_crates < 5:
        search_distance = 28

    if distance == float("inf") or distance > search_distance:
        return 0

    return distance / 28


def dead_end(field: np.array, agent_positions: List[Tuple[int, int]], position: Tuple[int, int], action: str) -> int:
    if action == 'WAIT' or action == 'BOMB':
        return 0

    max_straight = None

    for i in range(1, 4):
        if action == 'LEFT' and field[position[0] - i, position[1]] != 0 \
            or (position[0] - 1, position[1]) in agent_positions:
            max_straight = i - 1
            break
        elif action == 'RIGHT' and field[position[0] + i, position[1]] != 0 \
            or (position[0] + 1, position[1]) in agent_positions:
            max_straight = i - 1
            break
        elif action == 'UP' and field[position[0], position[1] - i] != 0 \
            or (position[0], position[1] - 1) in agent_positions:
            max_straight = i - 1
            break
        elif action == 'DOWN' and field[position[0], position[1] + i] != 0 \
            or (position[0], position[1] + 1) in agent_positions:
            max_straight = i - 1
            break

    if max_straight is None:
        return 0

    if max_straight > 2:
        return 0

    for i in range(max_straight + 1):
        if action == 'LEFT':
            # left and than up/down to escape bomb
            new_position = (position[0] - i, position[1])
            if (field[new_position[0], new_position[1] - 1] == 0
                and (new_position[0], new_position[1] - 1) not in agent_positions) \
                or (field[new_position[0], new_position[1] + 1] == 0
                    and (new_position[0], new_position[1] + 1) not in agent_positions):
                return 0
        elif action == 'RIGHT':
            # right and than up/down to escape bomb
            new_position = (position[0] + i, position[1])
            if (field[new_position[0], new_position[1] - 1] == 0
                and (new_position[0], new_position[1] - 1) not in agent_positions) \
                or (field[new_position[0], new_position[1] + 1] == 0
                    and (new_position[0], new_position[1] + 1) not in agent_positions):
                return 0
        elif action == 'UP':
            # up and than left/right to escape bomb
            new_position = (position[0], position[1] - i)
            if (field[new_position[0] - 1, new_position[1]] == 0
                and (new_position[0] - 1, new_position[1]) not in agent_positions) \
                or (field[new_position[0] + 1, new_position[1]] == 0
                    and (new_position[0] + 1, new_position[1]) not in agent_positions):
                return 0
        elif action == 'DOWN':
            # down and than left/right to escape bomb
            new_position = (position[0], position[1] + i)
            if (field[new_position[0] - 1, new_position[1]] == 0
                and (new_position[0], new_position[1] - 1) not in agent_positions) \
                or (field[new_position[0] + 1, new_position[1]] == 0
                    and (new_position[0], new_position[1] + 1) not in agent_positions):
                return 0

    return 1
