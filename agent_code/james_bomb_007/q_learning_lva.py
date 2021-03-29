from typing import Dict, Optional

import numpy as np

from .callbacks import state_to_features, get_q_value

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ALPHA = 0.1
GAMMA = 0.9


def update_q_function(
    model: dict, old_state: dict, action: str, new_state: Optional[dict], reward: float
) -> Dict[str, float]:
    """
    Method that updates the q function approximation model.

    :param model: The current q function model.
    :param old_state: The old state of the game before the chosen action.
    :param action: The chosen action.
    :param new_state: The new state of the game after the chosen action.
    :param reward: The reward for the current step.
    :return: The updated model.
    """
    temporal_difference = calculate_temporal_difference(model, old_state, action, new_state, reward)
    features = state_to_features(old_state, action)

    for feature in features:
        model[feature] += ALPHA * temporal_difference * features[feature]

    return model


def calculate_temporal_difference(
    model: dict, old_state: dict, action: str, new_state: Optional[dict], reward: float
) -> float:
    """
    Calculates the temporal difference between the last and the new step.

    :param model: The current q function model.
    :param old_state: The old state of the game before the chosen action.
    :param action: The chosen action.
    :param new_state: The new state of the game after the chosen action.
    :param reward: The reward for the current step.
    :return: Temporal difference between the last and the new step.
    """
    return reward + GAMMA * compute_value_from_q_values(model, new_state) - get_q_value(model, old_state, action)


def compute_value_from_q_values(model: dict, new_state: Optional[dict]) -> float:
    """
    Computes the maximum of the possible q values.

    :param model: The current q function model.
    :param new_state: The new state of the game after the chosen action.
    :return:
    """
    if new_state is None:
        return 0

    possible_q_values = [get_q_value(model, new_state, action) for action in ACTIONS]
    return np.max(possible_q_values)
