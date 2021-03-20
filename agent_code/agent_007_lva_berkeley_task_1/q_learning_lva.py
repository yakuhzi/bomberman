from typing import Dict

import numpy as np

from agent_code.agent_007_lva_berkeley_task_1.callbacks import state_to_features, get_q_value

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']  # , 'BOMB']
TRAINING_RATE = 0.001
GAMMA = 0.8
BATCH_SIZE = 100


def update_q_function(model: dict, old_state: dict, action: str, new_state: dict, reward: float) -> Dict[str, float]:
    temporal_difference = calculate_temporal_difference(model, old_state, action, new_state, reward)
    features = state_to_features(old_state, action)

    for feature in features:
        model[feature] += TRAINING_RATE * temporal_difference * features[feature]

    return model


def calculate_temporal_difference(model: dict, old_state: dict, action: str, new_state: dict, reward: float) -> float:
    return reward + GAMMA * compute_value_from_q_values(model, new_state) - get_q_value(model, old_state, action)


def compute_value_from_q_values(model: dict, new_state:dict) -> float:
    possible_q_values = [get_q_value(model, new_state, action) for action in ACTIONS]
    return np.max(possible_q_values)

