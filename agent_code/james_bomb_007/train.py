import pickle
from collections import namedtuple, deque
from math import isclose
from typing import List, Optional

import events as e
from .callbacks import state_to_features
from .q_learning_lva import update_q_function
from .statistics import Statistics

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

# Hyper parameter
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions

# Events
MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
WAITED_IN_DANGER = "WAITED_IN_DANGER"
USELESS_BOMB = "USELESS_BOMB"
USEFUL_BOMB = "USEFUL_BOMB"
DEAD_END = "DEAD_END"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model = initialize_model()
    self.statistics = Statistics()


def initialize_model():
    """
    Initialize the model with 0. The weights are per feature
    :return: initialized model as dictionary
    """
    return {
        "invalid_action": 0,
        "coin_distance": 0,
        "bomb_distance": 0,
        "agent_distance": 0,
        "crate_distance": 0,
        "drop_bomb": 0,
        "danger_ahead": 0,
        "dead_end": 0
    }


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if old_game_state is None:
        return

    # Update model
    update_model(self, old_game_state, self_action, new_game_state, events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    :param last_game_state: the last state of the game
    :param last_action: the last action the agent executed
    :param events: the events that occurred in the last action of the game
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Update model
    update_model(self, last_game_state, last_action, None, events)

    # Add statistic for the last round
    self.statistics.add_round_statistic(last_game_state)

    # Show statistics at the end of training
    if "n_rounds" in last_game_state and last_game_state["round"] == last_game_state["n_rounds"]:
        self.statistics.show(last_game_state["round"])

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def update_model(self, old_game_state: dict, self_action: str, new_game_state: Optional[dict], events: List[str]):
    """
    Update the model after a step in the game
    :param self: The same object that is passed to all of the callbacks.
    :param old_game_state: the game state before the action was executed
    :param self_action: the action the agent executed
    :param new_game_state: the new game state the agent is in now
    :param events: the events that occurred in that game step
    """
    # Convert state to features
    current_features = state_to_features(old_game_state, self_action)

    # Add auxiliary events
    if len(self.transitions) > 1:
        add_auxiliary_events(self, events, old_game_state, current_features)

    # Calculate reward
    reward = reward_from_events(self, events)

    # Add statistic for the last step
    self.statistics.add_step_statistic(reward, events)

    # Add transition to memory
    transition = Transition(current_features, self_action, reward)
    self.transitions.append(transition)

    # Update model
    self.model = update_q_function(self.model, old_game_state, self_action, new_game_state, reward)


def add_auxiliary_events(self, events: List[str], old_game_state: dict, current_features: dict):
    """
    Add auxiliary events that are not part of the "standard" event list
    :param self: The same object that is passed to all of the callbacks.
    :param events: the events that occurred in that game step
    :param old_game_state: the game state before the action was executed
    :param current_features: the features of the current game state
    """
    # Check if a dropped bomb was useful or useless, i.e. it was dropped near a crate or an agent
    if e.BOMB_DROPPED in events:
        agent_positons = list(map(lambda agent: agent[3], old_game_state["others"]))
        position = old_game_state["self"][3]

        if 0.11 < self.transitions[-1][0]["crate_distance"] < 0.15:
            events.append(USEFUL_BOMB)
        elif (position[0] - 1, position[1]) in agent_positons or (position[0] - 2, position[1]) in agent_positons:
            events.append(USEFUL_BOMB)
        elif (position[0] + 1, position[1]) in agent_positons or (position[0] + 2, position[1]) in agent_positons:
            events.append(USEFUL_BOMB)
        elif (position[0], position[1] - 1) in agent_positons or (position[0], position[1] - 2) in agent_positons:
            events.append(USEFUL_BOMB)
        elif (position[0], position[1] + 1) in agent_positons or (position[0], position[1] + 2) in agent_positons:
            events.append(USEFUL_BOMB)
        else:
            events.append(USELESS_BOMB)

    # Check if the agent waited near a bomb (in danger)
    old_bomb_distance = self.transitions[-1][0]["bomb_distance"]
    new_bomb_distance = current_features["bomb_distance"]

    if (old_bomb_distance != 0 and old_bomb_distance == new_bomb_distance) or (
        old_bomb_distance == 0 and new_bomb_distance == 1):
        events.append(WAITED_IN_DANGER)

    # Check if the agent moved away or towards a bomb
    if old_bomb_distance == 0 and new_bomb_distance == 0.75:
        events.append(MOVED_AWAY_FROM_BOMB)

    if old_bomb_distance != 0 and e.BOMB_EXPLODED not in events and e.BOMB_DROPPED not in events:
        if isclose(new_bomb_distance - old_bomb_distance, 0.25, abs_tol=0.05):
            events.append(MOVED_TOWARDS_BOMB)
        elif isclose(old_bomb_distance - new_bomb_distance, 0.25, abs_tol=0.05):
            events.append(MOVED_AWAY_FROM_BOMB)

    # Check if the agent went into a dead end
    if current_features["dead_end"] == 1 and e.INVALID_ACTION not in events and "WAITED" not in events:
        events.append(DEAD_END)


def reward_from_events(self, events: List[str]) -> int:
    """
    Rewards the agent gets so as to en/discourage certain behavior.
    :param self: The same object that is passed to all of the callbacks.
    :param events: the events that occurred in that game step
    :return: the sum of all rewards
    """
    # define how much a behaviour should be positively/ negatively rewarded
    game_rewards = {
        e.COIN_COLLECTED: 15,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -80,
        e.WAITED: -0.1,
        e.CRATE_DESTROYED: 4,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        MOVED_AWAY_FROM_BOMB: 3.5,
        MOVED_TOWARDS_BOMB: -3.5,
        USELESS_BOMB: -4,
        USEFUL_BOMB: 2,
        WAITED_IN_DANGER: -2,
        DEAD_END: -25
    }

    # sum up the rewards for all events that occurred
    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
