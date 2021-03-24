from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List, Tuple, Optional
from math import isclose

import events as e
from .visualization import Visualization
from .callbacks import state_to_features

# This is only an example!
from .q_learning_lva import update_q_function

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions
ACTION_HISTORY_SIZE = 10  # number of last actions for loop detection

# TODO: get dynamically
NUMBER_OF_FEATURES = 21

# Events
LOOP_EVENT = "LOOP_EVENT"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
SURVIVED_BOMB = "SURVIVED_BOMB"
MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"
USELESS_BOMB = "USELESS_BOMB"
USEFUL_BOMB = "USEFUL_BOMB"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.action_history = deque(maxlen=ACTION_HISTORY_SIZE)
    self.model = initialize_model()
    self.rewards = []
    self.steps = []
    self.average_rewards = []
    self.average_steps = []


def initialize_model():
    model = {
        'UP': np.zeros(NUMBER_OF_FEATURES),
        'RIGHT': np.zeros(NUMBER_OF_FEATURES),
        'DOWN': np.zeros(NUMBER_OF_FEATURES),
        'LEFT': np.zeros(NUMBER_OF_FEATURES),
        'WAIT': np.zeros(NUMBER_OF_FEATURES),
        'BOMB': np.zeros(NUMBER_OF_FEATURES),
    }
    return model


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
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Update model
    update_model(self, last_game_state, last_action, None, events)

    # Calculate total rewards
    round_transitions = np.array(self.transitions, dtype=object)[-last_game_state["step"]:]
    _, _, _, rewards = zip(*round_transitions)
    total_reward = np.sum(rewards)

    # Append statistics
    self.rewards.append(total_reward)
    self.steps.append(last_game_state['step'])

    # Calculate average statistics
    self.average_rewards.append(np.sum(self.rewards) / len(self.rewards))
    self.average_steps.append(np.sum(self.steps) / len(self.steps))

    # Plot training statistics after last round
    if "n_rounds" in last_game_state and last_game_state["round"] == last_game_state["n_rounds"]:
        Visualization.show_statistic("Reward", last_game_state["round"], self.rewards, self.average_rewards)
        Visualization.show_statistic("Steps", last_game_state["round"], self.steps, self.average_steps)

    # Reset action history
    self.action_history = deque(maxlen=ACTION_HISTORY_SIZE)
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def update_model(self, old_game_state: dict, action: str, new_game_state: Optional[dict], events: List[str]):
    # Convert state to features
    old_state_features = state_to_features(old_game_state)
    new_state_features = np.zeros(NUMBER_OF_FEATURES) if new_game_state is None else state_to_features(new_game_state)

    # Add auxiliary events
    if new_game_state is not None:
        add_auxiliary_events(self, events, old_state_features, new_state_features, action)

    # Calculate reward
    reward = reward_from_events(self, events)

    # Add transition to memory
    transition = Transition(old_state_features, action, new_state_features, reward)
    self.transitions.append(transition)

    # Update model
    self.model = update_q_function(self.model, self.transitions, action)


def add_auxiliary_events(
    self,
    events: List[str],
    old_state_features: List[int],
    new_state_features: List[int],
    action: Optional[str]
):
    if e.BOMB_EXPLODED not in events:
        old_bomb_distance = old_state_features[4]
        new_bomb_distance = new_state_features[4]

        if isclose(old_bomb_distance - new_bomb_distance, 1 / 20, abs_tol=0.01):
            events.append(MOVED_TOWARDS_BOMB)
        elif isclose(new_bomb_distance - old_bomb_distance, 1 / 20, abs_tol=0.01):
            events.append(MOVED_AWAY_FROM_BOMB)

    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(SURVIVED_BOMB)

    if e.BOMB_DROPPED in events and old_state_features[20] == 0:
        events.append(USELESS_BOMB)

    if e.BOMB_DROPPED in events and old_state_features[20] == 1:
        events.append(USEFUL_BOMB)

    if action is not None and loop_detected(self, action):
        events.append(LOOP_EVENT)

    # Add coin event if no bomb event
    if new_state_features[4] == 0 and new_state_features[5] == 0:
        old_coin_distance = old_state_features[9]
        new_coin_distance = new_state_features[9]

        if isclose(old_coin_distance - new_coin_distance, 1 / 20, abs_tol=0.01):
            events.append(MOVED_TOWARDS_COIN)
        elif isclose(new_coin_distance - old_coin_distance, 1 / 20, abs_tol=0.01):
            events.append(MOVED_AWAY_FROM_COIN)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -20,
        # e.GOT_KILLED: -10,
        # e.OPPONENT_ELIMINATED: 1,
        e.WAITED: -0.2,
        e.BOMB_DROPPED: -2,
        # e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 3,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        # e.SURVIVED_ROUND: 10,
        LOOP_EVENT: -2,
        MOVED_TOWARDS_COIN: 1.5,
        MOVED_AWAY_FROM_COIN: -1.5,
        SURVIVED_BOMB: 1.8,
        MOVED_AWAY_FROM_BOMB: 2.5,
        MOVED_TOWARDS_BOMB: -2.5,
        USELESS_BOMB: -5,
        USEFUL_BOMB: 3
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # print(reward_sum, events)

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def loop_detected(self, action: str) -> bool:
    # Append action to queue
    self.action_history.append(action)

    if len(self.action_history) <= 3:
        return False

    return self.action_history[-1] == self.action_history[-3] and self.action_history[-1] != \
           self.action_history[-2] and self.action_history[-2] == self.action_history[-4]
