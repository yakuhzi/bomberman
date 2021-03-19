from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List

import events as e
from agent_code.agent_007_lva_task_1.visualization import Visualization
from .callbacks import state_to_features

# This is only an example!
from .q_learning_lva import update_q_function

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions

# TODO: get dynamically
NUMBER_OF_FEATURES = 19

# Events
LOOP_EVENT = "LOOP_EVENT"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
SURVIVED_BOMB = "SURVIVED_BOMB"
MOVED_TOWARDS_BOMB = "MOVED_TOWARDS_BOMB"
MOVED_AWAY_FROM_BOMB = "MOVED_AWAY_FROM_BOMB"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.rewards = []
    self.steps = []
    self.average_rewards = []
    self.average_steps = []
    self.model = initialize_model()


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

    # Idea: Add your own events to hand out rewards

    if old_game_state is None:
        return

    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    auxillary_events(self, events, old_state_features, new_state_features)
    rewards = reward_from_events(self, events)

    transition = Transition(old_state_features, self_action, new_state_features, rewards)
    self.transitions.append(transition)

    self.model = update_q_function(self.model, self.transitions, self_action)


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

    last_state_features = state_to_features(last_game_state)

    auxillary_events(self, events, last_state_features, last_state_features)
    rewards = reward_from_events(self, events)

    transition = Transition(last_state_features, last_action, last_state_features, rewards)
    self.transitions.append(transition)

    self.model = update_q_function(self.model, self.transitions, last_action)

    _, _, _, rewards = zip(*self.transitions)
    total_reward = np.sum(rewards)

    self.rewards.append(total_reward)
    self.steps.append(last_game_state['step'])

    self.average_rewards.append(np.sum(self.rewards) / len(self.rewards))
    self.average_steps.append(np.sum(self.steps) / len(self.steps))

    if "n_rounds" in last_game_state and last_game_state["round"] == last_game_state["n_rounds"]:
        Visualization.show_statistic("Reward", last_game_state["round"], self.rewards, self.average_rewards)
        Visualization.show_statistic("Steps", last_game_state["round"], self.steps, self.average_steps)

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def auxillary_events(self, events: List[str], old_state_features: List[int], new_state_features: List[int]):
    old_distance = old_state_features[13]
    new_distance = new_state_features[13]

    if old_distance - new_distance == 1:
        events.append(MOVED_TOWARDS_COIN)
    elif new_distance - old_distance == 1:
        events.append(MOVED_AWAY_FROM_COIN)

    old_bomb_distance = old_state_features[8]
    new_bomb_distance = new_state_features[8]

    if e.BOMB_EXPLODED not in events:
        if old_bomb_distance - new_bomb_distance == 1:
            events.append(MOVED_TOWARDS_BOMB)
        elif new_bomb_distance - old_bomb_distance == 1:
            events.append(MOVED_AWAY_FROM_BOMB)

    if e.BOMB_EXPLODED in events and e.KILLED_SELF not in events:
        events.append(SURVIVED_BOMB)

    if loop_detected(self.transitions):
        events.append(LOOP_EVENT)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        # e.GOT_KILLED: -10,
        # e.OPPONENT_ELIMINATED: 1,
        e.WAITED: -2,
        e.BOMB_DROPPED: -2,
        # e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 5,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -5,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        LOOP_EVENT: -5,
        e.SURVIVED_ROUND: 10,
        MOVED_TOWARDS_COIN: 1.5,
        MOVED_AWAY_FROM_COIN: -1.5,
        SURVIVED_BOMB: 2,
        MOVED_AWAY_FROM_BOMB: 3.5,
        MOVED_TOWARDS_BOMB: -3.5,
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    if not self.train:
        print(reward_sum, events)

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def loop_detected(transitions: deque) -> bool:
    if len(transitions) <= 3:
        return False

    if transitions[-1].action == transitions[-3].action and transitions[-1].action != \
        transitions[-2].action and transitions[-2].action == transitions[-4].action:
        return True

    return False
