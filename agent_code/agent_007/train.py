from collections import namedtuple, deque

import numpy as np
import pickle
from typing import List

import events as e
from agent_code.agent_007.visualization import Visualization
from .callbacks import state_to_features

# This is only an example!
from .q_learning_lva import update_q_function

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 1000  # keep only ... last transitions

# TODO: get dynamically
NUMBER_OF_FEATURES = 9

# Events
LOOP_EVENT = "LOOP_EVENT"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"


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
    self.rounds_in_loop = 0
    self.rounds_waited = 0


def initialize_model():
    model = {
        'UP': np.zeros(NUMBER_OF_FEATURES),
        'RIGHT': np.zeros(NUMBER_OF_FEATURES),
        'DOWN': np.zeros(NUMBER_OF_FEATURES),
        'LEFT': np.zeros(NUMBER_OF_FEATURES),
        'WAIT': np.zeros(NUMBER_OF_FEATURES),
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

    if loop_detected(self):
        events.append(LOOP_EVENT)

    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    old_distance = old_state_features[4]
    new_distance = new_state_features[4]

    if new_distance < old_distance:
        events.append(MOVED_TOWARDS_COIN)
    elif new_distance - old_distance == 1:
        events.append(MOVED_AWAY_FROM_COIN)

    if 'WAITED' in events:
        self.rounds_waited += 1
    else:
        self.rounds_in_loop = 0

    rewards = reward_from_events(self, events)

    if old_state_features is not None and new_state_features is not None:
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


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        # e.KILLED_OPPONENT: 5,
        # e.KILLED_SELF: -10,
        # e.GOT_KILLED: -10,
        # e.OPPONENT_ELIMINATED: 1,
        e.WAITED: -2 - self.rounds_waited,
        # e.BOMB_DROPPED: 0,
        # e.BOMB_EXPLODED: 0,
        # e.CRATE_DESTROYED: 0.4,
        # e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -5,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_RIGHT: -1,
        LOOP_EVENT: -5,
        MOVED_TOWARDS_COIN: 1.5,
        MOVED_AWAY_FROM_COIN: -1.5,
        # e.SURVIVED_ROUND: 0.5,
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def loop_detected(self) -> bool:
    if len(self.transitions) <= 3:
        return False

    if self.transitions[-1].action == self.transitions[-3].action and self.transitions[-1].action != \
            self.transitions[-2].action and self.transitions[-2].action == self.transitions[-4].action:
        self.rounds_in_loop += 1
        return True
    self.rounds_in_loop = 0
    return False
