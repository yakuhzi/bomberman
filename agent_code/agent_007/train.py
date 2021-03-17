from collections import namedtuple, deque

import numpy as np
import pickle
import random
from typing import List

import events as e
from agent_code.agent_007.model import LinearQNet, QTrainer
from agent_code.agent_007.visualization import Visualization
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

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
    self.model = LinearQNet(9, 256, 5)
    self.trainer = QTrainer(self.model, lr=0.001, gamma=0.9)
    self.coins = []
    self.steps = []
    self.rounds = []
    self.average_coins = []
    self.average_steps = []
    self.highscore = 0
    self.highest_reward = -float('inf')


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

    mapped_action = map_action(self_action)
    rewards = reward_from_events(self, events)

    transition = Transition(old_state_features, mapped_action, new_state_features, rewards)
    self.transitions.append(transition)

    self.trainer.train_step(old_state_features, mapped_action, new_state_features, rewards)


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

    if len(self.transitions) > TRANSITION_HISTORY_SIZE:
        mini_sample = random.sample(self.transitions, TRANSITION_HISTORY_SIZE)
    else:
        mini_sample = self.transitions

    states, actions, next_states, rewards = zip(*mini_sample)
    self.trainer.train_step(states, actions, next_states, rewards)

    score = last_game_state["self"][1]
    total_reward = np.sum(rewards)

    if score >= self.highscore and total_reward > self.highest_reward:
        self.highscore = score
        self.highest_reward = total_reward

        # Store the model
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.model, file)

    self.coins.append(total_reward)
    self.steps.append(last_game_state['step'])
    self.rounds.append(last_game_state['round'])

    self.average_coins.append(np.sum(self.coins) / len(self.coins))
    self.average_steps.append(np.sum(self.steps) / len(self.steps))

    Visualization.show_rounds_statistic(self.rounds, self.coins, self.average_coins)
    Visualization.show_rounds_statistic(self.rounds, self.steps, self.average_steps)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        # e.KILLED_SELF: -10,
        # e.GOT_KILLED: -10,
        # e.OPPONENT_ELIMINATED: 1,
        e.WAITED: -2,
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
        MOVED_TOWARDS_COIN: 0.5,
        MOVED_AWAY_FROM_COIN: -1.5,
        # e.SURVIVED_ROUND: 0.5,
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def map_action(self_action: str) -> List[int]:
    if self_action == "UP":
        return [1, 0, 0, 0, 0]
    elif self_action == "RIGHT":
        return [0, 1, 0, 0, 0]
    elif self_action == "DOWN":
        return [0, 0, 1, 0, 0]
    elif self_action == "LEFT":
        return [0, 0, 0, 1, 0]
    else:
        return [0, 0, 0, 0, 1]


def loop_detected(self) -> bool:
    if len(self.transitions) <= 3:
        return False

    if self.transitions[-1].action == self.transitions[-3].action and self.transitions[-1].action != \
            self.transitions[-2].action and self.transitions[-2].action == self.transitions[-4].action:
        return True

    return False
