from collections import namedtuple, deque

import numpy as np
import pickle
from math import isclose
from typing import List, Optional

import events as e
from .callbacks import state_to_features
from .q_learning_lva import update_q_function
from .visualization import Visualization

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

# Hyper parameters -- DO modify
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

    self.rewards = []
    self.total_rewards = []
    self.average_rewards = []

    self.coins = 0
    self.total_coins = []
    self.average_coins = []

    self.steps = []
    self.average_steps = []

    self.invalid_actions = 0
    self.total_invalid_actions = []
    self.average_invalid_actions = []

    self.killed_agents = 0
    self.total_killed_agents = []
    self.average_killed_agents = []


def initialize_model():
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

    if e.INVALID_ACTION in events:
        self.invalid_actions += 1

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
    total_reward = np.sum(self.rewards)
    self.total_rewards.append(total_reward)
    self.rewards = []

    # Append statistics
    self.steps.append(last_game_state['step'])

    self.total_invalid_actions.append(self.invalid_actions)
    self.invalid_actions = 0

    self.total_coins.append(self.coins)
    self.coins = 0

    self.total_killed_agents.append(self.killed_agents)
    self.killed_agents = 0

    # Calculate average statistics
    self.average_rewards.append(np.sum(self.total_rewards) / len(self.total_rewards))
    self.average_steps.append(np.sum(self.steps) / len(self.steps))
    self.average_invalid_actions.append(np.sum(self.total_invalid_actions) / len(self.total_invalid_actions))
    self.average_coins.append(np.sum(self.total_coins) / len(self.total_coins))
    self.average_killed_agents.append(np.sum(self.total_killed_agents) / len(self.total_killed_agents))

    # Plot training statistics after last round
    if "n_rounds" in last_game_state and last_game_state["round"] == last_game_state["n_rounds"]:
        Visualization.show_statistic("Reward", last_game_state["round"], self.total_rewards, self.average_rewards)
        Visualization.show_statistic("Steps", last_game_state["round"], self.steps, self.average_steps)
        Visualization.show_statistic(
            "Invalid Actions", last_game_state["round"], self.total_invalid_actions, self.average_invalid_actions
        )
        Visualization.show_statistic("Coins Collected", last_game_state["round"], self.total_coins, self.average_coins)
        Visualization.show_statistic(
            "Agents killed", last_game_state["round"], self.total_killed_agents, self.average_killed_agents
        )

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def update_model(self, old_game_state: dict, self_action: str, new_game_state: Optional[dict], events: List[str]):
    # Convert state to features
    current_features = state_to_features(old_game_state, self_action)
    #print(self_action, current_features)

    # Add auxiliary events
    if len(self.transitions) > 1:
        add_auxiliary_events(self, events, current_features)

    # Calculate reward
    reward = reward_from_events(self, events)
    self.rewards.append(reward)

    # Add coin statistic
    if e.COIN_COLLECTED in events:
        self.coins += 1

    # Add kill statistic
    if e.KILLED_OPPONENT in events:
        self.killed_agents += 1

    # Add transition to memory
    transition = Transition(current_features, self_action, reward)
    self.transitions.append(transition)

    # Update model
    self.model = update_q_function(self.model, old_game_state, self_action, new_game_state, reward)


def add_auxiliary_events(self, events: List[str], current_features: dict):
    if e.BOMB_DROPPED in events:
        if 0.11 < self.transitions[-1][0]["crate_distance"] < 0.15:
            events.append(USEFUL_BOMB)
        else:
            events.append(USELESS_BOMB)

    old_bomb_distance = self.transitions[-1][0]["bomb_distance"]
    new_bomb_distance = current_features["bomb_distance"]

    if (old_bomb_distance != 0 and old_bomb_distance == new_bomb_distance) or (
        old_bomb_distance == 0 and new_bomb_distance == 1):
        events.append(WAITED_IN_DANGER)

    if old_bomb_distance == 0 and new_bomb_distance == 0.75:
        events.append(MOVED_AWAY_FROM_BOMB)

    if old_bomb_distance != 0 and e.BOMB_EXPLODED not in events and e.BOMB_DROPPED not in events:
        if isclose(new_bomb_distance - old_bomb_distance, 0.25, abs_tol=0.05):
            events.append(MOVED_TOWARDS_BOMB)
        elif isclose(old_bomb_distance - new_bomb_distance, 0.25, abs_tol=0.05):
            events.append(MOVED_AWAY_FROM_BOMB)

    if current_features["dead_end"] == 1 and e.INVALID_ACTION not in events and "WAITED" not in events:
        events.append(DEAD_END)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.GOT_KILLED: -80,
        # e.OPPONENT_ELIMINATED: 1,
        e.WAITED: -0.1,
        e.BOMB_DROPPED: 0,
        # e.BOMB_EXPLODED: 0,
        e.CRATE_DESTROYED: 4,
        e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        e.SURVIVED_ROUND: 0,
        MOVED_AWAY_FROM_BOMB: 3.5,
        MOVED_TOWARDS_BOMB: -3.5,
        USELESS_BOMB: -4,
        USEFUL_BOMB: 2,
        WAITED_IN_DANGER: -2,
        DEAD_END: -25
    }

    reward_sum = 0

    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    # print(reward_sum, events)

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
