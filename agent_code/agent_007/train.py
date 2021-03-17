import pickle
import random
from collections import namedtuple, deque
from typing import List
from agent_code.agent_007.model import Linear_QNet, QTrainer
from agent_code.agent_007.visualization import Visulatization
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 20  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
LOOP_EVENT = "LOOP_EVENT"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.model = Linear_QNet(9, 256, 5)
    self.trainer = QTrainer(self.model, lr=0.001, gamma=0.9)
    self.coins = []
    self.steps = []
    self.rounds = []


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
    if loop_detected(self):
        events.append(LOOP_EVENT)
    if old_game_state is not None:
        old_state = state_to_features(old_game_state)
        new_state = state_to_features(new_game_state)

        self.trainer.train_step(old_state, map_action(self_action), reward_from_events(self, events), new_state)
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def loop_detected(self) -> bool:
    if len(self.transitions)>3:
        if self.transitions[-1].action == self.transitions[-3].action and self.transitions[-1].action!=self.transitions[-2].action and self.transitions[-2].action == self.transitions[-4].action:
            return True
    return False


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


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.coins.append(last_game_state['self'][1])
    self.steps.append(last_game_state['step'])
    self.rounds.append(last_game_state['round'])
    Visulatization.show_rounds_statistic(self.rounds, self.coins)
    Visulatization.show_rounds_statistic(self.rounds, self.steps)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    if len(self.transitions) > 1000:
        mini_sample = random.sample(self.transitions, 1000)  # list of tuples
    else:
        mini_sample = self.transitions
    states, actions, next_states, rewards = zip(*mini_sample)

    mapped_actions = list(map(lambda action: map_action(action), actions))
    self.trainer.train_step(states, mapped_actions, rewards, next_states)
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
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
        e.COIN_COLLECTED: 10,
        # e.KILLED_OPPONENT: 5,
        # e.KILLED_SELF: -10,
        # e.GOT_KILLED: -10,
        # e.OPPONENT_ELIMINATED: 1,
        e.WAITED: -0.2,
        # e.BOMB_DROPPED: 0,
        # e.BOMB_EXPLODED: 0,
        # e.CRATE_DESTROYED: 0.4,
        # e.COIN_FOUND: 0.5,
        e.INVALID_ACTION: -0.5,
        e.MOVED_UP: -0.1,
        e.MOVED_DOWN: -0.1,
        e.MOVED_LEFT: -0.1,
        e.MOVED_RIGHT: -0.1,
        LOOP_EVENT: -0.5
        # e.SURVIVED_ROUND: 0.5,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
