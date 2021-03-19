from collections import namedtuple, deque

import numpy as np
import pickle
import random
import torch
from typing import List

import events as e
from agent_code.agent_007.model import LinearQNet, QTrainer
from agent_code.agent_007.ddqn_model import QNetwork, Memory
from agent_code.agent_007.visualization import Visualization
from .callbacks import state_to_features
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# This is only an example!
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 5  # keep only ... last transitions

# Events
LOOP_EVENT = "LOOP_EVENT"
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_EVERY = 4


def train_step(batch_size, current, target, optim, memory, gamma):
    states, actions, next_states, rewards = memory.sample(batch_size)

    q_values = current(states)

    next_q_values = current(next_states)
    next_q_state_values = target(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1).squeeze(1))
    next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = rewards + gamma * next_q_value

    loss = (q_value - expected_q_value.detach()).pow(2).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()


def evaluate(Qmodel, env, repeats):
    """
    Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
    episode reward.
    """
    Qmodel.eval()
    perform = 0
    for _ in range(repeats):
        state = env.reset()
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                values = Qmodel(state)
            action = np.argmax(values.cpu().numpy())
            state, reward, done, _ = env.step(action)
            perform += reward
    Qmodel.train()
    return perform / repeats


def update_parameters(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    #hidden: 256, input: features, output: actions
    #self.model = LinearQNet(9, 256, 5)
    self.state_size = 9
    self.action_size = 5

    # Q-Network
    self.qnetwork_local = QNetwork(self.action_size, self.state_size, 256).to(device)
    self.qnetwork_target = QNetwork(self.action_size, self.state_size, 256).to(device)
    #self.trainer = QTrainer(self.model, lr=1e-3, gamma=0.99)

    self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=LR)

    # Replay Memory
    self.memory = Memory(self.action_size, BUFFER_SIZE, BATCH_SIZE)

    # initialize time_step (for updating every UPDATE_EVERY steps)
    self.t_step = 0

    self.coins = []
    self.steps = []
    self.rounds = []
    self.average_coins = []
    self.average_steps = []


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

    #print(self_action)
    #if loop_detected(self):
    #    events.append(LOOP_EVENT)

    old_state_features = state_to_features(old_game_state)
    new_state_features = state_to_features(new_game_state)

    old_distance = old_state_features[4]
    new_distance = new_state_features[4]

    if new_distance < old_distance:
        events.append(MOVED_TOWARDS_COIN)
    elif new_distance - old_distance == 1:
        events.append(MOVED_AWAY_FROM_COIN)

    # select new action

    # map selected action to binary
    mapped_action = map_action(self_action)

    # calculate reward from events
    rewards = reward_from_events(self, events)

    # save experience in replay memory
    self.memory.add(old_state_features, mapped_action, rewards, new_state_features)

    # learn every UPDATE_EVERY time steps
    self.t_step = (self.t_step+1)% UPDATE_EVERY

    if self.t_step == 0:
        # if enough samples are available in memory, get random subset and learn
        if len(self.memory)>BATCH_SIZE:
            experience = self.memory.sample()
            learn(self, experience, GAMMA)

    #transition = Transition(old_state_features, mapped_action, new_state_features, rewards)
    #self.transitions.append(transition)

    #self.trainer.train_step(old_state_features, mapped_action, new_state_features, rewards)


def learn(self, experiences, gamma):
    """
    update value parameters using given batch of experience tuples
    :param self:
    :param experiences:
    :param gamma:
    :return:
    """
    states, actions, rewards, next_states = experiences
    criterion = torch.nn.MSELoss()
    self.qnetwork_local.train()
    self.qnetwork_target.eval()
    predicted_targets = self.qnetwork_local(states).gather(1, actions.long())
    # detach() -> returns new tensor, detached from current
    with torch.no_grad():
        labels_next = self.qnetwork_target(next_states)


    labels = rewards + (gamma * labels_next)
    # labels: 64, 1
    # predicted_targets: 64, 5
    loss = criterion(predicted_targets, labels).to(device)
    loss.backward()
    self.optimizer.step()

    # update target network
    soft_update(self, self.qnetwork_local, self.qnetwork_target, TAU)


def soft_update(self, local_model, target_model, tau):
    """
    θ_target = τ*θ_local + (1 - τ)*θ_target
    :param self:
    :param local_model:
    :param target_model:
    :param tau:
    :return:
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)


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
    if len(self.memory.memory) > BATCH_SIZE:
        mini_sample = self.memory.sample_experience()
    else:
        mini_sample = self.memory.memory

    #if last_game_state['round'] % self.measure_step == 0:
    #    self.performance.append([last_game_state['round'], evaluate(self.Q1, last_game_state, self.measure_repeats)])

    #self.memory.state.add(last_game_state)
    states, actions, next_states, rewards = zip(*mini_sample)
    #self.trainer.train_step(states, actions, next_states, rewards)

    total_reward = np.sum(rewards)

    self.coins.append(total_reward)
    self.steps.append(last_game_state['step'])
    self.rounds.append(last_game_state['round'])

    self.average_coins.append(np.sum(self.coins) / len(self.coins))
    self.average_steps.append(np.sum(self.steps) / len(self.steps))

    #Visualization.show_rounds_statistic(self.rounds, self.coins, self.average_coins)
    #Visualization.show_rounds_statistic(self.rounds, self.steps, self.average_steps)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.qnetwork_local, file)


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
    if len(self.memory) <= 3:
        return False

    if self.memory[-1].action == self.memory[-3].action and self.memory[-1].action != \
            self.memory[-2].action and self.memory[-2].action == self.memory[-4].action:
        return True

    return False
