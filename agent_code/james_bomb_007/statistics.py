from typing import List

import matplotlib.pyplot as plt
import numpy as np

import events as e


class Statistics:
    """Class that saves the statistics of the training and can show it in a plot"""

    def __init__(self):
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

    def add_step_statistic(self, reward: float, events: List[str]) -> None:
        """
        Adds the statistic of a step to the total statistic.

        :param reward: Reward of the step.
        :param events: Events occured in the step
        """
        # Add reward
        self.rewards.append(reward)

        # Add invalid action
        if e.INVALID_ACTION in events:
            self.invalid_actions += 1

        # Add collected coin
        if e.COIN_COLLECTED in events:
            self.coins += 1

        # Add killed opponent
        if e.KILLED_OPPONENT in events:
            self.killed_agents += 1

    def add_round_statistic(self, last_game_state: dict) -> None:
        """
        Adds the statistic of a round to the total statistic.

        :param last_game_state: Last state of the game.
        """
        total_reward = np.sum(self.rewards)
        self.total_rewards.append(total_reward)
        self.rewards = []

        self.steps.append(last_game_state['step'])

        self.total_invalid_actions.append(self.invalid_actions)
        self.invalid_actions = 0

        self.total_coins.append(self.coins)
        self.coins = 0

        self.total_killed_agents.append(self.killed_agents)
        self.killed_agents = 0

        # Calculate average statistics
        self.average_rewards.append(np.sum(self.total_rewards[-20:]) / len(self.total_rewards[-20:]))
        self.average_steps.append(np.sum(self.steps[-20:]) / len(self.steps[-20:]))
        self.average_invalid_actions.append(
            np.sum(self.total_invalid_actions[-20:]) / len(self.total_invalid_actions[-20:])
        )
        self.average_coins.append(np.sum(self.total_coins[-20:]) / len(self.total_coins[-20:]))
        self.average_killed_agents.append(np.sum(self.total_killed_agents[-20:]) / len(self.total_killed_agents[-20:]))

    def show(self, n_rounds: int) -> None:
        """
        Show training statistics

        :param n_rounds: Total number of trained rounds
        """
        self._show_plot(
            title="Total Reward",
            n_rounds=n_rounds,
            statistic=self.total_rewards,
            average=self.average_rewards,
            legend=['Total reward', 'Average over 20 rounds']
        )
        self._show_plot(
            title="Steps",
            n_rounds=n_rounds,
            statistic=self.steps,
            average=self.average_steps,
            legend=['Steps', 'Average over 20 rounds']
        )
        self._show_plot(
            title="Invalid Actions",
            n_rounds=n_rounds,
            statistic=self.total_invalid_actions,
            average=self.average_invalid_actions,
            legend=['Invalid actions', 'Average over 20 rounds']
        )
        self._show_plot(
            title="Coins Collected",
            n_rounds=n_rounds,
            statistic=self.total_coins,
            average=self.average_coins,
            legend=['Coins collected', 'Average over 20 rounds']
        )
        self._show_plot(
            title="Agents killed",
            n_rounds=n_rounds,
            statistic=self.total_killed_agents,
            average=self.average_killed_agents,
            legend=['Agents killed', 'Average over 20 rounds']
        )

    def _show_plot(
        self, title: str, n_rounds: int, statistic: List[int], average: List[int], legend: List[str]
    ) -> None:
        """
        Helper function to plot the statistics.
        :param title: Title of the plot.
        :param n_rounds: Total number of trained rounds.
        :param statistic: Data of the plot.
        :param average: Average data of the plot.
        :param legend: Legend of the plot.
        """
        rounds = range(n_rounds)
        plt.plot(rounds, statistic)
        plt.plot(rounds, average)
        plt.title(title)
        plt.xlabel('Round')
        plt.legend(legend)
        plt.show()
