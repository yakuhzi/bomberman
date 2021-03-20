import matplotlib.pyplot as plt


class Visualization:

    @staticmethod
    def show_statistic(title, n_rounds, statistic, average):
        rounds = range(n_rounds)
        plt.plot(rounds, statistic)
        plt.plot(rounds, average)
        plt.title(title)
        plt.show()
