import matplotlib.pyplot as plt


class Visualization:

    @staticmethod
    def show_rounds_statistic(rounds, statistic, average):
        plt.plot(rounds, statistic)
        plt.plot(rounds, average)
        plt.show()
