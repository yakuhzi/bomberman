import matplotlib.pyplot as plt

class Visulatization:
    def __init__(self):
        pass

    @staticmethod
    def show_rounds_statistic(rounds, statistic):
        plt.plot(rounds, statistic)
        plt.show()