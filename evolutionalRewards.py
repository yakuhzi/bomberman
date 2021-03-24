import numpy as np
import matplotlib as plt


def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    """

    :param fobj: function to optimize (can be defined with def or lambda expression)
    :param bounds: list with lower and upper bound for each parameter of function
    :param mut:
    :param crossp:
    :param popsize:
    :param its:
    :return:
    """
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    # next step has the population of vectors and can evaluate using foj --> map to rewards
    pop_denorm = min_b + pop * diff
    # evaluate vectors --> run game for each
    #fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    fitness = []
    for ind in pop_denorm:
        results = fobj(ind)
        print("first result ", results)
        fitness.append(results)
    fitness = np.asarray(fitness)
    print("fitness ", fitness)
    best_idx = np.argmax(fitness)
    best = pop_denorm[best_idx]
    print("first best ", best)
    for i in range(its):
        for j in range(popsize):
            # list with indexes of vectors in pop, excluding current one
            idxs = [idx for idx in range(popsize) if idx != j]
            # randomly choose 3 indexes
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            # create mutant by combining a, b, c --> clip to normalize again
            mutant = np.clip(a + mut * (b - c), 0, 1)
            # binomial crossover
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            # take values from mutant
            trial = np.where(cross_points, mutant, pop[j])
            # denormalize and evaluate, if better than current --> replace
            trial_denorm = min_b + trial * diff
            print("trial denorm ", trial_denorm)
            # evaluate mutant
            f = fobj(trial_denorm)
            print("evaluate mutant: ", f)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
                    print("best ", best)
        yield best, fitness[best_idx]
