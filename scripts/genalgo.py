#!/usr/bin/env python

"""
To build a genetic algorithm that maximizes enzyme inhibition
activity and returns the characteristics of the parent drug compound.
"""

import numpy as np
from operator import add
from random import randint, random
import pandas as pd
import pickle

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"

results_pickle = '../trained_networks/rr_25_data.pkl'
with open(results_pickle, 'rb') as result:
    CLF = pickle.load(result)


def main():
    target = 100
    p_count = 100
    i_length = 270
    i_min = -2
    i_max = 2
    p = np.array(population(p_count, i_length, i_min, i_max))  # Shape -> (p_count, i_length) = (100, 270)
    fitness_history = [grade(p, target), ]
    for i in range(100):
        p = evolve(p, target)
        print(p)
        fitness_history.append(grade(p, target))

    for datum in fitness_history:
        print(datum)

    p.to_csv('../data/genalgo_results.csv')


def individual(length, minimum, maximum):
    """
    Create a member of the population.
    :param length:
    :param minimum:
    :param maximum:
    :return:
    """
    return [np.random.uniform(minimum, maximum) for x in xrange(length)]


def population(count, length, minimum, maximum):
    """
    Create a number of individuals (i.e. a population).

    :param count: the number of individuals in the population
    :param length: the number of values per individual
    :param minimum: the minimum possible value in an individual's list of values
    :param maximum: the maximum possible value in an individual's list of values

    """
    return [individual(length, minimum, maximum) for x in xrange(count)]


def fitness(individuals, target):
    """
    Determine the fitness of an individual. Higher is better.

    :param individuals: the individual to evaluate
    :param target: the target number individuals are aiming for
    """
    individuals = individuals.reshape((1, -1))
    activity = CLF.predict(individuals)

    return abs(target - activity)


def grade(pop, target):
    """
    Find average fitness for a population.
    :param pop:
    :param target:
    :return:
    """
    summed = reduce(add, (fitness(x, target) for x in pop))
    return summed / (len(pop) * 1.0)


def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    """
    :param pop:
    :param target:
    :param retain:
    :param random_select:
    :param mutate:
    :return: parents
    """
    graded = pd.DataFrame([(fitness(x, target), x) for x in pop],
                          columns=['fitness', 'descriptors']).sort_values(by='fitness', ascending=0)
    # Shape -> (100, 2)
    graded.reset_index(drop=True, inplace=True)
    graded = np.array(graded['descriptors'])
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length].tolist()
    # randomly add other individuals to promote genetic diversity
    for individuals in graded[retain_length:]:
        if random_select > random():
            parents.append(individuals)
    # mutate some individuals
    for individuals in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individuals)-1)
            # this mutation is not ideal, because it restricts the range of possible values,
            # but the function is unaware of the min/max values used to create the individuals.
            individuals[pos_to_mutate] = randint(np.floor(min(individuals)),
                                                 np.ceil(max(individuals)))
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = len(male) / 2
            child = male[:half] + female[half:]
            children.append(child)
    parents.extend(children)
    return parents

if __name__ == "__main__":
    main()
