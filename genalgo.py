#!/usr/bin/env python

"""
To build a genetic algorithm that maximizes enzyme inhibition
activity and returns the characteristics of the parent drug compound.
"""

from random import randint, random
from operator import add

__author__ = "Pearl Philip"
__credits__ = "David Beck"
__license__ = "BSD 3-Clause License"
__maintainer__ = "Pearl Philip"
__email__ = "pphilip@uw.edu"
__status__ = "Development"


def individual(length, minimum, maximum):
    """
    Create a member of the population.
    :param length:
    :param minimum:
    :param maximum:
    :return:
    """
    return [randint(minimum, maximum) for x in xrange(length)]


def population(count, length, minimum, maximum):
    """
    Create a number of individuals (i.e. a population).
    :param count: the number of individuals in the population
    :param length: the number of values per individual
    :param minimum: the minimum possible value in an individual's list of values
    :param maximum: the maximum possible value in an individual's list of values
    :return:
    """
    return [individual(length, minimum, maximum) for x in xrange(count)]


def fitness(person, target):
    """
    Determine the fitness of a person. Higher is better.
    :param person: the person to evaluate
    :param target: the target number persons are aiming for
    :return:
    """
    sums = reduce(add, person, 0)
    return abs(target-sums)


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
    graded = [(fitness(x, target), x) for x in pop]
    graded = [x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other persons to
    # promote genetic diversity
    for person in graded[retain_length:]:
        if random_select > random():
            parents.append(person)
    # mutate some persons
    for person in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(person)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the persons
            person[pos_to_mutate] = randint(min(person), max(person))
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
