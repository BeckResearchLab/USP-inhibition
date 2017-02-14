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


def find_parents(n_rows, n_columns):

    # [Init pop, mutation rate, # generations, chromosome/solution length, # winners/per gen]
    initpop, mutrate, numgen, sol_len, num_win = n_rows, 0.1, 50, n_columns, 0.1 * n_rows
    # Initialize current population to random values within range
    curpop = np.random.choice(np.arange(-15, 15, step=0.01), size=(initpop, sol_len), replace=False)
    nextpop = np.zeros((curpop.shape[0], curpop.shape[1]))
    fitvec = np.zeros((initpop, 2))  # 1st col is indices, 2nd col is cost
    for i in range(numgen):  # Iterate through # generations
        # Create vector of all errors from cost function for each solution
        fitvec = np.array([np.array([x, np.sum(NN.costFunction(NN.X, NN.y, curpop[x].T))]) for x in range(initpop)])
        print("(Gen: #%s) Total error: %s\n" % (i, np.sum(fitvec[:, 1])))
        plt.pyplot.scatter(i, np.sum(fitvec[:, 1]))
        winners = np.zeros((num_win, sol_len))  # 10x3
        for n in range(len(winners)):  # For n in range(10)
            selected = np.random.choice(range(len(fitvec)), num_win/2, replace=False)
            wnr = np.argmin(fitvec[selected, 1])
            winners[n] = curpop[int(fitvec[selected[wnr]][0])]
        nextpop[:len(winners)] = winners  # Populate new gen with winners
        duplic_win = np.zeros(((initpop - len(winners)), winners.shape[1]))
        for x in range(winners.shape[1]):  # For each col in winners (3 cols)
            # Duplicate winners (10x3 matrix) 9 times to create a 90x3 matrix, then shuffle columns
            num_dups = ((initpop - len(winners))/len(winners))  # Num times to duplicate, needs to fill rest of nextpop
            duplic_win[:, x] = np.repeat(winners[:, x], num_dups, axis=0)  # Duplicate each col
            duplic_win[:, x] = np.random.permutation(duplic_win[:, x])  # Shuffle each col ("crossover")
        # Populate the rest of the generation with offspring of mating pairs
        nextpop[len(winners):] = np.matrix(duplic_win)
        # Create a mutation matrix, mostly 1s, but some elements are random numbers from a normal distribution
        mutmatrix = [np.float(np.random.normal(0, 2, 1)) if rn.random() < mutrate else 1 for x in range(nextpop.size)]
        # Randomly mutate part of the population by multiplying nextpop by our mutation matrix
        nextpop = np.multiply(nextpop, np.matrix(mutmatrix).reshape(nextpop.shape))
        curpop = nextpop
    plt.pyplot.ylabel('Total cost/err')
    plt.pyplot.xlabel('Generation #')
    parent = curpop[np.argmin(fitvec[:, 1])]

    return parent

