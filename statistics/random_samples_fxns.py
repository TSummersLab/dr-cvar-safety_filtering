"""
Helper functions to generate random samples

Author:
Sleiman Safaoui
Email:
sleiman.safaoui@utdallas.edu
GitHub:
@The-SS
Date:
April 13, 2023
"""
import numpy as np
import cvxpy as cp
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import laplace
from scipy.stats import bernoulli
import matplotlib.pyplot as plt


def generate_noise_samples(shape, loc, scale, dist='norm'):
    """
    :param shape: shape of random variables
    :param loc: mean value
    :param scale: standard deviation
    :param dist: distribution type
    :return:
    """
    if dist == "norm":
        xi = norm.rvs(loc=loc, scale=scale, size=shape)
    elif dist == 'expo':
        xi = expon.rvs(loc=loc, scale=scale, size=shape)
    elif dist == 'lap':
        xi = laplace.rvs(loc=loc, scale=scale, size=shape)
    elif dist == 'bern':
        p = 0.5
        xi = (bernoulli.rvs(p, loc=0, size=shape) - p) * scale + loc
    else:
        raise NotImplementedError('Chosen distribution not implemented')
    return xi


def main():
    xi = generate_noise_samples(shape=10, loc=1, scale=1, dist='norm')
    print(xi)
    xi = generate_noise_samples(shape=(1, 10), loc=1, scale=1, dist='norm')
    print(xi)
    xi = generate_noise_samples(shape=(3, 10), loc=30, scale=2, dist='expo')
    print(xi)
    xi = generate_noise_samples(shape=(3, 10), loc=1, scale=1, dist='lap')
    print(xi)
    xi = generate_noise_samples(shape=(3, 10), loc=2, scale=1, dist='bern')
    print(xi)


if __name__ == "__main__":
    main()
