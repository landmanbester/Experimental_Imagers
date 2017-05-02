#!/usr/bin/env python
"""
This is a collection of tools useful while implementing MCMC
Mon 6 March 2017
"""

import numpy as np

def get_suff_stats(xnow, xprev):
    """
    Here we compute sufficient statistics to sample from the Matrix Normal Inverse Wishart conjugate prior
    :param xnow: the state at time t+1
    :param xprev: the state at time t
    :return:
    """
