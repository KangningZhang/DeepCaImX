'''
With these functions you can generate fake 2 photon data, given simple inputs.

Author: S W Keemink, swkeemink@scimail.eu
'''
from __future__ import division
import numpy as np
import tifffile
from scipy.stats import multivariate_normal
from scipy.stats import norm


def spike_data_poisson(time=1000., dt=.1, rates=[0.], trials=1):
    ''' Generates spiking data (poisson)

    (from python course freiburg, slow but functional)

    Parameters
    ----------
    time : float
        Final time
    dt : float
        Timestep
    rates : list
        The firing rate in each block length 'time'
    trials : int
        How many trials to generate

    Returns
    -------
    array
        number of time-points by number of trials array with 0's and 1's
    '''
    trng = np.arange(0, time, dt)
    sp_data_all = []
    for t in range(int(trials)):
        sp_data = np.array([])
        for r in rates:
            sp_prob = r * dt
            sp_data = np.concatenate(
                (sp_data, np.random.binomial(1, sp_prob, len(trng)))
            )
        sp_data_all.append(sp_data)
    return np.array(sp_data_all)


def calcium_simple(n, taur, taud, dt, A=1, sigma=0.5, l=1000):
    ''' Generate calcium dye measurements of spike trains
    (assuming simple convolution with fixed difference of exponentials kernel)

    Parameters
    ----------
    n : array
        spiketrain
    taur : float
        time constant rise time calcium degradation/dye degradation
    taud : float
        time constant decay time calcium degradation/dye degradation
        for best results, taur<taud
    A  : floats
        Size of calcium jump following a spike
    sigma : float
        Noise magnitude
    l : int
        Length kernel
    dt : float
        timestep

    Returns
    -------
    array
        Calcium trace'''
    out = np.zeros(len(n))
    k = np.zeros(l * 2)
    t = np.arange(0, l) * dt
    k[l:] = A * (-np.exp(-t / taur) + np.exp(-t / taud))
    out = np.convolve(k, n, 'same')
    out += np.convolve(k, np.random.normal(0, sigma**2, len(n)), 'same')
    return out


def makeDendriteFilter(x0, y0, L, sigma, l, angle0, windiness, bg_noise=0):
    ''' For a given image size, make a spatial dendrite filter.
    From the starting point and starting angle the dendrite will
    be randomly wirling around.

    Parameters
    ----------
    x0,y0 : doubles
        Start point of dendrite
    L   : double
        number of steps the dendrite is long
    sigma : double
        width of the dendrite
    l : int
        width and height of image
    angle0 : double
        starting angle the dendrite will be going at
    windiness : double
        How windy the path should be. Will be between 0 (straight) and 1 (extremely windy)
    bg_noise : array
        Background noise

    Returns
    -------
    path
        The path in x and y coordinates (2d array)
    filt
        The spatial filter (2d array)
    '''
    # setup distance from dendrite distribution
    rn = norm(0, sigma)

    # initialize filter
    filt = np.zeros((l, l))

    # initialize path
    path = np.zeros((L, 2))  # path of dendrite
    path[0, :] = [x0, y0]

    #
    angle = angle0
    for j in range(1, L):
        angle += (rand() - 0.5) * np.pi * windiness * 2
        path[j, :] = path[j - 1, :] + 1 * [np.cos(angle), np.sin(angle)]

    # for each pixel, find out how far away from current path you are
    for x in arange(0, l):
        for y in arange(0, l):
            # find distance to closest point on path
            dx = abs(path[:, 0] - x)
            dy = abs(path[:, 1] - y)
            dr = sqrt(dx**2 + dy**2)
            mindr = min(dr)
            filt[x, y] += rn.pdf(mindr)

    # normalize filter so that max is 1
    filt /= np.max(filt)

    return path, filt


def makeGaussianFilter(x0, y0, cov, l, ring=False):
    ''' For a given image size, make a spatial Gaussian filter.

    Parameters
    ----------
    x0,y0 : doubles
        Start point of dendrite
    cov : array
        covariance matrix gaussian
    l : int
        width and height of image
    ring : binary, optional [False]
        if True, cell will be ring
    Returns
    -------
    filt
        The spatial filter (2d array)
    '''
    # get x,y values for gaussians
    x, y = np.mgrid[0:l, 0:l]
    pos = np.dstack((x, y))

    # generate filter
    mean = x0, y0
    rv = multivariate_normal(mean, cov)
    filt = rv.pdf(pos)  # spatial filter

    # normalize filter so that max is 1
    filt /= np.max(filt)

    # make ring
    if ring:
        rv = multivariate_normal(mean, cov / 2)
        filtmin = rv.pdf(pos)  # spatial filter
        # normalize filter so that max is 1
        filtmin /= np.max(filtmin)
        filt -= filtmin

    # renormalize filter so that max is 1
    filt /= np.max(filt)

    return filt


def gen2photon(calcium, bg, X, R, l, sigma_noise, freq):
    ''' generating 2 images for each timestep, using both dendrites and gaussian
    neurons

    inputs:
        calcium : array
            array with the calcium traces for each neuron
        bg : array
            array with the background trace
        X       : array
            array containing the x,y locations of each neuron
        R       : array
            array containing the width of each spatial filter
                 (as variance of a gaussian)
                 This value is divided by two for a ring neuron
                 This value is divided by 20 for a dendrite
                 These divisions are done so one can give an even
                 distribution of weights, and more or less keep similar
                 widths across filter types.
        l       : width and height of image
        sigma_noise : noise sigma
        freq    : array
            frequencies of each signal type. [gauss,ring,dendrite]
            sum(frequencies) should be 1, gauss,ring,dendrite give relative
            presence of each type. If sum is not 1, will be renormalized.
    '''
    # get number of each type of signal
    rlen = len(R)  # number of neurons
    if np.sum(freq) != 1:
        freq /= np.sum(freq)
    numbers = (freq * rlen).astype(int)

    # setup output
    # this is where we'll store the video
    out = zeros((l, l, len(calcium[0, :])))
    real = zeros((l, l))  # this is where we'll store the image pixels
    rois = []

    # start counter
    count = 0

    # add gaussian signals
    for i in range(numbers[0]):
        mean = X[i, 0], X[i, 1]
        var = R[i]
        cov = [[var, 0], [0, var]]
        filt = makeGaussianFilter(mean[0], mean[1], cov, l, False)
        filt[filt > 0.5] += 0.1
        for t in range(len(calcium[0, :])):
            out[:, :, t] += calcium[i, t] * filt
        real += filt
        count += 1
        rois += [filt > 0.5]

    # add ring signals
    for i in range(count, count + numbers[1]):
        mean = X[count, 0], X[count, 1]
        var = R[count]
        cov = [[var / 2, 0], [0, var / 2]]
        filt = makeGaussianFilter(mean[0], mean[1], cov, l, True)
        filt[filt > 0.65] += 0.1
        for t in range(len(calcium[0, :])):
            out[:, :, t] += calcium[count, t] * filt
        real += filt
        count += 1
        rois += [filt > 0.65]

    # add dendrite signals
    for i in range(count, rlen):
        length = np.random.randint(250, 500)
        sigma = R[count] / 100
        path, filt = makeDendriteFilter(
            X[count, 0], X[count, 1], length, sigma, l, rand() * np.pi * 2, 0.05)
        filt[filt > 0.5] += 0.1

        for t in range(len(calcium[0, :])):
            out[:, :, t] += calcium[i, t] * filt
        real += filt
        count += 1
        rois += [filt > 0.5]

    # make background spatial filter (random mixture of gaussians)
    x, y = randint(0, l, 2), randint(0, l, 2)

    bgfilt = np.zeros((l, l))
    for i in range(len(x)):
        cov = [[randint(60 * l, 80 * l), 0], [0, randint(60 * l, 80 * l)]]
        bgfilt += makeGaussianFilter(x[i], y[i], cov, l)
    # normalize filter
    bgfilt /= bgfilt.max()
    bgfilt += 1
    # bgfilt /= bgfilt.max()

    # add to global picture
    for x in range(l):
        for y in range(l):
            out[x, y, :] += bg * bgfilt[x, y]
    if out.min() < 0:
        out -= out.min()
    out = poisson(lam=out)
    return out, real, rois

# normalize function


def normalize(s, norm=None):
    ''' Removes minimum, then divides by new maximum'''

    out = s - np.median(s)
    if norm is None:
        return out / out.max()
    out /= norm
    return out
