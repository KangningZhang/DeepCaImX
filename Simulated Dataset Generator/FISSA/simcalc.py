"""This class will simulate a calcium dataset.

Author: s w keemink
"""
from __future__ import division
import numpy as np
import simcalclib as sclib
import tifffile
from PIL import Image, ImageDraw
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import cv2

def first_nonzero(arr, axis, invalid_val=-1):
        mask = arr!=0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    
    
class SimCalc():
    """Simlates a calcium imaging dataset given parameters.

    Parameters
    ----------
    stimprop : float
        proportion of cells that are tuned to the stimulus
    h : int
        Height and width of image in pixels
    N : int
        Number of neurons
    T : float
        Stimulus time. Total time will be StimCycles*2*T
    StimCycles : int, optional (default: 4)
        How many stimulus cycles within T
    dt : float, optional (default: 0.01)
        Simulation time
    """

    def __init__(self, h, N, T, StimCycles=4, dt=0.01):
        """Initialisation."""
        self.h = h
        self.N = N
        self.T = T
        self.StimCycles = StimCycles
        self.dt = dt

        
    
    
    def gen_calcium(self, Tau_r, Tau_d, A, p, rates=None, sp=None):
        """Generation of the spikes.

        Spikes are generated according to Poisson statistics.

        Calcium is first modelled with a rise and fall time, then passed
        through a polynomial to model nonlinearities:
        c_d' = -c_d/Tau_d + s(t)
        c_r' = -c_r/Tau_r + s(t)
        c = c_d + c_r

        F(c) = 1 + A[c+p2(c^2-c)) + p3(c^3-c)]

        Parameters
        ----------
        Tau_r, Tau_d : floats
            Rise and decay times of calcium
        A : float/array
            Spike response magnitude.
            If Array, should be of shape (N,1), to give every neuron a
            different value.
        p : array/list
            Calcium dynamics polynomial variables (p = [p2,p3])
        rates : array, optional (default: None)
            If None, the firing rates follow a normal distribution with
            a given mean and sigma. If given, should be an array giving the
            firing rates for every neuron.
        sp : array, optional (default: None)
            If None, will make spikes according to parameters. Otherwise,
            will use provided spikes to make calcium trace.
        """
        N = self.N
        N_neg = np.int8(np.floor(N/40))
        dt = self.dt
        if sp is None:
            # get basic variables
            T = self.T
            Cycles = self.StimCycles
            nFrames = int(T / dt) * 2 * Cycles
            times = np.arange(0, T * 2 * Cycles, dt)

            # set up firing rates
            if rates is None:
                rates = np.random.normal(0.5, 1, N)
                rates[rates < 0] = 0

            # Model Poisson firing
            sp = np.zeros((N, nFrames))
            for i in range(N):
                while np.sum(sp[i, :]) == 0:
                    sp[i, :] = sclib.spike_data_poisson(T, dt, [rates[i], rates[i] * 2] * Cycles, trials=1)
        else:
            nFrames = sp.shape[1]

        # Simulate Calcium dynamics
        c = np.zeros((N, nFrames))  # calcium traces
        F = np.zeros(c.shape)
        c_d = np.zeros(N)  # calcium decay dynamics
        c_r = np.zeros(N)  # calcium rise dynamics
        for i, t in enumerate(times):
            # iterate calcium levels
            c_d += dt * (sp[:, i] / dt - c_d / Tau_d)
            c_r += dt * (sp[:, i] / dt - c_r / Tau_r)
            c[:, i] = c_d - c_r

        # calculate basic dye levels
        # F = A*c

        maxc = (-2 * p[0] - np.sqrt(4 * p[0]**2 +
                                    12 * p[1] * (sum(p) - 1))) / (6 * p[1])
        maxf = (A * (maxc + p[0] * (maxc**2 - maxc) + p[1] * (maxc**3 - maxc)))
        F = A * (c + p[0] * (c**2 - c) + p[1] * (c**3 - c))
        if np.sum(c > maxc) > 0:
            F[c > maxc] = maxf
        
        order = np.argsort(first_nonzero(sp, axis=1))
        order_neg = np.random.permutation(N)
        order_neg = order_neg[0:N_neg]
        self.sp = sp[order, :]
        self.c = c[order, :]
        self.F = F[order, :]
        self.sp[order_neg,:] = np.ones((N_neg, nFrames))
        self.c[order_neg,:] = np.ones((N_neg, nFrames))
        self.F[order_neg,:] = np.ones((N_neg, nFrames))

    def gen_spat_kernels(self, locs=None, sizes=None, covs=None,
                         use_rings=None):
        """Generate the spatial kernels.

        Parameters
        ----------
        locs : array (number of neurons by 2), optional
            Locations of every neuron. If None, just randomizes.
        sizes : array, optional
            Sizes of every neuron. If None, just randomizes.
        covs : array, optional
            The covariance between width and height
            of each neuron. If None, sets all to 0.
        use_rings : array of bools, optional
            Whether to use Ring neurons or not. All True by default.

        """
        M = loadmat('Neuron_Library.mat')
        M = M['segs'] # Neuron Library loaded
        
        # get basic variables
        N = self.N
        h = self.h
        index = np.random.permutation(M.shape[2])
        spat_kernels = np.zeros((N,h,h))
        masks = np.zeros((N,h,h))

        # get locations and sizes
        if locs is None:
            locs = np.random.randint(0, h, (N, 2))
        if sizes is None:
            sizes = np.random.normal(h/3, np.sqrt(h/2), size=N)
        if covs is None:
            covs = np.zeros(N)
        if use_rings is None or use_rings:
            use_rings = np.ones(N, dtype=bool)

        # generate shapes
        for i in range(N):
                     
            #cov = np.array([[sizes[i], covs[i]], [covs[i], sizes[i]]])
            # make spatial kernel
            #spat_kernels[i] = sclib.makeGaussianFilter(
            #    locs[i, 0], locs[i, 1], cov, h)   ###, ring=use_rings[i]
            location = np.zeros([h,h])
            
            location[locs[i,0],locs[i,1]] = 1
            mask = M[:,:,index[i]]
            if np.random.uniform(0,1) > 0.7:
                spat_kernel = cv2.GaussianBlur(mask,(9,9),cv2.BORDER_DEFAULT)
            else:
                spat_kernel_dx = np.absolute(cv2.Sobel(mask, cv2.CV_64F, dx=1, dy=0, ksize=7))
                spat_kernel_dx = spat_kernel_dx / np.max(spat_kernel_dx)
                spat_kernel_dy = np.absolute(cv2.Sobel(mask, cv2.CV_64F, dx=0, dy=1, ksize=7))
                spat_kernel_dy = spat_kernel_dy / np.max(spat_kernel_dy)
                spat_kernel = (cv2.GaussianBlur(0.45*spat_kernel_dx+0.45*spat_kernel_dy+0.3*mask, (5,5), cv2.BORDER_DEFAULT))
            
            spat_kernels[i,:,:] = signal.convolve2d(location, spat_kernel, mode='same')

            # add offset to original kernel
            masks[i,:,:] = signal.convolve2d(location, mask, mode='same')
            #spat_kernels[i,:,:] += masks[i,:,:] / 5

        self.spat_kernels = spat_kernels
        self.masks = masks

    def gen_npil(self, eta=0.05, F0=1, nFilts=5):
        """Generation of neuropil, both its mask and responses.

        Temporal component:
        B' = eta*dW

        Spatial component:
        Sum of 2d-Gaussians with random locations and widths.
        The spatial component is normalized so the mean is 1.

        Parameters
        ----------
        eta : float
            Magnitude and timescale of brownian motion.
        F0 : float
            The drifting baseline
        nFilts : int
            Number of background filters to use for spatial component.
            Set to 0 to have flat background.
        """
        # get basic variables
        h = self.h
        T = self.T
        dt = self.dt
        Cycles = self.StimCycles
        nFrames = int(T / dt) * 2 * Cycles

        # run brownian motion for temporal component
        drive = np.concatenate([np.ones(int(T / dt)),
                                np.ones(int(T / dt)) * 1.05] * Cycles) * F0
        Bg = drive + eta * \
            np.cumsum(np.random.normal(size=nFrames)) * np.sqrt(dt)
        Bg[Bg < 0] = 0

        # start spatial component
        Bg_img = np.ones((1, h, h))

        if nFilts > 0:
            # random gaussian locations
            x, y = (np.random.randint(0, h, nFilts),
                    np.random.randint(0, h, nFilts))
            # loop over locations
            for i in range(len(x)):
                cov = [[np.random.normal(h * 15, np.sqrt(h * 15)), 0],
                       [0, np.random.normal(h * 15, np.sqrt(h * 15))]]

                mask_temp = sclib.makeGaussianFilter(x[i], y[i], cov, h)
                Bg_img[0, :, :] += mask_temp

            # normalize so mean is 1
            Bg_img /= Bg_img.mean()

        # make video
        video = np.repeat(Bg_img, nFrames, axis=0) * Bg[:, None, None]

        self.Bg_trace = Bg
        self.Bg_img = Bg_img[0, :, :]
        self.npil = video

    def gen_data(self, p=None, ampl=1):
        """Generation of final data, combining spikes, masks and neuropil.

        Run gen_npil, gen_spat_kernels, and gen_calcium first.

        Parameters
        ----------
        p : array/list
            Calcium dynamics polynomial variables (p = [p2,p3])
            not used at the moment
        ampl : float, optional
            Multiplication of output
        """
        # get basic variables
        N = self.N
        h = self.h
        T = self.T
        dt = self.dt
        Cycles = self.StimCycles
        nFrames = int(T / dt) * 2 * Cycles
        F = self.F
        kernels = self.spat_kernels

        # start cell video
        data = np.zeros((nFrames, h, h))
        #print('Of ' + str(N) + ' cells:')
        for cell in range(N):
            #print(cell)
            cell_img = kernels[cell].reshape((1, h, h))
            data += np.repeat(cell_img, nFrames, axis=0) * \
                F[cell, :, None, None]
            # TODO: speed up above two lines
        
        data[data < 0] = 0
        # data -= 1
        # data = (self.npil)*(1+data)-1
        # data[data<0]=0
        # data = data*ampl

        # optionally apply nonlinearity
        # data = 1 + (data + p[0]*(data**2 - data) + p[1]*(data**3 - data))

        # measure video through poisson
        self.data = data + self.npil
        video = np.random.poisson(data + self.npil)        
        self.video = video

    def gen_masks(self, threshold=0.5):
        """Generate the masks for each ROI.

        Parameters
        ----------
        threshold : float, optional
            Which threshold to use for mask generation.

        Returns
        -------
        list
            list of binary arrays representing the masks for each ROI
        """
        # get basic variables
        N = self.N
        masks = [0] * N

        # turn cell kernels into masks
        for i in range(N):
            # make ROI mask
            masks[i] = self.spat_kernels[i] > threshold

        return masks

    def save_tiff(self, fname='data.tif', dtype=np.float32):
        """Saving the generated data as a tiff file."""
        # shpe = self.video.shape
        # video = np.zeros((shpe[2], self.h, self.h), dtype=dtype)
        # for i in range(shpe[2]):
        #     video[i, :, :] = self.video.astype(dtype)[:, :, i]
        self.video = self.video/17*255
        video = self.video.astype(dtype)
        tifffile.imsave(fname, video, imagej=True)
    
    def save_gif(self, fname='data.tif', dtype=np.float32):
        """Saving the generated data as a gif file."""
        # shpe = self.video.shape
        # video = np.zeros((shpe[2], self.h, self.h), dtype=dtype)
        # for i in range(shpe[2]):
        #     video[i, :, :] = self.video.astype(dtype)[:, :, i]
        video = self.video.astype(dtype)
        video[0].save(fname, save_all=True, append_images=video[1:], optimize=False, duration=1000/20, loop=0)
