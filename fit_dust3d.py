#!/usr/bin/env python

from __future__ import print_function, division

import os

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib import ticker
from matplotlib.colors import CenteredNorm

import tensorflow as tf
if os.environ.get('RUN_FUNCTIONS_EAGERLY', False):
    print('Running all functions in eager mode.')
    tf.config.run_functions_eagerly(True)

import tensorflow.keras as keras
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import sonnet as snt

import itertools
from tqdm import tqdm
import json
from argparse import ArgumentParser


#
# Tensorflow setup
#

# Disable TensorFloat32, which is a reduced-precision float format
tf.config.experimental.enable_tensor_float_32_execution(False)
# Only allocate GPU memory as needed
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        print(f'Setting memory growth on {device}.')
        tf.config.experimental.set_memory_growth(device, True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('Failed to set memory growth!')
    pass

if os.getenv('VIRTUAL_GPUS'):
    n_vgpus = int(os.getenv('VIRTUAL_GPUS'))
    mem_per_vgpu = 1024 * 5
    for device in physical_devices:
        tf.config.set_logical_device_configuration(
            device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=mem_per_vgpu)
             for n in range(n_vgpus)]
        )
    print('logical devices:')
    print(tf.config.list_logical_devices('GPU'))


def batch_apply_tf(f, batch_size, *args,
                   function=False, progress=False, numpy=False):
    """
    Applies a tensorflow function f(*args), but internally batches the args,
    and then concatenates the results. The internal batching is usually
    meant to avoid running out of memory.

    Inputs:
      f (function): The function to apply.
      batch_size (int): Batch size.
      *args (array-like): The arguments to pass to `f`. Each argument will be
             batched along its zeroeth axis.
      function (Optional[bool]): If `True`, the function `f` will be
                                 wrapped using tf.function. Defaults to
                                 `False`.
      progress (Optional[bool]): If `True`, a progress bar will be shown.
                                 Defaults to `False`.
      numpy (Optional[bool]): If `True`, outputs will be converted to numpy
                              arrays.

    Outputs:
      res: Should be the same as the output of f(*args). The batching should
           have no impact on the final result.
    """
    res = []
    def f_batch(*x):
        return f(*x)
    if function:
        f_batch = tf.function(f_batch)
    iterator = range(0, len(args[0]), batch_size)
    if progress:
        iterator = tqdm(iterator)
    for i in iterator:
        batch = [a[i:i+batch_size] for a in args]
        res_batch = f_batch(*batch)
        if numpy:
            res_batch = res_batch.numpy()
        res.append(res_batch)
    if numpy:
        res = np.concatenate(res, axis=0)
    else:
        res = tf.concat(res, axis=0)
    return res


class MultiComponentModel(snt.Module):
    def __init__(self, *models):
        super(MultiComponentModel, self).__init__()
        self.models = models

    def __call__(self, x):
        ln_rho = 0.
        for m in self.models:
            ln_rho += m(x)
        return ln_rho

    def prior(self, x):
        ln_prior = 0.
        for m in self.models:
            ln_prior += m(x)
        return ln_prior


class DiskModel(snt.Module):
    def __init__(self, rho_0, h_z, h_R, x0):
        super(DiskModel, self).__init__()
        self.ln_rho0 = tf.Variable(
            np.log(rho_0),
            name='ln_rho0',
            dtype=tf.float32
        )
        self.ln_hz = tf.Variable(
            np.log(h_z),
            name='ln_hz',
            dtype=tf.float32
        )
        self.ln_hR = tf.Variable(
            np.log(h_R),
            name='ln_hR',
            dtype=tf.float32
        )
        self.x0 = tf.Variable(
            np.reshape(x0, (1,3)),
            name='x0',
            dtype=tf.float32,
            trainable=False
        )

    def __call__(self, x):
        x,y,z = tf.split(x-self.x0, 3, axis=1)
        R = tf.math.sqrt(x**2 + y**2)
        inv_rho_z = tf.math.cosh(z/tf.math.exp(ln_hz))
        inv_rho_R = tf.math.cosh(R/tf.math.exp(ln_hR))
        return self.ln_rho0 - tf.math.log(inv_rho_z) - tf.mathm.log(inv_rho_R)

    def prior(self):
        return 0.


class FourierSeriesND(snt.Module):
    def __init__(self, n_dim, max_order,
                 extent=1,
                 power_law_slope=0.,
                 mean=0., sigma=1.,
                 scale_init_sigma=1.,
                 k_ball=True,
                 phase_form=False,
                 seed=None):
        super(FourierSeriesND, self).__init__()
        self.n_dim = n_dim
        if hasattr(max_order, '__len__'):
            self.max_order = max_order
        else:
            self.max_order = [max_order]*n_dim
        if hasattr(extent, '__len__'):
            self.extent = extent
        else:
            self.extent = [extent]*n_dim
        self._k_ball = k_ball
        self._phase_form = phase_form
        self.power_law_slope = power_law_slope
        self._sigma = sigma
        self._initialize(mean=mean, scale_init_sigma=scale_init_sigma, seed=seed)

    def _get_k_matrix(self, region, flatten=False):
        k_axes = []
        for i,s in enumerate(region):
            scale = np.pi / self.extent[i]
            if s == -1:
                k_axes.append(-scale*np.arange(1,self.max_order[i]+1))
            elif s == 0:
                k_axes.append([0])
            elif s == 1:
                k_axes.append(scale*np.arange(1,self.max_order[i]+1))
            else:
                raise ValueError(f'Invalid region description: {region}')

        k = np.stack(np.meshgrid(*k_axes), axis=0)

        if flatten:
            k.shape = (self.n_dim,-1)

        return k

    @snt.once
    def _initialize(self, mean=0.0, scale_init_sigma=1.0, seed=None):
        rng = np.random.default_rng(seed)

        # Enumerate the regions of k-space that, combined with their mirror
        # (k -> -k), make up all of k-space, except for k = 0.
        regions = []
        mirrored = [(0,)*self.n_dim]
        for key in itertools.product(*([(-1,0,1)]*self.n_dim)):
            if key not in mirrored:
                regions.append(key)
                mirrored.append(tuple([-s for s in key]))

        # Construct k-tensor, containing all the non-degenerate modes
        k = np.concatenate(
            [self._get_k_matrix(key, flatten=True) for key in regions],
            axis=1
        )
        k2 = np.sum(k**2, axis=0)

        # Limit modes to spherical region in k-space?
        if self._k_ball:
            k_max = np.min(np.pi*np.array(self.max_order)/np.array(self.extent))
            idx = (k2 <= k_max**2 + 1e-5)
            k = k[:,idx]
            k2 = k2[idx]

        print(f'{k2.size} modes.')

        # Calculate normalization of sigma_k, based on formula:
        #   sigma^2 = sigma_1^2 \sum_k k^{powerlawslope}
        k_power = k2**(0.5*self.power_law_slope)
        sum_k_power = np.sum(k_power)
        prior_norm = sum_k_power / self._sigma**2
        sigma_k = self._sigma * np.sqrt(k_power / sum_k_power)

        # Generate coefficients and phases
        self.zp = tf.Variable(
            mean + scale_init_sigma*self._sigma*rng.normal(),
            name='zp',
            dtype=tf.float32
        )
        self.a = tf.Variable(
            scale_init_sigma*rng.normal(size=sigma_k.shape),
            name='a',
            dtype=tf.float32
        )
        if self._phase_form:
            self.phi = tf.Variable(
                rng.uniform(0, 2*np.pi, size=sigma_k.shape),
                name='phi',
                dtype=tf.float32,
                #constraint=lambda x: tf.math.mod(x, 2*np.pi)
            )
        else:
            self.b = tf.Variable(
                scale_init_sigma*rng.normal(size=sigma_k.shape),
                name='b',
                dtype=tf.float32
            )

        # Save constants
        self.k = tf.constant(k, name='k', dtype=tf.float32)
        self.k2 = tf.constant(k2, name='k2', dtype=tf.float32)
        self.sigma_k = tf.constant(sigma_k, name='sigma_k', dtype=tf.float32)

        # Power-law slope
        self.power_law_slope = tf.Variable(
            self.power_law_slope,
            name='power_law_slope',
            dtype=tf.float32,
            trainable=False
        )
        self.prior_norm = tf.Variable(
            prior_norm,
            name='prior_norm',
            dtype=tf.float32,
            trainable=False
        )

    def _recalc_sigma_k(self):
        k_power = self.k2**(0.5*self.power_law_slope)
        sum_k_power = tf.math.reduce_sum(k_power)
        sigma_k_new = self._sigma * np.sqrt(k_power / sum_k_power)
        self.sigma_k = tf.constant(
            sigma_k_new,
            name='sigma_k',
            dtype=tf.float32
        )

    def set_power_law_slope(self, power_law_slope):
        self.power_law_slope.assign(power_law_slope)

        # Update sigma_k, and rescale Fourier coefficients
        k_power = self.k2**(0.5*self.power_law_slope)
        sum_k_power = tf.math.reduce_sum(k_power)
        sigma_k_new = self._sigma * np.sqrt(k_power / sum_k_power)
        self.a.assign(self.a*self.sigma_k/sigma_k_new)
        self.b.assign(self.b*self.sigma_k/sigma_k_new)
        self.sigma_k = tf.constant(sigma_k_new, name='sigma_k', dtype=tf.float32)

        # Update prior normalization
        self.prior_norm.assign(sum_k_power / self._sigma**2)

    def copy_modes(self, model, transition_width=None):
        assert model._phase_form == self._phase_form
        assert self._k_ball and model._k_ball

        # Remove scaling from coefficients
        a_self = self.a.numpy() * self.sigma_k.numpy()
        b_self = self.b.numpy() * self.sigma_k.numpy()

        a_model = model.a.numpy() * model.sigma_k.numpy()
        b_model = model.b.numpy() * model.sigma_k.numpy()

        # Sort modes by k^2 in order to match
        k_model = model.k.numpy()
        k2_model = model.k2.numpy()
        n_model = k2_model.size

        k = self.k.numpy()
        k2 = self.k2.numpy()
        n = k2.size

        idx_model = np.argsort(k2_model, kind='stable') # Stability is critical!
        idx_self = np.argsort(k2, kind='stable')

        # Ensure that k-vectors of k^2-sorted modes match
        np.testing.assert_allclose(
            k[:,idx_self[:n_model]],
            k_model[:,idx_model[:n]],
            atol=1e-8, rtol=1e-8
        )

        # Determine how much to weight self and model amplitudes
        if transition_width is None:
            # Sharp transition between model and self
            w_model = np.ones(n_model, dtype='f4')
            w_self = np.zeros(n, dtype='f4')
            w_self[n_model:] = 1.
        else:
            if self._phase_form:
                # If in phase form, must first convert to (a,b) representation
                # before applying transition region weighting, and then
                # covert back afterwards.
                raise NotImplementedError(
                    '<transition_width> not implemented when phase_form==True'
                )

            # Smoothly transition over a width of Delta k = <transition_width>
            scale = np.exp(np.mean(np.log(np.pi/np.array(self.extent))))
            dk = scale * transition_width
            k1 = np.sqrt(np.max(k2_model))
            k0 = np.sqrt(np.max(k2_model)) - dk

            w_model = (k1-np.sqrt(k2_model[idx_model])) / (k1-k0)
            w_model = np.clip(w_model, 0., 1.)

            w_self = np.ones(n, dtype='f4')
            w_self[:n_model] = 1. - w_model

        # Copy over the amplitudes and phases
        k = k[:,idx_self]
        self.k = tf.constant(k, dtype=tf.float32, name='k')

        k2 = k2[idx_self]
        self.k2 = tf.constant(k2, dtype=tf.float32, name='k2')

        self._recalc_sigma_k()

        self.zp = tf.Variable(model.zp.numpy(), dtype=tf.float32, name='zp')

        a = w_self * a_self[idx_self]
        a[:n_model] += w_model * a_model[idx_model[:n]]
        a /= self.sigma_k.numpy()
        self.a = tf.Variable(a, dtype=tf.float32, name='a')

        if self._phase_form:
            phi = self.phi.numpy()
            phi = phi[idx_self]
            phi[:n_model] = model.phi.numpy()[idx_model[:n]]
            self.phi = tf.Variable(phi, dtype=tf.float32, name='phi')
        else:
            b = w_self * b_self[idx_self]
            b[:n_model] += w_model * b_model[idx_model[:n]]
            b /= self.sigma_k.numpy()
            self.b = tf.Variable(b, dtype=tf.float32, name='b')


    def __call__(self, x):
        """
        Evaluates the harmonic expansion at the points x.

        Inputs:
          x (tf.Tensor): Input coordinates. Has
            shape (# of points, # of dimensions).

        Outputs:
          f(x), the value of the harmonic expansion at the points x.
          Has shape (# of points, 1).
        """
        # z_{jk} = x_{ji} k_{ik} ( + phi_{jk} ),
        # where
        #   j := point,
        #   i := dimension,
        #   k := mode
        # y_{j} = [a_{k} cos(z_{jk}) ( + b_{k} sin(z_{jk}) )] sigma_{k}
        z = tf.tensordot(x, self.k, axes=[[1],[0]])
        if self._phase_form:
            z = z + self.phi

        res = self.zp + tf.tensordot(
            self.a * self.sigma_k, tf.math.cos(z),
            axes=[[0],[1]]
        )
        if not self._phase_form:
            res = res + tf.tensordot(
                self.b * self.sigma_k, tf.math.sin(z),
                axes=[[0],[1]]
            )

        res = tf.expand_dims(res, 1)
        return res

    def prior(self):
        # Penalties on amplitudes
        # (unit normal, because amplitudes are rescaled by sigma_k)
        if self._phase_form:
            p = tf.reduce_sum(self.a**2)
        else:
            p = tf.reduce_sum(self.a**2 + self.b**2)
        # No penalty on (0,0) term (the zero point)
        return p


def get_ray_ode(log_rho_fn, x_star):
    ds_dt = tf.norm(x_star, axis=1, keepdims=True, name='ds_dt')
    def ode(t, A):
        r"""
        t = fractional distance along ray
        A = \int \exp(\ln \rho) ds
        """
        #A,x = tf.split(y, [1,2], axis=1)
        #dy_dt = tf.concat([ds_dt*tf.math.exp(log_rho_fn(x)), dx_dt], 1)
        dA_dt = ds_dt * tf.math.exp(log_rho_fn(t*x_star))
        return dA_dt
    return ode


def plot_modes(model, title_extra=None):
    gamma = -float(model.power_law_slope.numpy())

    fig,ax = plt.subplots(
        1,1,
        subplot_kw=dict(aspect='equal'),
        figsize=(7.3,6),
        dpi=100
    )

    k = model.k.numpy()
    k2 = model.k2.numpy()
    a = model.a.numpy()

    n = a.size
    v = a
    vmax = np.max(np.abs(v))

    idx = np.argmin(np.pi*np.array(model.max_order)/np.array(model.extent))
    max_order = model.max_order[idx]
    kw = dict(
        edgecolors='none',
        s=(320/(2*max_order+1))**2,
        marker='s',
        vmin=-vmax,
        vmax=vmax,
        cmap='coolwarm'
    )

    im = ax.scatter(k[0], k[1], c=v, **kw)

    if not model._phase_form:
        b = model.b.numpy()
        v = k2**(0.5*gamma) * b
        ax.scatter(-k[0], -k[1], c=v, **kw)

    #cb = fig.colorbar(im, ax=ax, location='right', label=r'$a_{\vec{k}}$')
    fig.colorbar(im, label=fr'$k^{{{gamma}}} a_{{\vec{{k}}}}$')

    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')

    title = 'Fourier space'
    if title_extra is not None:
        title += f' ({title_extra})'
    ax.set_title(title)

    return fig


def plot_modes_imshow(model, title_extra=None, clip_order=None):
    gamma = -float(model.power_law_slope.numpy())

    fig,ax = plt.subplots(
        1,1,
        subplot_kw=dict(aspect='equal'),
        figsize=(7.3,6),
        dpi=100
    )

    k = model.k.numpy()
    k2 = model.k2.numpy()
    a = model.a.numpy()

    # Convert k vectors into integer (x,y,z) indices
    scale = np.array(model.extent)[:,None] / np.pi
    k_x,k_y,k_z = np.round(k*scale).astype('i8')

    # Select k_z == 0 modes
    idx = (k_z == 0)

    k_max = np.array(model.max_order)
    x_idx = k_x[idx]+k_max[0]
    y_idx = k_y[idx]+k_max[1]

    # Create image of Fourier modes
    img_shape = (2*k_max[0]+1, 2*k_max[1]+1)
    img = np.full(img_shape, np.nan, dtype=a.dtype)

    v = a[idx]
    img[x_idx,y_idx] = v
    img[img_shape[0]-x_idx-1,img_shape[1]-y_idx-1] = v

    x_max = k_max[0] + 0.5
    y_max = k_max[1] + 0.5

    # Clip image?
    if clip_order is not None:
        x0 = max(k_max[0] - clip_order[0], 0)
        y0 = max(k_max[1] - clip_order[1], 0)
        x1 = img.shape[0] - x0
        y1 = img.shape[1] - y0

        img = img[x0:x1,y0:y1]

        x_max -= x0
        y_max -= y0

    im = ax.imshow(
        img.T,
        origin='lower',
        extent=(-x_max,x_max,-y_max,y_max),
        interpolation='nearest',
        norm=CenteredNorm(),
        cmap='coolwarm_r',
        rasterized=True
    )

    fig.colorbar(im, label=fr'$k^{{{0.5*gamma}}} a_{{\vec{{k}}}}$')

    ax.set_xlabel(r'$\pi k_x / L_x$')
    ax.set_ylabel(r'$\pi k_y / L_y$')

    title = 'Fourier space'
    if title_extra is not None:
        title += f' ({title_extra})'
    ax.set_title(title)

    return fig


def plot_power(model, title_extra=None):
    k2 = model.k2.numpy()
    a2 = model.a.numpy()**2
    if not model._phase_form:
        a2 += model.b.numpy()**2
        a2 *= 0.5

    n_bins = max(10, int(np.sqrt(k2.size)/4))
    #ln_k2_min = np.log(np.min(k2))
    #ln_k2_max = np.log(np.max(k2))
    #k2_bins = np.exp(np.linspace(ln_k2_min, ln_k2_max, n_bins))
    k2_bins = np.linspace(np.min(k2), np.max(k2), n_bins)

    bin_idx = np.digitize(k2, k2_bins)
    power_bin = np.zeros(k2_bins.size)

    for j,i in enumerate(np.unique(bin_idx)):
        idx = (bin_idx == i)
        power_bin[j] = np.mean(a2[idx])

    k_mid = (k2_bins[1:]*k2_bins[:-1])**(1/4)
    #k_mid = 0.5 * (k2_bins[1:] + k2_bins[:-1])

    fig,ax = plt.subplots(1,1, figsize=(6,6), dpi=100)

    gamma = -float(model.power_law_slope.numpy())
    ax.semilogy(k_mid, power_bin[:-1], label='model')
    ax.axhline(
        1.,
        ls=':', alpha=0.5,
        label=fr'$P_k \propto k^{{-{gamma}}}$'
    )

    ax.set_xlabel(r'$k$')
    coeff_label = r'\left| a \right|^2'
    prefix = ''
    if not model._phase_form:
        coeff_label += r' + \left| b \right|^2'
        prefix = r'\frac{1}{2}\,'
    ax.set_ylabel(fr'${prefix} k^{{ {gamma} }} \left< {coeff_label} \right>$')

    title = 'Power spectrum'
    if title_extra is not None:
        title += f' ({title_extra})'
    ax.set_title(title)

    ax.legend()

    fig.subplots_adjust(left=0.16, right=0.94, bottom=0.12, top=0.90)

    return fig


def plot_A_residual_hist(log_rho, x_star, A_obs, A_err):
    A_pred = calc_A(log_rho, x_star, rtol=1e-5, atol=1e-5)
    dA = A_pred - A_obs
    chi = dA / A_err

    fig,(ax_d,ax_chi) = plt.subplots(1,2, figsize=(6,3), layout='constrained')

    d_max = 1.3 * np.percentile(np.abs(dA), 99.9)
    ax_d.hist(dA, bins=100, range=(-d_max,d_max), log=True)
    ax_d.set_xlabel(r'$\Delta A \ \left(\mathrm{pred - obs}\right)$')

    chi_max = 1.3 * np.percentile(np.abs(chi), 99.9)
    _,bins,_ = ax_chi.hist(chi, bins=100, range=(-chi_max,chi_max), log=True)
    ax_chi.set_xlabel(r'$\Delta A/\sigma_A \ \left(\mathrm{pred - obs}\right)$')

    ylim = ax_chi.get_ylim()

    x = np.linspace(bins[0], bins[-1], 1000)
    dx = bins[1] - bins[0]
    p_x = np.exp(-0.5*x**2) / np.sqrt(2*np.pi) * dx * len(x_star)
    ax_chi.plot(x, p_x, alpha=0.3, label=r'$\mathrm{ideal}$')
    ax_chi.legend(loc='upper right')

    ax_chi.set_ylim(ylim)

    for ax in (ax_d,ax_chi):
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which='major', alpha=0.2)
        ax.grid(True, which='minor', alpha=0.05)

    return fig


def plot_lnrho_A(img, x_star, A_star, extent,
                 star_extent=None,
                 title='',
                 diff_img=False, exp_img=False,
                 diff_stars=False,
                 z=0., dz=0.1):
    fig,ax_arr = plt.subplots(
        2,2,
        figsize=(6,6),
        dpi=100,
        gridspec_kw=dict(width_ratios=[1,0.1], height_ratios=[1,0.1])
    )

    ax,cax_im,cax_sc,ax_empty = ax_arr.flat
    ax_empty.axis('off')

    xlim, ylim = extent

    # Plot dust density
    idx = [slice(int(0.1*s), int(0.9*s)) for s in img.shape]
    if diff_img:
        vmax = 1.15 * np.percentile(np.abs(img[idx[0],idx[1]]), 99)
        kw = dict(cmap='coolwarm', vmin=-vmax, vmax=vmax)
        if exp_img:
            label = r'$\Delta \rho$'
        else:
            label = r'$\Delta \ln \rho$'
    else:
        if exp_img:
            vmin = 0.
            vmax = np.max(img[idx[0],idx[1]])
            label = r'$\rho$'
        else:
            vmin,vmax = np.percentile(img[idx[0],idx[1]], [1., 99.])
            w = vmax - vmin
            vmin -= 0.15 * w
            vmax += 0.15 * w
            label = r'$\ln \rho$'
        kw = dict(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        img.T,
        origin='lower',
        interpolation='nearest',
        extent=xlim+ylim,
        **kw
    )
    cb_im = fig.colorbar(im, cax=cax_im, orientation='vertical', label=label)

    # Plot stars
    n_stars = len(A_star)

    if n_stars:
        # If 3D, then only plot stars in a thin plane in z
        n_dim = x_star.shape[1]
        if n_dim == 3:
            idx = np.abs(x_star[:,2] - z) < dz
            x_star = x_star[idx]
            A_star = A_star[idx]

        if diff_stars:
            vmax = 1.2 * np.percentile(np.abs(A_star), 98)
            kw = dict(vmin=-vmax, vmax=vmax, edgecolors='w', linewidths=0.5)
            label = r'$\Delta A$'
        else:
            kw = dict(edgecolors='none')
            label = r'$A$'

        sc = ax.scatter(
            x_star[:,0],
            x_star[:,1],
            c=A_star,
            s=36 / max(np.sqrt(n_stars/1024), 1),
            cmap='coolwarm',
            **kw
        )
        cb_sc = fig.colorbar(sc, cax=cax_sc, orientation='horizontal', label=label)
    else:
        cax_sc.axis('off')

    # Plot box in which stars reside
    if star_extent is not None:
        xlim_s, ylim_s = star_extent
        ax.plot(
            [xlim_s[1], xlim_s[1], xlim_s[0], xlim_s[0], xlim_s[1]],
            [ylim_s[1], ylim_s[0], ylim_s[0], ylim_s[1], ylim_s[1]],
            ls=':',
            c='k',
            alpha=0.5
        )

    #ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax.set_title(title)

    fig.subplots_adjust(
        left=0.10,
        right=0.88,
        bottom=0.10,
        top=0.92,
        wspace=0.04,
        hspace=0.12
    )

    return fig


def train(log_rho_fit, dataset,
          n_stars, batch_size, n_epochs,
          lr0=1e-3, lr1=1e-6, n_lr_drops=9,
          log_w0=-2, log_w1=-3,
          use_dist_err=False,
          chi2_outlier=12.25,
          rtol=1e-7,
          atol=1e-5,
          checkpoint_every=1,
          checkpoint_hours=1,
          max_checkpoints=16,
          checkpoint_dir=r'checkpoints',
          checkpoint_name='log_rho',
          callback=None):
    # Get current distribution strategy (for working with multiple GPUs)
    strategy = tf.distribute.get_strategy()

    # Get the number of devices (such as GPUs) to be used
    n_devices = strategy.num_replicas_in_sync
    global_batch_size = batch_size * n_devices

    # Break the dataset up into batches
    dataset = dataset.shuffle(
                      buffer_size=64*global_batch_size
                  ).batch(
                      global_batch_size, drop_remainder=True
                  ).repeat(n_epochs)
    # Adapt dataset for given distribution strategy (across multiple GPUs)
    dataset = strategy.experimental_distribute_dataset(dataset)

    # Calculate the number of steps from the given
    # dataset size and requested # of epochs
    n_steps = (n_stars // global_batch_size) * n_epochs

    print(  'Batching & step calculation:')
    print(fr'  * devices: {n_devices}')
    print(fr'  * stars: {n_stars}')
    print(fr'  * epochs: {n_epochs}')
    print(fr'  * local batch size: {batch_size}')
    print(fr'  -> {n_steps} steps (= (n_stars//global_batch_size)*n_epochs)')

    # Smoothly increase the weight given to the prior during training
    def get_prior_weight(step):
        log_w = log_w0 + step/n_steps * (log_w1-log_w0) # base 10
        return tf.constant(10**log_w)

    prior_weight = tf.Variable(get_prior_weight(0), name='prior_weight')

    # Optimizer, with staircase-exponential learning rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        lr0,
        decay_steps=int(n_steps/(n_lr_drops+1)),
        decay_rate=(lr1/lr0)**(1/n_lr_drops),
        staircase=True
    )
    #opt = keras.optimizers.SGD(
    #    learning_rate=lr_schedule,
    #    momentum=0.5,
    #    global_clipnorm=100. # Guard-rails to prevent fitter from going haywire
    #)
    opt = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        global_clipnorm=100. # Guard-rails to prevent fitter from going haywire
    )

    # Function that takes one gradient-descent step
    #@tf.function
    def grad_step(*args):
        # Get the loss function, with a given integrator tolerance
        loss_fn = get_loss_function(
            rtol=rtol, atol=atol,
            use_dist_err=use_dist_err,
            chi2_outlier=tf.constant(chi2_outlier)
        )

        if use_dist_err:
            A_obs, A_err, d_err, x_star = args
            # Integrate out to max dist of d_obs + n*d_err for each star
            d_obs = tf.norm(x_star, axis=1, name='d_obs')
            dx_ds = x_star / tf.expand_dims(d_obs, axis=1)
            ds_dt = tf.expand_dims(d_obs+5*d_err, axis=1, name='ds_dt')
            inputs = (A_obs, A_err, d_obs, d_err, dx_ds, ds_dt, log_rho_fit)
        else:
            A_obs, A_err, x_star = args
            # Integrate out to distance of each star
            ds_dt = tf.norm(x_star, axis=1, keepdims=True, name='ds_dt')
            inputs = (A_obs, A_err, x_star, ds_dt, log_rho_fit)

        # Calculate loss
        #tf.print('loss_fn')
        with tf.GradientTape() as g:
            loss, likelihood, prior, diagnostics = loss_fn(
                *inputs, prior_weight=prior_weight
            )

        # Calculate and apply gradients of loss w.r.t. training variables
        variables = log_rho_fit.trainable_variables
        #tf.print('Calculating gradients')
        grads = g.gradient(loss, variables)
        #tf.print('Applying gradients')
        opt.apply_gradients(zip(grads, variables))

        # Return some useful diagnostics
        #tf.print('Calculating norm')
        norm = tf.linalg.global_norm(grads)
        #tf.print('Calculating n_eval')
        n_eval = diagnostics.num_ode_fn_evaluations
        #tf.print('global norm:', norm)
        #tf.print('# of evaluations:', n_eval)

        return loss, likelihood, prior, norm, n_eval

    # Adapt training step for distribution strategy
    @tf.function
    def distributed_grad_step(*args):
        print('Tracing <distributed_grad_step()> ...')

        per_replica_results = strategy.run(
            grad_step,
            args=args
        )
        return strategy.reduce(
            tf.distribute.ReduceOp.MEAN,
            per_replica_results,
            axis=None
        )

    # Keep track of history of loss, chi^2 and prior during training
    history = {
        'loss': [],
        'likelihood': [],
        'prior': [],
        'norm': [],
        'n_eval': []
    }

    # Step counter (needed for checkpointing)
    step = tf.Variable(0, name='step')
    checkpoint_steps = int(np.ceil(checkpoint_every*n_stars/global_batch_size))

    # Checkpointer
    checkpoint = tf.train.Checkpoint(log_rho=log_rho_fit, opt=opt, step=step)
    chkpt_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        max_to_keep=max_checkpoints,
        keep_checkpoint_every_n_hours=checkpoint_hours
    )

    # Look for latest existing checkpoint
    latest = chkpt_manager.latest_checkpoint
    if latest is not None:
        print(f'Restoring from checkpoint {latest} ...')
        checkpoint.restore(latest).assert_existing_objects_matched(
                                   ).expect_partial()
        print(f'Resuming from step {int(step.numpy())} (of {n_steps}).')

        # Try to load stats history
        history_fname = f'{latest}_stats.json'
        with open(history_fname, 'r') as f:
            history = json.load(f)

    physical_devices = tf.config.list_physical_devices('GPU')
    print('physical_devices:', physical_devices)
    if physical_devices:
        for device in physical_devices:
            device_name = device.name.removeprefix('/physical_device:')
            tf.config.experimental.reset_memory_stats(device_name)

    # Take gradient-descent steps on training batches
    step_iter = tqdm(
        enumerate(dataset, int(step.numpy())),
        initial=int(step.numpy()),
        total=n_steps
    )
    for i,data_batch in step_iter:
        if i >= n_steps:
            break

        # Update the weight and slope of the power-spectrum prior
        prior_weight.assign(get_prior_weight(i))

        # Take a single gradient step
        loss, likelihood, prior, norm, n_eval = distributed_grad_step(
            *data_batch
        )

        history['loss'].append(float(loss.numpy()))
        history['likelihood'].append(float(likelihood.numpy()))
        history['prior'].append(float(prior.numpy()))
        history['norm'].append(float(norm.numpy()))
        history['n_eval'].append(int(n_eval.numpy()))

        # Display diagnostics on progress bar
        progress_info = {
            'loss': float(loss.numpy()),
            'prior': float(prior.numpy()),
            #'lr': float(opt._decayed_lr(tf.float32)),
            'norm': float(norm.numpy()),
            'n_eval': int(n_eval.numpy()),
            #'gamma': float(gamma.numpy()),
            #'weight': float(prior_weight.numpy()),
            'zp': float(log_rho_fit.zp.numpy())
        }
        if physical_devices:
            mem_info = tf.config.experimental.get_memory_info(device_name)
            progress_info['mem_peak'] = f'{mem_info["peak"]/1024**3:.1f}GB'
        step_iter.set_postfix(progress_info)

        # Checkpoint
        if checkpoint_every is not None:
            if (i and not (i % checkpoint_steps)) or (i >= n_steps-1):
                print('Checkpointing ...')
                step.assign(i+1)
                chkpt_fname = chkpt_manager.save()
                print(f'  --> {chkpt_fname}')

                history_fname = f'{chkpt_fname}_stats.json'
                with open(history_fname, 'w') as f:
                    json.dump(history, f)

        # Call the given callback function, which may, for example, plot
        # the current ln(rho) field
        if callback is not None:
            callback(i)

    return history


def gen_mock_data(n_modes, star_extent, box_extent, n_stars,
                  mu_lnrho=-1.0, sigma_lnrho=1.0, gamma=3.6,
                  sigma_A=0, sigma_r=0,
                  batch_size=1024, seed=None):
    ## Calculate box extent
    #box_extent = [(1+box_buffer)*w for w in star_extent]
    n_dim = len(n_modes)

    # Generate log(rho) field
    log_rho = FourierSeriesND(
        n_dim, n_modes,
        extent=box_extent,
        power_law_slope=-gamma,
        sigma=sigma_lnrho,
        seed=seed
    )
    log_rho.zp.assign(mu_lnrho)

    # Draw stellar positions
    if seed is not None:
        seed = 2 * seed + 1 # Use different seed for locations of stars
    rng = np.random.default_rng(seed)

    n_dim = len(star_extent)
    x_star = rng.uniform(-1, 1, size=(n_stars,n_dim)).astype('f4')
    for k,w in enumerate(star_extent):
        x_star[:,k] *= w

    # Calculate stellar extinctions
    A = calc_A(log_rho, x_star, batch_size=batch_size, atol=1e-5, rtol=1e-5)

    # Add noise into the stellar extinctions.
    # First, choose an uncertainty for each star:
    df = 9.0
    A_err = sigma_A/df * rng.chisquare(df,size=A.shape).astype('f4')
    A_err = np.sqrt(A_err**2 + (0.05*sigma_A)**2)
    # Draw Gaussian noise for each star
    A_obs = A + A_err * rng.normal(size=A.shape).astype('f4')

    # Add noise into the stellar distances.
    # First, choose a fractional distance uncertainty for each star:
    df = 9.0
    d_err = sigma_r/df * rng.chisquare(df,size=A.shape).astype('f4')
    d_err = np.sqrt(d_err**2 + (0.05*sigma_r)**2)
    # Draw Gaussian noise for each star
    dr = d_err * rng.normal(size=A.shape).astype('f4')
    # Observed position of star
    x_obs = x_star * (1+dr)[:,None]
    # Scale fractional uncertainty to absolute uncertainty
    d_star = np.linalg.norm(x_star, axis=1)
    d_err *= d_star

    # Return a dictionary of results
    res = dict(
        log_rho=log_rho,
        star_extent=star_extent,
        box_extent=box_extent,
        x_star=x_star,
        x_obs=x_obs,
        d_err=d_err,
        A=A,
        A_obs=A_obs,
        A_err=A_err
    )
    return res


def calc_image(log_rho, n_dim, extent, z=[0.], batch_size=1024):
    xlim, ylim = extent
    x_grid = np.linspace(xlim[0], xlim[1], 400, dtype='f4')
    y_grid = np.linspace(ylim[0], ylim[1], 400, dtype='f4')
    if n_dim == 2:
        x_grid,y_grid = np.meshgrid(y_grid, y_grid, indexing='ij')
        coord_grid = np.stack([x_grid, y_grid], axis=2)
    elif n_dim == 3:
        z_grid = np.array(z).astype('f4')
        x_grid,y_grid,z_grid = np.meshgrid(
            y_grid, y_grid, z_grid,
            indexing='ij'
        )
        coord_grid = np.stack([x_grid, y_grid, z_grid], axis=3)
    shape = x_grid.shape
    coord_grid.shape = (-1, n_dim)

    @tf.function
    def calc_rho_batch(x):
        print('Tracing calc_rho_batch ...')
        return log_rho(x)

    img = np.empty(coord_grid.shape[0], dtype='f4')
    for i in tqdm(range(0,coord_grid.shape[0],batch_size)):
        img[i:i+batch_size] = calc_rho_batch(
            coord_grid[i:i+batch_size]
        ).numpy().flat

    img.shape = shape
    if n_dim == 3:
        img = np.log(np.mean(np.exp(img), axis=2))

    return img, [xlim, ylim]


def calc_A(log_rho, x_star, batch_size=1024, rtol=1e-8, atol=1e-7):
    n_stars = x_star.shape[0]
    A = np.empty(n_stars, dtype='f4')

    @tf.function
    def calc_batch(x):
        print('Tracing calc_batch ...')
        n = x.shape[0]
        ode_fn = get_ray_ode(log_rho, x)
        solver = tfp.math.ode.DormandPrince(
            rtol=rtol, atol=atol, name='ray_integrator'
        )
        res = solver.solve(
            ode_fn,
            0, tf.zeros([n,1]),
            tf.constant([1])
        )
        #A,_ = tf.split(res.states, [1,2], axis=2)
        return tf.squeeze(res.states)

    for i in tqdm(range(0,n_stars,batch_size)):
        A[i:i+batch_size] = calc_batch(x_star[i:i+batch_size]).numpy()

    return A


def plot_sky(log_rho, dist, extent, A_reference=None,
             nside=64, batch_size=1024, title=''):
    from astropy_healpix import HEALPix
    from skyplot_utils import plot_healpix_map

    #@tf.function
    #def calc_A_batch(x):
    #    return calc_A(log_rho, x)

    # Create grid of points to plot
    hpx = HEALPix(nside=nside, order='nested')
    lon,lat = hpx.healpix_to_lonlat(np.arange(hpx.npix))
    lon = lon.to('rad').value
    lat = lat.to('rad').value
    x_sky = np.empty((lon.size, 3), dtype='f4')
    x_sky[:,0] = np.cos(lon) * np.cos(lat)
    x_sky[:,1] = np.sin(lon) * np.cos(lat)
    x_sky[:,2] = np.sin(lat)
    x_sky *= dist
    #The following line works in the dev version of astropy_healpix
    #x_sky = dist * np.stack(hpx.healpix_to_xyz(np.arange(hpx.npix)), axis=1)

    # Limit grid-point distances to fall within given extent
    if not hasattr(extent, '__len__'):
        extent = [extent]*3
    for k,w in enumerate(extent):
        x_sky *= np.clip(extent[k]/np.abs(x_sky[:,k]), 0., 1.)[:,None]

    # Calculate extinctions at sky locations (at given distance)
    A_sky = np.empty(hpx.npix, dtype='f4')

    print('Calculating extinction over sky ...')
    A_sky = calc_A(log_rho, x_sky)
    #for i in tqdm(range(0,hpx.npix,batch_size)):
    #    A_sky[i:i+batch_size] = calc_A_batch(x_sky[i:i+batch_size]).numpy()

    # Plot extinction over sky
    def make_fig(A, kw, label):
        fig = plt.figure(figsize=(6,3.1))
        fig.suptitle(title)
        ax,im = plot_healpix_map(fig, A, imshow_kwargs=kw)
        fig.colorbar(im, ax=ax, label=label)
        fig.subplots_adjust(left=0.05, right=0.98)
        return fig

    figs = make_fig(A_sky, {}, rf'$A \left(r={dist:.1f}\right)$')

    if A_reference is not None:
        kw = dict(
            cmap='coolwarm',
            norm=CenteredNorm()
        )
        label = rf'$\Delta A \left(r={dist:.1f}\right)$'
        figs = [figs, make_fig(A_sky-A_reference, kw, label)]

    return figs, A_sky


def plot_loss(history):
    n_steps = len(history['loss'])

    fig,(ax_u,ax_l) = plt.subplots(2,1, figsize=(6,6))

    ax_u.plot(history['loss'], label='loss')
    ax_u.plot(history['likelihood'], alpha=0.5, label='likelihood')

    ax_u.plot([], [], alpha=0.5, label='prior') # dummy plot, for legend
    ax_u.set_ylabel(r'$\mathrm{loss}$, $\mathrm{likelihood}$')

    ax2 = ax_u.twinx()
    for i in range(2):
        ax2.plot([],[]) # dummy plots, to cycle colors on ax2
    ax2.plot(history['prior'], alpha=0.5)
    ax2.set_ylabel(r'$\mathrm{prior}$')

    ax_l.plot(history['norm'])
    ax_l.set_ylabel(r'$\left|\nabla\left(\mathrm{loss}\right)\right|$')

    if 'n_eval' in history:
        ax2 = ax_l.twinx()
        ax2.plot( # dummy
            [], [],
            label=r'$\left|\nabla\left(\mathrm{loss}\right)\right|$'
        )
        ax2.plot(history['n_eval'], alpha=0.7, label=r'$\mathrm{\#\ evaluations}$')
        ax2.set_ylabel(r'$\mathrm{\#\ evaluations}$')

    ax_u.legend(loc='upper right')
    ax2.legend(loc='center right') # lower plot

    ax_l.set_xlabel(r'$\mathrm{training\ step}$')
    ax_u.set_title(r'$\mathrm{training\ history}$')

    ax_u.set_xticklabels([])

    ax_u.grid('on', axis='x', alpha=0.1)
    ax_l.grid('on', axis='x', alpha=0.1)

    for ax in (ax_u,ax_l):
        ax.grid('on', axis='x', alpha=0.1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(n_steps//7))

    fig.subplots_adjust(
        top=0.94,
        bottom=0.10,
        left=0.14,
        right=0.86,
        hspace=0.05
    )

    return fig


def get_loss_function(use_dist_err=False, chi2_outlier=tf.constant(12.25),
                      rtol=1e-7, atol=1e-5):
    """
    Returns a function that calculates the loss of
    a model of ln(rho), given a set of stellar observations.

    Inputs:
      use_dist_err (Optional[bool]): Take distance uncertainties into account
        (default: False).
      chi2_outlier(Optional[float]): Suppress effect of chi-square values
        beyond this value (default: 25).
      rtol (Optional[float]): Relative tolerance of the integrator
        (default: 1e-7).
      atol (Optional[float]): Absolute tolerance of the integrator
        (default: 1e-5).

    Returns:
      `calc_loss`, the loss function.
    """
    # ODE solver
    solver = tfp.math.ode.DormandPrince(
        rtol=rtol, atol=atol, name='ray_integrator',
        #max_num_steps=10000
    )

    if use_dist_err:
        def calc_loss(A_obs, A_err, d_obs, d_err,
                      dx_ds, ds_dt, log_rho_model,
                      prior_weight=tf.constant(1e-3),
                      lnL_outlier=tf.constant(-25.)):
            def ode(t, AL, dx_ds, ds_dt, d_obs, d_err, A_obs, A_err):
                r"""
                Calculates d(A,L)/dt along the ray.
                  t = fractional distance along ray
                  AL = (A, L) = (extinction, likelihood) = ODE state
                  dx_ds = unit vector pointing to star
                  ds_dt = path length per unit time (t) = distance to star
                """
                #print('')
                #print('============================')
                #print('t =', t)
                #print('AL =', AL)
                #print('dx/ds =', dx_ds)
                #print('ds/dt =', ds_dt)
                #print('d_obs =', d_obs)
                #print('d_err =', d_err)
                #print('A_obs =', A_obs)
                #print('A_err =', A_err)

                A_t,L_t = tf.split(AL, [1,1], axis=1)
                A_t = tf.squeeze(A_t, axis=1)
                #print('A(t) =', A_t)
                L_t = tf.squeeze(L_t, axis=1)
                #print('L(t) =', L_t)

                # Current position and distance
                d = t * ds_dt
                x = dx_ds * d
                #print('x =', x)
                d = tf.squeeze(d, axis=1)
                #print('d =', d)

                # dA/ds = rho(x(s))
                dA_ds = tf.math.exp(log_rho_model(x))
                dA_ds = tf.squeeze(dA_ds, axis=1)
                #print('dA/ds =', dA_ds)

                # dL/ds = N(A|A_obs,A_err) * N(d|d_obs,d_err)
                chi2 = (
                    ((d_obs-d)/d_err)**2
                  + ((A_obs-A_t)/A_err)**2
                )
                chi2 = (
                    chi2_outlier * tf.math.asinh(chi2/chi2_outlier)
                )
                dL_ds = tf.math.exp(-0.5*chi2)

                #dL_ds = tf.math.exp(-0.5 * (
                #    ((d_obs-d)/d_err)**2
                #  + ((A_obs-A_t)/A_err)**2
                #))
                #print('dL/ds =', dL_ds)
                
                # d(A,L)/dt = ds/dt * d(A,L)/ds
                dAL_dt = ds_dt * tf.stack([dA_ds,dL_ds], axis=1)
                #print('d(AL)/dt =', dAL_dt)

                #if np.any(~np.isfinite(t.numpy())):
                #    return None

                return dAL_dt

            # Initial state: (A,L) = (0,0)
            #AL0 = tf.zeros((A_obs.shape[0], 2))
            L_outlier = tf.math.exp(lnL_outlier)
            A0 = tf.zeros([A_obs.shape[0]])
            L0 = tf.ones([A_obs.shape[0]]) * L_outlier
            #AL0 = tf.zeros_like(A_obs)
            AL0 = tf.stack([A0,L0], axis=1)
            #print('AL0 =', AL0)

            # Solve ODE
            res = solver.solve(
                ode,
                0, AL0,
                tf.constant([1]),
                constants={
                    'dx_ds':dx_ds, 'ds_dt':ds_dt,
                    'd_obs':d_obs, 'd_err':d_err,
                    'A_obs':A_obs, 'A_err':A_err
                }
            )

            # Extract (A,L) as distance -> infinity
            AL_final = tf.squeeze(res.states, axis=0)
            #print('AL(t=1) =', AL_final)
            A_final,L_final = tf.split(AL_final, [1,1], axis=1)
            A_final = tf.squeeze(A_final, axis=1)
            L_final = tf.squeeze(L_final, axis=1)

            # Require L >= 0 (numerical integration of strictly positive
            # functions can yield slightly negative values).
            L_final = tf.clip_by_value(L_final, L_outlier, np.inf)
            #print('A(t=1) =', A_final)
            #print('L(t=1) =', L_final)

            # Calculate loss from L and prior
            lnL = tf.math.log(L_final) - tf.math.log(A_err*d_err)
            #idx = np.isnan(lnL.numpy())
            #if np.any(idx):
            #    print('NaN lnL!')
            #    print('  lnL =', lnL.numpy()[idx])
            #    print('  L_final =', L_final.numpy()[idx])
            #    print('  A_err =', A_err.numpy()[idx])
            #    print('  d_err =', d_err.numpy()[idx])
            #n_above_min = np.count_nonzero(lnL.numpy()>lnL_outlier.numpy())

            # Add in floor on L (using stable addition method, similar to
            # tf.math.reduce_logsumexp)
            lnL = tfp.math.log_add_exp(lnL, lnL_outlier)
            #lnL0 = tf.minimum(lnL, lnL_outlier)
            #lnL1 = tf.maximum(lnL, lnL_outlier)
            #lnL = lnL1 + tf.math.log(1 + tf.math.exp(lnL0-lnL1))
            #lnL = tf.clip_by_value(lnL, lnL_outlier, np.inf)
            #if np.any(idx):
            #    print('  clipped lnL =', lnL.numpy()[idx])
            #print(f'{n_above_min} stars above minimum likelihood.')
            #print('ln(L) =', lnL)
            #idx = ~np.isfinite(lnL.numpy())
            #if np.any(idx):
            #    print('Non-finite lnL!')
            #    print('  lnL =', lnL.numpy()[idx])
            likelihood = -2 * tf.reduce_mean(lnL)
            #print('likelihood =', likelihood)
            prior = log_rho_model.prior()
            loss = likelihood + prior_weight * prior

            return loss, likelihood, prior, res.diagnostics
    else:
        def calc_loss(A_obs, A_err, x_star, ds_dt, log_rho_model,
                      prior_weight=tf.constant(1e-3)):
            def ode(t, A, dx_dt, ds_dt):
                r"""
                t = fractional distance along ray
                A = \int \exp(\ln \rho) ds = extinction
                dx_dt = change in position per unit time (t) = position of star
                ds_dt = path length per unit time (t) = distance to star
                """
                # dA/dt = dA/ds * ds/dt = rho(x(s)) ds/dt
                dA_dt = ds_dt * tf.math.exp(log_rho_model(t*dx_dt))
                return dA_dt

            initial_state = tf.expand_dims(tf.zeros_like(A_obs), 1)
            res = solver.solve(
                ode,
                0, initial_state,
                tf.constant([1]),
                constants={'dx_dt':x_star, 'ds_dt':ds_dt}
            )
            A_model = tf.squeeze(res.states)
            #log_chi2 = tf.math.log(tf.reduce_mean(((A_obs - A_model)/A_err)**2))
            chi2 = ((A_obs - A_model) / A_err)**2
            # Soften chi^2, so that when chi^2 >~ chi2_outlier, the loss no longer
            # grows quadratically with chi.
            chi2 = tf.reduce_mean(
                chi2_outlier * tf.math.asinh(chi2/chi2_outlier)
            )
            prior = log_rho_model.prior()
            loss = tf.math.log(chi2 + prior_weight * prior)
            return loss, tf.math.log(chi2), prior, res.diagnostics

    return calc_loss


def plot_dust_and_stars(
            log_rho, x_star, A_obs, A_err,
            box_extent, star_extent,
            log_rho_true=None,
            ln_rho_img_true=None,
            ln_rho_intz_img_true=None,
            fig_dir='plots/', fn_suffix='',
            title='model'
        ):
    n_dim = len(box_extent)

    # Plotting settings
    n_stars_plot = 1024*8
    xlim = [-box_extent[0], box_extent[0]]
    ylim = [-box_extent[1], box_extent[1]]
    xlim_s = [-star_extent[0], star_extent[0]]
    ylim_s = [-star_extent[1], star_extent[1]]

    # Power spectrum
    fig = plot_power(log_rho, title_extra=title)
    fig.savefig(
        os.path.join(fig_dir, f'power_spectrum{fn_suffix}')
    )
    plt.close(fig)

    fig = plot_modes_imshow(log_rho, title_extra=title)
    fig.savefig(
        os.path.join(fig_dir, f'fourier_modes{fn_suffix}')
    )
    plt.close(fig)

    if log_rho_true is not None:
        fig = plot_modes_imshow(
            log_rho_true,
            title_extra='mock clipped',
            clip_order=np.array(log_rho.max_order)
        )
        fig.savefig(
            os.path.join(fig_dir, f'fourier_modes_mockclipped{fn_suffix}')
        )
        plt.close(fig)

    # A residual histograms
    n_hist_max = 1024*64
    fig = plot_A_residual_hist(
        log_rho,
        x_star[:n_hist_max],
        A_obs[:n_hist_max],
        A_err[:n_hist_max]
    )
    fig.savefig(
        os.path.join(fig_dir, f'A_residuals{fn_suffix}')
    )
    plt.close(fig)

    # rho
    ln_rho_img,_ = calc_image(
        log_rho,
        n_dim,
        (xlim,ylim),
        z=[0.]
    )
    rho_img = np.exp(ln_rho_img)
    fig = plot_lnrho_A(
        rho_img, [], [],
        (xlim,ylim), star_extent=(xlim_s,ylim_s),
        exp_img=True,
        title=title
    )
    fig.savefig(
        os.path.join(fig_dir, f'rho{fn_suffix}')
    )
    plt.close(fig)

    # \int \rho dz
    z_slices = np.linspace(-star_extent[2], star_extent[2], 201)
    ln_rho_intz_img,_ = calc_image(
        log_rho,
        n_dim,
        (xlim,ylim),
        z=z_slices
    )
    rho_intz_img = np.exp(ln_rho_intz_img)
    fig = plot_lnrho_A(
        rho_intz_img, [], [],
        (xlim,ylim), star_extent=(xlim_s,ylim_s),
        exp_img=True,
        title=f'{title} (integrated in z)'
    )
    fig.savefig(
        os.path.join(fig_dir, f'rho_intz{fn_suffix}')
    )
    plt.close(fig)

    # Sky projection
    for k,d in enumerate(np.arange(0.1, 1.01, 0.1)):
        fig,_ = plot_sky(
            log_rho,
            d, # kpc
            star_extent
        )
        fig.savefig(
            os.path.join(fig_dir, f'A_sky_d{k:02d}{fn_suffix}')
        )
        plt.close(fig)

    # rho and predicted stellar A
    x_plot = x_star[:n_stars_plot]
    A_est_plot = calc_A(log_rho, x_plot)
    A_obs_plot = A_obs[:n_stars_plot]
    fig = plot_lnrho_A(
        rho_img, x_plot, A_est_plot,
        (xlim,ylim), star_extent=(xlim_s,ylim_s),
        exp_img=True,
        title=title
    )
    fig.savefig(
        os.path.join(fig_dir, f'rho_stars{fn_suffix}')
    )
    plt.close(fig)

    # rho and predicted-observed stellar A
    fig = plot_lnrho_A(
        rho_img, x_plot, A_est_plot-A_obs_plot,
        (xlim,ylim), star_extent=(xlim_s,ylim_s),
        diff_stars=True, exp_img=True,
        title=f'{title} - observed'
    )
    fig.savefig(
        os.path.join(fig_dir, f'rho_dstars{fn_suffix}')
    )
    plt.close(fig)

    if log_rho_true is not None:
        # rho and predicted-true stellar A
        A_true_plot = calc_A(log_rho_true, x_plot)
        fig = plot_lnrho_A(
            rho_img, x_plot, A_est_plot-A_true_plot,
            (xlim,ylim), star_extent=(xlim_s,ylim_s),
            diff_stars=True, exp_img=True,
            title=f'{title} - true'
        )
        fig.savefig(
            os.path.join(fig_dir, f'rho_dstars_true{fn_suffix}')
        )
        plt.close(fig)

        # rho
        if ln_rho_img_true is None:
            ln_rho_img_true,_ = calc_image(
                log_rho_true,
                n_dim,
                (xlim,ylim),
                z=[0.]
            )
        drho_img = rho_img - np.exp(ln_rho_img_true)
        fig = plot_lnrho_A(
            drho_img, [], [],
            (xlim,ylim), star_extent=(xlim_s,ylim_s),
            diff_img=True, exp_img=True,
            title=f'{title} - true'
        )
        fig.savefig(
            os.path.join(fig_dir, f'drho_true{fn_suffix}')
        )
        plt.close(fig)

        # \int \rho dz
        if ln_rho_intz_img_true is None:
            ln_rho_intz_img_true,_ = calc_image(
                log_rho_true,
                n_dim,
                (xlim,ylim),
                z=z_slices
            )
        drho_intz_img = rho_intz_img - np.exp(ln_rho_intz_img_true)
        fig = plot_lnrho_A(
            drho_intz_img, [], [],
            (xlim,ylim), star_extent=(xlim_s,ylim_s),
            diff_img=True, exp_img=True,
            title=f'{title} - true (integrated in z)'
        )
        fig.savefig(
            os.path.join(fig_dir, f'drho_intz_true{fn_suffix}')
        )
        plt.close(fig)

    return ln_rho_img, ln_rho_intz_img


def main():
    parser = ArgumentParser(
        add_help=True,
        description='Fit 3D dust map in Fourier space.'
    )
    input_parser = parser.add_mutually_exclusive_group(required=True)
    input_parser.add_argument(
        '-i', '--input',
        type=str,
        metavar='INPUT.(h5|fits)',
        help=(
            'Astropy table with stellar positions (xyz) '
            'and extinctions (E, sigma_E).'
        )
    )
    input_parser.add_argument(
        '--mock',
        type=int,
        metavar='SEED',
        help='Generate mock data using given (int) seed.'
    )
    parser.add_argument(
        '-dir', '--directory',
        type=str,
        required=True,
        metavar='path/to/output',
        help='Directory in which to store output.'
    )
    parser.add_argument(
        '-opt', '--options',
        type=str,
        required=True,
        metavar='OPTIONS.json',
        help='Training settings, in JSON format (see example).'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=('default','mirrored','multiworkermirrored'),
        default='default',
        help='Which TF distribution strategy to use (for multiple GPUs)'
    )
    parser.add_argument(
        '--use-dist-err',
        action='store_true',
        help='Take distance uncertainties into account.'
    )
    parser.add_argument(
        '--omit-plots',
        action='store_true',
        help='Do not plot results (for quick tests).'
    )
    args = parser.parse_args()

    # Ensure that various required directories exist
    run_dir = args.directory
    fig_dir = os.path.join(run_dir, 'plots')
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    if not os.path.isdir(fig_dir):
        os.mkdir(fig_dir)

    tf.random.set_seed(1)

    # Read options file
    with open(args.options, 'r') as f:
        options = json.load(f)

    extent = options.pop('extent')
    star_extent = extent.pop('stars')
    box_extent = extent.pop('box')
    n_dim = len(star_extent)

    # Define distribution strategy (for working with multiple GPUs).
    # This must be done before any other TensorFlow operations are performed.
    strategy = {
        'default': tf.distribute.get_strategy(),
        'mirrored': tf.distribute.MirroredStrategy(),
        'multiworkermirrored': tf.distribute.MultiWorkerMirroredStrategy()
    }[args.strategy]
    print(f'Using {args.strategy} distribution strategy.')

    # Plotting settings
    n_stars_plot = 1024*8
    xlim = [-box_extent[0], box_extent[0]]
    ylim = [-box_extent[1], box_extent[1]]
    xlim_s = [-star_extent[0], star_extent[0]]
    ylim_s = [-star_extent[1], star_extent[1]]

    # Check if mock data requested
    if args.mock is not None:
        print(f'Generating mock data with seed {args.mock} ...')
        mock_options = options.pop('mock')
        mock_modes = mock_options.pop('n_modes')
        n_stars = mock_options.pop('n_stars')
        sigma_A_mock = mock_options.get('sigma_A', 0.05)
        sigma_r_mock = mock_options.get('sigma_r', 0.05)

        res = gen_mock_data(
            mock_modes, star_extent, box_extent, n_stars,
            mu_lnrho=mock_options.get('mu_lnrho',-1.0),
            sigma_lnrho=mock_options.get('sigma_lnrho',1.0),
            gamma=mock_options.get('gamma', 3.6),
            sigma_A=sigma_A_mock,
            sigma_r=sigma_r_mock,
            seed=args.mock
        )

        A_obs = res['A_obs']
        A_err = res['A_err']
        x_star = res['x_star']
        if args.use_dist_err:
            x_obs = res['x_obs']
            d_err = res['d_err']
        else:
            x_obs = x_star
        log_rho_true = res['log_rho']
        A_true = res['A']

        if not args.omit_plots:
            print('Plotting mock data ...')
            ln_rho_img_true, ln_rho_intz_img_true = plot_dust_and_stars(
                log_rho_true, x_star, A_obs, A_err,
                box_extent, star_extent,
                fig_dir=fig_dir, fn_suffix='_mock',
                title='true'
            )
    else:
        # Load input data
        data_fn = args.input
        print(f'Loading {data_fn} ...')

        from astropy.table import Table
        t = Table.read(data_fn)

        A_obs = t['E'].data.astype('f4')
        A_err = t['sigma_E'].data.astype('f4')
        x_star = t['xyz'].data.astype('f4')
        if args.use_dist_err:
            d_err = t['sigma_r'].data.astype('f4')
        n_stars = len(A_obs)
        print(f'Loaded {n_stars} sources.')

        # Shuffle stars
        idx = np.arange(n_stars)
        rng = np.random.default_rng(3)
        rng.shuffle(idx)
        A_obs = A_obs[idx]
        A_err = A_err[idx]
        x_star = x_star[idx]

        log_rho_true = None
        ln_rho_img_true = None
        ln_rho_intz_img_true = None

    # Take distance uncertainties into account?
    if args.use_dist_err:
        dataset = tf.data.Dataset.from_tensor_slices((A_obs,A_err,d_err,x_star))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((A_obs,A_err,x_star))

    log_rho_old = None
    options = options['training_rounds']

    for train_round,opts in enumerate(options):
        n_modes = opts.pop('n_modes')
        print(f'Training round {train_round}:')
        print(f'# of modes: {n_modes}')
        print('Options:')
        print(json.dumps(opts, indent=2))

        # All variables must be initialized within the strategy scope
        with strategy.scope():
            # Initialize model
            log_rho_fit = FourierSeriesND(
                n_dim, n_modes,
                extent=box_extent,
                power_law_slope=-opts.pop('gamma',3.5),
                sigma=opts.pop('sigma',1.0),
                scale_init_sigma=0.1,
                phase_form=False,
                seed=17*(train_round+1)
            )

            n_trainable = sum([
                tf.size(v) for v in log_rho_fit.trainable_variables
            ])
            print(f'{n_trainable} trainable variables.')

            # Copy in modes of lower-resolution model, if available
            if train_round > 0:
                log_rho_fit.copy_modes(log_rho_old, transition_width=4)
            log_rho_old = log_rho_fit

            # Ensure that checkpoint directory exists
            checkpoint_dir = os.path.join(
                run_dir,
                f'checkpoints_{train_round}'
            )
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)

            # Optimize model
            print('Training model ...')
            batch_size = opts.pop('batch_size')
            n_epochs = opts.pop('n_epochs')
            history = train(
                log_rho_fit, dataset,
                n_stars, batch_size, n_epochs,
                use_dist_err=args.use_dist_err,
                checkpoint_name='log_rho',
                checkpoint_dir=checkpoint_dir,
                **opts
            )

        # Plot results
        print('Plotting results ...')

        fig = plot_loss(history)
        fig.savefig(os.path.join(fig_dir, f'loss_history_{train_round}'))
        plt.close(fig)

        if not args.omit_plots:
            plot_dust_and_stars(
                log_rho_fit, x_star, A_obs, A_err,
                box_extent, star_extent,
                log_rho_true=log_rho_true,
                ln_rho_img_true=ln_rho_img_true,
                ln_rho_intz_img_true=ln_rho_intz_img_true,
                fig_dir=fig_dir, fn_suffix=f'_{train_round}',
                title='predicted'
            )

    return 0


if __name__ == '__main__':
    main()

