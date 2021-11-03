#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib import ticker

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import sonnet as snt

from tqdm import tqdm
import os


class HarmonicExpansion2D(snt.Module):
    def __init__(self, max_order, extent=[1,1], k_slope=0):
        super(HarmonicExpansion2D, self).__init__()
        self.max_order = max_order
        self.extent = extent
        self._initialize(k_slope)

    @snt.once
    def _initialize(self, k_slope):
        # k-vector
        N = 2 * self.max_order + 1
        shape = [N,N]
        k = np.pi * (np.indices(shape) - self.max_order)
        for i in range(2):
            k[i] /= self.extent[i]
        k2 = np.sum(k**2, axis=0)
        log_k = 0.5 * np.log(k2)
        log_k[self.max_order,self.max_order] = 0
        self.k = tf.constant(k.astype('f4'), name='k')
        self.k2 = tf.constant(k2.astype('f4'), name='k2')
        self.log_k = tf.constant(log_k.astype('f4'), name='log_k')

        # Mask defining which coefficients are for cos and sin
        a_mask = np.zeros(shape, dtype='f4')
        for i in range(self.max_order+1):
            a_mask[i,:N-i] = 1.0
        b_mask = 1 - a_mask
        self.a_mask = tf.constant(a_mask)
        self.b_mask = tf.constant(b_mask)

        # Initialize coefficients
        sigma = (k2+1)**(-0.5*(k_slope-1))
        sigma /= np.sqrt(np.sum(sigma**2))
        ab = sigma * np.random.normal(size=shape)
        # sigma = 1 / self.max_order
        self.ab = tf.Variable(ab.astype('f4'), name='ab')

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
        # z_{jnm} = x_{ji} k_{inm}
        z = tf.tensordot(x, self.k, axes=[[1],[0]])
        a = self.ab * self.a_mask
        b = self.ab * self.b_mask
        res = (
            tf.tensordot(a, tf.math.cos(z), axes=[[0,1],[1,2]])
          + tf.tensordot(b, tf.math.sin(z), axes=[[0,1],[1,2]])
        )
        res = tf.expand_dims(res, 1)
        return res
    
    def prior(self, slope):
        return tf.reduce_sum(tf.math.pow(self.k2,0.5*slope) * self.ab**2)


def get_ray_ode(log_rho_fn, dx_dt):
    ds_dt = tf.expand_dims(
        tf.norm(dx_dt, axis=1),
        1,
        name='ds_dt'
    )
    def ode(t, y):
        """
        t = fractional distance along ray
        y = (\int \exp(\rho) ds, *x)
        """
        A,x = tf.split(y, [1,2], axis=1)
        dy_dt = tf.concat([ds_dt*tf.math.exp(log_rho_fn(x)), dx_dt], 1)
        return dy_dt
    return ode


def plot_lnrho_A(lnrho_img, x_star, A_star, title='', diff=False):
    fig,ax_arr = plt.subplots(
        2,2,
        figsize=(6,6),
        dpi=100,
        gridspec_kw=dict(width_ratios=[1,0.1], height_ratios=[1,0.1])
    )

    ax,cax_im,cax_sc,ax_empty = ax_arr.flat
    ax_empty.axis('off')

    idx = [slice(int(0.1*s), int(0.9*s)) for s in lnrho_img.shape]
    if diff:
        vmax = 1.15 * np.percentile(np.abs(lnrho_img[idx[0],idx[1]]), 99)
        kw = dict(cmap='coolwarm', vmin=-vmax, vmax=vmax)
        label = r'$\Delta \ln \rho$'
    else:
        vmin,vmax = np.percentile(lnrho_img[idx[0],idx[1]], [2., 98.])
        w = vmax - vmin
        vmin -= 0.15 * w
        vmax += 0.15 * w
        kw = dict(vmin=vmin, vmax=vmax)
        label = r'$\ln \rho$'
    
    im = ax.imshow(
        lnrho_img,
        origin='lower',
        interpolation='nearest',
        extent=[-10,10,-10,10],
        **kw
    )
    cb_im = fig.colorbar(im, cax=cax_im, orientation='vertical', label=label)

    n_stars = len(A_star)

    if n_stars:
        if diff:
            vmax = 1.2 * np.percentile(np.abs(A_star), 98)
            kw = dict(vmin=-vmax, vmax=vmax, edgecolors='w', linewidths=0.5)
            label = r'$\Delta A$'
        else:
            kw = dict(edgecolors='none')
            label = r'$A$'

        sc = ax.scatter(
            x_star.numpy()[:,0],
            x_star.numpy()[:,1],
            c=A_star,
            s=16 / max(np.sqrt(n_stars/1024), 1),
            cmap='coolwarm',
            **kw
        )
        cb_sc = fig.colorbar(sc, cax=cax_sc, orientation='horizontal', label=label)
    else:
        cax_sc.axis('off')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
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


def train(log_rho_fit, x_star, A_true, n_steps, callback=None):
    #prior_weight = tf.constant(1e-2)

    def get_prior_weight(step):
        log_w0, log_w1 = -2, -1 # base 10
        log_w = log_w0 + step/n_steps * (log_w1-log_w0)
        return tf.constant(10**log_w)

    n_stars = x_star.shape[0]

    ode_fn = get_ray_ode(log_rho_fit, x_star)
    solver = tfp.math.ode.DormandPrince(
        rtol=1e-3, atol=1e-3, name='ray_integrator'
    )

    # opt = tfa.optimizers.RectifiedAdam(
    #     lr=1e-3,
    #     min_lr=1e-4,
    #     total_steps=n_steps,
    #     warmup_proportion=0.1,
    #     epsilon=1e-5 # Increase stability at late times
    # )
    # opt = keras.optimizers.Adadelta(learning_rate=1e-3)
    #lr_values = [1e-3, 1e-4, 1e-5, 1e-6]
    lr_values = [1e-3, 1e-4, 1e-5]
    #lr_values = [1e-4, 1e-5]
    n = len(lr_values)
    lr_boundaries = [int((k+1)*n_steps/n) for k in range(n-1)]
    lr_schedule = keras.optimizers.schedules.PiecewiseConstantDecay(
        lr_boundaries, lr_values
    )
    opt = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.5)

    @tf.function
    def grad_step(prior_weight):
        print('Tracing <grad_step()> ...')

        with tf.GradientTape() as g:
            print('Solving ODE ...')
            res = solver.solve(
                ode_fn,
                0, tf.zeros([n_stars,3]),
                tf.constant([1])
            )
            print('Calculating chi^2 ...')
            A_fit,_ = tf.split(res.states, [1,2], axis=2)
            A_fit = tf.squeeze(A_fit)
            log_chi2 = tf.math.log(tf.reduce_mean((A_true - A_fit)**2))
            prior = log_rho_fit.prior(3.0)
            loss = log_chi2 + prior_weight * prior

        variables = log_rho_fit.trainable_variables
        print('Calculating gradients ...')
        grads = g.gradient(loss, variables)
        print('Applying gradients ...')
        opt.apply_gradients(zip(grads, variables))

        print('Returning ...')
        return loss, A_fit, log_chi2, prior

    loss_hist = []
    ln_chi2_hist = []
    prior_hist = []

    step_iter = tqdm(range(n_steps))
    for i in step_iter:
        prior_weight = get_prior_weight(i)
        loss, A_fit, ln_chi2, prior = grad_step(prior_weight)

        loss_hist.append(loss)
        ln_chi2_hist.append(ln_chi2)
        prior_hist.append(prior)
        step_iter.set_postfix({
            'ln(chi2)': float(ln_chi2),
            'prior': float(prior),
            'loss': float(loss)
        })

        if callback is not None:
            callback(i)

    return loss_hist, ln_chi2_hist, prior_hist


def gen_mock_data(max_order, n_stars, sigma_A=0, k_slope=4):
    log_rho = HarmonicExpansion2D(max_order, extent=[10,10], k_slope=k_slope)

    rng = np.random.default_rng()
    x_star = tf.constant(rng.uniform(-8, 8, size=(n_stars,2)).astype('f4'))

    A = calc_A(log_rho, x_star)
    
    A_obs = A + sigma_A * rng.normal(size=A.shape)

    return log_rho, x_star, A, A_obs


def calc_image(log_rho):
    x_grid = np.linspace(-10, 10, 200, dtype='f4')
    x_grid,y_grid = np.meshgrid(x_grid,x_grid)
    shape = x_grid.shape
    xy_grid = np.stack([x_grid, y_grid], axis=2)
    xy_grid.shape = (np.prod(shape), 2)

    img = log_rho(xy_grid).numpy()
    img.shape = shape

    return img


def calc_A(log_rho, x_star):
    n_stars = x_star.shape[0]
    ode_fn = get_ray_ode(log_rho, x_star)
    solver = tfp.math.ode.DormandPrince(
        rtol=1e-4, atol=1e-4, name='ray_integrator'
    )
    res = solver.solve(
        ode_fn,
        0, tf.zeros([n_stars,3]),
        tf.constant([1])
    )
    A,_ = tf.split(res.states, [1,2], axis=2)
    A = tf.squeeze(A)

    return A


def plot_loss(loss_hist, ln_chi2_hist, prior_hist):
    fig,ax = plt.subplots(1,1, figsize=(6,4))

    ax.plot(loss_hist, label='loss')
    ax.plot(ln_chi2_hist, alpha=0.5, label=r'$\ln \chi^2$')
    ax.plot([], [], alpha=0.5, label='prior') # dummy plot, for legend
    ax.set_ylabel(r'loss, $\ln \chi^2$')

    ax2 = ax.twinx()
    for i in range(2):
        ax2.plot([],[]) # dummy plots, to cycle colors on ax2
    ax2.plot(prior_hist, alpha=0.5)
    ax2.set_ylabel(r'prior')

    ax.legend(loc='upper right')

    ax.set_xlabel('training step')
    ax.set_title('training history')
    
    return fig


def main():
    fig_dir = 'plots_slope3_err1_increaseprior2/'

    # Generate mock data
    log_rho_true, x_star, A_true, A_obs = gen_mock_data(
        60, 4096,
        sigma_A=1,
        k_slope=3
    )

    img_true = calc_image(log_rho_true)
    fig = plot_lnrho_A(img_true, x_star, A_true, title='truth')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_true'))
    plt.close(fig)
    fig = plot_lnrho_A(img_true, x_star, [], title='truth')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_true_nostars'))
    plt.close(fig)

    # Initialize model
    log_rho_fit = HarmonicExpansion2D(40, extent=[10,10], k_slope=6)

    n_trainable = sum([tf.size(v) for v in log_rho_fit.trainable_variables])
    print(f'{n_trainable} trainable variables.')

    #A_init = calc_A(log_rho_fit, x_star)
    #img_init = calc_image(log_rho_fit)
    #fig = plot_lnrho_A(img_init, x_star, A_init, title='fit (step 0)')
    #fig.savefig(os.path.join(fig_dir, 'ln_rho_stars_step00000'))
    #plt.close(fig)
    #fig = plot_lnrho_A(img_init, x_star, [], title='fit (step 0)')
    #fig.savefig(os.path.join(fig_dir, 'ln_rho_nostars_step00000'))
    #plt.close(fig)

    # Optimize model
    n_steps = 2048
    plot_every = 2

    def plot_callback(step):
        if (step % plot_every != plot_every-1) and (step != -1):
            return
        img = calc_image(log_rho_fit)
        A = calc_A(log_rho_fit, x_star)
        fig = plot_lnrho_A(img, x_star, A, title=f'fit (step {step+1})')
        fig.savefig(os.path.join(fig_dir, f'ln_rho_fit_stars_step{step+1:05d}'))
        plt.close(fig)
        fig = plot_lnrho_A(img, x_star, [], title=f'fit (step {step+1})')
        fig.savefig(os.path.join(fig_dir, f'ln_rho_fit_nostars_step{step+1:05d}'))
        plt.close(fig)
        fig = plot_lnrho_A(
            img-img_true, x_star, A-A_true,
            title=f'fit - truth (step {step+1})',
            diff=True
        )
        fig.savefig(os.path.join(fig_dir, f'ln_rho_diff_stars_step{step+1:05d}'))
        plt.close(fig)
        fig = plot_lnrho_A(
            img-img_true, x_star, [],
            title=f'fit - truth (step {step+1})',
            diff=True
        )
        fig.savefig(os.path.join(fig_dir, f'ln_rho_diff_nostars_step{step+1:05d}'))
        plt.close(fig)

    plot_callback(-1)

    loss_hist, ln_chi2_hist, prior_hist = train(
        log_rho_fit, x_star, A_obs, n_steps,
        callback=plot_callback
    )

    img_fit = calc_image(log_rho_fit)
    A_fit = calc_A(log_rho_fit, x_star)
    fig = plot_lnrho_A(img_fit, x_star, A_fit, title='fit')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_fit'))
    plt.close(fig)
    fig = plot_lnrho_A(img_fit, x_star, [], title='truth')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_fit_nostars'))
    plt.close(fig)
    fig = plot_lnrho_A(
        img_fit-img_true, x_star, A_fit-A_true,
        title='fit - truth', diff=True
    )
    fig.savefig(os.path.join(fig_dir, 'ln_rho_diff_true'))
    plt.close(fig)
    fig = plot_lnrho_A(
        img_fit-img_true, x_star, A_fit-A_obs,
        title='fit - observed', diff=True
    )
    fig.savefig(os.path.join(fig_dir, 'ln_rho_diff_obs'))
    plt.close(fig)
    fig = plot_lnrho_A(
        img_fit-img_true, x_star, [],
        title='fit - observed', diff=True
    )
    fig.savefig(os.path.join(fig_dir, 'ln_rho_diff_nostars'))
    plt.close(fig)
    fig = plot_loss(loss_hist, ln_chi2_hist, prior_hist)
    fig.savefig(os.path.join(fig_dir, 'loss_history'))
    plt.close(fig)

    return 0

if __name__ == '__main__':
    main()

