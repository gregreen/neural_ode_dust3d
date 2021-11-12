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
    def __init__(self, max_order, extent=[1,1], k_slope=0, seed=None):
        super(HarmonicExpansion2D, self).__init__()
        self.max_order = max_order
        self.extent = extent
        self._initialize(k_slope, seed=seed)

    @snt.once
    def _initialize(self, k_slope, seed=None):
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
        sigma = k2**(-0.5*k_slope)
        sigma[self.max_order,self.max_order] = 1.0
        sigma /= np.sqrt(np.sum(sigma**2))
        rng = np.random.default_rng(seed)
        ab = sigma * rng.normal(size=shape)
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


def plot_lnrho_A(img, x_star, A_star, title='', diff=False, exp=False):
    fig,ax_arr = plt.subplots(
        2,2,
        figsize=(6,6),
        dpi=100,
        gridspec_kw=dict(width_ratios=[1,0.1], height_ratios=[1,0.1])
    )

    ax,cax_im,cax_sc,ax_empty = ax_arr.flat
    ax_empty.axis('off')

    # Plot dust density
    idx = [slice(int(0.1*s), int(0.9*s)) for s in img.shape]
    if diff:
        vmax = 1.15 * np.percentile(np.abs(img[idx[0],idx[1]]), 99)
        kw = dict(cmap='coolwarm', vmin=-vmax, vmax=vmax)
        if exp:
            label = r'$\Delta \rho$'
        else:
            label = r'$\Delta \ln \rho$'
    else:
        if exp:
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
        img,
        origin='lower',
        interpolation='nearest',
        extent=[-10,10,-10,10],
        **kw
    )
    cb_im = fig.colorbar(im, cax=cax_im, orientation='vertical', label=label)

    # Plot stars
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
            x_star[:,0],
            x_star[:,1],
            c=A_star,
            s=16 / max(np.sqrt(n_stars/1024), 1),
            cmap='coolwarm',
            **kw
        )
        cb_sc = fig.colorbar(sc, cax=cax_sc, orientation='horizontal', label=label)
    else:
        cax_sc.axis('off')

    # Plot box in which stars reside
    ax.plot(
        [8, 8, -8, -8, 8],
        [8, -8, -8, 8, 8],
        ls=':',
        c='k',
        alpha=0.5
    )

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


def train(log_rho_fit, dataset,
          n_stars, batch_size, n_epochs,
          callback=None):
    # Break the dataset up into batches
    dataset_batches = dataset.shuffle(
                          buffer_size=64*batch_size
                      ).batch(batch_size).repeat(n_epochs)

    # Calculate the number of steps from the given
    # dataset size and requested # of epochs
    n_steps = (n_stars // batch_size) * n_epochs

    # Smoothly increase the weight given to the prior during training
    def get_prior_weight(step):
        log_w0, log_w1 = -3, -3 # base 10
        log_w = log_w0 + step/n_steps * (log_w1-log_w0)
        return tf.constant(10**log_w)

    gamma = tf.constant(1.8) # Slope of penalty on high-k modes

    # Optimizer, with staircase-exponential learning rate
    lr_init, lr_final = 1e-3, 1e-6
    n_lr_drops = 9
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        lr_init,
        decay_steps=int(n_steps/(n_lr_drops+1)),
        decay_rate=(lr_final/lr_init)**(1/n_lr_drops),
        staircase=True
    )
    opt = keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.5,
        global_clipnorm=100. # Guard-rails to prevent fitter from going haywire
    )

    # Get the loss function, with a given integrator tolerance
    loss_fn = get_loss_function(rtol=1e-7, atol=1e-5)

    # Function that takes one gradient-descent step
    @tf.function
    def grad_step(A_obs, x_star, prior_weight):
        print('Tracing <grad_step()> ...')

        # Calculate distance to each star
        #tf.print('Calculating ds_dt')
        ds_dt = tf.norm(x_star, axis=1, keepdims=True, name='ds_dt')

        # Calculate loss
        #tf.print('loss_fn')
        with tf.GradientTape() as g:
            loss, log_chi2, prior, A_fit, diagnostics = loss_fn(
                A_obs, x_star, ds_dt, log_rho_fit,
                batch_size=batch_size,
                prior_weight=prior_weight,
                gamma=gamma
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

        return loss, A_fit, log_chi2, prior, norm, n_eval

    # Keep track of history of loss, chi^2 and prior during training
    history = {
        'loss': [],
        'ln_chi2': [],
        'prior': [],
        'norm': [],
        'n_eval': []
    }

    # Take gradient-descent steps on training batches
    step_iter = tqdm(enumerate(dataset_batches), total=n_steps)
    for i,(A_obs,x_star) in step_iter:
        prior_weight = get_prior_weight(i)
        loss, A_fit, ln_chi2, prior, norm, n_eval = grad_step(
            A_obs, x_star, prior_weight
        )

        history['loss'].append(float(loss))
        history['ln_chi2'].append(float(ln_chi2))
        history['prior'].append(float(prior))
        history['norm'].append(float(norm))
        history['n_eval'].append(int(n_eval))

        # Display diagnostics on progress bar
        step_iter.set_postfix({
            'ln(chi2)': float(ln_chi2),
            'prior': float(prior),
            'loss': float(loss),
            'lr': float(opt._decayed_lr(tf.float32)),
            'norm': float(norm),
            'n_eval': int(n_eval)
        })

        # Call the given callback function, which may, for example, plot
        # the current ln(rho) field
        if callback is not None:
            callback(i)

    return history


def gen_mock_data(max_order, n_stars,
                  sigma_A=0, k_slope=4,
                  batch_size=1024, seed=None):
    log_rho = HarmonicExpansion2D(
        max_order,
        extent=[10,10],
        k_slope=k_slope,
        seed=seed
    )

    if seed is not None:
        seed = 2 * seed + 1 # Use different seed for locations of stars
    rng = np.random.default_rng(seed)
    x_star = rng.uniform(-8, 8, size=(n_stars,2)).astype('f4')

    A = np.empty(n_stars, dtype='f4')

    for i in tqdm(range(0,n_stars,batch_size)):
        A[i:i+batch_size] = calc_A(log_rho, x_star[i:i+batch_size]).numpy()

    A_obs = A + sigma_A * rng.normal(size=A.shape).astype('f4')

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
        rtol=1e-8, atol=1e-7, name='ray_integrator'
    )
    res = solver.solve(
        ode_fn,
        0, tf.zeros([n_stars,1]),
        tf.constant([1])
    )
    #A,_ = tf.split(res.states, [1,2], axis=2)
    A = tf.squeeze(res.states)

    return A


def plot_loss(history):
    fig,(ax_u,ax_l) = plt.subplots(2,1, figsize=(6,6))

    ax_u.plot(history['loss'], label='loss')
    ax_u.plot(history['ln_chi2'], alpha=0.5, label=r'$\ln \chi^2$')

    ax_u.plot([], [], alpha=0.5, label='prior') # dummy plot, for legend
    ax_u.set_ylabel(r'loss, $\ln \chi^2$')

    ax2 = ax_u.twinx()
    for i in range(2):
        ax2.plot([],[]) # dummy plots, to cycle colors on ax2
    ax2.plot(history['prior'], alpha=0.5)
    ax2.set_ylabel(r'prior')

    ax_l.plot(
        history['norm'],
        label=r'$\left|\nabla\left(\mathrm{loss}\right)\right|$'
    )
    ax_l.set_ylabel(r'$\left|\nabla\left(\mathrm{loss}\right)\right|$')

    if 'n_eval' in history:
        ax_l.plot([], [], label='# evaluations') # dummy
        ax2 = ax_l.twinx()
        ax2.plot([], []) # dummy
        ax2.plot(history['n_eval'])
        ax2.set_ylabel(r'# evaluations')

    ax_u.legend(loc='upper right')
    ax_l.legend(loc='center right')

    ax_l.set_xlabel('training step')
    ax_u.set_title('training history')

    ax_u.set_xticklabels([])

    ax_u.grid('on', axis='x', alpha=0.1)
    ax_l.grid('on', axis='x', alpha=0.1)

    fig.subplots_adjust(
        top=0.94,
        bottom=0.10,
        left=0.14,
        right=0.86,
        hspace=0.05
    )

    return fig


def get_loss_function(rtol=1e-7, atol=1e-5):
    """
    Returns a function that calculates the loss of
    a model of ln(rho), given a set of stellar observations.

    Inputs:
      rtol (float): Relative tolerance of the integrator (default: 1e-7).
      atol (float): Absolute tolerance of the integrator (default: 1e-5).

    Returns:
      `calc_loss`, the loss function.
    """
    # ODE solver
    solver = tfp.math.ode.DormandPrince(
        rtol=rtol, atol=atol, name='ray_integrator'
    )

    def calc_loss(A_obs, x_star, ds_dt, log_rho_model,
                  batch_size=1024,
                  prior_weight=tf.constant(1e-3),
                  gamma=tf.constant(2.0)):
        def ode(t, A, dx_dt, ds_dt):
            r"""
            t = fractional distance along ray
            A = \int \exp(\ln \rho) ds = extinction
            dx_dt = change in position per unit time (t) = position of star
            ds_dt = path length per unit time (t) = distance to star
            """
            dA_dt = ds_dt * tf.math.exp(log_rho_model(t*dx_dt))
            return dA_dt

        res = solver.solve(
            ode,
            0, tf.zeros([batch_size,1]),
            tf.constant([1]),
            constants={'dx_dt':x_star, 'ds_dt':ds_dt}
        )
        A_model = tf.squeeze(res.states)
        log_chi2 = tf.math.log(tf.reduce_mean((A_obs - A_model)**2))
        prior = log_rho_model.prior(gamma)
        loss = log_chi2 + prior_weight * prior
        return loss, log_chi2, prior, A_model, res.diagnostics

    return calc_loss


def main():
    fig_dir = 'plots_test/'
    seed_mock, seed_fit, seed_tf = 17, 31, 101 # Fix psuedorandom seeds

    tf.random.set_seed(seed_tf)

    # Generate mock data
    print('Generating mock data ...')
    n_stars = 1024 * 512
    log_rho_true, x_star, A_true, A_obs = gen_mock_data(
        40, n_stars,
        sigma_A=0.1,
        k_slope=1.8,
        batch_size=8*1024,
        seed=seed_mock
    )
    dataset = tf.data.Dataset.from_tensor_slices((A_obs,x_star))

    # Calculate chi^2 and prior of truth
    ln_chi2_true = np.log(np.mean((A_obs-A_true)**2))
    prior_true = log_rho_true.prior(1.8)
    print('Using true model:')
    print(f'  ln(chi^2) = {ln_chi2_true:.5f}')
    print(f'      prior = {prior_true:.5f}')

    # For plotting, only keep a subset of stars
    x_star = x_star[:4096]
    A_true = A_true[:4096]
    A_obs = A_obs[:4096]

    img_true = calc_image(log_rho_true)
    fig = plot_lnrho_A(img_true, x_star, A_true, title='truth')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_true'))
    plt.close(fig)
    fig = plot_lnrho_A(img_true, x_star, [], title='truth')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_true_nostars'))
    plt.close(fig)
    fig = plot_lnrho_A(np.exp(img_true), x_star, [], title='truth', exp=True)
    fig.savefig(os.path.join(fig_dir, 'rho_true_nostars'))
    plt.close(fig)

    # Initialize model
    log_rho_fit = HarmonicExpansion2D(
        40,
        extent=[10,10],
        k_slope=6,
        seed=seed_fit
    )

    n_trainable = sum([tf.size(v) for v in log_rho_fit.trainable_variables])
    print(f'{n_trainable} trainable variables.')

    # Optimize model
    plot_every = 32

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
        fig = plot_lnrho_A(np.exp(img), x_star, [], title=f'fit (step {step+1})', exp=True)
        fig.savefig(os.path.join(fig_dir, f'rho_fit_nostars_step{step+1:05d}'))
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
        fig = plot_lnrho_A(
            np.exp(img)-np.exp(img_true), x_star, [],
            title=f'fit - truth (step {step+1})',
            diff=True,
            exp=True
        )
        fig.savefig(os.path.join(fig_dir, f'rho_diff_nostars_step{step+1:05d}'))
        plt.close(fig)

    plot_callback(-1)

    batch_size = 1024 * 8
    n_epochs = 16
    history = train(
        log_rho_fit, dataset,
        n_stars, batch_size, n_epochs,
        callback=plot_callback
    )

    img_fit = calc_image(log_rho_fit)
    A_fit = calc_A(log_rho_fit, x_star)
    fig = plot_lnrho_A(img_fit, x_star, A_fit, title='fit')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_fit'))
    plt.close(fig)
    fig = plot_lnrho_A(img_fit, x_star, [], title='fit')
    fig.savefig(os.path.join(fig_dir, 'ln_rho_fit_nostars'))
    plt.close(fig)
    fig = plot_lnrho_A(np.exp(img_fit), x_star, [], title='fit', exp=True)
    fig.savefig(os.path.join(fig_dir, 'rho_fit_nostars'))
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
    fig = plot_loss(history)
    fig.savefig(os.path.join(fig_dir, 'loss_history'))
    plt.close(fig)

    return 0

if __name__ == '__main__':
    main()

