#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from astropy_healpix import HEALPix
from astropy import units
from astropy.io import fits
from reproject import reproject_from_healpix


def get_mol_wcs(size):
    """
    Returns a WCS header for a Galactic Mollweide projection,
    centered on the Galactic Center.

    Inputs:
      size (int): width (in pixels) of the projected image.

    Returns: FITS header describing the WCS projection.
    """
    crpix1 = f'{size/2+0.5:.1f}'
    crpix2 = f'{size/4+0.5:.1f}'
    pix_scale = 0.675*480/size
    crdelt1 = f'{-pix_scale:.8f}'
    crdelt2 = f'{pix_scale:.8f}'
    target_header = fits.Header.fromstring(f"""
        NAXIS   =                    2
        NAXIS1  = {size: >20d}
        NAXIS2  = {size//2: >20d}
        CTYPE1  = 'GLON-MOL'
        CRPIX1  = {crpix1: >20s}
        CRVAL1  =                  0.0
        CDELT1  = {crdelt1: >20s}
        CUNIT1  = 'deg     '
        CTYPE2  = 'GLAT-MOL'
        CRPIX2  = {crpix2: >20s}
        CRVAL2  =                  0.0
        CDELT2  = {crdelt2: >20s}
        CUNIT2  = 'deg     '
        COORDSYS= 'galactic'
        """,
    sep='\n        ')
    return target_header


def plot_healpix_map(fig, m, size=1024,
                     subplot=(1,1,1), nested=True,
                     imshow_kwargs={}):
    target_header = get_mol_wcs(size)

    m_proj, footprint = reproject_from_healpix(
        (m, 'galactic'),
        target_header,
        nested=nested,
        order='nearest-neighbor'
    )

    ax = fig.add_subplot(
        *subplot,
        projection=WCS(target_header),
        frame_class=EllipticalFrame
    )

    kw = imshow_kwargs.copy()
    if 'cmap' not in kw:
        kw['cmap'] = 'viridis'

    if isinstance(kw['cmap'], str):
        kw['cmap'] = plt.get_cmap(kw['cmap'])
    cmap = kw['cmap'].copy()
    cmap.set_bad('lightgray')
    cmap.set_under('lightgray')
    kw['cmap'] = cmap

    im = ax.imshow(
        m_proj,
        **imshow_kwargs
    )

    ax.coords.grid(color='w', alpha=0.2)
    ax.coords['glon'].set_ticks_visible(False)
    ax.coords['glon'].set_ticklabel_visible(False)
    ax.coords['glat'].set_ticks_visible(False)
    ax.coords['glat'].set_ticklabel_visible(False)

    return ax, im


def main():
    fig = plt.figure()
    m = np.arange(12 * 4**2)
    ax,im = plot_healpix_map(fig, m)
    fig.savefig('healpix_example.png', dpi=300)

    return 0

if __name__ == '__main__':
    main()

