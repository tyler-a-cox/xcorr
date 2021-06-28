import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from astropy import units as u
import tqdm
import glob
import os


def tau_s(z_s):
    """ """
    return (
        6.45e5
        * (cosmo.Ob0 * cosmo.h / 0.03)
        * (cosmo.Om0 / 0.3) ** -0.5
        * ((1 + z_s) / 10)
    )


def helper(x):
    """ """
    return (
        x ** 4.5 / (1.0 - x)
        + 9 / 7 * x ** 3.5
        + 9.0 / 5.0 * x ** 2.5
        + 3 * x ** 1.5
        + 9 * x ** 0.5
        - 4.5 * np.log((1 + x ** 0.5) / (1 - x ** 0.5))
    )


def tau_lya(halo_pos, xH, z, z_reion=6.0, dim=256, width=200 * u.Mpc):
    """
    xH: (np.array, int)
        Average neutral fraction

    z: float
        Source redshift

    halo_file: float
        asdf

    Returns:
    -------

    """
    D = rand_average_bubble_size(halo_pos, xH, dim=dim, width=width)
    z_obs = z + hand_wavy_redshift(z, D)
    h_diff = helper((1 + z) / (1 + z_obs)) - helper((1 + z_reion) / (1 + z_obs))
    return (
        np.mean(xH)
        * tau_s(z)
        * (2.02e-8 / np.pi)
        * ((1 + z) / (1 + z_obs)) ** 1.5
        * h_diff
    )


def hand_wavy_redshift(z, D=6.6 * u.Mpc):
    """ """
    return (cosmo.H(z) * D / const.c).to(u.dimensionless_unscaled)


def bubble_size(pos, xH):
    """

    Return the ionized bubble size in voxels

    Parameters:
    ----------

    pos : tuple, np.array
        LAE halo positions

    xH : np.array
        Neutral fraction cube

    """
    try:
        return np.abs(
            pos[2]
            - np.array(
                np.nonzero(
                    xH[
                        pos[0],
                        pos[1],
                    ]
                )
            )
        ).min()

    except:
        return -1


def average_bubble_size(halo_pos, xH, dim=256.0, width=200.0 * u.Mpc):
    """
    Calculates the mean of the whole sample
    """
    pix = 0
    count = 0
    for i in tqdm.tqdm(
        range(halo_pos.shape[0]),
        desc="Calculating Mean Bubble Size",
        unit="halo",
        total=halo_pos.shape[0],
    ):
        size = bubble_size(halo_pos[i, :], xH)
        if size > 0:
            pix += size
            count += 1
    return (pix / count) * (width / dim)


def rand_average_bubble_size(halo_pos, xH, dim=256.0, width=200.0 * u.Mpc):
    """
    Randomly selects ~1% of the population to take the mean
    """
    pix = 0
    count = 0
    s = halo_pos.shape[0]
    idx = np.random.choice(np.arange(s), replace=False, size=int(s / 100.0))
    pos = halo_pos[idx, :]

    for i in tqdm.tqdm(
        range(pos.shape[0]),
        desc="Calculating Mean Bubble Size",
        unit="halo",
        total=pos.shape[0],
    ):
        size = bubble_size(pos[i, :], xH)
        if size > 0:
            pix += size
            count += 1
    return (pix / count) * (width / dim)


class Interferometer:
    """
    Noise for 21cm and CO intensity mappers
    """

    def __init__(self, dnu=0.1):
        """ """
        self.dnu = dnu

    def __repr__(self):
        """ """
        pass


class Mapper:
    """
    Noise for Halpha, LymanAlpha intensity mappers
    """

    def __init__(self, R=41.5, dnu=0.1):
        """ """
        self.dnu = dnu
        self.R = R

    def __repr__(self):
        """ """
        pass
