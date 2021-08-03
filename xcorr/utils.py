import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from powerbox import get_power
from scipy.interpolate import interp1d
from .xcorr import *


def dimensional_ps(cube, boxlength, deltax2=None, get_variance=False, **kwargs):
    """
    Dimensional Power Spectrum

    """
    if deltax2 is None:
        deltax2 = cube

    if get_variance:
        ps, k, var = power_spectra(
            cube, boxlength, get_variance=get_variance, deltax2=deltax2, **kwargs
        )
    else:
        ps, k = power_spectra(cube, boxlength, deltax2=deltax2, **kwargs)

    return cube.mean() * deltax2.mean() * ps, k


def power_spectra(cube, boxlength, get_variance=False, deltax2=None, **kwargs):
    """
    Light wrapper over get_power
    """

    deltax = cube / cube.mean() - 1.0

    if deltax2 is None:
        deltax2 = deltax

    else:
        deltax2 = deltax2 / deltax2.mean() - 1.0

    if get_variance:
        ps, k, var = get_power(
            deltax, boxlength, get_variance=get_variance, deltax2=deltax2, **kwargs
        )
        return ps, k, var

    else:
        ps, k = get_power(deltax, boxlength, deltax2=deltax2, **kwargs)

        return ps * k ** 3 / (2 * np.pi ** 2), k


def r(deltax, deltax2, boxlength, get_variance=False, **kwargs):
    """
    Cross-correlation coefficient
    """
    PS_1, k = power_spectra(deltax, boxlength, **kwargs)
    PS_2, _ = power_spectra(deltax2, boxlength, **kwargs)
    PS_x, _ = power_spectra(deltax, boxlength, deltax2=deltax2, **kwargs)
    return PS_x / np.sqrt(PS_1 * PS_2), k


def H(z):
    """
    Returns astropy Hubble constant at given redshift

    Units: km Mpc^-1 s^-1
    """
    return cosmo.H(z)


def y(z):
    """
    wl_lya -> Lyman-alpha wavelength in units of km

    Returns value in units of Mpc s
    """
    l_lya = 1.215e-7 * u.m
    return l_lya * (1.0 + z) ** 2 / H(z)


def scale_factor(z, csn=True):
    """
    Common scale factor that appears fairly often.
    """
    # TODO: originally was using cosmo.comoving_transverse_distance
    # changed to cosmo.angular_diameter_distance
    if csn:
        return (
            y(z)
            * cosmo.angular_diameter_distance(z) ** 2
            / (4 * np.pi * cosmo.luminosity_distance(z) ** 2)
        )
    else:
        return (
            y(z)
            * cosmo.angular_diameter_distance(z) ** 2
            / (4 * np.pi * cosmo.lookback_distance(z) ** 2)
        )
