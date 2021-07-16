import numpy as np
import tqdm
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from powerbox import get_power


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


class IntensityMapper:
    """
    Noise for Halpha, LymanAlpha intensity mappers
    """

    def __init__(self, R=41.5, noise=1e-20):
        """

        Args:
            R : float
                asdf
            noise : float
                asdf
        """
        self.noise = noise
        self.R = R

    def __repr__(self):
        """ """
        pass
