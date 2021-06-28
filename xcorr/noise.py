import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from astropy import units as u
from twentyonecmFAST import load_binary_data
import tqdm
import glob
import os


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
