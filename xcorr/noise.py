import numpy as np


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
