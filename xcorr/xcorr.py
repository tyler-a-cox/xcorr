import numpy as np
import powerbox
import scipy


class Cube:
    """ """

    def __init__(self, **kwargs):
        """ """
        self.kwargs = kwargs

    def cross(self, a):
        """
        Use powerbox to cross-correlate with another cube object
        """
        pass

    def pspec(self):
        """
        Calculate the power spectrum
        """
        pass


class Hyperfine(Cube):
    """ """

    def __init__(self, **kwargs):
        """ """
        name = "21cm"
        super().__init__(**kwargs)

    def simulate(self):
        """
        Use inputs to simulate 21cm - 21cmFAST. Note this will probably just end
        up inheriting lots of the properties of 21cmFAST
        """
        pass

    def __repr__(self):
        """ """
        pass


class LymanAlpha(Cube):
    """ """

    def __init__(self):
        """ """
        name = "Lyman Alpha"
        super().__init__(**kwargs)

    def simulate(self, attenuation=False, method="skewer"):
        """
        Simulate lyman alpha
        """
        assert (method in ["skewer", "bubble"], "Not a valid attenuation model method")
        pass

    def __repr__(self):
        """ """
        pass


class CarbonMonoxide(Cube):
    """ """

    def __init__(self):
        """ """
        name = "CO"
        super().__init__(**kwargs)

    def simulate(self, attenuation=False, method="skewer"):
        """
        Simulate lyman alpha
        """
        assert (method in ["skewer", "bubble"], "Not a valid attenuation model method")
        pass

    def __repr__(self):
        """ """
        pass


class HAlpha(Cube):
    """ """

    def __init__(self):
        """ """
        name = "H-alpha"
        super().__init__(**kwargs)

    def simulate(self, attenuation=False, method="skewer"):
        """
        Simulate hydrogen-alpha
        """
        assert (method in ["skewer", "bubble"], "Not a valid attenuation model method")
        pass

    def __repr__(self):
        """ """
        pass
