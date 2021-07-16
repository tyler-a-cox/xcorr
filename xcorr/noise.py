import numpy as np
import tqdm
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from powerbox import get_power
import py21cmsense as p21s


class Interferometer(p21s):
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

    def __init__(self, R=41.5, noise=1e-20, xpix=1.0):
        """

        Args:
            R : float

            noise : float
                asdf
        """
        self.noise = noise
        self.R = R

    def k_par_res(self, z):
        """k parallel mode resolution for SPHEREx"""
        return 2 * np.pi * (self.R * cosmo.H(z) / (c * (1 + z))).to(u.Mpc ** -1)

    """

    Perpendicular Mode Resolution

    """

    def k_perp_res(self, z, x_pix=1.0 * u.arcsecond):
        """Perpendicular Resolution for"""
        theta = x_pix.to(u.radian)
        return 2 * np.pi / (cosmo.comoving_distance(z) * theta)

    """

    Window Function to Introduce Resolution Error to SPHEREx

    """

    def W(self, kperp, kpar, z):
        """Window function to handle resolution limitations of SPHEREx

        Parameters
        ----------
        kperp: float
        kpar: float
        z: float

        Returns:

        """
        kpar_res = self.k_par_res(z).value
        kperp_res = self.k_perp_res(z).value
        return np.exp((kperp / kperp_res) ** 2 + (kpar / kpar_res) ** 2)

    def V_pix(self, z, x_pix=1.0 * u.arcsecond):
        """ """
        return ((cosmo.kpc_comoving_per_arcmin(z) * x_pix) ** 2 / k_par_res(z)).to(
            u.Mpc ** 3
        ) / 2.0

    def lyman_noise(
        self, ps_interp, kperp, kpar, z, thermal=True, sample=True, thermal_noise=3e-21
    ):
        """
        Noise contribution from SPHEREx-like experiment
        """
        k = np.sqrt(kperp ** 2 + kpar ** 2)
        var = 0
        nu = 2.466e15 / (1.0 + z)
        if sample:
            try:
                var += self.ps_interp(k)
            except ValueError:
                return np.inf

        if thermal:
            var += (
                k ** 3
                / (2 * np.pi ** 2)
                * self.V_pix(z).value
                * (nu * thermal_noise) ** 2
                * self.W(kperp, kpar, z)
            )

        return var

    def x_var(self, kperp, kpar, pspec):
        """ """
        k = np.sqrt(kperp ** 2 + kpar ** 2)
        try:
            return pspec(k)
        except ValueError:
            return np.inf
        return

    def x_power_spec(self, ks, pspec):
        """ """
        ps = []
        for k in ks:
            try:
                ps.append(self.pspec(k))
            except ValueError:
                ps.append(np.inf)
        return np.array(ps)

    def __repr__(self):
        """ """
        pass
