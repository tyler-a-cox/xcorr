import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy import constants as const
from powerbox import get_power
import tqdm
from scipy.interpolate import interp1d
from functools import lru_cache
from .utils import *


class Cube:
    """ """

    def __init__(self, **kwargs):
        """ """
        self.kwargs = kwargs
        self.cube = cube
        self.boxlength = boxlength
        self.N = cube.shape[0]

    def cross(self, a):
        """
        Use powerbox to cross-correlate with another cube object
        """
        pass

    # @lru_cache
    def power_spectra(
        self, cube, boxlength, get_variance=False, deltax2=None, **kwargs
    ):
        """
        Light wrapper over get_power

        TODO: get_power has all the functionality needed to compute the power spectrum
        except for adding the leading factor. Consolidate power_spectra and dimensional_ps
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

    def dimensional_ps(
        self, cube, boxlength, deltax2=None, get_variance=False, **kwargs
    ):
        """
        Dimensional Power Spectrum

        """
        if deltax2 is None:
            deltax2 = cube

        if get_variance:
            ps, k, var = self.power_spectra(
                cube, boxlength, get_variance=get_variance, deltax2=deltax2, **kwargs
            )
        else:
            ps, k = self.power_spectra(cube, boxlength, deltax2=deltax2, **kwargs)

        return cube.mean() * deltax2.mean() * ps, k

    def r(self, deltax, deltax2, boxlength, get_variance=False, **kwargs):
        """
        Cross-correlation coefficient
        """
        PS_1, k = self.power_spectra(deltax, boxlength, **kwargs)
        PS_2, _ = self.power_spectra(deltax2, boxlength, **kwargs)
        PS_x, _ = self.power_spectra(deltax, boxlength, deltax2=deltax2, **kwargs)
        return PS_x / np.sqrt(PS_1 * PS_2), k


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

    def simulate(self, diffuse=True, halo=True, attenuation=False, method="skewer"):
        """
        Simulate lyman alpha
        """
        if attenuation:
            assert method in [
                "skewer",
                "bubble",
            ], "Not a valid attenuation model method"

            tau = self.attenuate(
                self.run, self.halos.halos_masses, self.halos.halos_coords
            )

        pass

    def attenuate(self, run, halomasses, halocoords):
        """
        Calculate tau for lyman alpha halos
        """
        return

    def star_formation_rate(self, M, z=7, sim_num=1):
        """
        Returns the star-formation rate for a dark-matter halo of a given mass
        and redshift

        Units: M_sun per year


        Note: Zero-out redshift for now. Other versions of this equation use
        redshift but the current
              sim that I am basing this equation off of does not use redshift.

        https://arxiv.org/pdf/1205.1493.pdf

        """

        if sim_num == 1:
            a, b, d, c1, c2 = 2.8, -0.94, -1.7, 1e9, 7e10
            sfr = 2.8e-28 * (M ** a) * (1.0 + M / c1) ** b * (1.0 + M / c2) ** d

        if sim_num == 2:
            a, b, d, e, c1, c2, c3 = 2.59, -0.62, 0.4, -2.25, 8e8, 7e9, 1e11
            sfr = (
                1.6e-26
                * (M ** a)
                * (1.0 + M / c1) ** b
                * (1.0 + M / c2) ** d
                * (1.0 + M / c3) ** e
            )

        if sim_num == 3:
            a, b, d, e, c1, c2, c3 = 2.59, -0.62, 0.4, -2.25, 8e8, 7e9, 1e11
            sfr = (
                2.25e-26
                * (1.0 + 0.075 * (z - 7))
                * (M ** a)
                * (1.0 + M / c1) ** b
                * (1.0 + M / c2) ** d
                * (1.0 + M / c3) ** e
            )

        return sfr * u.M_sun / u.year

    def f_lya(self, z, C_dust=3.34, zeta=2.57):
        """
        Fraction of lyman-alpha photons not absorbed by dust

        https://arxiv.org/pdf/1010.4796.pdf
        """
        return C_dust * 1e-3 * (1.0 + z) ** zeta

    def f_esc(self, M, z):
        """
        Escape fraction of ionizing photons
        """

        def alpha(z):
            """
            Alpha/beta values found in:

            https://arxiv.org/pdf/0903.2045.pdf
            """
            zs = np.array([10.4, 8.2, 6.7, 5.7, 5.0, 4.4])
            a = np.array([2.78e-2, 1.30e-2, 5.18e-3, 3.42e-3, 6.68e-5, 4.44e-5])
            b = np.array([0.105, 0.179, 0.244, 0.262, 0.431, 0.454])
            fa = interp1d(zs, a, kind="cubic")
            fb = interp1d(zs, b, kind="cubic")
            return (fa(z), fb(z))

        a, b = alpha(z)
        return np.exp(-a * M ** b)

    def L_gal_rec(self, M, z, sim_num=1):
        """
        Luminosity due to galactic recombinations

        Args:
            M: (float, np.array)
                Masses of dark matter halos
            z: (float)
                Redshift of observation
        """
        sf_rate = self.star_formation_rate(M, z=z, sim_num=sim_num)
        return (
            1.55e42
            * (1 - self.f_esc(M, z))
            * self.f_lya(z)
            * sf_rate
            * u.erg
            / u.s
            * u.year
            / u.Msun
        )

    def L_gal_exc(self, M, z, sim_num=1):
        """
        Luminosity due to galactic excitations

        Args:
            M: (float, np.array)
                Masses of dark matter halos
            z: (float)
                Redshift of observation
        """
        sf_rate = self.star_formation_rate(M, z=z, sim_num=sim_num)
        return (
            4.03e41
            * self.f_lya(z)
            * (1 - self.f_esc(M, z))
            * sf_rate
            * u.erg
            / u.s
            * u.year
            / u.Msun
        )

    def L_gal(self, M, z, sim_num=1):
        """
        Args:
            M: (float, np.array)
                Masses of dark matter halos
            z: (float)
                Redshift of observation
        """
        return self.L_gal_exc(M, z, sim_num=sim_num) + self.L_gal_rec(
            M, z, sim_num=sim_num
        )

    def I_gal(self, M, z, n=200.0, cube_volume=300.0):
        """
        Lyman Alpha surface brightness due to galactic emission
        """
        V = (cube_volume * u.Mpc / n) ** 3
        nu = 2.47e15 / u.s / (1 + z)
        return (nu * scale_factor(z) * L_gal(M, z) / V).to(u.erg / u.cm ** 2 / u.s)

    def cube_brightness(self, M, halo_pos, z, n=200):
        """
        Surface brightness of a
        """
        lya_field = np.zeros((n, n, n))
        I_vals = I_gal(M, z, n=n).value
        lya_field[halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2]] += I_vals
        return lya_field

    def cube_brightness_change(self, M, halo_pos, z, n=200):
        """
        Surface brightness of a
        """
        lya_field = np.zeros((n, n, n))
        I_vals = I_gal(M, z, n=n).value
        np.add.at(lya_field, (halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2]), I_vals)
        return lya_field

    """
    Diffuse Component
    """

    def n_rec_dot(self, T_k, x, delta_x, z):
        """ """
        return alpha(T_k, z) * n_e(x, delta_x, z) * n_HII(x, delta_x, z)

    def n_e(self, x, delta_x, z):
        """ """
        return x * n_b(delta_x, z)

    def n_b(self, delta_x, z):
        """ """
        n_b0 = 1.905e-7 * u.cm ** -3
        return delta_x * (1 + z) ** 3 * n_b0

    def n_HII(self, x, delta_x, z, Y_He=0.24):
        """ """
        return n_e(x, delta_x, z) * (4.0 - 4.0 * Y_He) / (4.0 - 3 * Y_He)

    def alpha(self, T_k, z):
        """
        Recombination coefficient
        """
        units = u.cm ** 3 / u.s
        return 4.2e-13 * (T_k / 1e4) ** -0.7 * (1 + z) ** 3 * units

    def L_diffuse(self, T_k, x, delta_x, z, f_rec=0.66):
        """ """
        E_lya = 1.637e-11 * u.erg
        return f_rec * n_rec_dot(T_k, x, delta_x, z) * E_lya

    def I_diffuse(self, T_k, x, delta_x, z):
        """ """
        c = (
            y(z)
            * cosmo.comoving_transverse_distance(z) ** 2
            / (4 * np.pi * cosmo.luminosity_distance(z) ** 2)
        )
        nu = 2.47e15 / u.s / (1 + z)
        return (L_diffuse(T_k, x, delta_x, z) * c * nu).to(u.erg / u.cm ** 2 / u.s)

    def tau_s(self, z_s):
        """ """
        return (
            6.45e5
            * (cosmo.Ob0 * cosmo.h / 0.03)
            * (cosmo.Om0 / 0.3) ** -0.5
            * ((1 + z_s) / 10)
        )

    def helper(self, x):
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
        D = self.rand_average_bubble_size(halo_pos, xH, dim=dim, width=width)
        z_obs = z + self.hand_wavy_redshift(z, D)
        h_diff = self.helper((1 + z) / (1 + z_obs)) - self.helper(
            (1 + z_reion) / (1 + z_obs)
        )
        return (
            np.mean(xH)
            * self.tau_s(z)
            * (2.02e-8 / np.pi)
            * ((1 + z) / (1 + z_obs)) ** 1.5
            * h_diff
        )

    def hand_wavy_redshift(self, z, D=6.6 * u.Mpc):
        """ """
        return (cosmo.H(z) * D / const.c).to(u.dimensionless_unscaled)

    def bubble_size(self, pos, xH):
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

    def average_bubble_size(self, halo_pos, xH, dim=256.0, width=200.0 * u.Mpc):
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
            size = self.bubble_size(halo_pos[i, :], xH)
            if size > 0:
                pix += size
                count += 1
        return (pix / count) * (width / dim)

    def rand_average_bubble_size(self, halo_pos, xH, dim=256.0, width=200.0 * u.Mpc):
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
            size = self.bubble_size(pos[i, :], xH)
            if size > 0:
                pix += size
                count += 1
        return (pix / count) * (width / dim)

    def __repr__(self):
        """ """
        return


class CarbonMonoxide(Cube):
    """ """

    def __init__(self):
        """ """
        super().__init__(**kwargs)

    def simulate(self, attenuation=False, method="skewer"):
        """
        Simulate lyman alpha
        """
        assert method in ["skewer", "bubble"], "Not a valid attenuation model method"
        pass

    def __repr__(self):
        """ """
        pass


class HAlpha(Cube):
    """ """

    def __init__(self):
        """ """
        super().__init__(**kwargs)

    def simulate(self, attenuation=False, method="skewer"):
        """
        Simulate hydrogen-alpha
        """
        if attenuation:
            assert method in [
                "skewer",
                "bubble",
            ], "Not a valid attenuation model method"

            tau = self.attenuate(
                self.run, self.halos.halos_masses, self.halos.halos_coords
            )

        pass

    def __repr__(self):
        """ """
        pass
