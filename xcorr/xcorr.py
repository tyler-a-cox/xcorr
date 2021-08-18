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

    def __init__(self, N=200, boxlength=300):
        """ """
        self.boxlength = boxlength
        self.N = N

    def cross(self, a, b, **kwargs):
        """
        Use powerbox to cross-correlate with another cube object
        """
        self.power_spectra(a, deltax2=b, **kwargs)

    # @lru_cache
    def power_spectra(self, cube, get_variance=False, deltax2=None, **kwargs):
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
                deltax,
                self.boxlength,
                get_variance=get_variance,
                deltax2=deltax2,
                **kwargs
            )
            return ps, k, var

        else:
            ps, k = get_power(deltax, self.boxlength, deltax2=deltax2, **kwargs)

            return ps * k ** 3 / (2 * np.pi ** 2), k

    def dimensional_ps(self, cube, deltax2=None, get_variance=False, **kwargs):
        """
        Dimensional Power Spectrum

        """
        if deltax2 is None:
            deltax2 = cube

        if get_variance:
            ps, k, var = self.power_spectra(
                cube, get_variance=get_variance, deltax2=deltax2, **kwargs
            )
        else:
            ps, k = self.power_spectra(cube, deltax2=deltax2, **kwargs)

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

    def __init__(self, **kwargs):
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
            # Note A = 3e-28, not 2e-28
            sfr = 2e-27 * (M ** a) * (1.0 + M / c1) ** b * (1.0 + M / c2) ** d

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

    def I_gal(self, M, z, sim_num=1, csn=True):
        """
        Lyman Alpha surface brightness due to galactic emission
        """
        V = (self.boxlength * u.Mpc / self.N) ** 3
        nu = 2.47e15 / u.s / (1 + z)
        return (
            nu * scale_factor(z, csn=csn) * self.L_gal(M, z, sim_num=sim_num) / V
        ).to(u.erg / u.cm ** 2 / u.s)

    def cube_brightness(self, M, halo_pos, z, csn=True):
        """
        Surface brightness of a
        """
        lya_field = np.zeros((self.N, self.N, self.N))
        I_vals = self.I_gal(M, z, csn=csn).value
        lya_field[halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2]] += I_vals
        return lya_field

    def cube_brightness_change(self, M, halo_pos, z, sim_num=1, csn=True):
        """
        Surface brightness of a
        """
        lya_field = np.zeros((self.N, self.N, self.N))
        I_vals = self.I_gal(M, z, sim_num=sim_num, csn=csn).value
        np.add.at(lya_field, (halo_pos[:, 0], halo_pos[:, 1], halo_pos[:, 2]), I_vals)
        return lya_field

    """
    Diffuse Component

    Check:
        - alpha looks good
        - nHII looks good
        - nb might need some work
        - n_e looks good
        - nrec dot looks good
        - frec looks good

    """

    def n_rec_dot(self, T_k, x, delta_x, z, plus=True):
        """ """
        return (
            self.alpha(T_k, z)
            * self.n_e(x, delta_x, z, plus=plus)
            * self.n_HII(x, delta_x, z)
        )

    def n_e(self, x, delta_x, z, plus=True):
        """ """
        return x * self.n_b(delta_x, z, plus=plus)

    def n_b(self, delta_x, z, plus=True):
        """ """
        n_b0 = 1.905e-7 * u.cm ** -3
        if plus:
            return (1 + delta_x) * (1 + z) ** 3 * n_b0

        else:
            return delta_x * (1 + z) ** 3 * n_b0

    def n_HII(self, x, delta_x, z, Y_He=0.24, plus=True):
        """ """
        return (
            self.n_e(x, delta_x, z, plus=plus) * (4.0 - 4.0 * Y_He) / (4.0 - 3 * Y_He)
        )

    def alpha(self, T_k, z):
        """
        Recombination coefficient
        """
        units = u.cm ** 3 / u.s
        return 4.2e-13 * (T_k / 1e4) ** -0.7 * (1 + z) ** 3 * units

    def f_rec(self, T_k):
        """ """
        return 0.686 - 0.106 * np.log10(T_k / 1e4) - 0.009 * (T_k / 1e4) ** -0.4

    def L_diffuse(self, T_k, x, delta_x, z, plus=True):
        """ """
        E_lya = 1.637e-11 * u.erg
        return self.f_rec(T_k) * self.n_rec_dot(T_k, x, delta_x, z, plus=plus) * E_lya

    def I_diffuse(self, T_k, x, delta_x, z, csn=True, plus=True):
        """ """
        c = scale_factor(z, csn=csn)
        nu = 2.47e15 / u.s / (1 + z)
        return (self.L_diffuse(T_k, x, delta_x, z, plus=plus) * c * nu).to(
            u.erg / u.cm ** 2 / u.s
        )

    """
    attention
    """

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
            return np.abs(pos[2] - np.array(np.nonzero(xH[pos[0], pos[1],]))).min()

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

    def tau_gp(self, zs):
        """
        """
        tgp = 7.16e5 * ((1 + zs) / 10) ** (3 / 2)
        return tgp

    def tau_LOS(self, coords, xH, z, N=200, boxlength=300):
        """
        """
        taus = np.zeros(coords.shape[0])

        for k in tqdm.tqdm(range(N)):
            ind = coords[:, 2] == k
            halos = coords[ind]
            skewers = xH[halos[:, 0], halos[:, 1]]

            if k > N // 2:
                k = N - k
                skewer = skewers[:, ::-1]

            zsource = z - self.hand_wavy_redshift(z, boxlength / N * u.Mpc * k)
            si = np.arange(k, N - 1)
            zbi = zsource - self.hand_wavy_redshift(zsource, boxlength / N * si * u.Mpc)
            zei = zsource - self.hand_wavy_redshift(
                zsource, boxlength / N * (si + 1) * u.Mpc
            )
            tau = self.tau_gp(zsource) * (2.02e-8 / np.pi)
            special = self.helper((1 + zbi) / (1 + zsource)) - self.helper(
                (1 + zei) / (1 + zsource)
            )
            # This used to be k+1
            taus[ind] = np.sum(
                tau
                * skewers[:, k:-1]
                * special
                * ((1 + zbi) / (1 + zsource)) ** (3 / 2),
                axis=1,
            )

        return np.array(taus)

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
