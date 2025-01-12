"""
This module contains the `PointLens` class, which provides
functions to analyze microlensing effects caused by
an isolated point-mass lens.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import numpy as np
from mpmath import hyp1f1, gamma

from .constants import C_SI, G_SI, MSUN_SI


class PointLens:
    """
    A class containing functions for studying microlensing
    due to an isolated point-mass lens.
    In such cases, two images are formed:
    one minima (type I) and one saddle (type II).

    """

    # image positions
    def x_minima(self, y):
        """
        Returns the image position for
        the minima (type I) image.

        Parameters
        ----------
        y : float
           The impact parameter.

        Returns
        -------
        float:
           Image position.

        """

        return (y + np.sqrt(y**2 + 4)) / 2

    def x_saddle(self, y):
        """
        Returns the image position for the
        saddle-point (type II) image.

        Parameters
        ----------
        y : float
           The impact parameter.

        Returns
        -------
        float :
           Image position.

        """

        return (y - np.sqrt(y**2 + 4)) / 2

    # image magnifications
    def magification_minima(self, y):
        """
        Returns the image magnification for
        the minima (type I) image.

        Parameters
        ----------
        y : float
           The impact parameter.

        Returns
        -------
        float :
           Image magnification.

        """

        return 1 / 2 + (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4))

    def magification_saddle(self, y):
        """
        Returns the image magnification for
        the saddle-point (type II) image.

        Parameters
        ----------
        y : float
           The impact parameter.

        Returns
        -------
        float :
           Image magnification.

        """

        return 1 / 2 - (y**2 + 2) / (2 * y * np.sqrt(y**2 + 4))

    # time delay between the two micro-images
    def time_delay_dimensionless(self, y):
        """
        Returns the dimensionless time-delay
        between the two micro-images.

        Parameters
        ----------
        y : float
           The impact parameter.

        Returns
        -------
        float :
           dimensionless time-delay between micro-images.

        """

        return (y * np.sqrt(y**2 + 4)) / 2.0 + np.log(
            (np.sqrt(y**2 + 4) + y) / (np.sqrt(y**2 + 4) - y)
        )

    def time_delay(self, ml, y, zl=0):
        """
        Returns the time-delay between the two micro-images in seconds.

        Parameters
        ----------
        ml : float
            Microlens mass.
        y : float
           The impact parameter.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making it as the redshifted lens mass).

        Returns
        -------
        float:
           time-delay between micro-images in seconds.

        """

        return (
            4 * G_SI * MSUN_SI * ml * (1 + zl) / C_SI**3
        ) * self.time_delay_dimensionless(y)

    # Geometric and Quasi Geometric approximations
    def Fw_geometric_optics(self, w, y):
        """
        Returns the lensing amplification factor, F(w),
        assuming geometric optics approximation.

        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter.

        Returns
        -------
        complex :
            Amplification factor F(w).

        """

        return np.sqrt(np.abs(self.magification_minima(y))) - 1j * np.sqrt(
            np.abs(self.magification_saddle(y))
        ) * np.exp(1j * w * self.time_delay_dimensionless(y))

    def Fw_quasigeometric_optics(self, w, y):
        """
        Returns the lensing amplification factor, F(w),
        assuming Quasi-geometric optics approximation.
        References - arXiv:0402165

        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter.

        Returns
        -------
        complex :
           Amplification factor, F(w).

        """

        return (
            self.Fw_geometric_optics(w, y)
            + (1j / (3 * w))
            * (
                (4 * self.x_minima(y) ** 2 - 1)
                / (pow((self.x_minima(y) ** 2 + 1), 3) * (self.x_minima(y) ** 2 - 1))
            )
            * np.sqrt(np.abs(self.magification_minima(y)))
            + (1 / (3 * w))
            * (
                (4 * self.x_saddle(y) ** 2 - 1)
                / (pow((self.x_saddle(y) ** 2 + 1), 3) * (self.x_saddle(y) ** 2 - 1))
            )
            * np.sqrt(np.abs(self.magification_saddle(y)))
            * np.exp(1j * w * self.time_delay_dimensionless(y))
        )

    # dimensionless frequency (w) in terms of dimensionful frequency (f), w(f)
    def w_of_f(self, f, ml, zl=0):
        """
        Converts a dimensionful frequency (f)
        to the dimensionless frequency (w).

        Parameters
        ----------
        f : float
           The dimensionful frequency which is to be converted.
        ml : float
            Microlens mass.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making it as the redshifted lens mass).

        Returns
        -------
        float :
           The dimensionless frequency, w(f).

        """

        wf = f * (8.0 * np.pi * G_SI * MSUN_SI / C_SI**3) * ml * (1 + zl)
        return wf

    # dimensionful frequency (f) in terms of dimensionless frequency (w), f(w)
    def f_of_w(self, w, ml, zl=0):
        """
        Converts a dimensionless frequency (w)
        to the dimensionful frequency (f).

        Parameters
        ----------
        w : float
           The dimensionless frequency which is to be converted.
        ml : float
            Microlens mass.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making it as the redshifted lens mass).

        Returns
        -------
        float :
           The dimensionful frequency, f(w).

        """

        fw = w / ((8.0 * np.pi * G_SI * MSUN_SI / C_SI**3) * ml * (1 + zl))
        return fw

    # cutoff frequencies for transition to geometric optics

    def w_cutoff_geometric_optics_tolerance_0p1(self, y, print_warning=True):
        """
        Returns a cutoff dimensionless frequency (wc) for a given
        y such that w > wc gives relative error < 0.1 %
        when geometric optics approximation is used.
        Valid for y in range (0.01, 5.00).

        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00).

        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        if y <= 0.12:
            wc = 15112.5 - 52563.5 * y
        elif 0.12 < y <= 1.5:
            wc = (
                -34.08
                - 12.84 * pow(y, -1.0)
                + 114.33 * pow(y, -2.0)
                + 0.89 * pow(y, -3.0)
            )
        else:
            wc = -15.02 + 18.25 * y - 2.66 * y**2

        if print_warning:
            if y < 0.01 or round(y, 3) > 5.00:
                print(
                    f"Warning: y = {y} is outside the interpolation\
                    range (0.01, 5.00). Thus, Extrapolating!"
                )
        return wc

    def w_cutoff_geometric_optics_tolerance_1p0(self, y, print_warning=True):
        """
        Returns a cutoff dimensionless frequency (wc) for a given
        y such that w > wc gives relative error < 1.0 %
        when geometric optics approximation is used.
        Valid for y in range (0.01, 5.00).

        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00).

        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        if y <= 0.071:
            wc = 16604 - 202686 * y
        else:
            wc = 0.64 + 0.97 * pow(y, -1.0) + 6 * pow(y, -2.0) + 0.38 * pow(y, -3.0)

        if print_warning:
            if y < 0.01 or round(y, 3) > 5.00:
                print(
                    f"Warning: y = {y} is outside the interpolation\
                    range (0.01, 5.00). Thus, Extrapolating!"
                )
        return wc

    # cutoff frequencies for transition to Quasi-geometric optics
    def w_cutoff_quasigeometric_optics_tolerance_0p1(self, y, print_warning=True):
        """
        Returns a cutoff dimensionless frequency (wc) for a given
        y such that w > wc gives relative error < 0.1 %
        when Quasi-geometric optics approximation is used.
        Valid for y in range (0.01, 5.00).

        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00).

        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        wc = 9 * pow(y, -1.0) + 0.04 * pow(y, -2.0)

        if print_warning:
            if y < 0.01 or round(y, 3) > 5.00:
                print(
                    f"Warning: y = {y} is outside the interpolation\
                    range (0.01, 5.00). Thus, Extrapolating!"
                )
        return wc

    def w_cutoff_quasigeometric_optics_tolerance_1p0(self, y, print_warning=True):
        """
        Returns a cutoff dimensionless frequency (wc) for a given
        y such that w > wc gives relative error < 1.0 % when
        Quasi-geometric optics approximation is used.
        Valid for y in range (0.01, 5.00).

        Parameters
        ----------
        y : float
           The impact parameter, preferably in range (0.01, 5.00).

        Returns
        -------
        float :
           The cutoff dimensionless frequency, wc.

        """

        wc = 4 * pow(y, -1.0) - np.log(y) / 5.0
        if print_warning:
            if y < 0.01 or round(y, 3) > 5.00:
                print(
                    f"Warning: y = {y} is outside the interpolation\
                    range (0.01, 5.00). Thus, Extrapolating!"
                )
        return wc

    # Amplification factor related functions
    def Fw_analytic(self, w, y):
        """
        Returns the amplification factor, F(w, y),
        for point lens using the analytical formula.
        It breaks down, or is difficult to compute,
        when the system approaches geometrical optics regime.
        In cases where it is not converging,
        use _Fw_effective()_.

        Parameters
        ----------
        w : float
           The dimensionless frequency.
        y : float
           The impact parameter.

        Returns
        -------
        complex:
           The Amplification Factor, F(w, y).

        """

        if w == 0:
            return 1

        w = np.float128(w)
        xm = np.float128((y + np.sqrt(y * y + 4.0)) / 2.0)
        pm = np.float128(pow(xm - y, 2) / 2.0 - np.log(xm))
        hp = np.log(w / 2.0) - (2.0 * pm)
        h = np.exp((np.pi * w / 4.0) + 1j * (hp * w / 2.0))
        gm = gamma(1.0 - (1j * w / 2.0))
        hf = hyp1f1((1j * w / 2.0), 1.0, (1j * y * y * w / 2.0))
        Ff = h * gm * hf
        return complex(Ff.real, Ff.imag)

    def Fw_effective(self, w, y):
        """
        An efficient computation of the point-lens amplification factor,
        F(w, y), that uses analytical expression along with
        the knowledge of the Quasi-geometric and the geometric optics limit.
        It can handle any value, but is efficient within regime of y in (0.01, 5).

        For dimensionful variant of this function, use: _point_Ff_eff()_.

        For mapping this function to an array of frequencies,
        use _Fw_effective_map()_
        Parameters
        ----------
        w : float
           The dimensionless frequency which is to be converted.
        y : float
           The impact parameter.

        Returns
        -------
        complex :
           The Amplification Factor, F(w, y).

        """
        if y < 0.01:
            return self.Fw_analytic(w, y)

        wc_geo = self.w_cutoff_geometric_optics_tolerance_0p1(y)
        wc_Qgeo = self.w_cutoff_quasigeometric_optics_tolerance_0p1(y)
        if wc_Qgeo < wc_geo:
            if w < wc_Qgeo:
                return self.Fw_analytic(w, y)
            if wc_Qgeo <= w < wc_geo:
                return self.Fw_quasigeometric_optics(w, y)
            return self.Fw_geometric_optics(w, y)

        return self.Fw_quasigeometric_optics(w, y)

    def Ff_effective(self, f, ml, y, zl=0):
        """
        Returns an efficient computation of the point-lens amplification factor,
        F(f, ml, y, zl=0), that uses analytical expression along with
        the knowledge of the Quasi-geometric and the geometric optics limit.
        It can handle any value, but is efficient within regime of y in (0.01, 5).

        For dimensionless variant of this function,
        use: _Fw_effective()_.

        For mapping this function to an array of frequencies,
        use _point_Ff_eff_map()_.

        Parameters
        ----------
        f : float
           Frequency (in Hz).
        ml : float
           Microlens mass.
        y : float
           The impact parameter.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making it as the redshifted lens mass).

        Returns
        -------
        complex:
           The Amplification Factor, F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.Fw_effective(w, y)

    def Fw_effective_map(self, ws, y):
        """
        Mapping fucntion for Fw_effective().
        This takes an array of (dimensionless) frequencies as input.

        Parameters
        ----------
        ws : float array.
            Array of dimensionless frequencies.
        y : float
           The impact parameter.

        Returns
        -------
        complex array :
           Array containing the amplification factors, F(ws, y).

        """

        return np.array(list(map(lambda w: self.Fw_effective(w, y), ws)))

    def Ff_effective_map(self, fs, ml, y, zl=0):
        """
        Mapping fucntion for point_Ff_eff().
        This takes an array of frequencies as input.

        Parameters
        ----------
        f : float array
           Array of Frequencies (in Hz).
        ml : float
            Microlens mass.
        y : float
           The impact parameter.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making ml as the redshifted lens mass).

        Returns
        -------
        complex array :
            Array containing the amplification factors, F(fs, ml, y, zl).

        """

        return np.array(list(map(lambda f: self.Ff_effective(f, ml, y, zl), fs)))

    def Fw_analytic_map(self, ws, y):
        """
        Mapping fucntion for Fw_analytic().
        This takes an array of (dimensionless) frequencies as input.

        Parameters
        ----------
        ws : float array
            Array of dimensionless frequencies.
        y : float
           The impact parameter.

        Returns
        -------
        complex array :
           Array containing the amplification factors, F(ws, y).

        """

        return np.array(list(map(lambda w: self.Fw_analytic(w, y), ws)))

    def Ff_analytic(self, f, ml, y, zl=0):
        """
        Returns amplification factor, F(f, ml, y, zl=0),
        for point lens using actual analytic formula.
        It breaks down, or is difficult to compute,
        when the system approaches geometrical optics regime.

        This function is dimensionful variant of Fw_analytic().

        Parameters
        ----------
        fs : float
            Frequency (in Hz).
        ml : float
            Microlens mass.
        y : float
           The impact parameter.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making it as the redshifted lens mass).

        Returns
        -------
        complex :
           The Amplification Factor, F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.Fw_analytic(w, y)

    def Ff_analytic_map(self, fs, ml, y, zl=0):
        """
        Mapping fucntion for _Ff_analytic()_. This takes an array of frequencies as input.

        Parameters
        ----------
        fs : float array
            Array of Frequencies (in Hz).
        ml : float
            Microlens mass.
        y : float
           The impact parameter.
        zl : float, optional
            Lens-redshift. Default = 0
            (this is equivalent to absorbing the (1+zl) term into
            ml thereby making it as the redshifted lens mass).

        Returns
        -------
        complex array :
           Array containing the amplification factors, F(fs, ml, y, zl).

        """

        return np.array(list(map(lambda f: self.Ff_analytic(f, ml, y, zl), fs)))

    def Ff_geometric_optics(self, f, ml, y, zl=0):
        """
        Returns the lensing amplification factor, F(f, ml, y, zl=0),
        assuming geometric optics approximation.

        This function is dimensionful variant of Fw_geometric_optics().

        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter.

        Returns
        -------
        complex :
            Amplification factor F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.Fw_geometric_optics(w, y)

    def Ff_quasigeometric_optics(self, f, ml, y, zl=0):
        """
        Returns the lensing amplification factor, F(f, ml, y, zl=0),
        assuming Quasi-geometric optics approximation.

        This function is dimensionful variant of Ff_quasigeometric_optics().

        Parameters
        ----------
        w : float
           Dimensionless frequency.
        y : float
           The impact parameter.

        Returns
        -------
        complex :
            Amplification factor F(f, ml, y, zl).

        """

        w = self.w_of_f(f, ml, zl)
        return self.Fw_geometric_optics(w, y)
