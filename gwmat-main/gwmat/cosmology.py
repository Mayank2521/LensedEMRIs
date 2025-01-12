"""
This module contains the `Cosmology` class, which provides
essential functions for computing cosmological distances
(such as luminosity distance and comoving distance)
based on redshift and vice-versa.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import numpy as np
from scipy.integrate import quad

from .constants import C_SI


class Cosmology:
    """
    Contains basic functions to compute cosmological distances
    as functions of redshift, and vice-versa.

    """

    def __init__(
        self, omega_m=0.315, omega_lambda=0.685, omega_k=0.0, hubble_constant=69.8e3
    ):
        """
        Fixing cosmology.

        Parameters
        ----------
        omega_m : float, optional
            matter_density, by default 0.315
        omega_lambda : float, optional
            dark_energy_density, by default 0.685
        omega_k : float, optional
            determines curvature, by default 0.
        hubble_constant : float, optional
            Hubbles constant (ms^-1 Mpc^-1), by default 69.8e3

        """
        self.omega_m = omega_m  # matter_density
        self.omega_lambda = omega_lambda  # dark_energy_density
        self.omega_k = omega_k  # determines curvature
        self.omega_r = (
            1 - self.omega_m - self.omega_lambda - self.omega_k
        )  # radiation_energy_density
        self.hubble_constant = hubble_constant  # (ms^-1 Mpc^-1)
        self.hubble_distance = C_SI / self.hubble_constant  # Mpc

    def inverse_dimensionless_hubble_parameter(self, z):
        """
        Returns hubble_constant/H(z) = 1/E(z), inverse
        of the dimensionless Hubble_parameter.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            hubble_constant/H(z) = 1/E(z)

        """

        return self.hubble_distance * pow(
            self.omega_r * (1 + z) ** 4
            + self.omega_m * (1 + z) ** 3
            + self.omega_k * (1 + z) ** 2
            + self.omega_lambda,
            -1 / 2,
        )

    def compute_comoving_distance(self, z):
        """
        Returns the comoving distance
        as a function of redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            The comoving distance, in Mpc.

        """

        return quad(self.inverse_dimensionless_hubble_parameter, 0, z)[0]

    def compute_angular_diameter_distance(self, z):
        """
        Returns the angular diameter distance
        as a function of redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            The angular diameter distance, in Mpc.

        """

        return pow(1 + z, -1) * self.compute_comoving_distance(z)

    def angular_diameter_distance_between_redshifts(self, z1, z2):
        """
        Returns the angular diameter distance
        between two redshifts z1 and z2.

        Parameters
        ----------
        z1 : float
            First Redshift
        z2 : float
            Second Redshift

        Returns
        -------
        float
            The angular diameter distance between
            the two redshift, in Mpc.

        """

        return pow(1 + z2, -1) * (
            self.compute_comoving_distance(z2)
            * np.sqrt(
                1
                + self.omega_r
                * pow(self.compute_comoving_distance(z1) / self.hubble_distance, 2)
            )
            - self.compute_comoving_distance(z1)
            * np.sqrt(
                1
                + self.omega_r
                * pow(self.compute_comoving_distance(z2) / self.hubble_distance, 2)
            )
        )

    def compute_luminosity_distance(self, z):
        """
        Returns the luminosity distance
        as a function of redshift z.

        Parameters
        ----------
        z : float
            Redshift

        Returns
        -------
        float
            The luminosity distance, in Mpc.

        """

        return (1 + z) * self.compute_comoving_distance(z)

    def find_redshift_from_comoving_distance(self, dc):
        """
        Finds the redshift corresponding a given
        comoving distance using binary search.

        Parameters
        ----------
        dc : float
            The comoving distance, in Mpc.

        Returns
        -------
        float
            Redshift

        """

        z_min, z_max = 0, 10
        z = (z_min + z_max) / 2.0
        dist = self.compute_comoving_distance(z=z)
        while abs(1 - dc / dist) > 0.001:
            z = (z_min + z_max) / 2.0
            dist = self.compute_comoving_distance(z=z)
            if dist < dc:
                z_min = z
            else:
                z_max = z
        return z

    def find_redshift_from_luminosity_distance(self, luminosity_distance):
        """
        Finds the redshift corresponding a given
        luminosity distance using binary search.

        Parameters
        ----------
        luminosity_distance : float
            The luminosity distance, in Mpc.

        Returns
        -------
        float
            Redshift

        """

        z_min, z_max = 0, 10
        z = (z_min + z_max) / 2.0
        dist = self.compute_luminosity_distance(z=z)
        while abs(1 - luminosity_distance / dist) > 0.001:
            z = (z_min + z_max) / 2.0
            dist = self.compute_luminosity_distance(z=z)
            if dist < luminosity_distance:
                z_min = z
            else:
                z_max = z
        return z
