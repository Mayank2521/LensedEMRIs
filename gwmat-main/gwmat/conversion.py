"""
This module contains the class `CBCParameterConversion`, which provides
useful functions for converting parameters related to binary systems.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import numpy as np

class CBCParameterConversion:
    """
    Contains functions related to parameter conversions relevant for binaries.

    """

    def component_masses_to_chirp_mass(self, m1, m2):
        """
        Returns the chirp mass corresponding to the binary component masses.

        Parameters
        ----------
        m1 : float or np.ndarray
            Mass of the first component of the binary, in Msun.
        m2 : float or np.ndarray
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        float or np.ndarray
            Chirp mass, in Msun.

        """

        return (m1 * m2) ** (3 / 5.0) / (m1 + m2) ** (1 / 5.0)

    def component_masses_to_mass_ratio(self, m1, m2):
        """
        Returns the mass ratio of the binary component masses.
        Convention: q = m_secondary/m_primary < 1

        Parameters
        ----------
        m1 : float or np.ndarray
            Mass of the first component of the binary, in Msun.
        m2 : float or np.ndarray
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        float or np.ndarray
            Mass ratio.

        """

        return np.minimum(m1, m2) / np.maximum(m1, m2)

    def chirp_mass_and_mass_ratio_to_component_masses(self, m_chirp, q):
        """
        Converts a given chirp mass and mass ratio
        to the binary component masses.

        Parameters
        ----------
        m_chirp : float or np.ndarray
            Chirp mass of the binary.
        q : float or np.ndarray
            Mass ratio of the binary.

        Returns
        -------
        float, float or np.ndarray, np.ndarray
            The binary component masses

        """

        m2 = (m_chirp * (1.0 + q) ** (1 / 5.0)) / q ** (3 / 5.0)
        m1 = q * m2
        return m1, m2

    def component_masses_to_symmetric_mass_ratio(self, m1, m2):  # symmetric mass ratio
        """
        Returns the symmetric mass ratio for a given binary system.

        Parameters
        ----------
        m1 : float or np.ndarray
            Mass of the first component of the binary, in Msun.
        m2 : float or np.ndarray
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        float or np.ndarray
            The symmetric mass ratio.

        """

        return m1 * m2 / (m1 + m2) ** 2

    def component_masses_and_aligned_spins_to_chi_eff(
        self, m1, m2, sz1, sz2
    ):  # effective spin
        """
        Returns the effective spin as a function of the
        componenet masses and aligned spin vlaues.

        Parameters
        ----------
        m1 : float or np.ndarray
            Mass of the first component of the binary, in Msun.
        m2 : float or np.ndarray
            Mass of the second component of the binary, in Msun.
        sz1 : float or np.ndarray
            z-component of the spin of the first component of the binary.
        sz2 : float or np.ndarray
            z-component of the spin of the second component of the binary.

        Returns
        -------
        float or np.ndarray
            The effective spin (chi_eff) of the binary.

        """

        return (m1 * sz1 + m2 * sz2) / (m1 + m2)

    def mchirp_q_aligned_spins_to_chi_eff(self, m_chirp, q, sz1, sz2):
        """
        Returns the effective spin as a function of the
        chirp mass and mass ratio of the binary.

        Parameters
        ----------
        m_chirp : float or np.ndarray
            The chirp mass of the binary.
        q : float or np.ndarray
            The mass ratio of the binary.
        sz1 : float or np.ndarray
            z-component of the spin of the first component of the binary.
        sz2 : float or np.ndarray
            z-component of the spin of the second component of the binary.

        Returns
        -------
        float or np.ndarray
            The effective spin (chi_eff) of the binary.

        """

        m1, m2 = self.chirp_mass_and_mass_ratio_to_component_masses(m_chirp, q)
        return (m1 * sz1 + m2 * sz2) / (m1 + m2)

    def mchirp_and_eta_to_component_masses(self, m_chirp, eta):
        """
        Converts a given chirp mass and symmetric mass ratio
        of the binary to the individual componenet masses.

        Parameters
        ----------
        m_chirp : float or np.ndarray
            The chirp mass of the binary.
        eta : float
            The symmetric mass ratio of the binary (0< eta <=0.25).

        Returns
        -------
        float, float or np.ndarray
            Binary componenet masses

        """

        assert (
            eta <= 0.25
        ), f"Symmetric mass ratio should lie between 0 and 0.25, but the given eta was {eta}"
        mtot = m_chirp * eta ** (-3 / 5)
        m1 = 0.5 * mtot * (1 + (1 - 4 * eta) ** 0.5)
        m2 = 0.5 * mtot * (1 - (1 - 4 * eta) ** 0.5)
        return m1, m2

    def total_mass_and_mass_ratio_to_component_masses(self, mtot, q):
        """
        Converts a given total mass and mass ratio of the binary to the
        individual componenet masses. Supports array as well.

        Parameters
        ----------
        mtot : float or np.ndarray
            The total mass of the binary.
        q : float or np.ndarray
            The mass ratio of the binary.

        Returns
        -------
        float, float or np.ndarray, np.ndarray
            Binary componenet masses.

        """

        m1 = mtot / (1 + q)
        m2 = q * m1
        return np.maximum(m1, m2), np.minimum(m1, m2)

    def total_mass_and_mass_ratio_to_chirp_mass(self, mtot, q):
        """
        Converts a given total mass and mass ratio
        of the binary to its chirp mass.

        Parameters
        ----------
        mtot : float or np.ndarray
            The total mass of the binary.
        q : float or np.ndarray
            The mass ratio of the binary.

        Returns
        -------
        float or np.ndarray
            Chirp mass of the binary.

        """

        m1 = mtot / (1 + q)
        m2 = q * m1
        return self.component_masses_to_chirp_mass(m1, m2)

    def component_masses_to_total_mass_and_mass_ratio(self, m1, m2):
        """
        Converts a given binary componenet masses
        to its total mass and mass ratio.

        Parameters
        ----------
        m1 : float or np.ndarray
            Mass of the first component of the binary, in Msun.
        m2 : float or np.ndarray
            Mass of the second component of the binary, in Msun.

        Returns
        -------
        total mass : float or np.ndarray
            Total mass of the binary.
        mass ratio : float or np.ndarray
            Mass ratio of the binary.

        """

        q = np.minimum(m1, m2) / np.maximum(m1, m2)
        return m1 + m2, q

    def compute_chi_p(self, chi_1_perp, chi_2_perp, q):
        """
        Compute the effective precession spin parameter (chi_p)
        for a binary black hole system. Supports array as well.

        The chi_p parameter characterizes the contribution of spin
        to the orbital precession of a binary black hole system.
        It is computed from the components of the spin vectors
        perpendicular to the orbital angular momentum for the two
        black holes, taking into account the mass ratio of the system.
        References: https://arxiv.org/pdf/1408.1810,
        https://cplberry.com/2020/04/18/gw190412/

        Parameters
        ----------
        chi_1_perp : float or np.ndarray
            The component of the dimensionless spin vector
            perpendicular to the orbital angular
            momentum for the primary (heavier) black hole.

        chi_2_perp : float or np.ndarray
            The component of the dimensionless spin vector
            perpendicular to the orbital angular
            momentum for the secondary (lighter) black hole.

        q : float or np.ndarray
            The mass ratio of the black hole binary (0 < q <= 1).
            If q > 1, it is inverted (1/q) to
            ensure q <= 1 for consistency.

        Returns
        -------
        chi_p : float or np.ndarray
            The effective precession spin parameter,
            computed as the maximum of two components:
            - chi_1_perp (spin perpendicular to the orbital plane
            for the heavier black hole)
            - A weighted spin component for the lighter black hole,
            adjusted by the mass ratio.

        Notes
        -----
        The chi_p calculation uses the following steps:
        1. If the input mass ratio (q) is greater than 1,
        it is inverted to 1/q to ensure q <= 1.
        2. chi_p is then calculated as the maximum between:
           - chi_1_perp: the spin perpendicular to the orbital plane for the primary black hole.
           - chi_p_v2: a weighted combination of chi_2_perp and the mass ratio q, accounting
             for the influence of the lighter black hole's spin on precession.
        """
        q = np.minimum(q, 1.0 / q)
        chi_p_v1 = (
            chi_1_perp  # chi_1_perpendicular for the heavier (or primary) black hole.
        )
        chi_p_v2 = q * ((4 * q + 3) / (4 + 3 * q)) * chi_2_perp
        chi_p = np.maximum(chi_p_v1, chi_p_v2)
        return chi_p
