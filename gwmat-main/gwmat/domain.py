"""
This module contains the class `CBCParameterDomain`, which provides some
useful functions to wrap BBH parameters into their respeticve domains.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import numpy as np


class CBCParameterDomain:
    """
    Contains functions to wrap BBH parameters into their respeticve domains.
    These are mainly used during fitting factor computations.

    """

    def wrap_reflective(self, x, x1, x2):
        """
        Function to wrap and reflect a real number around the points x1 and x2.
        Example - For spins, we will have 'wrap_reflective(1.1, -1, 1) = 0.9',
        'wrap_reflective(-1.1, -1, 1) = -0.9', and so on.

        Parameters
        ----------
        x : float
            Value to be reflected.
        x1 : float
            The LHS reflective point.
        x2 : float
            The RHS reflective point.

        Returns
        -------
        float
            Wrapped and reflective value of x around x1 and x2.

        """

        if x2 is None:  # i.e., x2 can be unbounded
            x2 = (
                1e50  # assign a big number (but not np.inf as it will bias the algebra)
            )
        period = 2 * (x2 - x1)
        x_frac = (x - x1) % period
        if x_frac <= period / 2:
            x_ref = x_frac + x1
        else:  # x_frac > period / 2
            x_ref = (period - x_frac) + x1
        return x_ref

    # function to wrap a real number periodically around the points x1 and x2.
    # Ex- for a sine func, we will have 'periodic(2*np.pi + x, 0, 2*np.pi) = x'.
    def wrap_periodic(self, x, x1, x2):
        """
        Function to wrap a real number periodically around the points x1 and x2.
        Example - For spins, we will have 'wrap_periodic(2*np.pi + x, 0, 2*np.pi) = x'.

        Parameters
        ----------
        x : float
            Value to be reflected.
        x1 : float
            The LHS coordinate of boundary.
        x2 : float
            The RHS coordinate of boundary.

        Returns
        -------
        float
            Periodically wrapped value of x around x1 and x2.

        """

        period = x2 - x1
        x_frac = (x - x1) % period
        x_p = x_frac + x1
        return x_p

    # wraps x between (x1, x2) assuming boundaries to be either periodic or reflective.
    def wrap(self, x, x1, x2, boundary="reflective"):
        """
        Function to wrap a real number around the points
        x1 and x2 either periodically or reflectively.
        Example - (i) For spins, we will have
        'wrap(2*np.pi + x, 0, 2*np.pi, 'periodic') = x',
        'wrap(1.1, -1, 1, 'reflective') = 0.9',
        'wrap(-1.1, -1, 1, 'reflective') = -0.9'.

        Parameters
        ----------
        x : float
            Value to be reflected.
        x1 : float
            The LHS coordinate of boundary.
        x2 : float
            The RHS coordinate of boundary.
        boundary : {'reflective', 'periodic'}, optional.
            Boundary type to conisder while wrapping. Default = 'reflective'.
        Returns
        -------
        float
            Periodically wrapped value of x around x1 and x2.

        Raises
        ------
        KeyError
            Allowed keywords for boundary are: {'reflective', 'periodic'}

        """

        if boundary in ("reflective", "Reflective"):
            return self.wrap_reflective(x, x1, x2)

        if boundary in ("periodic", "Periodic"):
            return self.wrap_periodic(x, x1, x2)

        raise KeyError(
            "Incorrect keyword provided for the argument 'boundary'.\
            Allowed keywords are: {'reflective', 'periodic'}."
        )

    def check_spin_component_magnitude(self, a):
        """
        Checks if the dimensionless spin magnitude a <  0.998,
        otherwise assign a = 0.9.

        """

        if abs(round(a, 3)) > 0.998:
            return 0.9 * a / np.abs(a)

        return a

    def check_spin_component_domain(self, x):
        """
        Domain of an individual spin component:
        wrapping and reflection of real line around (-1, 1).

        """

        sp_ref = self.wrap(x, -1.0, 1.0, boundary="reflective")
        sp_ref = self.check_spin_component_magnitude(sp_ref)
        return sp_ref

    def check_total_spin_magnitude(self, sp):
        """
        Domain of three spin components: ensures that spin magnitude
        is less than one for a given set of three spin components.

        """

        try:
            assert (
                len(sp) == 3
            ), f"Spin should have three components [s_x, s_y, s_z], \
            but entered spin has length = {len(sp)} instead"
            sp = np.array(sp)
            a = np.linalg.norm([sp[0], sp[1], sp[2]])
            if a != 0:
                a_new = self.check_spin_component_magnitude(a)
                sp = a_new * sp / a
            return sp
        except TypeError:
            return self.check_spin_component_domain(sp)

    def check_spin_vector_domain(self, sp):
        """
        Final combined function for wrapping of spin values:
        can handle both 3-component and 1-component spin values.

        Parameters
        ----------
        sp : {float, list}
            Spin value(s).

        Returns
        -------
        {float, list}
            Wrapped spin value(s).

        """

        try:
            sp = list(map(self.check_spin_component_domain, sp))
            sp = self.check_total_spin_magnitude(sp)
        except TypeError:
            sp = self.check_spin_component_domain(sp)
        return sp

    def check_mass_domain(self, x, m_min=3.5, m_max=None):
        """
        Returns wrapped mass value(s): wrapping and reflection of
        real line around (3.2, 1e50), where 1e50 is used so that
        `m > m_min` is the only real restriction.

        Parameters
        ----------
        x : float
            Mass value to be wrapped within domain.
        m_min : float, optional
            Minimum mass to consider while wrapping. Default = 3.5.
        m_max : {None, float}, optional
            Maximum mass to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped Mass value within domain.

        """
        m_ref = self.wrap(x, m_min, m_max, boundary="reflective")
        return m_ref

    def check_chirp_mass_domain(self, x, cm_min=3.05, cm_max=None):
        """
        Returns wrapped Chirp Mass value(s):
        wrapping and reflection of real line around (3, 1e50),
        where 1e50 is a large enough number so that
        `CM > cm_min` is the only real restriction.

        Parameters
        ----------
        x : float
            Chirp mass value to be wrapped within domain.
        cm_min : float, optional
            Minimum Chirp mass to consider while wrapping.
            Default = 3.05 (because chirp_mass(m1=3.5, m2=3.5) ~ 3.05).
        cm_max : {None, float}, optional
            Maximum Chirp mass to consider while wrapping.
            Default = None(which uses 1e50 as a proxy).

        Returns
        -------
        float
            Wrapped Chirp Mass value within domain.

        """

        cm_ref = self.wrap(x, cm_min, cm_max, boundary="reflective")
        return cm_ref

    # domain of Mass Ratio values: wrapping and reflection of real line around (~0, 1).
    def check_mass_ratio_domain(self, x, q_min=1 / 18.0, q_max=1):
        """
        Returns wrapped mass ratio value(s):
        wrapping and reflection of real line around (~0, 1),
        as q = min(m1/m2, m2/m1) lies in (0, 1).

        Parameters
        ----------
        x : float
            Mass ratio value to be wrapped within domain.
        q_min : float, optional
            Minimum mass ratio to consider while wrapping. Default = 3.5.
        q_max : {None, float}, optional
            Maximum mass ratio to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped mass ratio value within domain.

        """

        x_wrap = self.wrap(x, q_min, q_max, boundary="reflective")
        return x_wrap

    # domain of Symmetric Mass Ratio values: wrapping and reflection of real line around (~0, 1/4).
    def check_symmetric_mass_ratio_domain(self, x, eta_min=0.05, eta_max=1 / 4.0):
        """
        Returns wrapped symmetric mass ratio value(s):
        wrapping and reflection of real line around (~0, 1/4.).

        Parameters
        ----------
        x : float
            Mass ratio value to be wrapped within domain.
        eta_min : float, optional
            Minimum symmetric mass ratio to consider while wrapping. Default = 3.5.
        eta_max : {None, float}, optional
            Maximum symmetric mass ratio to consider while wrapping. Default = None.

        Returns
        -------
        float
            Wrapped symmetric mass ratio value within domain.

        """

        x_wrap = self.wrap(x, eta_min, eta_max, boundary="reflective")
        return x_wrap
