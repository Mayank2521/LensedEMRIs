"""
This module contains important physical constants used throughout
the GWMAT package. These constants are defined in SI units and
can be utilized for various calculations.

Constants:
- G_SI: Gravitational constant in m^3/kg/s^2.
- C_SI: Speed of light in m/s.
- MSUN_SI: Solar mass in kg.
- DIST_PC: Distance of one parsec in meters.
- DIST_MPC: Distance of one megaparsec in meters.

References:
- Various scientific articles.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

# Useful Constants
G_SI = (
    6.67430 * 1e-11  # (m^3/Kg/s^2)
)  # ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9890581/
C_SI = 299792458  # (m/s) ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9890581/
MSUN_SI = (
    1.98847 * 1e30  # (Kg)
)  # ref: asa.usno.navy.mil: https://tinyurl.com/2c77mv4t
DIST_PC = 3.0856775714409184e16  # One parsec (in meters)
DIST_MPC = 3.0856775714409184e22  # One Mega-parsec (in meters)
