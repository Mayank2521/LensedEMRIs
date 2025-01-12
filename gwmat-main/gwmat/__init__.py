# gwmat/__init__.py

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]
__version__ = "1.0.0"

# Import the classes from the individual files
from .point_lens import PointLens
from .cosmology import Cosmology
from .domain import CBCParameterDomain
from .conversion import CBCParameterConversion
from .gw_utils import GWUtils
from .general_utils import GeneralUtils
from .injection import CBCInjection
from . import cythonized_point_lens
from . import methods
from . import bilby_custom_FD_source_models

# Initialize default instances of the classes
point_lens = PointLens()
cosmology = Cosmology()  # Default parameters
domain = CBCParameterDomain()
conversion = CBCParameterConversion()
gw_utils = GWUtils()
general_utils = GeneralUtils()
injection = CBCInjection()  # Default parameters


# Function to configure parameters
def configure_parameters(
    omega_m=0.315, omega_lambda=0.685, omega_k=0.0, hubble_constant=69.8e3
):
    global cosmology
    cosmology = Cosmology(
        omega_m=omega_m,
        omega_lambda=omega_lambda,
        omega_k=omega_k,
        hubble_constant=hubble_constant,
    )

# NRInjection is excluded from default import due to specific installation requirements
# from .nr_injection import NRInjection
# nr_injection = NRInjection()
