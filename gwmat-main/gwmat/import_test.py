import pycbc
from pycbc.psd.analytical import *

print(pycbc.psd.analytical.get_psd_model_list())
print(pycbc.psd.analytical.get_pycbc_psd_list())

print(pycbc.psd.analytical_space.analytical_psd_lisa_tdi_AE(1000,1,1e-3))