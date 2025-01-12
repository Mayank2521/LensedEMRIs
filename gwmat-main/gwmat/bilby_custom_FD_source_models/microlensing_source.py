"""
Author: Anuj Mishra

This module contains a Bilby source file designed for 
performing parameter estimation under the assumption 
of an isolated point lens model.

**Methodology for Efficient Lensing Amplification Factor Computation** :

1. **Precompute the Wave Optics Part**: A lookup table is generated to 
precompute the "wave optics" contribution.

2. **Geometrical Optics with Cython**: Cython is employed to compute 
the "geometrical optics" part efficiently using the power of C.

3. **Dynamic Frequency Cutoff**: A frequency cutoff, dependent on the 
impact parameter, shifts the computation from wave to geometrical optics. 
This cutoff is based on a numerical fit to ensure that the relative error 
between the analytic \( F(f) \) and the geometrical optics approximation 
is less than 1% for frequencies above the cutoff.

**Lookup Table** :
The lookup table must be generated before using the source function. A script to generate the table is available at `'gwmat/scripts/generate_point_lens_Ff_lookup_table.py'`. Once generated, the table can be provided to the sampler via the `"lookup_table_path"` keyword argument in **kwargs.

**Handling Lens Mass** :
The `point_lens_Ff_lookup_table` function, which uses the lookup table, can handle any lens mass, provided it falls within the impact parameter range used when building the table. Typically, a lens mass range of \( (10^{-1}, 10^5) \) is sufficient for most use cases.

**Non-Lookup Table Version** :
This module also includes a source model that does not rely on the lookup table, though it is slower due to the absence of precomputed values.
"""


from bilby.gw.source import lal_binary_black_hole
import numpy as np
from scipy.interpolate import interp1d

import gwmat


## functions
def y_w_grid_data(Ff_grid):
    ys_grid = np.array([Ff_grid[str(i)]["y"] for i in range(len(Ff_grid))])
    ws_grid = Ff_grid[str(np.argmin(ys_grid))]["ws"]
    return ys_grid, ws_grid


def y_index(yl, ys_grid):
    return np.argmin(np.abs(ys_grid - yl))


def w_index(w, ws_grid):
    return np.argmin(np.abs(ws_grid - w))


def point_lens_Ff_lookup_table(
    fs, Mlz, yl, Ff_grid, ys_grid, ws_grid, extrapolate=False
):
    wfs = np.array([gwmat.cythonized_point_lens.w_of_f(f, Mlz) for f in fs])

    if yl >= 1e-2:
        wc = gwmat.cythonized_point_lens.w_cutoff_geometric_optics_tolerance_1p0(yl)
    else:
        wc = np.max(wfs)

    wfs_1 = wfs[wfs <= np.min(ws_grid)]
    Ffs_1 = np.array([gwmat.cythonized_point_lens.Fw_analytic(w, y=yl) for w in wfs_1])

    wfs_2 = wfs[(wfs > np.min(ws_grid)) & (wfs <= np.max(ws_grid))]
    wfs_2_wave = wfs_2[wfs_2 <= wc]
    wfs_2_geo = wfs_2[wfs_2 > wc]

    i_y = y_index(yl, ys_grid)
    tmp_Ff_dict = Ff_grid[str(i_y)]
    ws = tmp_Ff_dict["ws"]
    Ffs = tmp_Ff_dict["Ffs_real"] + 1j * tmp_Ff_dict["Ffs_imag"]
    fill_val = ["interpolate", "extrapolate"][extrapolate]
    i_Ff = interp1d(ws, Ffs, fill_value=fill_val)
    Ffs_2_wave = i_Ff(wfs_2_wave)

    Ffs_2_geo = np.array(
        [gwmat.cythonized_point_lens.Fw_geometric_optics(w, yl) for w in wfs_2_geo]
    )

    wfs_3 = wfs[wfs > np.max(ws_grid)]
    Ffs_3 = np.array(
        [gwmat.cythonized_point_lens.Fw_geometric_optics(w, yl) for w in wfs_3]
    )

    Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))
    assert len(Ffs) == len(fs), "len(Ffs) = {} does not match len(fs) = {}".format(
        len(Ffs), len(fs)
    )
    return Ffs


# Define a global variable to hold the lookup table
LOOKUP_TABLE = None


def load_lookup_table(lookup_table_path):
    global LOOKUP_TABLE
    if LOOKUP_TABLE is None:
        print("## Loading and setting up the lookup table ##")
        import pickle

        with open(lookup_table_path, "rb") as f:
            Ff_grid = pickle.load(f)
        ys_grid, ws_grid = y_w_grid_data(Ff_grid)
        print("## Done ##")
        LOOKUP_TABLE = (Ff_grid, ys_grid, ws_grid)
    return LOOKUP_TABLE


### S1. Source Model for point lens microlensing using lookup table ###
def point_lens_microlensing_binary_black_hole(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    Log_Mlz,
    yl,
    **kwargs
):
    """
    This is a Bilby Frequency Domain Source model for 
    performing parameter estimations assuming 
    isolated point lens model.

    It returns microlensed frequency-domain plus and cross 
    polarized waveforms evaluated at the required 
    frequency_array.

    Parameters
    ----------
    * frequency_array : float
        Array of frequency values where the WF will be evaluated.
    * mass_1 : float
        The mass of the primary component object
        in the binary (in solar masses).
    * mass_2 : float
        The mass of the secondary component object in
        the binary (in solar masses).
    * a_1 : float, optional
        The dimensionless spin magnitude of object 1.
    * a_2 : float, optional
        The dimensionless spin magnitude of object 2.
    * tilt_1 : ({0.,float}), optional
        Angle between L and the spin magnitude of object 1.
    * tilt_2 : float, optional
        Angle between L and the spin magnitude of object 2.
    * phi_12 : float, optional
        Difference between the azimuthal angles of
        the spin of the object 1 and 2.
    * phi_jl : float, optional
        Azimuthal angle of L on its cone about J.
    * Log_Mlz : float
        Redshifted Mass of the point-lens in
        log10 scale (in solar masses).
    * yl : float
        The dimensionless impact parameter between
        the lens and the source.
    * theta_jn : float, optional
        Angle between the line of sight and
        the total angular momentum J.
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between
        the orbital angular momentum J(or, L) and the
        line-of-sight at the reference frequency.
        Default = 0.
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).

    Returns
    -------
    Dictionary:
        * plus: A numpy array.
            Strain values of the plus polarized WF in Frequency Domain..
        * cross: A numpy array.
            Strain values of the cross polarized WF in Frequency Domain.

    """
    waveform_kwargs = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=20.0,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        pn_spin_order=-1,
        pn_tidal_order=-1,
        pn_phase_order=-1,
        pn_amplitude_order=0,
        lookup_table_path="/home/anuj.mishra/git_repos/gwmat/data/lookup_table_data/\
        point_lens_Ff_lookup_table_Geo_relErr_1p0_Mlz_1e-1_1e5_ys_1e-3_5.pkl",
    )
    waveform_kwargs.update(kwargs)

    global LOOKUP_TABLE
    if LOOKUP_TABLE is None:
        LOOKUP_TABLE = load_lookup_table(
            lookup_table_path=waveform_kwargs["lookup_table_path"]
        )
    Ff_grid, ys_grid, ws_grid = LOOKUP_TABLE

    lal_res = lal_binary_black_hole(
        frequency_array,
        mass_1,
        mass_2,
        luminosity_distance,
        a_1,
        tilt_1,
        phi_12,
        a_2,
        tilt_2,
        phi_jl,
        theta_jn,
        phase,
        **waveform_kwargs
    )
    
    if Log_Mlz < -3:
        return dict(plus=lal_res["plus"], cross=lal_res["cross"])

    Mlz = np.power(10., Log_Mlz)
    Ff = point_lens_Ff_lookup_table(
        fs=frequency_array,
        Mlz=Mlz,
        yl=yl,
        Ff_grid=Ff_grid,
        ys_grid=ys_grid,
        ws_grid=ws_grid,
    )
    lhp = Ff * lal_res["plus"]
    lhc = Ff * lal_res["cross"]
    return dict(plus=lhp, cross=lhc)


### S2. Source Model for point lens microlensing using Analytic func + Geo Optics approx. (more accurate but slower; not recommended for PE) ###
def point_lens_microlensing_binary_black_hole_analytic(
    frequency_array,
    mass_1,
    mass_2,
    luminosity_distance,
    a_1,
    tilt_1,
    phi_12,
    a_2,
    tilt_2,
    phi_jl,
    theta_jn,
    phase,
    Log_Mlz,
    yl,
    **kwargs
):
    """
    This is similar to above source model 
    "point_lens_microlensing_binary_black_hole" but uses 
    analytic function to compute the wave optics part and 
    doesn't require a lookup table. 
    This is more accurate but slower, 
    hence not recommended for doing an extensive PE.

    """
    waveform_kwargs = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=20.0,
        maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False,
        pn_spin_order=-1,
        pn_tidal_order=-1,
        pn_phase_order=-1,
        pn_amplitude_order=0,
    )
    waveform_kwargs.update(kwargs)

    lal_res = lal_binary_black_hole(
        frequency_array,
        mass_1,
        mass_2,
        luminosity_distance,
        a_1,
        tilt_1,
        phi_12,
        a_2,
        tilt_2,
        phi_jl,
        theta_jn,
        phase,
        **waveform_kwargs
    )

    
    if Log_Mlz < -3:
        return dict(plus=lal_res["plus"], cross=lal_res["cross"])
    Mlz = np.power(10., Log_Mlz)
    Ff = np.array(
        [
            gwmat.cythonized_point_lens.Ff_effective(f, ml=Mlz, y=yl)
            for f in frequency_array
        ]
    )
    lhp = Ff * lal_res["plus"]
    lhc = Ff * lal_res["cross"]
    return dict(plus=lhp, cross=lhc)


