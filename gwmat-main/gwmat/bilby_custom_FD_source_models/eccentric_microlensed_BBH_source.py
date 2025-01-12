"""
Author: Anuj Mishra

A Bilby source file for performing parameter estimations assuming isolated point lens model in presence of eccentricity. This can be used, for example, to break the degeneracy between eccentricity and microlensing, as first shown by Anuj and Apratim, 2023 (in preparation).

For microlensing, we use GWMAT for an efficient computation of the amplification factor due to an isolated point lens. The methodology is breifly described as follows: 
1. Precompute the "wave optics" part in a lookup table. 
2. Use Cython to compute the "geometrical-optics" part with the power of C.   
3. Use a dynamic frequency cutoff depenidng on the impact parameter to shift from wave to geometrical optics. This cutoff uses a numerical fit such that the relative errors between the analytic F(f) and the one obtained from geometrical optics limit at any frequency above the cutoff is less than 1%.

Note: For step 1, it assumes a look up table has already been generated before using this source file. The script to generate the lookup table can be found at 'GWMAT/point_lens_Ff_lookup_table/pnt_Ff_lookup_table_gen.py'. Once generated, add the path to the lookup table in the object 'lookup_table_file' below.

For Eccentircity, we use GWEAT where TEOBResumS WF model is employed for generating eccentric IMR signals coming from BBH.

This Bilby source file is the combination of the source files in GWMAT and GWEAT.

"""


from bilby.gw.source import lal_binary_black_hole
import numpy as np
from scipy.interpolate import interp1d


###############################
## 1. Microlensing Computation
###############################

import gwmat

## functions
def y_w_grid_data(Ff_grid):
    ys_grid = np.array([Ff_grid[str(i)]['y'] for i in range(len(Ff_grid))])
    ws_grid = Ff_grid[str(np.argmin(ys_grid))]['ws']
    return ys_grid, ws_grid

def y_index(yl, ys_grid):
    return np.argmin(np.abs(ys_grid - yl))

def w_index(w, ws_grid):
    return np.argmin(np.abs(ws_grid - w))


def point_lens_Ff_lookup_table(fs, Mlz, yl, ys_grid, ws_grid, extrapolate=False):
    wfs = np.array([gwmat.cythonized_point_lens.w_of_f(f, Mlz) for f in fs])
    
    if yl >= 1e-2:
        wc = gwmat.cythonized_point_lens.wc_geo_re1p0(yl)
    else:
        wc = np.max(wfs)

    wfs_1 = wfs[wfs <= np.min(ws_grid)]
    Ffs_1 =  np.array([gwmat.cythonized_point_lens.point_Fw(w, y=yl) for w in wfs_1]) 

    wfs_2 = wfs[(wfs > np.min(ws_grid))&(wfs <= np.max(ws_grid))]
    wfs_2_wave = wfs_2[wfs_2 <= wc]
    wfs_2_geo = wfs_2[wfs_2 > wc]

    i_y  = y_index(yl, ys_grid)
    tmp_Ff_dict = Ff_grid[str(i_y)]
    ws = tmp_Ff_dict['ws']
    Ffs = tmp_Ff_dict['Ffs_real'] + 1j*tmp_Ff_dict['Ffs_imag']
    fill_val = ['interpolate', 'extrapolate'][extrapolate]
    i_Ff = interp1d(ws, Ffs, fill_value=fill_val)
    Ffs_2_wave = i_Ff(wfs_2_wave)

    Ffs_2_geo = np.array([gwmat.cythonized_point_lens.point_Fw_geo(w, yl) for w in wfs_2_geo])

    wfs_3 = wfs[wfs > np.max(ws_grid) ]
    Ffs_3 = np.array([gwmat.cythonized_point_lens.point_Fw_geo(w, yl) for w in wfs_3])

    Ffs = np.concatenate((Ffs_1, Ffs_2_wave, Ffs_2_geo, Ffs_3))
    assert len(Ffs)==len(fs), 'len(Ffs) = {} does not match len(fs) = {}'.format(len(Ffs), len(fs))
    return Ffs


# Define a global variable to hold the lookup table
LOOKUP_TABLE = None

def load_lookup_table(lookup_table_path):
    global LOOKUP_TABLE
    if LOOKUP_TABLE is None:        
        print('## Loading and setting up the lookup table ##')
        import pickle
        with open(lookup_table_path, 'rb') as f:
            Ff_grid = pickle.load(f)
        ys_grid, ws_grid = y_w_grid_data(Ff_grid) 
        print('## Done ##')
        LOOKUP_TABLE = (ys_grid, ws_grid)
    return LOOKUP_TABLE   


##############################################################################################
## 2.Utilizing above microlensing computation in eccentric signals generated using TEOBResumS
##############################################################################################

import pycbc
from pycbc.waveform import utils
from scipy.interpolate import interp1d
import sys
sys.path.append('/home/anuj/git_repos/gweat/src/')
sys.path.append('/home/anuj.mishra/git_repos/gweat/src/')
import TEOBResumS_utils as ecc_gen

def cyclic_time_shift_of_WF(wf, rwrap=0.2):
        """
        Inspired by PyCBC's function pycbc.types.TimeSeries.cyclic_time_shift(), 
        it shifts the data and timestamps in the time domain by a given number of seconds (rwrap). 
        Difference between this and PyCBCs function is that this function preserves the sample rate of the WFs while cyclically rotating, 
        but the time shift cannot be smaller than the intrinsic sample rate of the data, unlike PyCBc's function.
        To just change the time stamps, do ts.start_time += dt.
        Note that data will be cyclically rotated, so if you shift by 2 seconds, the final 2 seconds of your data will be patched to the initial part 
        of the signal, therevy also changing the start_time by -2 seconds.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            The waveform for cyclic rotation.
        rwrap : float, optional
            Amount of time to shift the vector. Default = 0.2.

        Returns
        -------
        pycbc.types.TimeSeries
            The time shifted time series.

        """        

        # This function does cyclic time shift of a WF.
        # It is similar to PYCBC's "cyclic_time_shift" except for the fact that it also preserves the Sample Rate of the original WF.
        if rwrap is not None and rwrap != 0:
            sn = abs(int(rwrap/wf.delta_t))     # number of elements to be shifted 
            cycles = int(sn/len(wf))

            cyclic_shifted_wf = wf.copy()

            sn_new = sn - int(cycles * len(wf))

            if rwrap > 0:
                epoch = wf.sample_times[0] - sn_new * wf.delta_t
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[-sn_new:]
                    tmp_wf_p2 = wf_arr[:-sn_new] 
                    shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)
            else:
                epoch = wf.sample_times[sn_new]
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[sn_new:] 
                    tmp_wf_p2 = wf_arr[:sn_new]
                    shft_wf_arr = np.concatenate(( tmp_wf_p1, tmp_wf_p2 ))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(shft_wf_arr, delta_t = wf.delta_t, epoch = epoch)  

            for i in range(cycles):        
                    epoch = epoch - np.sign(rwrap)*wf.duration
                    wf_arr = np.array(cyclic_shifted_wf)[:]
                    cyclic_shifted_wf = pycbc.types.TimeSeries(wf_arr, delta_t = wf.delta_t, epoch = epoch)

            assert len(cyclic_shifted_wf) == len(wf), 'Length mismatch: cyclic time shift added extra length to WF.'
            return cyclic_shifted_wf
        else:
            return wf
        
def determine_time_shift(wf):
    wf_dt = wf.sample_times[1] - wf.sample_times[0]
    wf_end_time = wf.sample_times[-1] + wf_dt
    peak_time = wf.sample_times[np.argmax(np.array(wf))]
    t_shift = wf_end_time - peak_time
    return t_shift     


# TD TEOBResumS WF generator
def eccentric_TEOBResumS_BBH_TD(mass_1, mass_2, luminosity_distance, chi_1z, chi_2z,
                          theta_jn, phase, ecc, **kwargs):
    """
    This is a wrapper function to call TOBResumS WF generator through the utils source file `TEOBResumS_utils.py`:
    https://gitlab.com/anuj-mishra/GWEAT/-/blob/main/src/TEOBResumS_utils.py?ref_type=heads
    It returns tapered TD plus and cross polarized WFs.

    Note: This is not a TD source model for Bilby. 

    Parameters
    ----------
    * mass_1 : float 
        The mass of the primary component object in the binary (in solar masses).
    * mass_2 : float 
        The mass of the secondary component object in the binary (in solar masses).
    * chi1z : float
        The z component of the first binary component. Default = 0.
    * chi2z : float
        The z component of the second binary component. Default = 0.            
    * ecc : float
        Eccentricity of the binary at a reference frequency of f_start. 
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between the orbital angular momentum J(or, L) and the
        line-of-sight at the reference frequency. Default = 0.              
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).
    * kwargs: dictionary
        * f_start : ({10., float}), optional 
            Reference frequency for defining eccentricity. This is also the starting frequency of the (2, 2) mode for waveform generation (in Hz). 
        * ecc_freq : ({2, int}), optional 
             Use periastron (0), average (1) or apastron (2) frequency for initial condition computation. Default = 2.                
        * sample_rate : ({4096, int}), optional 
            Sample rate for the TEOBResumS WF generation.
            It should be more than the "sampling-frequency" used by Bilby for PE. Ideally, twice the value works fine.                
        * mode_array : ({[[2,2]], 2D list}), optional 
            Mode array for the WF generation.
        * ode_abstol : ({1e-8, float}), optional 
            Absolute numerical tolerance for ODE compuations used in TEOBResumS. 
            Default=1e-8, which is more than the original default of 1e-13 for speeding up WF evaluations.
        * ode_reltol : ({1e-7, float}), optional 
            Relative numerical tolerance for ODE compuations used in TEOBResumS. 
            Default=1e-7, which is more than the original default of 1e-11 for speeding up WF evaluations.
    Returns
    -------
    Dictionary:
        * plus: A tapered PyCBC Timeseries object.
            The plus polarized WF.
        * cross: A tapered PyCBC Timeseries object.
            The cross polarized WF.

    """    
    
    waveform_kwargs = dict(
    f_start=10, sample_rate=4096, mode_array=[[2,2]],
    ecc_freq=2, ode_abstol=1e-8, ode_reltol=1e-7,
    )
    waveform_kwargs.update(kwargs)
    waveform_kwargs['mode_array'] = [[int(a[0]), int(a[1])] for a in waveform_kwargs['mode_array']]
    waveform_kwargs['modes_list'] = ecc_gen.modes_to_k( waveform_kwargs['mode_array'])
    #https://bitbucket.org/eob_ihes/teobresums/wiki/Conventions,%20parameters%20and%20output 
    pars = {
            'mass_1'             : mass_1,
            'mass_2'             : mass_2,
            'chi1z'              : chi_1z,
            'chi2z'              : chi_2z,
            'luminosity_distance': luminosity_distance,
            'inclination'        : theta_jn,  
            'coa_phase'          : phase,
            'use_mode_lm'        : waveform_kwargs['modes_list'],   #List of modes to use/output through EOBRunPy
            'output_lm'          : waveform_kwargs['modes_list'],   #List of modes to print on file
            'srate_interp'       : waveform_kwargs['sample_rate'],  #srate at which to interpolate. Default = 4096.
            'initial_frequency'  : waveform_kwargs['f_start'],      #in Hz if use_geometric_units = 0, else in geometric units
            'ecc'                : ecc,     #Eccentricity. Default = 0.
            'ecc_freq'           : waveform_kwargs['ecc_freq'],      #Use periastron (0), average (1) or apastron (2) frequency for initial condition computation. Default = 1
            'ode_abstol'         : waveform_kwargs['ode_abstol'],
            'ode_reltol'         : waveform_kwargs['ode_reltol']
           }
    
    pars.update(waveform_kwargs)
    wfs_res = ecc_gen.teobresums_td_pure_polarized_wf_gen(**pars)
    hp, hc = wfs_res['hp'], wfs_res['hc']

    # shifting the peak of the WF to t=0.
    wf = hp - 1j * hc
    wf = cyclic_time_shift_of_WF(wf, rwrap=determine_time_shift(wf)) 
    wf.start_time=0
    hp, hc = wf.real(), wf.imag()
    return dict(plus = hp, cross = hc)
    

# FD Microlensed TEOBResumS Source model for Bilby   
def eccentric_TEOBResumS_point_lens_MicL_BBH_FD(frequency_array, mass_1, mass_2, luminosity_distance, chi_1z, chi_2z,
                     theta_jn, phase, ecc, Log_Mlz, yl, **kwargs):
    """
    This is a Bilby Frequency Domain Source model for performing parameter estimations with TOBResumS waveforms, including eccentricity and microlensing.

    It returns FD plus and cross polarized WFs interpolated at the required frequency_array.

    The arguments in the kwargs can be updated through `waveform-arguments-dict` in the config ini file of bilby-pipe. For example, the default values are:
    waveform-arguments-dict = {f_start=10, ecc_freq=2, sample_rate=4096, mode_array=[[2,2]],  ode_abstol=1e-8, ode_reltol=1e-7}

    Parameters
    ----------
    * frequency_array : float 
        Array of frequency values where the WF will be evaluated.        
    * mass_1 : float 
        The mass of the primary component object in the binary (in solar masses).
    * mass_2 : float 
        The mass of the secondary component object in the binary (in solar masses).
    * chi1z : float
        The z component of the first binary component. Default = 0.
    * chi2z : float
        The z component of the second binary component. Default = 0.            
    * ecc : float
        Eccentricity of the binary at a reference frequency of f_start. 
    * Log_Mlz : float
        Redshifted Mass of the point-lens in log10 scale (in solar masses).
    * yl : float
        The dimensionless impact parameter between the lens and the source.          
    * luminosity_distance : ({100.,float}), optional
        Luminosity distance to the binary (in Mpc).
    * theta_jn : float
        Inclination (rad), defined as the angle between the orbital angular momentum J(or, L) and the
        line-of-sight at the reference frequency. Default = 0.              
    * phase : ({0.,float}), optional
        Coalesence phase of the binary (in rad).
    * kwargs: dictionary
        * f_start : ({10., float}), optional 
            Reference frequency for defining eccentricity. This is also the starting frequency of the (2, 2) mode for waveform generation (in Hz). 
        * ecc_freq : ({2, int}), optional 
             Use periastron (0), average (1) or apastron (2) frequency for initial condition computation. Default = 2.                
        * sample_rate : ({4096, int}), optional 
            Sample rate for the TEOBResumS WF generation.
            It should be more than the "sampling-frequency" used by Bilby for PE. Ideally, twice the value works fine.                
        * mode_array : ({[[2,2]], 2D list}), optional 
            Mode array for the WF generation.
        * ode_abstol : ({1e-8, float}), optional 
            Absolute numerical tolerance for ODE compuations used in TEOBResumS. 
            Default=1e-8, which is more than the original default of 1e-13 for speeding up WF evaluations.
        * ode_reltol : ({1e-7, float}), optional 
            Relative numerical tolerance for ODE compuations used in TEOBResumS. 
            Default=1e-7, which is more than the original default of 1e-11 for speeding up WF evaluations.

    Returns
    -------
    Dictionary:
        * plus: A numpy array.
            Strain values of the plus polarized WF in Frequency Domain..
        * cross: A numpy array.
            Strain values of the cross polarized WF in Frequency Domain.

    """       
    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomPv2', reference_frequency=20.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        f_start=10, sample_rate=4096, mode_array=[[2,2]],
        ecc_freq=2, ode_abstol=1e-8, ode_reltol=1e-7,
        lookup_table_path="/home/anuj.mishra/git_repos/gwmat/data/lookup_table_data/point_lens_Ff_lookup_table_Geo_relErr_1p0_Mlz_1e-1_1e5_ys_1e-3_5.pkl")
    waveform_kwargs.update(kwargs)
    waveform_kwargs['mode_array'] = [[int(a[0]), int(a[1])] for a in waveform_kwargs['mode_array']] # Bilby treats them as string so converting them to int manually.

    global LOOKUP_TABLE
    if LOOKUP_TABLE is None:
        LOOKUP_TABLE = load_lookup_table(lookup_table_path=waveform_kwargs["lookup_table_path"])
    ys_grid, ws_grid = LOOKUP_TABLE
    
    #print('\n######',waveform_kwargs,'######\n')   # for testing purpose
    wfs_res = eccentric_TEOBResumS_BBH_TD(mass_1, mass_2, luminosity_distance, chi_1z, chi_2z,
                          theta_jn, phase, ecc, **waveform_kwargs)

    df = frequency_array[1] - frequency_array[0]

    res = dict()
    for k in wfs_res.keys():
        wf = wfs_res[k]
        ## converting TD WF -> FD WF 
        fd_wf = wf.to_frequencyseries(delta_f=wf.delta_f)
        ## interpolating for given freqeuncy array
        fd_wf_arr = np.log10(np.array(fd_wf, dtype=np.complex128))
        ifd_wf = interp1d(fd_wf.sample_frequencies[:], fd_wf_arr[:], kind='linear')
        fd_wf_arr = np.concatenate(([0], 10**ifd_wf(frequency_array[1:])))
        #frequency_bounds = (frequency_array >=frequency_array['minimum_frequency']) * (frequency_array <= waveform_kwargs['maximum_frequency'])
        frequency_bounds = (frequency_array >=frequency_array[0]) * (frequency_array <= frequency_array[-1])
        fd_wf_arr *= frequency_bounds

        assert len(fd_wf_arr) == len(frequency_array), 'length mismatch between the required Bilby frequency array and TEOBResumS output'      
        
        # adding microlensing effects
        Mlz = np.power(10., Log_Mlz)
        if round(Mlz, 3) == 0:
            res[k] = fd_wf_arr 
        else:
            Ff = point_lens_Ff_lookup_table(fs=frequency_array, Mlz=Mlz, yl=yl, ys_grid=ys_grid, ws_grid=ws_grid)
            res[k] =  Ff*fd_wf_arr

    return res
























































