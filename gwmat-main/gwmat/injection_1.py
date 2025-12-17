"""
This module defines the `CBCInjection` class, which provides useful 
functions for generating injection frame files of gravitational wave 
signals from Compact Binary Coalescences (CBCs).
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import os
from copy import deepcopy

import numpy as np
from scipy.interpolate import interp1d
import pycbc
from pycbc import types
from pycbc.waveform import get_td_waveform, get_fd_waveform, taper_timeseries
from pycbc.detector import Detector
from pycbc.frame import frame
import pycbc.noise
import pycbc.psd

from few.amplitude.romannet import RomanAmplitude
from few.waveform import FastSchwarzschildEccentricFlux
from few.utils.utility import get_fundamental_frequencies
from few.utils.ylm import GetYlms
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from scipy import special 
import matplotlib.pyplot as plt
from pycbc.types import TimeSeries
from pycbc.types import FrequencySeries


from .point_lens import PointLens
from .gw_utils import GWUtils
from .general_utils import GeneralUtils

psd_directory_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "detector_PSDs/"
)


class CBCInjection1(PointLens, GWUtils, GeneralUtils):
    """
        This class contains functions for injecting gravitational wave (GW) signals from
        compact binary coalescences (CBCs), both microlensed and unlensed, with or without
        noise. It also provides functions for signal-to-noise ratio (SNR) computations,
        distance estimates from SNR, and related tasks.

        Parameters for generating simulated signals accept any key compatible with PyCBC.
        Lensing-specific parameters can be added when generating microlensed signals.
        Additional parameters may be useful for injections and saving them as frame files.

        While all PyCBC parameters are valid, the following dictionary templates provide
        the typical structure for generating injections:

        - `init_params`: Initialization parameters for waveform generation. This dictionary
          varies based on whether the waveform is generated in the time domain (TD) or
          frequency domain (FD).
        - `lens_params`: Parameters specific to microlensing.
        - `cbc_params`: Binary coalescence parameters, including options for both L-frame
          and J-frame coordinate systems.
        - `misc_params`: Additional settings for waveform generation, padding, tapering,
          and saving data.
        - `psd_params`: Parameters related to noise and PSD settings for detectors.

        **Example Params dictionary:**

        .. code-block:: python

            init_params = dict(wf_domain="FD", f_start=20, snr_f_min=20., snr_f_max=None,
            f_ref=20., delta_f=None, wf_approximant="IMRPhenomXO4a",
            ifo_list = ['H1', 'L1', 'V1'])  # when wf_domain="FD"

            init_params = dict(wf_domain="TD", f_start=20, snr_f_min=20., snr_f_max=None,
            f_ref=20., sample_rate=2048, delta_t=None, wf_approximant="IMRPhenomXO4a",
            ifo_list = ['H1', 'L1', 'V1'])  # when wf_domain="TD"

            lens_params = dict(m_lens=0., y_lens=5., z_lens=0.,
            lens_mass_lower_limit=1.e-3, Ff_data=None)

            cbc_params = dict(mass_1="user-defined", mass_2="user-defined", a_1=0, a_2=0,
            tilt_1=0, tilt_2=0, phi_jl=0, phi_12=0, theta_jn=0,
            luminosity_distance="user-defined", ra=0., dec=0.,
            polarization=0., coa_phase=0., trigger_time=0.)  # in J-frame.

            cbc_params = dict(mass_1="user-defined", mass_2="user-defined", spin1x=0,
            spin1y=0, spin1z=0, spin2x=0, spin2y=0, spin2z=0, inclination=0,
            luminosity_distance="user-defined", ra=0., dec=0.,
            polarization=0., coa_phase=0., trigger_time=0.)  # in L-Frame.

            misc_params = dict(rwrap=0, cyclic_time_shift_method="gwmat",
           taper_hp_hc=True, hp_hc_extra_padding_at_start=0,
           make_hp_hc_duration_power_of_2=True,
           extra_padding_at_start=1, extra_padding_at_end=2,
           save_data=False, data_outdir='./',
           data_label=None, data_channel='PyCBC-Injection')

            psd_params = dict(Noise=True, psd_H1="O4", psd_L1="O4", psd_V1="O4",
            noise_seed=127, is_asd_file=False, psd_f_low=None)

            params = {**init_params, **lens_params, **cbc_params, **psd_params, **misc_params}


        **CAUTION:**
        1. `snr_f_min` and `snr_f_max` control the frequency range for SNR computations
        and are not used during waveform generation. To set frequency bounds for
        waveform generation, use `f_start/f_lower` and `f_final` instead.\n
        2. Parameters should be provided in either the L-frame (used by LAL/PyCBC) or the
        J-frame (used by Bilby), but not both.

    Parameters
    ----------
    params : dict
        Dictionary of parameters for generating simulated signals.
        The dictionary can contain the following keys:

        **Initial Parameters:**

            * f_start : ({20., float}), optional
                For time-domain (TD) waveform models, `f_start` is the
                starting frequency of the (l=2, m=2) mode.
                For frequency-domain (FD) waveform models, it corresponds to the
                lowest frequency from where all available modes start. (in Hz)
                Example: To generate a waveform containing modes up to l=4 starting
                from 20Hz, set f_start <= 20 * (2/4) = 10 Hz for TD WF models,
                or set f_start <= 20 Hz for FD WF models.

            * snr_f_min : ({20., float}), optional
                Lower frequency cutoff for computing the SNR (in Hz).
                NOTE: This is different from `f_start`/`f_lower`, which sets the
                lower frequency for waveform generation.

            * snr_f_max : ({None., float}), optional
                Upper frequency cutoff for computing the SNR (in Hz).
                NOTE: This is different from `f_final`, which can be used to set
                the upper frequency for waveform generation.

            * f_ref : ({20., float}), optional
                Reference frequency (in Hz).

            * wf_domain : {"TD", "FD", "time-domain", "frequency-domain"}, optional
                Specifies the domain for waveform generation using PyCBC.
                - If set to "TD" or "time-domain", PyCBC's `get_td_waveform` is called.
                - If set to "FD" or "frequency-domain", PyCBC's `get_fd_waveform` is called.
                Default is "TD" because even for FD models like "IMRPhenomXO4a",
                the FD->TD conversion may produce spurious results, whereas TD->FD conversion
                gives reasonable outputs. Since simulated signals are primarily generated in
                the time domain, "TD" is the default.
                To explicitly generate in the frequency domain,
                set `wf_domain="FD"` or `"frequency-domain"`.

            * sample_rate : int, optional
                Sample rate for time-domain waveforms (`wf_domain="TD"` or "time-domain").
                The default is based on the total mass.

            * delta_t : float, optional
                Time spacing for time-domain waveforms.
                If provided, `sample_rate` is ignored. Otherwise, `delta_t=1/sample_rate`.

            * delta_f : float, optional
                Frequency spacing for frequency-domain waveforms
                (`wf_domain="FD"` or "frequency-domain").
                The default is based on the total mass.

            * wf_approximant : str, optional
                Name of the LAL waveform approximant to use. Default is "IMRPhenomXO4a".

            * ifo_list : list of strings, optional
                List of interferometers to consider. Default = ['H1', 'L1', 'V1'].

        **Microlensing Related Parameters:**

            * m_lens : ({0., float}), optional
                Point-lens mass (in solar masses).

            * y_lens : ({5., float}), optional
                Dimensionless impact parameter between the lens and the source.

            * z_lens : ({0., float}), optional
                Lens redshift. Default is 0.
                NOTE: When `z_lens=0`, `m_lens` represents the redshifted lens mass.

            * lens_mass_lower_limit : ({1.e-3, float}), optional
                Minimum allowed lens mass for computing microlensing effects.
                If `m_lens < 1.e-3`, microlensing effects are not computed.

            * Ff_data : float, optional
                Frequency-domain data to modify the waveform.
                If provided, `m_lens`, `y_lens`, and `z_lens` will be ignored.
                Default is None.

        **CBC Parameters:**

            * mass_1 : float
                Detector frame mass of the first component object
                in the binary (in solar masses).

            * mass_2 : float
                Detector frame mass of the second component object
                in the binary (in solar masses).

        (J-Frame Coordinates):

            * a_1 : float, optional
                Dimensionless spin magnitude of the first component. Default is 0.

            * a_2 : float, optional
                Dimensionless spin magnitude of the second component. Default is 0.

            * tilt_1 : ({0., float}), optional
                Angle between the orbital angular momentum (L) and the spin magnitude
                of object 1. Default is 0.

            * tilt_2 : float, optional
                Angle between the orbital angular momentum (L) and the spin magnitude
                of object 2. Default is 0.

            * phi_12 : float, optional
                Difference between the azimuthal angles of the spins of objects 1 and 2.
                Default is 0.

            * phi_jl : float, optional
                Azimuthal angle of L on its cone about J. Default is 0.

            * theta_jn : float, optional
                Angle between the line of sight and the total angular momentum (J).
                Default is 0.

        (L-Frame Coordinates):

            * inclination : float
                Inclination (in radians), defined as the angle between the
                orbital angular momentum (L) and the line of sight at the
                reference frequency. Default is 0.

            * spin1x : float
                X component of the spin for the first binary component. Default is 0.

            * spin1y : float
                Y component of the spin for the first binary component. Default is 0.

            * spin1z : float
                Z component of the spin for the first binary component. Default is 0.

            * spin2x : float
                X component of the spin for the second binary component. Default is 0.

            * spin2y : float
                Y component of the spin for the second binary component. Default is 0.

            * spin2z : float
                Z component of the spin for the second binary component. Default is 0.

        (Other Extrinsic Parameters):**

            * luminosity_distance : None, optional
                Luminosity distance to the binary (in Mpc).

            * ra : ({0., float}), optional
                Right ascension of the source (in radians).

            * dec : ({0., float}), optional
                Declination of the source (in radians).

            * polarization : ({0., float}), optional
                Polarization angle of the source (in radians).

            * coa_phase : ({0., float}), optional
                Coalescence phase of the binary (in radians).

            * trigger_time : ({0., float}), optional
                Trigger time of the gravitational wave event (in GPS time).


        **Noise Related Parameters:**

            * Noise : str, use one of the following {True, 'True', 'true'}
                to indicate whether to add noise to the projected signals.
                This boolean value specifies if noise should be included.

            * psd_H1, psd_L1, psd_V1 : str
                Path to the respective Power Spectral Density (PSD) files.
                Default is 'O4' (PSDs corresponding to O4 are pre-saved and
                linked to the keyword `O4`).

            * noise_seed : int
                Seed for generating random Gaussian noise.

            * is_asd_file : bool, optional
                Set it to True if the provided PSD files are actually ASD files.
                Default is False.

            * psd_f_low : float, optional
                Lower frequency for noise generation from PSD.
                Default is to use the value of "snr_f_min".

        **Miscellaneous Parameters:**

            * rwrap : ({0., float}), optional
                Cyclic time shift value (in seconds).

            * cyclic_time_shift_method : str, optional
                Method to use for cyclic time shift: "gwmat" or "pycbc".
                Default is "gwmat".

            * taper_hp_hc : bool, optional
                Indicates whether to taper the generated h_p and h_c.
                Default is True.

            * hp_hc_extra_padding_at_start : float, optional
                Duration of padding at the start of h_p and h_c.
                Default is 0 for frequency domain waveform models
                and 32 seconds for time-domain models.

            * make_hp_hc_duration_power_of_2 : bool, optional
                Indicates whether to make the duration
                of h_p and h_c a power of 2.
                Default is True.

            * extra_padding_at_start : ({1., float}), optional
                Duration of padding at the start of the
                projected interferometer signal.

            * extra_padding_at_end : ({2., float}), optional
                Duration of padding at the end of the
                projected interferometer signal.

            * save_data : str, use one of the following {True, 'True', 'true'}
                Indicates whether to save the data.
                This boolean value specifies if the detector data
                should be saved as a frame file.
                Default is False.

            * data_outdir : str
                Output directory where the data will be saved.
                Default is './', which corresponds to the current
                directory from which the command is run.

            * data_label : str
                Label to use while saving data. Default is 'data'.

            * data_channel : str
                Detector channel name (used while reading the data).
                Default is 'PyCBC_Injection'.
    """

    def __init__(self):
        super().__init__()
        self.params = None

    def initialize_params(self, **params):
        """
        Initialize and update default parameters.

        This function accepts a dictionary of parameters and updates
        the default parameters with the provided values. If no parameters
        are provided, the default parameters are returned as is.

        Parameters:
        ----------
        params : dict, optional
            A dictionary containing parameters to update the defaults.
            If not provided, the function will return the default parameters.

        Returns:
        -------
        dict
            The updated dictionary of parameters after applying the
            provided updates to the defaults.
        """
        params_default = {
            "snr_f_min": 20.0,  # this is used only for SNR computation.
            "snr_f_max": None,  # this is used only for SNR computation.
            "f_ref": 20.0,
            "sample_rate": None,
            "delta_t": None,
            "delta_f": None,
            "ifo_list": ["H1", "L1", "V1"],
            "m_lens": 0.0,
            "y_lens": 5.0,
            "z_lens": 0.0,
            "lens_mass_lower_limit": 1.0e-3,
            "Ff_data": None,
            "ra": 0.0,
            "dec": 0.0,
            "rwrap": 0.0,
            "wf_domain": "TD",
            "taper_hp_hc": True,
            "hp_hc_extra_padding_at_start": 0.0,
            "make_hp_hc_duration_power_of_2": True,
            "extra_padding_at_start": 1.0,
            "extra_padding_at_end": 2.0,
            "Noise": False,
            "psd_H1": "O4",
            "psd_L1": "O4",
            "psd_V1": "O4",
            "noise_seed": 127,  # just a default random value
            "is_asd_file": False,
            "psd_f_low": None,
            "save_data": False,
            "data_outdir": "./",
            "data_label": None,
            "data_channel": "PyCBC_Injection",
            "cyclic_time_shift_method": "gwmat",
        }
        params_default.update(params)
        params = params_default.copy()

        # Syncing parameter names to support multiple input formats,
        # such as Bilby, PyCBC, and LAL dictionaries, while also
        # adhering to the naming conventions defined in this class.
        key_pairs_with_defaults = [
            ("mass_1", "mass1", None),
            ("mass_2", "mass2", None),
            ("luminosity_distance", "distance", None),
            ("coa_phase", "phase", 0.0),
            ("polarization", "psi", 0.0),
            ("trigger_time", "geocent_time", 0.0),
            ("wf_approximant", "approximant", "IMRPhenomXO4a"),
            ("f_start", "f_lower", 20.0),
        ]
        for key1, key2, default in key_pairs_with_defaults:
            params = self.sync_keys_in_dict(key1, key2, default, **params)

        # convert from bilby params to lal params, which PyCBC uses, if needed.
        params = self.convert_jframe_params_to_lframe_if_present(**params)

        # set_default delta_t/delta_f and rwrap values.
        total_mass = params["mass_1"] + params["mass_2"]
        if params["wf_domain"] in ["FD", "frequency-domain"]:
            if params["rwrap"] is None:
                params["rwrap"] = -2.1
                # rwrap is chosen to be 2.1s, and not 2s, because:
                # (i) we don't expect a post-merger signal > 2s,
                # (ii) exact 2s of rwrap might cause different starting times
                # of the signal as trigger times of signals vary in each ifo.
            if params["delta_f"] is None:
                params["delta_f"] = self.set_delta_f_based_on_total_mass(total_mass)

        elif params["wf_domain"] in ["TD", "time-domain"]:
            if params["rwrap"] is None:
                params["rwrap"] = -0.1
                # this value usually works for all masses.
                # But still, be careful as some part of the inspiral can end up
                # being wrapped towards the end of the waveform.
            if params.get("delta_t") is None:
                if params.get("sample_rate") is None:
                    params["sample_rate"] = self.set_sample_rate_based_on_total_mass(
                        total_mass
                    )
                params["delta_t"] = 1.0 / params["sample_rate"]
                if not params.get("snr_f_max"):
                    params["snr_f_max"] = (1.0 / params["delta_t"]) // 2
        else:
            raise KeyError("`wf_domain` must be one of ['FD', 'TD'].")
        if not params.get("psd_f_low"):
            params["psd_f_low"] = params["snr_f_min"]
        return params

    def convert_jframe_params_to_lframe_if_present(self, **params):
        """
        Convert parameters from J-Frame to L-Frame if J-Frame parameters are present.

        This method checks the provided parameters for the presence of L-Frame and
        J-Frame keys. If only L-Frame keys are present, it ensures all required
        parameters have default values set to zero. If J-Frame parameters are provided,
        it converts them to L-Frame parameters using the appropriate conversion function
        and updates the input parameters with the converted values.

        Parameters:
        -----------
        **params : keyword arguments
            A dictionary of parameters that may contain keys related to either L-Frame
            or J-Frame parameters.

        Returns:
        --------
        dict
            A dictionary of parameters updated to include L-Frame values.

        Raises:
        -------
        ValueError
            If neither L-Frame nor J-Frame parameters are provided, or if both types
            of parameters are given.

        Notes:
        ------
        L-Frame keys include: ["spin1x", "spin1y", "spin1z", "spin2x",
        "spin2y", "spin2z", "inclination"].

        J-Frame keys include: ["a_1", "a_2", "tilt_1", "tilt_2",
        "phi_12", "phi_jl", "theta_jn"].

        """

        lframe_keys = [
            "spin1x",
            "spin1y",
            "spin1z",
            "spin2x",
            "spin2y",
            "spin2z",
            "inclination",
        ]
        jframe_keys = ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn"]

        # Check for the presence of keys from L-Frame or J-Frame in params
        lframe_present = any(k in params for k in lframe_keys)
        jframe_present = any(k in params for k in jframe_keys)

        # Set missing values to 0 for both frames and handle conversions
        if lframe_present and not jframe_present:
            for key in lframe_keys:
                params.setdefault(key, 0)
            return params

        if jframe_present and not lframe_present:
            for key in jframe_keys:
                params.setdefault(key, 0)

            # Convert J-Frame parameters to L-Frame
            lframe_dict = self.jframe_to_l0frame(
                mass_1=params["mass_1"],
                mass_2=params["mass_2"],
                f_ref=params["f_ref"],
                phi_ref=params["coa_phase"],
                theta_jn=params["theta_jn"],
                phi_jl=params["phi_jl"],
                a_1=params["a_1"],
                a_2=params["a_2"],
                tilt_1=params["tilt_1"],
                tilt_2=params["tilt_2"],
                phi_12=params["phi_12"],
            )
            params.update(lframe_dict)
            return params

        if not lframe_present and not jframe_present:
            print(
                "Warning: No L-Frame related keys were provided. Setting them to 0 by default."
            )
            for key in lframe_keys:
                params.setdefault(key, 0)
            return params

        # Raise an error if both frame parameters are present
        raise ValueError(
            "Provide parameter values in either the L-Frame or the J-Frame, not both."
        )

    def perform_cyclic_time_shift_on_hp_hc(self, hp, hc, rwrap, method="gwmat"):
        """
        Perform a cyclic time shift on the given gravitational waveforms (hp and hc).

        This function applies a cyclic time shift to the provided
        gravitational waveforms to avoid wrapping effects.
        The shifting can be done using different methods,
        specified by the `method` parameter.

        Parameters:
            hp: FrequencySeries or TimeSeries
                The gravitational waveform for h_plus (hp) to be shifted.

            hc: FrequencySeries or TimeSeries
                The gravitational waveform for h_cross (hc) to be shifted.

            rwrap: float
                The wrap-around time in seconds. This parameter determines
                how much time is shifted in the cyclic operation.

            method: str, optional
                The method used for cyclic time shifting. Options are:
                - "gwmat": Uses the gwmat library's cyclic time shift function.
                - "pycbc": Uses the PyCBC library's built-in cyclic time shift method.
                Default is "gwmat".

        Returns:
            tuple: A tuple containing the shifted waveforms (hp, hc).

        Raises:
            ValueError: If an invalid method is specified.
        """
        if method == "gwmat":
            hp = self.cyclic_time_shift_of_wf(hp, rwrap)
            hc = self.cyclic_time_shift_of_wf(hc, rwrap)
            return hp, hc

        if method == "pycbc":
            hp = hp.cyclic_time_shift(rwrap)
            hc = hc.cyclic_time_shift(rwrap)
            return hp, hc

        raise KeyError("argument `method` can only take values 'gwmat' or 'pycbc'.")

    def perform_tapering_on_hp_hc(self, hp, hc, tapermethod="TAPER_STARTEND"):
        """
        Apply tapering to the given gravitational waveforms using PyCBC.
    
        This function applies a tapering function to the `hp` (h_plus) and `hc` (h_cross)
        time series. The tapering method can be specified to control how the waveforms 
        are modified, which is useful for reducing artifacts at the edges of the waveforms.
    
        Parameters:
        ----------
        hp : FrequencySeries
            The frequency-domain h_plus waveform to be tapered.
        hc : FrequencySeries
            The frequency-domain h_cross waveform to be tapered.
        tapermethod : str, optional
            The tapering method to apply. Defaults to "TAPER_STARTEND".
            Possible values are (‘TAPER_NONE’, ‘TAPER_START’, ‘TAPER_END’,\
            ‘TAPER_STARTEND’, ‘start’, ‘end’, ‘startend’) 
            - NB ‘TAPER_NONE’ will not change the series!]
        Returns:
        -------
        hp : FrequencySeries
            The tapered h_plus waveform.
        hc : FrequencySeries
            The tapered h_cross waveform.
        """
        hp = taper_timeseries(hp, tapermethod=tapermethod, return_lal=False)
        hc = taper_timeseries(hc, tapermethod=tapermethod, return_lal=False)
        return hp, hc

    def pad_hp_hc_at_start(self, hp, hc, pad_length):
        """
        Pad the start of the gravitational waveforms with zeros.

        This function modifies the start time of the `hp` (h_plus) and `hc` (h_cross)
        waveforms by adding a specified length of padding at the beginning.
        This can be useful for ensuring that there is sufficient time before the
        start of the signal for processing or analysis.

        Parameters:
        ----------
        hp : TimeSeries
            The time-domain h_plus waveform to be padded.
        hc : TimeSeries
            The time-domain h_cross waveform to be padded.
        pad_length : float
            The length of time (in seconds) to pad at the start of the waveforms.

        Returns:
        -------
        hp : TimeSeries
            The padded h_plus waveform.
        hc : TimeSeries
            The padded h_cross waveform.
        """
        hp = self.modify_signal_start_time(hp, extra=pad_length)
        hc = self.modify_signal_start_time(hc, extra=pad_length)
        return hp, hc

    def process_td_wf_models_output(self, hp, hc, **params):
        """
        Post-process time-domain plus and cross polarizations (hp and hc)
        generated using PyCBC's `get_td_waveform` for time-domain waveform models.
        """
        if params["taper_hp_hc"]:
            hp, hc = self.perform_tapering_on_hp_hc(
                hp=hp,
                hc=hc,
                tapermethod="TAPER_START",  # no need to taper end in this case.
            )
        else:
            print(
                "WARNING: The provided waveform approximant is a\
 Time-Domain model. It is recommended that it be tapered,\
 by setting `taper_hp_hc=True`, before performing any operations."
            )
        if not params.get("hp_hc_extra_padding_at_start"):
            params["hp_hc_extra_padding_at_start"] = 32
            print(
                "WARNING: The provided waveform approximant is a\
 Time-Domain model. It is recommended that its duration\
 be increased, by setting a non-zero value of\
 `hp_hc_extra_padding_at_start`, before performing any\
 operations to increase resolution in the frequency-domain.\
 As a precaution, a value `hp_hc_extra_padding_at_start=32`\
 has been assigned by default."
            )

        hp, hc = self.pad_hp_hc_at_start(
            hp=hp,
            hc=hc,
            pad_length=params["hp_hc_extra_padding_at_start"],
        )

        if params["make_hp_hc_duration_power_of_2"]:
            hp = self.adjust_signal_length_to_power_of_2(hp)
            hc = self.adjust_signal_length_to_power_of_2(hc)
        return hp, hc

    def process_fd_wf_models_output(self, hp, hc, **params):
        """
        Post-process time-domain plus and cross polarizations (hp and hc)
        generated using PyCBC's `get_td_waveform` for frequency-domain waveform models .
        """
        # performing cyclic time shift to avoid wrapping up of WFs
        if params.get("rwrap"):
            hp, hc = self.perform_cyclic_time_shift_on_hp_hc(
                hp=hp,
                hc=hc,
                rwrap=params["rwrap"],
                method=params["cyclic_time_shift_method"],
            )
        # check whether to taper the WFs.
        if params["taper_hp_hc"]:
            hp, hc = self.perform_tapering_on_hp_hc(
                hp=hp,
                hc=hc,
                tapermethod="TAPER_STARTEND",
            )
        if params.get("hp_hc_extra_padding_at_start"):
            hp, hc = self.pad_hp_hc_at_start(
                hp=hp,
                hc=hc,
                pad_length=params["hp_hc_extra_padding_at_start"],
            )
        if params["make_hp_hc_duration_power_of_2"]:
            hp = self.adjust_signal_length_to_power_of_2(hp)
            hc = self.adjust_signal_length_to_power_of_2(hc)
        return hp, hc

    def process_get_td_waveform_object(self, hp, hc, **params):
        """
        Post-process time-domain plus and cross polarizations (hp and hc)
        generated using PyCBC's `get_td_waveform`.
        """
        # check if the WF approximant is a TD WF model.
        # If yes, first taper it and increase its duration.
        wf_approximants = self.gw_wf_approximants_in_lal()
        if (
            params["wf_approximant"] in wf_approximants["td_approximants"]
            and params["wf_approximant"] not in wf_approximants["fd_approximants"]
        ):
            hp, hc = self.process_td_wf_models_output(hp, hc, **params)
        else:  # if the WF is an FD model.
            hp, hc = self.process_fd_wf_models_output(hp, hc, **params)

        fd_hp, fd_hc = self.convert_td_hp_hc_to_fd_wfs(hp, hc)
        return hp, hc, fd_hp, fd_hc

    def process_get_fd_waveform_object(self, fd_hp, fd_hc, **params):
        """
        Post-process frequency-domain plus and cross polarizations
        (fd_hp and fd_hc) generated using PyCBC's `get_fd_waveform`.
        """
        # generating TD WFs from FD WFs
        hp, hc = self.convert_fd_hp_hc_to_td_wfs(fd_hp, fd_hc)

        # performing cyclic time shift to avoid wrapping up of WFs
        if params.get("rwrap"):
            hp, hc = self.perform_cyclic_time_shift_on_hp_hc(
                hp=hp,
                hc=hc,
                rwrap=params["rwrap"],
                method=params["cyclic_time_shift_method"],
            )

        # check whether to taper the WFs.
        if params["taper_hp_hc"]:
            hp, hc = self.perform_tapering_on_hp_hc(
                hp=hp,
                hc=hc,
                tapermethod="TAPER_START",
            )
        return hp, hc, fd_hp, fd_hc

    def convert_td_hp_hc_to_fd_wfs(self, td_hp, td_hc):
        """
        Convert time-domain plus and cross polarizations
        (hp and hc) to frequency-domain using PyCBC.
        """
        # generating FD WFs from TD WFs
        fd_hp = td_hp.to_frequencyseries(delta_f=td_hp.delta_f)
        fd_hc = td_hc.to_frequencyseries(delta_f=td_hc.delta_f)
        return fd_hp, fd_hc

    def convert_fd_hp_hc_to_td_wfs(self, fd_hp, fd_hc):
        """
        Convert frequency-domain plus and cross polarizations
        (hp and hc) to time-domain using PyCBC.
        """
        # generating TD WFs from FD WFs
        td_hp = fd_hp.to_timeseries(delta_t=fd_hp.delta_t)
        td_hc = fd_hc.to_timeseries(delta_t=fd_hc.delta_t)
        return td_hp, td_hc

    def generate_unlensed_waveforms(self, **params):
        """
        Generate unlensed time-domain and frequency-domain polarization modes using PyCBC.
        """
        # Generate waveforms in TD or FD
        if params["wf_domain"] == "TD":
            td_hp, td_hc = get_td_waveform({**params, "rwrap": 0.0})
            td_hp, td_hc, fd_hp, fd_hc = self.process_get_td_waveform_object(
                td_hp, td_hc, **params
            )
        else:
            fd_hp, fd_hc = get_fd_waveform({**params, "rwrap": 0.0})
            td_hp, td_hc, fd_hp, fd_hc = self.process_get_fd_waveform_object(
                fd_hp, fd_hc, **params
            )

        unlensed_waveforms = {
            "hp_TD_Unlensed": td_hp,
            "hc_TD_Unlensed": td_hc,
            "hp_FD_Unlensed": fd_hp,
            "hc_FD_Unlensed": fd_hc,
        }
        return unlensed_waveforms

    def generate_unlensed_emri_waveforms(self, **params):
        """
        Generating emri waveforms in time-domain and frequency-domain polarization modes using FEW.
        Parameters expected in params:
        - M: Primary mass (solar masses)
        - mu: Secondary mass (solar masses) 
        - p0: Initial semi-latus rectum
        - e0: Initial eccentricity
        - theta: Polar viewing angle
        - phi: Azimuthal viewing angle
        - dist: Distance (Gpc)
        - dt: Time step (seconds)
        - T: Duration (years)
        - mode_selection: List of (l, m, n) modes to include
        - use_gpu: Whether to use GPU acceleration
        """
        use_gpu = False

        inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
            }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
            "use_gpu": use_gpu  # GPU is available in this class
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {
            "use_gpu": use_gpu,  # GPU is availabel for this type of summation
            "pad_output": False,
        }

        amp = RomanAmplitude()
 
        M = 1e6
        mu = 5e1
        p0 = 10.0
        e0 = 0.4
        theta = np.pi/4.
        phi = np.pi/3.

        dist =1
        dt =10.0
        T = 2.0

        l_sel = 2
        m_sel = 2
        n_sel = 0

        specific_modes = [(l_sel, m_sel, n_sel)]
        
        ylm_gen = GetYlms(assume_positive_m=True, use_gpu=False)

        few_base = FastSchwarzschildEccentricFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
        )

        wave_22 = few_base(M, mu, p0, e0, theta, phi, dist=dist, dt=dt, T=T, mode_selection=specific_modes) # time domain waveform
        
        td_hp = TimeSeries(
            initial_array=np.real(wave_22),
            delta_t=dt,
            dtype=np.float64
        )
        td_hc = TimeSeries(
            initial_array=np.imag(wave_22),
            delta_t=dt,
            dtype=np.float64
        )

        fd_hp = td_hp.to_frequencyseries()
        fd_hc = td_hc.to_frequencyseries()

        unlensed_emri_waveforms = {
            "frequency_array": fd_hp.sample_frequencies,
            "hp_TD_Unlensed": td_hp,
            "hc_TD_Unlensed": td_hc, 
            "hp_FD_Unlensed": fd_hp,
            "hc_FD_Unlensed": fd_hc,
        }
        return unlensed_emri_waveforms

    def generate_unlensed_emri_waveforms_FD(self, **params):
        """
        UNDER PROCESS
        Generating emri waveforms in frequency-domain polarization modes using FEW.
        Parameters expected in params:
        - M: Primary mass (solar masses)
        - mu: Secondary mass (solar masses) 
        - p0: Initial semi-latus rectum
        - e0: Initial eccentricity
        - theta: Polar viewing angle
        - phi: Azimuthal viewing angle
        - dist: Distance (Gpc)
        - dt: Time step (seconds)
        - T: Duration (years)
        - mode_selection: List of (l, m, n) modes to include
        - use_gpu: Whether to use GPU acceleration
        """
        use_gpu = False

        inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
            }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
            "use_gpu": use_gpu  # GPU is available in this class
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {
            "use_gpu": use_gpu,  # GPU is availabel for this type of summation
            "pad_output": False,
        }

        amp = RomanAmplitude()
 
        M = 1e6
        mu = 5e1
        p0 = 10.0
        e0 = 0.4
        theta = np.pi/4.
        phi = np.pi/3.

        dist =1
        dt =10.0
        T = 2.0

        l_sel = 2
        m_sel = 2
        n_sel = 0

        specific_modes = [(l_sel, m_sel, n_sel)]
        
        ylm_gen = GetYlms(assume_positive_m=True, use_gpu=False)

        few_base = FastSchwarzschildEccentricFlux(
        inspiral_kwargs=inspiral_kwargs,
        amplitude_kwargs=amplitude_kwargs,
        Ylm_kwargs=Ylm_kwargs,
        sum_kwargs=sum_kwargs,
        use_gpu=use_gpu,
        )

        wave_22 = few_base(M, mu, p0, e0, theta, phi, dist=dist, dt=dt, T=T, mode_selection=specific_modes) # time domain waveform
        fft_wave = np.fft.fft(wave_22)*dt # frequency domain waveform from fft of time domain waveoform 
        freq_fft = np.fft.fftfreq(len(wave_22),dt)
        print(f"frequencies : {freq_fft},waveform : {fft_wave}")
        print(f"frequency size : {len(freq_fft)}, waveform size : {len(fft_wave)}")
        
        td_hp = TimeSeries(
            initial_array=np.real(wave_22),
            delta_t=dt,
            dtype=np.float64
        )
        td_hc = TimeSeries(
            initial_array=np.imag(wave_22),
            delta_t=dt,
            dtype=np.float64
        )
        df = 1.0 / (len(wave_22) * dt)

        fd_hp = FrequencySeries(
            initial_array=np.real(fft_wave),
            delta_f=df,
            dtype=np.float64
        )

        fd_hc = FrequencySeries(
            initial_array=np.imag(fft_wave),
            delta_f=df,
            dtype=np.float64
        )

        unlensed_emri_waveforms = {
            "frequency_array": freq_fft,
            "hp_TD_Unlensed": td_hp,
            "hc_TD_Unlensed": td_hc, 
            "hp_FD_Unlensed": fd_hp,
            "hc_FD_Unlensed": fd_hc,
        }
        return unlensed_emri_waveforms

    def modify_signal_start_time(self, wf, extra=1):
        """
        Function to modify the starting time of a signal so that it
        starts on an integer GPS time (in sec) + add extra length
        as specified by the user.

        Parameters
        ----------
        wf :  pycbc.types.TimeSeries
            WF whose length is to be modified.
        extra : int, optional
            Extra length to be added in the beginning after making
            the WF to start from an integer GPS time (in sec).
            Default = 1.

        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Modified waveform starting form an integer time.

        """

        sr = wf.sample_rate
        diff = wf.sample_times[0] - np.floor(wf.sample_times[0])
        dlen = round(sr * (extra + diff))
        wf_strain = np.concatenate((np.zeros(dlen), wf))
        t0 = wf.sample_times[0]
        dt = wf.delta_t
        n = dlen
        tnn = t0 - (n + 1) * dt
        wf_stime = np.concatenate(
            (np.arange(t0 - dt, tnn, -dt)[::-1], np.array(wf.sample_times))
        )
        nwf = pycbc.types.TimeSeries(wf_strain, delta_t=wf.delta_t, epoch=wf_stime[0])
        return nwf

    def modify_signal_end_time(self, wf, extra=2):
        """
        Function to modify the signal end time so that it ends
        on an integer GPS time (in sec) + add extra length as
        specified by the user.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            WF whose length is to be modified.
        extra : int, optional
            Extra length to be added towards the end after making the
            WF to end from an integer GPS time (in sec).
            Default = 2, which makes sure post-trigger duration
            is of at least 2 seconds.

        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Modified waveform ending on an integer time.

        """

        sr = wf.sample_rate
        olen = len(wf)
        dt = abs(wf.sample_times[-1] - wf.sample_times[-2])
        diff = np.ceil(wf.sample_times[-1]) - (
            wf.sample_times[-1] + dt
        )  # wf.sample_times[-1]-int(wf.sample_times[-1])
        nlen = round(olen + sr * (extra + diff))
        wf.resize(nlen)
        return wf

    def adjust_signal_length_to_power_of_2(self, wf):
        """
        Function to modify the length of a signal so that
        its duration is a power of 2.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            WF whose length is to be modified.
            Modified waveform with duration a power of 2.
        Returns
        -------
        pycbc.types.timeseries.TimeSeries
            Returns the waveform with length a power of 2.

        """

        dur = wf.duration
        wf.resize(int(round(wf.sample_rate * np.power(2, np.ceil(np.log2(dur))))))
        wf = self.cyclic_time_shift_of_wf(wf, rwrap=wf.duration - dur)
        return wf

    def _validate_waveforms(self, **unlensed_waveforms):
        """Validate if both hp and hc waveforms are provided."""
        if all(unlensed_waveforms.get(k) for k in ["hp_FD_Unlensed", "hc_FD_Unlensed"]):
            return (
                unlensed_waveforms["hp_FD_Unlensed"],
                unlensed_waveforms["hc_FD_Unlensed"],
            )

        raise KeyError(
            "Provide the unlensed frequency domain polarization modes\
        h_plus and h_cross, using keys ['hp_FD_Unlensed', 'hc_FD_Unlensed']."
        )

    def _compute_Ff_from_Ff_data(self, Ffs, fd_hp, fd_hc):
        """
        Compute the amplification factor array to be applied
        to fd_hp and fd_hc using the provided F(f) data.
        """
        if np.asarray(Ffs).size > 1 and np.asarray(Ffs[0]).size == 3:
            iFf = interp1d(Ffs[:, 0], Ffs[:, 1] + 1j * Ffs[:, 2], kind="linear")
            if np.allclose(fd_hp.sample_frequencies, fd_hc.sample_frequencies):
                fs = fd_hp.sample_frequencies
                Ff = np.asarray(list(map(iFf, fs)))
                return Ff, Ff

            print(
                "Warning: Different frequency arrays for hp and hc. Computing separately."
            )
            return np.asarray(list(map(iFf, fd_hp.sample_frequencies))), np.asarray(
                list(map(iFf, fd_hc.sample_frequencies))
            )

        raise ValueError(
            "Please provide 'Ff_data' in the correct format.\
            The data should be a three columned array\
            containing {frequencies, Re[F(f)], Im[F(f)]}."
        )

    def _compute_Ff_for_point_mass_microlensing(
        self, fd_hp, fd_hc, m_lens, y_lens, z_lens
    ):
        """
        Compute the amplification factor array to be applied to
        fd_hp and fd_hc using the point-mass model for microlensing.
        """
        if np.allclose(fd_hp.sample_frequencies, fd_hc.sample_frequencies):
            fs = fd_hp.sample_frequencies
            Ff = self.Ff_effective_map(fs=fs, ml=m_lens, y=y_lens, zl=z_lens)
            return Ff, Ff

        print(
            "Warning: Different frequency arrays for hp and hc. Computing separately."
        )
        Ff_hp = self.Ff_effective_map(
            fs=fd_hp.sample_frequencies, ml=m_lens, y=y_lens, zl=z_lens
        )
        Ff_hc = self.Ff_effective_map(
            fs=fd_hc.sample_frequencies, ml=m_lens, y=y_lens, zl=z_lens
        )
        return Ff_hp, Ff_hc

    def apply_lensing_amplification_factor(self, unlensed_waveforms, **params):
        """
        Apply lensing amplification factor on unlensed waveforms,
        either using F(f) data or point lens parameters.
        """
        fd_hp, fd_hc = self._validate_waveforms(**unlensed_waveforms)
        if params.get("Ff_data") is not None:
            Ff_hp, Ff_hc = self._compute_Ff_from_Ff_data(
                params["Ff_data"], fd_hp, fd_hc
            )
        else:
            m_lens, y_lens, z_lens = (
                params["m_lens"],
                params["y_lens"],
                params["z_lens"],
            )
            if m_lens < params["lens_mass_lower_limit"]:
                return unlensed_waveforms
            Ff_hp, Ff_hc = self._compute_Ff_for_point_mass_microlensing(
                fd_hp, fd_hc, m_lens, y_lens, z_lens
            )
        # Apply lensing to the waveforms
        print(f"amplification factor : {Ff_hc.shape} , {Ff_hp.shape}")
        lensed_fd_hp_array = Ff_hp * np.asarray(fd_hp)
        lensed_fd_hc_array = Ff_hc * np.asarray(fd_hc)
        print(f"Lensed_arrays : {lensed_fd_hp_array.shape, lensed_fd_hc_array.shape}")
        # Convert to PyCBC FrequencySeries object
        lensed_fd_hp = types.FrequencySeries(
            lensed_fd_hp_array, delta_f=fd_hp.delta_f, epoch=fd_hp.start_time
        )
        print(f"Lensed_Pycbc_Object_fd_hp : {lensed_fd_hp.shape}")
        lensed_fd_hc = types.FrequencySeries(
            lensed_fd_hc_array, delta_f=fd_hc.delta_f, epoch=fd_hc.start_time
        )
        print(f"Lensed_Pycbc_Object_fd_hp : {lensed_fd_hc.shape}")
        # Convert to corresponding TimeSeries object.
        lensed_td_hp, lensed_td_hc = self.convert_fd_hp_hc_to_td_wfs(
            lensed_fd_hp, lensed_fd_hc
        )

        # Apply cyclic time shift
        if params.get("rwrap"):
            lensed_td_hp, lensed_td_hc = self.perform_cyclic_time_shift_on_hp_hc(   
                hp=lensed_td_hp,
                hc=lensed_td_hc,
                rwrap=params["rwrap"],
                method=params["cyclic_time_shift_method"],
            )
        print(f"rwrap value : {self.params["rwrap"]}")
        print(f"lensed_td_hp : {lensed_td_hp}, lensed_fd_hp: {lensed_fd_hp}")
        print(f"lensed_td_hp shape: {lensed_td_hp.shape}, lensed_fd_hp shape: {lensed_fd_hp.shape}, unlensed_td_hp_shape: {unlensed_waveforms["hp_TD_Unlensed"].shape}, unlensed_fd_hp_shape: {unlensed_waveforms["hp_FD_Unlensed"].shape}")
        lensed_waveforms = {
            "hp_FD_Lensed": lensed_fd_hp,
            "hc_FD_Lensed": lensed_fd_hc,
            "hp_TD_Lensed": lensed_td_hp,
            "hc_TD_Lensed": lensed_td_hc,
        }

        return lensed_waveforms

    def generate_gw_polarizations_hp_hc(self, **params):
        """
        Function to generate h_plus and h_cross polarization modes.

        Parameters
        ----------
        params : dict
            Dictionary of parameters as described in the definition
            of this class. To quickly generate WFs with default settings,
            use:

            .. code-block:: python

                params = dict(mass_1=, mass_2=, luminosity_distance=)


        Returns
        -------
        Dictionary of:
        * unlensed_FD_WF_hp, unlensed_FD_WF_hc : pycbc.types.FrequencySeries
            Unlensed pure polarized frequency-domain (FD) waveforms.
        * unlensed_TD_WF_hp, unlensed_TD_WF_hp : pycbc.types.TimeSeries
            Unlensed pure polarized time-domain (TD) waveforms.
        * lensed_FD_WF_hp, lensed_FD_WF_hc : pycbc.types.FrequencySeries
            Lensed pure polarized frequency-domain (FD) waveforms.
        * lensed_TD_WF_hp, lensed_TD_WF_hc : pycbc.types.TimeSeries
            Lensed pure polarized time-domain (TD) waveforms.

        """
        if self.params is not None and self.check_if_dict_is_subset(
            params, self.params
        ):
            params = self.params.copy()
        else:
            params = self.initialize_params(**params)
            self.params = params

        unlensed_waveforms = self.generate_unlensed_waveforms(**params)
        print(unlensed_waveforms["hp_TD_Unlensed"].shape)
        lensed_waveforms = self.apply_lensing_amplification_factor(
            unlensed_waveforms, **params
        )
        print(lensed_waveforms["hp_TD_Lensed"].shape)
        return {**unlensed_waveforms, **lensed_waveforms}

    def generate_gw_polarizations_hp_hc_emri(self, **params):
        """
        Generate h_plus and h_cross polarization modes for EMRI waveforms.
        Parameters
        ----------

        
        Returns
        -------
        Dictionary of:
        * unlensed_FD_WF_hp, unlensed_FD_WF_hc : pycbc.types.FrequencySeries
            Unlensed pure polarized frequency-domain (FD) waveforms.
        * unlensed_TD_WF_hp, unlensed_TD_WF_hp : pycbc.types.TimeSeries
            Unlensed pure polarized time-domain (TD) waveforms.
        * lensed_FD_WF_hp, lensed_FD_WF_hc : pycbc.types.FrequencySeries
            Lensed pure polarized frequency-domain (FD) waveforms.
        * lensed_TD_WF_hp, lensed_TD_WF_hc : pycbc.types.TimeSeries
            Lensed pure polarized time-domain (TD) waveforms.
        """
        if self.params is not None and self.check_if_dict_is_subset(
            params, self.params
        ):
            params = self.params.copy()
        else:
            params = self.initialize_params(**params)
            self.params = params

        unlensed_waveforms = self.generate_unlensed_emri_waveforms()
        print("step1 done")
        print(type(unlensed_waveforms))
        print(unlensed_waveforms)
        print(f"Print unlensed waveform emri TD shape {unlensed_waveforms["hp_TD_Unlensed"].shape}")
        lensed_waveforms = self.apply_lensing_amplification_factor(
            unlensed_waveforms, **params
        )
        print("step2 done")
        print(f"Print lensed waveform emri TD shape {lensed_waveforms["hp_TD_Lensed"].shape}")
        return {**unlensed_waveforms, **lensed_waveforms}
    
    def simulate_zero_noise_injection(self, **params):
        """
        Simulated zero-noise (lensed-)injections.

        Parameters
        ----------
        params : dict
            Dictionary of parameters as described in the definition
            of this class. To quickly generate WFs with default settings,
            use:

            .. code-block:: python

                params = dict(mass_1=, mass_2=, luminosity_distance=)

        Returns
        -------
        Dictionary of :
        * pure_polarized_wfs : dict
            A dictionary containing plus and cross polarizaions of WF.
            keys = ['hp', 'hc'].
        * pure_ifo_signal : dict
            A dictionary containing projected WFs onto detectors without noise.
            keys = ifo_list.

        """

        # Choose a GPS end time, sky location, and polarization phase for the merger
        # NOTE: Right ascension and polarization phase runs from 0 to 2pi
        #       Declination runs from pi/2 to -pi/2 with the poles at pi/2 and -pi/2.

        # Check if either 'hp' or 'hc' is missing
        if any(params.get(key) is None for key in ("hp", "hc")):
            hp_hc_data = self.generate_gw_polarizations_hp_hc(**params)
            if self.params is not None:
                params = self.params.copy()

            # Assign hp and hc to unlensed by default
            hp = hp_hc_data["hp_TD_Unlensed"]
            hc = hp_hc_data["hc_TD_Unlensed"]

            # Overwrite with lensed versions if relevant conditions are met
            if all(hp_hc_data.get(key) for key in ("hp_TD_Lensed", "hc_TD_Lensed")):
                hp = hp_hc_data["hp_TD_Lensed"]
                hc = hp_hc_data["hc_TD_Lensed"]

        else:  # this allows passing hp/hc separately to this function.
            hp = params["hp"]
            hc = params["hc"]

        end_time = params["trigger_time"]
        hp.start_time += end_time
        hc.start_time += end_time

        # projection onto detectors
        det = {}
        ifo_signal = {}
        for ifo in params["ifo_list"]:
            det[ifo] = Detector(ifo)
            ifo_signal[ifo] = det[ifo].project_wave(
                hp, hc, params["ra"], params["dec"], params["polarization"]
            )
            ifo_signal[ifo] = taper_timeseries(
                ifo_signal[ifo], tapermethod="TAPER_STARTEND", return_lal=False
            )  # remove edge effects

        # We modify the length of the WF so that its time starts
        # and ends in integer seconds.
        # For this, we first make ends integer then add some extra
        # seconds towards the end and the start.
        for ifo in params["ifo_list"]:
            wf = deepcopy(ifo_signal[ifo])
            wf = self.modify_signal_start_time(
                wf, extra=params["extra_padding_at_start"]
            )
            wf = self.modify_signal_end_time(
                wf, extra=params["extra_padding_at_end"]
            )  # extra=1 => post-trigger duration is 2 seconds.
            wf = self.adjust_signal_length_to_power_of_2(
                wf
            )  # making total segment lenght a power of 2 by adding
            # zeros towards the start of the WF.
            ifo_signal[ifo] = wf

        res = {
            "pure_polarized_wfs": {"hp": hp, "hc": hc},
            "pure_ifo_signal": ifo_signal,
        }
        return res

    def replace_zeros_with_small_values(self, psd):
        """
        Function to replace zeros with rather small values in a given psd.

        """
        for i, val in enumerate(psd):
            if val == 0:
                psd[i] = 1e-52
        return psd

    # make psd_dict: duration, files, default=O4 psds
    def generate_psd(self, psd_file, psd_sample_rate=None, psd_duration=32, **params):
        """
        A tailored function to generate PSD.

        """

        # The PSD will be interpolated to the requested frequency spacing
        delta_f = 1.0 / psd_duration
        length = int(psd_sample_rate / delta_f)
        psd_file_f_low = np.loadtxt(psd_file)[0, 0]
        low_freq_cutoff = max(psd_file_f_low, params["psd_f_low"])
        psd = pycbc.psd.from_txt(
            psd_file,
            length,
            delta_f,
            low_freq_cutoff,
            is_asd_file=params["is_asd_file"],
        )
        psd = self.replace_zeros_with_small_values(psd)
        return psd

    # Adds noise to a WF
    def add_noise_to_gw_signal(
        self, wf, psd, noise_seed=127
    ):  # takes a pure timeseries WF as input
        """
        Function to add noise to a given GW signal based
        on the provided PSD and a realization seed.

        Parameters
        ----------
        wf : pycbc.types.TimeSeries
            _description_
        psd : _type_
            pycbc.types.timeseries.TimeSeries
        noise_seed : int, optional
            Seed value to generate noise realisation.
            Default = 127.

        Returns
        -------
        pycbc.types.TimeSeries
            WF with added noise.

        """

        # remove edge effects
        wf = taper_timeseries(wf, tapermethod="TAPER_STARTEND", return_lal=False)
        # generate a noise timeseries with duration equal to that of template
        delta_t = wf.delta_t
        t_samples = len(wf)
        ts = pycbc.noise.noise_from_psd(t_samples, delta_t, psd, seed=noise_seed)
        noisy_sig = types.TimeSeries(
            np.array(wf) + np.array(ts), delta_t=delta_t, epoch=wf.start_time
        )  # adding noise to the pure wf
        return noisy_sig

    def get_psd_file_dict(self, **params):
        """
        Retrieve the paths to Power Spectral Density (PSD) files
        for the given interferometers (IFOs).

        This function checks whether specific PSD files are provided
        for each IFO (H1, L1, V1).
        If a PSD file is not provided or set to "default" or "O4",
        it assigns the default O4 projected PSD file for that IFO
        from the `detector_PSDs` directory.
        Otherwise, it checks if the provided PSD file path exists,
        either relative to the `detector_PSDs` directory or
        as an absolute path.

        Parameters:
        -----------
        path_to_PSD_file : str
            Path to the directory containing the default
            PSD files for each interferometer.

        **params : dict**
            Keyword arguments including:

            - **ifo_list** : list of str
              List of interferometers, e.g., ["H1", "L1", "V1"].

            - **psd_H1** : str, optional
              Path to the PSD file for H1. Defaults to "aLIGO_O4_high_psd.txt" if not provided.

            - **psd_L1** : str, optional
              Path to the PSD file for L1. Defaults to "aLIGO_O4_high_psd.txt" if not provided.

            - **psd_V1** : str, optional
              Path to the PSD file for V1. Defaults to "aVirgo_O4_high_psd.txt" if not provided.

        Returns:
        --------
        psd_file_dict : dict
            Dictionary where keys are IFOs (e.g., "H1", "L1", "V1") and values are the full paths
            to the corresponding PSD files.

        Raises:
        -------
        ValueError
            If a specified PSD file cannot be found at the provided or default path.

        """

        psd_file_dict = {}

        for ifo in params.get("ifo_list", []):
            psd_key = f"psd_{ifo}"
            psd_value = params.get(psd_key)

            # Default to O4 projected PSD if not provided or is "default" or "O4"
            if not psd_value or psd_value in ["default", "O4"]:
                if not psd_value:
                    print(
                        f"No PSD file provided. Using the default O4 projected PSD for {ifo}."
                    )
                if ifo in ["H1", "L1"]:
                    psd_file_dict[ifo] = os.path.join(
                        psd_directory_path, "aLIGO_O4_high_psd.txt"
                    )
                elif ifo == "V1":
                    psd_file_dict[ifo] = os.path.join(
                        psd_directory_path, "aVirgo_O4_high_psd.txt"
                    )

            # If a specific PSD file is provided relative to the default PSD directory
            elif os.path.exists(os.path.join(psd_directory_path, psd_value)):
                psd_file_dict[ifo] = os.path.join(psd_directory_path, psd_value)

            # Use the provided PSD path if it exists
            elif os.path.exists(psd_value):
                psd_file_dict[ifo] = psd_value

            # If the provided file is not found
            else:
                raise ValueError(f"PSD file not found: {psd_value}")

        return psd_file_dict

    # simulated lensed noisy signal
    def simulate_noisy_injection(self, **params):
        """
        Simulated (lensed-)injections with added gaussian noise
        colored with the provided PSDs.

        Parameters
        ----------
        params : dict
            Dictionary of parameters as described in the
            definition of this class.
            To quickly generate WFs with added Noise
            and default settings, use:

            .. code-block:: python

                cbc_params = dict(mass_1=, mass_2=, luminosity_distance=)
                psd_params = dict(Noise=True, psd_H1="O4", psd_L1="O4", psd_V1="O4")
                params = {**cbc_params, **psd_params}


        Returns
        -------
        Dictionary of following keys:
            * pure_polarized_wfs : dict
                A dictionary containing plus and cross polarizaions of WF.
                Keys = ['hp', 'hc'].
            * pure_ifo_signal : dict
                A dictionary containing projected WFs onto detectors without noise.
                Keys = ifo_list
            * noisy_ifo_signal : dict
                A dictionary containing projected WFs onto detectors with added noise.
                Keys = ifo_list
            * psd : dict
                A dictionary containing generated PSDs in each detector.
                Keys = ifo_list

        """

        wfs_res = self.simulate_zero_noise_injection(**params)
        if self.params is not None:
            params = self.params.copy()

        pure_ifo_signal = wfs_res["pure_ifo_signal"]

        psd_file_dict = self.get_psd_file_dict(**params)
        psd = {}
        for ifo in params["ifo_list"]:
            if psd_file_dict[ifo] is not None:
                psd[ifo] = self.generate_psd(
                    psd_file=psd_file_dict[ifo],
                    psd_sample_rate=1.0 / pure_ifo_signal[ifo].delta_t,
                    psd_duration=pure_ifo_signal[ifo].duration,
                    **params,
                )
            else:
                psd[ifo] = None

        if (
            params["Noise"] is True
            or params["Noise"] == "True"
            or params["Noise"] == "true"
        ):
            noisy_ifo_signal = {}
            for ifo in params["ifo_list"]:
                n_sig = self.add_noise_to_gw_signal(
                    pure_ifo_signal[ifo], psd[ifo], noise_seed=params["noise_seed"]
                )
                noisy_ifo_signal[ifo] = deepcopy(n_sig)
        else:
            noisy_ifo_signal = pure_ifo_signal.copy()

        wfs_res.update(noisy_ifo_signal=noisy_ifo_signal, psd=psd)

        if (
            params["save_data"] is True
            or params["save_data"] == "True"
            or params["save_data"] == "true"
        ):
            for ifo in params["ifo_list"]:
                noisy_sig = wfs_res["noisy_ifo_signal"][ifo]
                data = pycbc.types.TimeSeries(
                    np.array(noisy_sig),
                    delta_t=1 / noisy_sig.sample_rate,
                    epoch=round(noisy_sig.sample_times[0]),
                )
                label = ifo[0] + "-" + ifo + "_" + params["data_channel"] + "-"
                start_time = float(data.sample_times[0])
                label += str(round(start_time)) + "-" + str(round(data.duration))
                if params["data_label"] is not None:
                    label += "_" + params["data_label"]

                print(f"Saving {ifo} data: {params['data_outdir'] + label + '.gwf'}")
                frame.write_frame(
                    params["data_outdir"] + label + ".gwf",
                    ifo + ":" + params["data_channel"],
                    data,
                )
        return wfs_res

    def simulate_injection_with_comprehensive_output(self, **params):
        """
        Function to simulate (lensed-)injections with complete data:
        [(polarization modes hp, hc), (pure zero-noise injected signals),
        (noisy injected signal), etc.], in addition to SNR information in each IFO.


        # Computing SNRs:
        # In general, since s(t) = h(t) + n(t),
        # matched_filter_SNR = (s|h_T) = (s|h)/sqrt(h|h),
        where h_T is normalized template of a waveform h.
        # optimal_matched_filter_SNR = (h|h_T) = (h|h)/sqrt(h|h)
        # Since we assume noise is gaussian, this MF_SNR is usually a gaussian
        with mean at the optimal SNR (because of the term (n|h) ).
        # Optimal SNR becomes important in cases of "very low SNRs",
        when (n|h) dominates over (h|h). In that case,
        # (n|h) dominates over (h|h) significantly thus biasing the MF_value.
        # However, optimal SNR still returns the correct value as it is only
        weighted by the psd, rather than the inner product with it (unlike MF).

        Parameters
        ----------
        params : dict
            Dictionary of parameters as described in the
            definition of this class.
            To quickly generate WFs with added Noise
            and default settings, use:

                .. code-block:: python

                    cbc_params = dict(mass_1=, mass_2=, luminosity_distance=)
                    psd_params = dict(Noise=True, psd_H1="O4", psd_L1="O4", psd_V1="O4")
                    params = {**cbc_params, **psd_params}

        Returns
        -------
        dict :
            Dictionary of following keys:
                * pure_polarized_wfs : dict
                    A dictionary containing plus and cross polarizaions of WF.
                    Keys = ['hp', 'hc'].
                * pure_ifo_signal : dict
                    A dictionary containing projected WF onto detector(s) without noise.
                    Keys = ifo_list.
                * noisy_ifo_signal : dict
                    A dictionary containing projected WF onto detector(s) with added noise.
                    Keys = ifo_list.
                * psd : dict
                    A dictionary containing generated PSD(s) in each detector.
                    Keys = ifo_list.
                * signal_templates : dict
                    A dictionary containing template(s) corresponding to the
                    projected signal(s), i.e., signals with peak at one end
                    of the WF (using cyclic shifting).
                    The peak of the template WF needs to be at its edge (start/end)
                    to recover the correct trigger time of the event.
                    Keys = ifo_list.
                * optimal_snr : dict
                    A dictionary containing the Optimal SNR of the injected signal(s).
                * match_filter_snr : dict
                    A dictionary containing the Matched-filter SNR of the injected signal(s).
                * network_optimal_snr : float
                    Network optimal SNR.
                * network_matched_filter_snr : float
                    Network Matched Filter SNR.

        """

        wfs_res = self.simulate_noisy_injection(**params)
        if self.params is not None:
            params = self.params.copy()

        signal_templates = {}
        match_filter_snr_timeseries = {}
        match_filter_snr = {}
        optimal_snr = {}
        for ifo in params["ifo_list"]:
            signal = deepcopy(wfs_res["pure_ifo_signal"][ifo])
            dt_end = (
                signal.sample_times[-1]
                - params["trigger_time"]
                + (signal.sample_times[1] - signal.sample_times[0])
            )
            signal_templates[ifo] = self.cyclic_time_shift_of_wf(signal, rwrap=dt_end)
            template, pure_data, noisy_data, psd = (
                signal_templates[ifo],
                wfs_res["pure_ifo_signal"][ifo],
                wfs_res["noisy_ifo_signal"][ifo],
                wfs_res["psd"][ifo],
            )
            mf_snr_ts = pycbc.filter.matchedfilter.matched_filter(
                template,
                noisy_data,
                psd=psd,
                low_frequency_cutoff=params["snr_f_min"],
                high_frequency_cutoff=params["snr_f_max"],
            )
            match_filter_snr_val = max(np.abs(mf_snr_ts))
            opt_snr_ts = pycbc.filter.matchedfilter.matched_filter(
                template,
                pure_data,
                psd=psd,
                low_frequency_cutoff=params["snr_f_min"],
                high_frequency_cutoff=params["snr_f_max"],
            )
            opt_snr_val = max(np.abs(opt_snr_ts))
            match_filter_snr_timeseries[ifo] = mf_snr_ts
            match_filter_snr[ifo] = match_filter_snr_val
            optimal_snr[ifo] = opt_snr_val

        network_optimal_snr = np.linalg.norm(
            np.array(list(optimal_snr.items()), dtype=object)[:, 1]
        )
        network_matched_filter_snr = np.linalg.norm(
            np.array(list(match_filter_snr.items()), dtype=object)[:, 1]
        )

        wfs_res.update(
            {
                "signal_templates": signal_templates,
                "matched_filter_snr": match_filter_snr,
                "optimal_snr": optimal_snr,
            }
        )
        wfs_res.update(
            {
                "network_optimal_snr": network_optimal_snr,
                "matched_filter_snr_timeseries": match_filter_snr_timeseries,
                "network_matched_filter_snr": network_matched_filter_snr,
            }
        )
        return wfs_res

    def network_optimal_snr_to_distance(self, net_optimal_snr, **params):
        """
        Converts a given net_opt_snr value to effective distance
        for a given set of binary and lens params.

        Parameters
        ----------
        net_optimal_snr : float
            Required Network optimal SNR of the signal.
        params : dict
            Dictionary of parameters to generate WFs as
            described in the definition of this class.

        Returns
        -------
        float :
            Distance corresponding to the provided Network optimal SNR.

        """

        if params.get("luminosity_distance") is None:
            params["luminosity_distance"] = 100  # a fiducial value
        params.update(save_data=False)  # just to deal with cases where it is true.
        wfs_res = self.simulate_injection_with_comprehensive_output(**params)
        req_dist = (
            params["luminosity_distance"]
            * wfs_res["network_optimal_snr"]
            / net_optimal_snr
        )
        return req_dist

    # converts a given net_matched_filter_snr value to effective distance
    # for a given set of binary and lens params.
    def network_matched_filter_snr_to_distance(self, net_mf_snr, threshold=1, **params):
        """
        Converts a given net_opt_snr value to effective distance
        for a given set of binary and lens params.

        Parameters
        ----------
        net_mf_snr : float
            Required Network matched-filter SNR of the signal.
        threshold : float
            Threshold on the relative error to reach before
            stop searching (in %). Default = 10.
        params : dict
            Dictionary of parameters to generate WFs as
            described in the definition of this class.

        Returns
        -------
        float :
            Distance corresponding to the provided
            Network matched-filter SNR.

        """

        params.update(save_data=False)  # just to deal with cases where it is true.
        dist_opt = self.network_optimal_snr_to_distance(net_mf_snr, **params)
        params["luminosity_distance"] = dist_opt
        wfs_res = self.simulate_injection_with_comprehensive_output(**params)
        tmp_net_mf_snr = wfs_res["network_matched_filter_snr"]

        # binary search around dist_opt
        dist_min, dist_max = dist_opt * np.array([0.5, 1.5])
        dist_avg = dist_opt
        k = 0
        rel_err = np.abs((tmp_net_mf_snr - net_mf_snr) / net_mf_snr)
        while rel_err > threshold / 100:
            k += 1
            # print("iter: ", k)
            rel_err = np.abs((tmp_net_mf_snr - net_mf_snr) / net_mf_snr)
            # print("relativr_error: ", rel_err)
            tmp_net_mf_snr_0 = tmp_net_mf_snr
            if tmp_net_mf_snr > net_mf_snr:
                dist_min = dist_avg
                dist_avg = (dist_min + dist_max) / 2
                params["luminosity_distance"] = dist_avg
                wfs_res = self.simulate_injection_with_comprehensive_output(**params)
                tmp_net_mf_snr = wfs_res["network_matched_filter_snr"]
            else:
                dist_max = dist_avg
                dist_avg = (dist_min + dist_max) / 2
                params["luminosity_distance"] = dist_avg
                wfs_res = self.simulate_injection_with_comprehensive_output(**params)
                tmp_net_mf_snr = wfs_res["network_matched_filter_snr"]
            del_mfr = np.abs(tmp_net_mf_snr_0 - tmp_net_mf_snr)
            if del_mfr < 1e-4:
                print(
                    f"Warning: Required matched filter value {net_mf_snr:.4f} is lower than\
 the minimum value {tmp_net_mf_snr:.4f} possible for given f_low.\
\nReturning distance correpsonding to the optimal SNR instead."
                )
                return dist_opt
        return dist_avg
