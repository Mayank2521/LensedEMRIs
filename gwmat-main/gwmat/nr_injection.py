"""
This module defines the `NRInjection` class, which provides functions 
for generating injection frame files from SXS Numerical Relativity (NR) 
simulations of gravitational wave signals from Compact Binary Coalescences (CBCs).

The class relies on the official SXS module and primarily utilizes `nr-catalog-tools`. 
For installation, the specific version of `nr-catalog-tools` used is:
https://github.com/anuj137/nr-catalog-tools/tree/bugfix_remove_junk.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

from glob import glob
from subprocess import call
import json
import pickle

import numpy as np
import sxs
from nrcatalogtools.sxs import SXSCatalog
import pandas as pd

import gwmat
from .constants import C_SI, G_SI, MSUN_SI


class NRInjection:
    """
    Class for generating injection frame files for SXS Numerical Relativity (NR)
    simulations of gravitational wave signals from
    Compact Binary Coalescences (CBCs).
    """

    def __init__(self):
        # Create an instance of classes to be used from gwmat
        self.cbc_injection = gwmat.injection
        self.gw_utils = gwmat.gw_utils
        self.general_utils = gwmat.general_utils
        self.conversion = gwmat.conversion
        # Initialize attributes specific to NRInjection
        self.sxs_cache_dir = str(sxs.sxs_directory("cache"))
        self._sxs_catalog = None
        self._nrcatalogtools_sxscatalog = None
        self.sxs_simulations_dataframe = None
        self.sxs_wf_modes = None
        self.solar_mass_to_seconds = G_SI * MSUN_SI / C_SI**3
        self.nr_inj_params = None

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
        nr_wf_gen_default_params = {
            "simulation_name": None,
            "simulation_id": None,
            "total_mass": None,
            "ra": 0.0,
            "dec": 0.0,
            "sample_rate": 16384,
            "delta_t": None,
            "f_ref": None,
            "t_ref": None,
            "k": 3,
            "kind": None,
            "tol": 1e-6,
            "remove_junk": True,
        }

        gwmat_related_default_params = {
            "ifo_list": ["H1", "L1", "V1"],
            "snr_f_min": 20,  # this is used only for SNR computation.
            "snr_f_max": None,  # this is used only for SNR computation.
            "f_ref": None,
            "rwrap": 0.0,
            "taper_hp_hc": True,
            "hp_hc_extra_padding_at_start": 8,
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
            "m_lens": 0.0,
            "y_lens": 5.0,
            "z_lens": 0.0,
            "lens_mass_lower_limit": 1e-3,
            "Ff_data": None,
        }
        params_default = {**nr_wf_gen_default_params, **gwmat_related_default_params}
        params_default.update(params)
        params = params_default.copy()

        if not params.get("total_mass"):
            raise ValueError("No value provided for `total_mass`.")

        if params.get("theta_jn") is not None:
            raise ValueError(
                "`theta_jn` is not a valid parameter for NR Injections.\
 Provide only `inclination` or `iota`."
            )
        if (
            params.get("simulation_name") is None
            and params.get("simulation_id") is None
        ):
            raise ValueError("Provide either `simulation_name` or `simulation_id`.")
        if params.get("simulation_name") is None:
            df = (
                self.sxs_simulations_dataframe
                if self.sxs_simulations_dataframe is not None
                else self.get_sxs_dataframe()
            )
            params["simulation_name"] = df.index[params.get("simulation_id")]
        else:
            pass

        self.sxs_wf_modes = self.nrcatalogtools_sxscatalog.get(
            params["simulation_name"]
        )

        # To deal with cases where inclination is provided as iota.
        params = self.general_utils.sync_keys_in_dict(
            "inclination", "iota", 0.0, **params
        )

        # get instrinsic parameters in L and J frames.
        useful_params = self.get_useful_parameters(**params)
        params.update(useful_params)

        ## Syncing parameter names to support multiple input formats,
        ## such as Bilby, PyCBC, and LAL dictionaries, while also
        ## adhering to the naming conventions defined in this class.
        key_pairs_with_defaults = [
            ("mass_1", "mass1", None),
            ("mass_2", "mass2", None),
            ("luminosity_distance", "distance", None),
            ("coa_phase", "phase", 0.0),
            ("polarization", "psi", 0.0),
            ("trigger_time", "geocent_time", 0.0),
            ("f_start", "f_lower", None),
        ]
        for key1, key2, default in key_pairs_with_defaults:
            params = self.general_utils.sync_keys_in_dict(key1, key2, default, **params)

        if params.get("delta_t") is None:
            if params.get("sample_rate") is None:
                params["sample_rate"] = (
                    self.gw_utils.set_sample_rate_based_on_total_mass(
                        params["total_mass"]
                    )
                )
            params["delta_t"] = 1.0 / params["sample_rate"]

        if not params.get("psd_f_low"):
            params["psd_f_low"] = params["snr_f_min"]
        return params

    @property
    def sxs_catalog(self):
        """Returns WaveformModes object with junks removed."""
        if self._sxs_catalog is None:
            self.load_sxs_catalog()
        return self._sxs_catalog

    @property
    def nrcatalogtools_sxscatalog(self):
        """Returns WaveformModes object with junks removed."""
        if self._nrcatalogtools_sxscatalog is None:
            self.load_sxs_catalog_through_nrcatalogtools()
        return self._nrcatalogtools_sxscatalog

    def determine_minimum_total_mass_for_a_given_f22_start(
        self, f_ref_orbital=None, f_start_22=10
    ):
        """
        Determine the minimum total binary mass for a given
        starting frequency and reference_orbital_frequency.
        """
        if f_ref_orbital is None:
            f_ref_orbital = np.linalg.norm(
                self.sxs_wf_modes.metadata.reference_orbital_frequency
            )
        return f_ref_orbital / np.pi / f_start_22 / self.solar_mass_to_seconds

    def load_sxs_catalog(self):
        """Loading SXS data if available, otherwise download before loading."""
        # Check if the SXS catalog file is available in the cache directory
        # If it exists, load the catalog.json file from the cache.
        sxs_catalog_path = glob(self.sxs_cache_dir + "/catalog.json")

        if sxs_catalog_path:
            print(f"Loading SXS data from cache directory: {sxs_catalog_path[0]}")
        else:
            # If the file is not found, download the catalog.json from the SXS website.
            print("Downloading SXS data")
            call(
                f"wget https://data.black-holes.org/catalog.json -P {self.sxs_cache_dir}",
                shell=True,
            )
            sxs_catalog_path = glob(self.sxs_cache_dir + "/catalog.json")

        self._sxs_catalog = sxs.load(location=sxs_catalog_path[0])
        return self._sxs_catalog

    def get_sxs_dataframe(
        self,
        enforce_mass_ratio_consistency=True,
        add_chi_p=True,
        add_minimum_total_mass_for_fstart20=True,
        add_minimum_total_mass_for_fstart13p33=True,
        add_minimum_total_mass_for_fstart10=True,
    ):
        """
        Get metadata of all the SXS simulations as DataFrame object.
        Additionaly, we add the information regarding chi_p and the
        minimum total mass corresponding to f_start={10, 13.33, 20} Hz.
        """
        df = self.sxs_catalog.simulations_dataframe

        if enforce_mass_ratio_consistency:
            for k in ["initial_mass_ratio", "reference_mass_ratio"]:
                qs = df[k].values
                qs = np.asarray([1.0 / q if q < 1 else q for q in qs])
                df[k] = qs

        if add_chi_p:
            df["reference_chi_p"] = self.conversion.compute_chi_p(
                chi_1_perp=df["reference_chi1_perp"],
                chi_2_perp=df["reference_chi2_perp"],
                q=df["reference_mass_ratio"],
            )
        if add_minimum_total_mass_for_fstart20:
            # Define a lambda function to apply to each row
            df["minimum_total_mass_for_fstart20"] = df[
                "reference_orbital_frequency"
            ].apply(
                lambda f_ref: self.determine_minimum_total_mass_for_a_given_f22_start(
                    f_ref_orbital=np.linalg.norm(f_ref), f_start_22=20
                )
            )
        if add_minimum_total_mass_for_fstart13p33:
            # Define a lambda function to apply to each row
            df["minimum_total_mass_for_fstart13p33"] = df[
                "reference_orbital_frequency"
            ].apply(
                lambda f_ref: self.determine_minimum_total_mass_for_a_given_f22_start(
                    f_ref_orbital=np.linalg.norm(f_ref), f_start_22=20 * 2 / 3
                )
            )
        if add_minimum_total_mass_for_fstart10:
            # Define a lambda function to apply to each row
            df["minimum_total_mass_for_fstart10"] = df[
                "reference_orbital_frequency"
            ].apply(
                lambda f_ref: self.determine_minimum_total_mass_for_a_given_f22_start(
                    f_ref_orbital=np.linalg.norm(f_ref), f_start_22=10
                )
            )
        self.sxs_simulations_dataframe = df
        return self.sxs_simulations_dataframe

    def get_useful_parameters(self, **params):
        """
        Retrieve all the relevant parameters, especially the intrinsic
        parameters in both the L and J frames,
        for a specific SXS NR injection.
        """
        params_in_l_frame = self.sxs_wf_modes.get_parameters(
            total_mass=params["total_mass"]
        )
        params_in_j_frame = self.gw_utils.l0frame_to_jframe(
            mass_1=params_in_l_frame["mass1"],
            mass_2=params_in_l_frame["mass2"],
            spin1x=params_in_l_frame["spin1x"],
            spin1y=params_in_l_frame["spin1y"],
            spin1z=params_in_l_frame["spin1z"],
            spin2x=params_in_l_frame["spin2x"],
            spin2y=params_in_l_frame["spin2y"],
            spin2z=params_in_l_frame["spin2z"],
            inclination=params["inclination"],
            f_ref=params_in_l_frame["f_lower"],
            phi_ref=0.0,
        )
        useful_params_dict = {**params_in_l_frame, **params_in_j_frame}
        sxs_metadata = self.sxs_wf_modes.metadata
        chirp_mass = self.conversion.component_masses_to_chirp_mass(
            params_in_l_frame["mass1"], params_in_l_frame["mass2"]
        )
        symmetric_mass_ratio = self.conversion.component_masses_to_symmetric_mass_ratio(
            params_in_l_frame["mass1"], params_in_l_frame["mass2"]
        )
        chi_p = self.conversion.compute_chi_p(
            chi_1_perp=sxs_metadata["reference_chi1_perp"],
            chi_2_perp=sxs_metadata["reference_chi2_perp"],
            q=sxs_metadata["reference_mass_ratio"],
        )
        useful_params_dict.update(
            chirp_mass=chirp_mass,
            mass_ratio=sxs_metadata["reference_mass_ratio"],
            symmetric_mass_ratio=symmetric_mass_ratio,
            total_mass=params["total_mass"],
            chi_eff=sxs_metadata["reference_chi_eff"],
            chi_p=chi_p,
            eccentricity=sxs_metadata["reference_eccentricity"],
        )
        return useful_params_dict

    def load_sxs_catalog_through_nrcatalogtools(self):
        """
        Load SXS catalog through NRCatalogtools
        and save it as pickle for future use.
        """
        # Define the path to the nrcatalogtools SXSCatalog
        # pickle file in the cache directory
        nrcatalogtools_sxscatalog_path = glob(
            self.sxs_cache_dir + "/nrcatalogtools_sxscatalog.pkl"
        )
        # If the nrcatalogtools.sxs.SXSCatalog object is not saved in the cache,
        # create it from the catalog.json
        if len(nrcatalogtools_sxscatalog_path) == 0:
            print(
                "Loading SXS catalog through `nrcatalogtools.sxs.SXSCatalog`.\
                This will take some time."
            )
            # Load the catalog.json data.
            with open(self.sxs_cache_dir + "/catalog.json", "r", encoding="utf-8") as f:
                sxs_cat_json = json.load(f)

            # Create the SXSCatalog object using the loaded JSON data
            self._nrcatalogtools_sxscatalog = SXSCatalog(catalog=sxs_cat_json)

            # Save the SXSCatalog object to a pickle file
            # in the cache directory for future use
            with open(nrcatalogtools_sxscatalog_path[0], "wb") as f:
                pickle.dump(self._nrcatalogtools_sxscatalog, f)

        # If the SXSCatalog object is already saved in the cache
        # (i.e., the pickle file exists), load the object from
        # the cache to avoid recomputing it.
        else:
            print(
                f"Loading the `nrcatalogtools.sxs.SXSCatalog` object\
 from cache directory: {nrcatalogtools_sxscatalog_path[0]}"
            )
            with open(nrcatalogtools_sxscatalog_path[0], "rb") as f:
                self._nrcatalogtools_sxscatalog = pickle.load(f)

        return self._nrcatalogtools_sxscatalog

    def generate_nr_gw_polarizations_hp_hc(self, **params):
        """
        Generate gravitational wave polarizations (h_plus and h_cross)
        for a given SXS Numerical Relativity (NR) simulation.

        This function retrieves the polarization modes based on the
        specified parameters (total_mass and other extrinsic parameters
        required for generating h_plus and h_cross) and generates the
        time-domain waveforms for the polarizations.
        It allows for various adjustments, such as tapering of the
        waveforms and moving the peak of the waveforms to time zero.

        Parameters:
            **params: dict
                Additional parameters for waveform generation.
                
                - **simulation_name (str)** : Name of the simulation.
                
                - **simulation_id (int)** : ID of the simulation name\
                (required if simulation_name is not provided).
                
                - **total_mass (float)** : Total mass of the binary system\
                (in solar masses).
                
                - **distance (float)** : Distance to the source (in Mpc).
                
                - **inclination (float)** : Inclination angle (in radians, default is 0).
                
                - **coa_phase (float)** : Coalescence phase (in radians, default is 0).
                
                - **delta_t (float)** : Time step between samples\
                (default is 1 / 16384 seconds).
                
                - **sample_rate (float)** : Sample rate for waveform generation\
                (overrides delta_t if provided).
                
                - **f_ref (float)** : Reference frequency (default is None).
                
                - **t_ref (float)** : Reference time (default is None).
                
                - **k (int)** : Parameter for waveform generation (default is 3).
                
                - **kind (str)** : Type of waveform to generate (default is None).
                
                - **tol (float)** : Tolerance for waveform generation (default is 1e-6).
                
                - **remove_junk (bool)** : Flag to remove junk radiation (default is True).
                
                - **taper_hp_hc (bool)** : Flag to apply tapering to hp and hc waveforms.\
                Default is True.
                
                - **hp_hc_extra_padding_at_start** : float, optional
                Duration of padding at the start of h_p and h_c.
                Default is 8 seconds.
                
                - **make_hp_hc_duration_power_of_2** : bool, optional
                Indicates whether to make the duration
                of h_p and h_c a power of 2.
                Default is True.

        Returns:
            dict: dict
                A dictionary containing the time-domain gravitational wave polarizations:

                - **hp** : Time-domain waveform for h_plus (hp).
                
                - **hc** : Time-domain waveform for h_cross (hc).

        Raises:
            ValueError: If neither `simulation_name` nor `simulation_id` is provided.

        Notes:
            It is assumed that the `sxs_simulations_dataframe` and
            `nrcatalogtools_sxscatalog` have been appropriately initialized
            prior to calling this function.
        """
        if (
            self.nr_inj_params is not None
            and self.general_utils.check_if_dict_is_subset(params, self.nr_inj_params)
        ):
            params = self.nr_inj_params.copy()
        else:
            params = self.initialize_params(**params)
            self.nr_inj_params = params.copy()

        print(
            f"Starting frequency of the (l=2, m=2) waveform mode = {params['f_start']:.2f} Hz"
        )

        pols = self.sxs_wf_modes.get_td_waveform(
            total_mass=params["total_mass"],
            distance=params["distance"],
            inclination=params["inclination"],
            coa_phase=params["coa_phase"],
            delta_t=params["delta_t"],
            f_ref=params["f_ref"],
            t_ref=params["t_ref"],
            k=params["k"],
            kind=params["kind"],
            tol=params["tol"],
            remove_junk=params["remove_junk"],
        )

        hp, hc = pols.real(), -1 * pols.imag()

        if params["taper_hp_hc"]:
            hp, hc = self.cbc_injection.perform_tapering_on_hp_hc(
                hp=hp,
                hc=hc,
                tapermethod="TAPER_START",  # no need to taper end in this case.
            )
        else:
            print(
                "WARNING: It is recommended that NR waveforms be tapered,\
                by setting `taper_hp_hc=True`, before performing any operations."
            )

        hp, hc = self.cbc_injection.pad_hp_hc_at_start(
            hp=hp,
            hc=hc,
            pad_length=params["hp_hc_extra_padding_at_start"],
        )

        if params["make_hp_hc_duration_power_of_2"]:
            hp = self.cbc_injection.adjust_signal_length_to_power_of_2(hp)
            hc = self.cbc_injection.adjust_signal_length_to_power_of_2(hc)

        # moving the peak to t=0, similar to PyCBC convention.
        time_at_peak = hp.sample_times[np.argmax(abs(hp))]
        hp.start_time += -time_at_peak
        hc.start_time += -time_at_peak

        # add lensing effects, if any. Default is None.
        # if params.get("Ff_data") or params["m_lens"] > params["lens_mass_lower_limit"]:
        hp, hc = self.apply_lensing_effects_if_any(hp=hp, hc=hc, **params)
        return {"hp": hp, "hc": hc}

    def apply_lensing_effects_if_any(self, hp, hc, **params):
        """
        Apply lensing effects to gravitational wave polarizations, if specified.

        Parameters
        ----------
        hp : array_like
            The plus polarization waveform in the time domain.
        hc : array_like
            The cross polarization waveform in the time domain.
        params : dict
            Additional parameters including lens parameters.

        Returns
        -------
        hp, hc : array_like
            The potentially lensed gravitational wave polarizations in the time domain.
        """
        # Return the passed hp and hc if lensing parameters are not provided.
        if (
            params.get("Ff_data") is None
            and params["m_lens"] < params["lens_mass_lower_limit"]
        ):
            return hp, hc

        print("Adding lensing effects to NR waveforms.")
        # Convert time-domain waveforms to frequency-domain
        fd_hp, fd_hc = self.cbc_injection.convert_td_hp_hc_to_fd_wfs(hp, hc)

        # Store the original frequency-domain waveforms
        unlensed_waveforms = {"hp_FD_Unlensed": fd_hp, "hc_FD_Unlensed": fd_hc}

        # Apply lensing effects if parameters are provided. Default is None (Unlensed).
        tmp_lensed_waveforms = self.cbc_injection.apply_lensing_amplification_factor(
            unlensed_waveforms, **params
        )

        # Check if lensed time-domain waveforms are available
        hp_lensed = tmp_lensed_waveforms.get("hp_TD_Lensed")
        hc_lensed = tmp_lensed_waveforms.get("hc_TD_Lensed")
        if hp_lensed is not None and hc_lensed is not None:
            hp, hc = hp_lensed, hc_lensed

        return hp, hc

    def simulate_zero_noise_nr_injection(self, **params):
        """
        Simulate zero-noise Numerical Relativity (NR) injections
        based on a particular SXS NR simulation.
        This function returns GW polarization modes h_plus and h_cross
        projected onto the interferometers (IFO) based on the
        specified total_mass and other extrinsic parameters.
        """
        if any(
            params.get(key) is None for key in ("hp", "hc")
        ):  # this allows passing hp/hc separately to this function.
            wfs_res = self.generate_nr_gw_polarizations_hp_hc(**params)
            if self.nr_inj_params is not None:
                params = self.nr_inj_params.copy()
            params.update(wfs_res)
        return self.cbc_injection.simulate_zero_noise_injection(**params)

    def simulate_noisy_nr_injection(self, **params):
        """
        Simulate noisy Numerical Relativity (NR) injections
        (NR GW signal + interfermoter noise)
        based on a particular SXS NR simulation.
        This function returns GW polarization modes h_plus and
        h_cross projected onto the interferometers (IFO) with added
        gaussian noise colored with the provided PSDs for IFOs.
        For a given SXS simulation, one can specify different
        total_mass and extrinsic parameters.
        """
        if any(params.get(key) is None for key in ("hp", "hc")):
            wfs_res = self.generate_nr_gw_polarizations_hp_hc(**params)
            if self.nr_inj_params is not None:
                params = self.nr_inj_params.copy()
            params.update(wfs_res)
        return self.cbc_injection.simulate_noisy_injection(**params)

    def simulate_nr_injection_with_comprehensive_output(self, **params):
        """
        Simulate Numerical Relativity (NR) injections and return a comprehensive set
        of data including gravitational wave polarizations, injected signals,
        and signal-to-noise ratio (SNR) information.

        This function generates NR-based waveforms and injects them into detector data
        (with or without noise) across multiple interferometers (IFOs). It returns
        useful data such as the polarization modes h_plus (hp) and h_cross (hc),
        the injected signals, both zero-noise and with added gaussian noise, and the SNR
        information in each interferometer.
        """
        if any(params.get(key) is None for key in ("hp", "hc")):
            wfs_res = self.generate_nr_gw_polarizations_hp_hc(**params)
            if self.nr_inj_params is not None:
                params = self.nr_inj_params.copy()
            params.update(wfs_res)
        return self.cbc_injection.simulate_injection_with_comprehensive_output(**params)

    def network_optimal_snr_to_distance(self, net_optimal_snr, **params):
        """
        Determine the luminosity distance required to achieve a specific `network_optimal_snr`
        for a given Numerical Relativity (NR) injection.

        This function calculates the luminosity distance that will result in the desired
        network optimal signal-to-noise ratio (SNR) across a network of detectors for
        a given NR waveform injection.
        """
        if any(params.get(key) is None for key in ("hp", "hc")):
            wfs_res = self.generate_nr_gw_polarizations_hp_hc(**params)
            if self.nr_inj_params is not None:
                params = self.nr_inj_params.copy()
            params.update(wfs_res)
        return self.cbc_injection.network_optimal_snr_to_distance(
            net_optimal_snr, **params
        )

    def network_matched_filter_snr_to_distance(self, net_mf_snr, threshold=1, **params):
        """
        Determine the luminosity distance required to achieve
        a specific `network_matched_filter_snr` for a
        given Numerical Relativity (NR) injection.

        This function calculates the luminosity distance that will result
        in the desired network optimal signal-to-noise ratio (SNR) across
        a network of detectors for a given NR waveform injection.
        """
        if any(params.get(key) is None for key in ("hp", "hc")):
            wfs_res = self.generate_nr_gw_polarizations_hp_hc(**params)
            if self.nr_inj_params is not None:
                params = self.nr_inj_params.copy()
            params.update(wfs_res)
        return self.cbc_injection.network_matched_filter_snr_to_distance(
            net_mf_snr, threshold, **params
        )

    # Function with AND conditions
    def filter_sxs_simulations_by_AND_conditions(
        self,
        df=None,
        reference_mass_ratio=None,
        reference_chi_eff=None,
        reference_chi_p=None,
        reference_chi1_mag=None,
        reference_chi2_mag=None,
        reference_chi1_perp=None,
        reference_chi2_perp=None,
        reference_eccentricity=None,
        minimum_total_mass_for_fstart10=None,
        minimum_total_mass_for_fstart13p33=None,
        minimum_total_mass_for_fstart20=None,
    ):
        """
        Filter SXS simulations based on multiple conditions using logical AND.

        This function applies the specified conditions to the SXS simulations
        dataframe and returns a filtered dataframe that meets all specified
        criteria. Each condition must be satisfied for a simulation to be
        included in the result.

        Parameters: dict
            df (pd.DataFrame, optional): The dataframe of SXS simulations to filter.
                If None, the function uses the default dataframe.
            reference_mass_ratio (tuple, optional): A tuple of minimum and maximum
                mass ratios to filter by. It automatically brings all the available
                mass-ratios to one convention before masking.
            reference_chi_eff (tuple, optional): A tuple of minimum and maximum
                effective spins to filter by.
            reference_chi_p (tuple, optional): A tuple of minimum and maximum
                effective precession parameters to filter by.
            reference_chi1_mag (tuple, optional): A tuple of minimum and maximum
                spin magnitudes of the primary component of the binary to filter by.
            reference_chi2_mag (tuple, optional): A tuple of minimum and maximum
                spin magnitudes of the secondary component of the binary to filter by.
            reference_chi1_perp (tuple, optional): A tuple of minimum and maximum
                perpendicular spin components of the primary component of the binary to filter by.
            reference_chi2_perp (tuple, optional): A tuple of minimum and maximum
                perpendicular spin components of the secondary component of the binary to filter by.
            reference_eccentricity (tuple, optional): A tuple of minimum and
                maximum eccentricities to filter by.
                NOTE: Providing this also excludes dataframe rows
                having NAN entires, which is a safer thing to do.
            minimum_total_mass_for_fstart10 (tuple, optional): A tuple of minimum
                and maximum total masses for f_start = 10 Hz to filter by.
            minimum_total_mass_for_fstart13p33 (tuple, optional): A tuple of minimum
                and maximum total masses for f_start = 13.33 Hz to filter by.
            minimum_total_mass_for_fstart20 (tuple, optional): A tuple of minimum
                and maximum total masses for f_start = 20 Hz to filter by.

        Returns:
            pd.DataFrame: A filtered dataframe containing only the simulations that
            meet all specified conditions.
        """
        if df is None:
            df = (
                self.sxs_simulations_dataframe
                if self.sxs_simulations_dataframe is not None
                else self.get_sxs_dataframe()
            )

        # Start with a mask of all True
        mask = pd.Series([True] * len(df), index=df.index)

        # Apply conditions if arguments are not None
        if reference_mass_ratio is not None:
            # Ensuring that mass-ratio values have consistent convention: (m1/m2)>1.
            qs = df["reference_mass_ratio"].values
            qs = np.asarray([1.0 / q if q < 1 else q for q in qs])

            reference_mass_ratio = np.asarray(reference_mass_ratio)
            if np.all(reference_mass_ratio < 1):
                reference_mass_ratio = 1 / reference_mass_ratio[::-1]
            elif np.sum(reference_mass_ratio >= 1) == 1:
                raise ValueError(
                    "Provide the mass ratios in a consistent format: either both < 1 or both >= 1."
                )

            mask &= (qs >= reference_mass_ratio[0]) & (qs <= reference_mass_ratio[1])

        if reference_chi_eff is not None:
            mask &= (df["reference_chi_eff"] >= reference_chi_eff[0]) & (
                df["reference_chi_eff"] <= reference_chi_eff[1]
            )

        if reference_chi_p is not None:
            mask &= (df["reference_chi_p"] >= reference_chi_p[0]) & (
                df["reference_chi_p"] <= reference_chi_p[1]
            )

        if reference_chi1_mag is not None:
            mask &= (df["reference_chi1_mag"] >= reference_chi1_mag[0]) & (
                df["reference_chi1_mag"] <= reference_chi1_mag[1]
            )

        if reference_chi2_mag is not None:
            mask &= (df["reference_chi2_mag"] >= reference_chi2_mag[0]) & (
                df["reference_chi2_mag"] <= reference_chi2_mag[1]
            )

        if reference_chi1_perp is not None:
            mask &= (df["reference_chi1_perp"] >= reference_chi1_perp[0]) & (
                df["reference_chi1_perp"] <= reference_chi1_perp[1]
            )

        if reference_chi2_perp is not None:
            mask &= (df["reference_chi2_perp"] >= reference_chi2_perp[0]) & (
                df["reference_chi2_perp"] <= reference_chi2_perp[1]
            )

        if reference_eccentricity is not None:
            mask &= (df["reference_eccentricity"] >= reference_eccentricity[0]) & (
                df["reference_eccentricity"] <= reference_eccentricity[1]
            )

        if minimum_total_mass_for_fstart10 is not None:
            mask &= (
                df["minimum_total_mass_for_fstart10"]
                >= minimum_total_mass_for_fstart10[0]
            ) & (
                df["minimum_total_mass_for_fstart10"]
                <= minimum_total_mass_for_fstart10[1]
            )

        if minimum_total_mass_for_fstart13p33 is not None:
            mask &= (
                df["minimum_total_mass_for_fstart13p33"]
                >= minimum_total_mass_for_fstart13p33[0]
            ) & (
                df["minimum_total_mass_for_fstart13p33"]
                <= minimum_total_mass_for_fstart13p33[1]
            )

        if minimum_total_mass_for_fstart20 is not None:
            mask &= (
                df["minimum_total_mass_for_fstart20"]
                >= minimum_total_mass_for_fstart20[0]
            ) & (
                df["minimum_total_mass_for_fstart20"]
                <= minimum_total_mass_for_fstart20[1]
            )

        # Return the filtered DataFrame
        return df[mask]

    # Function with OR conditions
    def filter_sxs_simulations_by_OR_conditions(
        self,
        df=None,
        reference_mass_ratio=None,
        reference_chi_eff=None,
        reference_chi_p=None,
        reference_chi1_mag=None,
        reference_chi2_mag=None,
        reference_chi1_perp=None,
        reference_chi2_perp=None,
        reference_eccentricity=None,
        minimum_total_mass_for_fstart10=None,
        minimum_total_mass_for_fstart13p33=None,
        minimum_total_mass_for_fstart20=None,
    ):
        """
        Filter SXS simulations based on multiple conditions using logical OR.

        This function applies the specified conditions to the SXS simulations
        dataframe and returns a filtered dataframe that meets at least one of
        the specified criteria. A simulation will be included in the result
        if it satisfies any of the conditions.

        Parameters: dict
            df (pd.DataFrame, optional): The dataframe of SXS simulations to filter.
                If None, the function uses the default dataframe.
            reference_mass_ratio (tuple, optional): A tuple of minimum and maximum
                mass ratios to filter by. It automatically brings all the available
                mass-ratios to one convention before masking.
            reference_chi_eff (tuple, optional): A tuple of minimum and maximum
                effective spins to filter by.
            reference_chi_p (tuple, optional): A tuple of minimum and maximum
                effective precession parameters to filter by.
            reference_chi1_mag (tuple, optional): A tuple of minimum and maximum
                spin magnitudes of the primary component of the binary to filter by.
            reference_chi2_mag (tuple, optional): A tuple of minimum and maximum
                spin magnitudes of the secondary component of the binary to filter by.
            reference_chi1_perp (tuple, optional): A tuple of minimum and maximum
                perpendicular spin components of the primary component of the binary to filter by.
            reference_chi2_perp (tuple, optional): A tuple of minimum and maximum
                perpendicular spin components of the secondary component of the binary to filter by.
            reference_eccentricity (tuple, optional): A tuple of minimum and
                maximum eccentricities to filter by.
                NOTE: Providing this also excludes dataframe rows
                having NAN entires, which is a safer thing to do.
            minimum_total_mass_for_fstart10 (tuple, optional): A tuple of minimum
                and maximum total masses for f_start = 10 Hz to filter by.
            minimum_total_mass_for_fstart13p33 (tuple, optional): A tuple of minimum
                and maximum total masses for f_start = 13.33 Hz to filter by.
            minimum_total_mass_for_fstart20 (tuple, optional): A tuple of minimum
                and maximum total masses for f_start = 20 Hz to filter by.

        Returns:
            pd.DataFrame: A filtered dataframe containing only the simulations that
            meet at least one of the specified conditions.
        """

        if df is None:
            df = (
                self.sxs_simulations_dataframe
                if self.sxs_simulations_dataframe is not None
                else self.get_sxs_dataframe()
            )

        # Start with a mask of all False (we will OR conditions)
        mask = pd.Series([False] * len(df), index=df.index)

        # Apply conditions if arguments are not None
        if reference_mass_ratio is not None:
            # Ensuring that mass-ratio values have consistent convention: (m1/m2)>1.
            qs = df["reference_mass_ratio"].values
            qs = np.asarray([1.0 / q if q < 1 else q for q in qs])

            reference_mass_ratio = np.asarray(reference_mass_ratio)
            if np.all(reference_mass_ratio < 1):
                reference_mass_ratio = 1 / reference_mass_ratio[::-1]
            elif np.sum(reference_mass_ratio >= 1) == 1:
                raise ValueError(
                    "Provide the mass ratios in a consistent format: either both < 1 or both >= 1."
                )

            mask |= (qs >= reference_mass_ratio[0]) & (qs <= reference_mass_ratio[1])

        if reference_chi_eff is not None:
            mask |= (df["reference_chi_eff"] >= reference_chi_eff[0]) & (
                df["reference_chi_eff"] <= reference_chi_eff[1]
            )

        if reference_chi_p is not None:
            mask |= (df["reference_chi_p"] >= reference_chi_p[0]) & (
                df["reference_chi_p"] <= reference_chi_p[1]
            )

        if reference_chi1_mag is not None:
            mask |= (df["reference_chi1_mag"] >= reference_chi1_mag[0]) & (
                df["reference_chi1_mag"] <= reference_chi1_mag[1]
            )

        if reference_chi2_mag is not None:
            mask |= (df["reference_chi2_mag"] >= reference_chi2_mag[0]) & (
                df["reference_chi2_mag"] <= reference_chi2_mag[1]
            )

        if reference_chi1_perp is not None:
            mask |= (df["reference_chi1_perp"] >= reference_chi1_perp[0]) & (
                df["reference_chi1_perp"] <= reference_chi1_perp[1]
            )

        if reference_chi2_perp is not None:
            mask |= (df["reference_chi2_perp"] >= reference_chi2_perp[0]) & (
                df["reference_chi2_perp"] <= reference_chi2_perp[1]
            )

        if reference_eccentricity is not None:
            mask |= (df["reference_eccentricity"] >= reference_eccentricity[0]) & (
                df["reference_eccentricity"] <= reference_eccentricity[1]
            )

        if minimum_total_mass_for_fstart10 is not None:
            mask |= (
                df["minimum_total_mass_for_fstart10"]
                >= minimum_total_mass_for_fstart10[0]
            ) & (
                df["minimum_total_mass_for_fstart10"]
                <= minimum_total_mass_for_fstart10[1]
            )

        if minimum_total_mass_for_fstart13p33 is not None:
            mask |= (
                df["minimum_total_mass_for_fstart13p33"]
                >= minimum_total_mass_for_fstart13p33[0]
            ) & (
                df["minimum_total_mass_for_fstart13p33"]
                <= minimum_total_mass_for_fstart13p33[1]
            )

        if minimum_total_mass_for_fstart20 is not None:
            mask |= (
                df["minimum_total_mass_for_fstart20"]
                >= minimum_total_mass_for_fstart20[0]
            ) & (
                df["minimum_total_mass_for_fstart20"]
                <= minimum_total_mass_for_fstart20[1]
            )

        # Return the filtered DataFrame
        return df[mask]
