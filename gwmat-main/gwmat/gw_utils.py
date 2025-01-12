"""
This module contains the class `GWUtils`, which provides some
useful utility functions for analysing CBC singals.
"""

__author__ = ["Anuj Mishra <anuj.mishra@ligo.org>"]

import os
from glob import glob
from tqdm import tqdm

import pandas as pd
import numpy as np

import lalsimulation
import pycbc
from pycbc.detector import Detector
from pycbc.waveform import get_td_waveform
from pycbc.filter import match
from pycbc.psd.analytical import aLIGOZeroDetHighPower
import pycbc.noise
import pycbc.psd
import lalinference.imrtgr.nrutils as nr

from .constants import C_SI, G_SI, MSUN_SI
from .general_utils import GeneralUtils


class GWUtils(GeneralUtils):
    """
    Some utility functions related to analysing CBCs.

    """

    def __init__(self):
        super().__init__()
        self.solar_mass_to_seconds = G_SI * MSUN_SI / C_SI**3

    def r_isco(self, m_tot, a_f):
        """
        Returns the equatorial Innermost Stable Circular Orbit (ISCO),
        also known as radius of the marginally stable orbit.
        For Kerr metric, it depends on whether the orbit is
        prograde (negative sign) or retrograde (positive sign).

        References:
        Eq.2.21 of _Bardeen et al.
        <https://ui.adsabs.harvard.edu/abs/1972ApJ...178..347B/abstract>_,

        Eq.1 of _Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>_

        Parameters
        ----------
        m_tot = m1+m2 : float
            Binary mass (in solar masses).
        a_f : float
            Dimensionless spin parameter of the remnant compact object.

        Returns
        -------
        dict:
            Dictionary of:

            * r_isco_retrograde: float
                ISCO radius for a particle in retrogade motion (in solar masses).

        """

        fac = m_tot
        z1 = 1 + np.cbrt(1 - a_f**2) * (np.cbrt(1 + a_f) + np.cbrt(1 - a_f))
        z2 = np.sqrt(3 * a_f**2 + z1**2)
        risco_n = fac * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
        risco_p = fac * (3 + z2 + np.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
        r_dict = {"r_isco_retrograde": risco_p, "r_isco_prograde": risco_n}
        return r_dict

    def gw_frequency_at_kerr_isco(self, m_f, a_f):
        """
        Returns GW frequency at ISCO for a spinning BH Binary.
        References: Eq.4 of `Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>`


        Parameters
        ----------
        m_f: float
            Provide either the total binary Mass, or
            the remnant BH mass (in solar masses).
            The reference suggests using total binary mass.
            However, using remnant mass is preferred as
            that is what is used in the collaboration.

        a_f : float
            Final dimensionless spin magnitude of the remnant BH.

        Returns
        -------
        dict:
            Dictionary of:

            * f_ISCO_retrograde: float
                GW frequency at ISCO for binaries in retrogade motion (in solar masses).
            * f_ISCO_prograde: float
                ISCO radius for a particle in prograde motion (in solar masses).

        """
        fac = 1 / (2 * np.pi * m_f * self.solar_mass_to_seconds)
        r_res = self.r_isco(m_f, a_f)
        r_n = r_res["r_isco_prograde"]
        r_p = r_res["r_isco_retrograde"]
        f_orb_isco_n = fac * (a_f + pow(r_n / m_f, 3 / 2)) ** (-1)
        f_orb_isco_p = fac * (a_f + pow(r_p / m_f, 3 / 2)) ** (-1)
        f_n, f_p = (
            2 * f_orb_isco_n,
            2 * f_orb_isco_p,
        )  # because of quadrupolar contributions, f_gw = 2 * f_orb
        f_dict = {"f_ISCO_retrograde": f_p, "f_ISCO_prograde": f_n}
        return f_dict

    def gw_frequency_at_kerr_isco_from_bbh_params(self, **params):
        """
        Returns GW frequency at ISCO for a spinning BH Binary when given the BBH parameters.

        Parameters
        ----------
        Dictionary:
            params: dict
                containing all the relevant parameters of binary.

        Returns
        -------
        dict:
            Dictionary of:

            * f_ISCO_retrograde: float
                GW frequency at ISCO for binaries in retrogade motion (in solar masses).
            * f_ISCO_prograde: float
                ISCO radius for a particle in prograde motion (in solar masses).

        """
        remnant_params = self.remnant_mass_and_spin_from_bbh_params(**params)
        fisco_res = self.gw_frequency_at_kerr_isco(
            remnant_params["m_f"], remnant_params["a_f"]
        )
        return fisco_res

    def gw_frequency_at_schwarzschild_isco(self, m_tot):
        """
        Returns GW frequency at ISCO for a non-spinning BH Binary.
        References: Eq.4 of `Chad Hanna et al. <https://arxiv.org/pdf/0801.4297.pdf>`

        Parameters
        ----------
        m_tot = m1+m2: float
           Binary Mass (in solar masses).

        Returns
        -------
        float
          GW Frequency at ISCO (in Hz).

        """
        return self.gw_frequency_at_kerr_isco(m_tot, a_f=0)["f_ISCO_prograde"]

    def gw_frequency_at_bkl_isco(self, m1, m2):
        """
        Mass ratio dependent GW frequency at ISCO derived from estimates of the final spin
        of a merged black hole in a paper by Buonanno, Kidder, Lehner
        (arXiv:0709.3839).  See also arxiv:0801.4297v2 eq.(5)

        Parameters
        ----------
        m1 : float or numpy.array
            The mass of the first component object in the binary (in solar masses).
        m2 : float or numpy.array
            The mass of the second component object in the binary (in solar masses).

        Returns
        -------
        f : float or numpy.array
            GW Frequency at ISCO (in Hz).

        """

        # q is defined to be in [0,1] for this formula
        q = np.minimum(m1 / m2, m2 / m1)
        return (
            0.5
            * self.gw_frequency_at_schwarzschild_isco(m1 + m2)
            * (1 + 2.8 * q - 2.6 * q**2 + 0.8 * q**3)
        )

    def gw_frequency_at_meco(self, total_mass, q, spin1z, spin2z):
        """
        Compute an estimate of the minimum energy
        circular orbit (MECO) frequency.
        Uses lalsimulation/lib/LALSimIMRPhenomXUtilities.c#L44
        Phenomenological fit to hybrid minimum energy circular orbit (MECO) function.
        Uses 3.5PN hybridised with test-particle limit.
        Reference: M Cabero et al, PRD, 95, 064016, (2017), arXiv:1602.03134

        Parameters
        ----------
            total_mass : float
                    Total mass in solar masses
            q : float
                Mass ratio >=1
            spin1z : float
                    Dimensionless spin of primary
            spin2z : float
                    Dimensionless spin of secondary

        Returns
        -------
            float
                MECO frequency in Hz.
        """
        eta = np.minimum(q / (1.0 + q) ** 2, 0.25)

        if np.isscalar(total_mass):
            # For scalar inputs, directly compute and return the result
            f_meco_tmp = lalsimulation.SimIMRPhenomXfMECO(eta, spin1z, spin2z)
            f_meco = self.mass_rescaled_frequency_to_hertz(f_meco_tmp, total_mass)
        else:
            # For array inputs, compute for each element
            f_meco_tmp = np.zeros_like(total_mass)
            for i in range(total_mass.shape[0]):
                f_meco_tmp[i] = lalsimulation.SimIMRPhenomXfMECO(
                    eta[i], spin1z[i], spin2z[i]
                )

        f_meco = self.mass_rescaled_frequency_to_hertz(f_meco_tmp, total_mass)
        return f_meco

    def recommended_reference_frequency(
        self, total_mass, q, spin1z, spin2z, f_ref=20.0, f_start=10.0, fudge_factor=0.97
    ):
        """
        Recommend the appropriate reference frequency for gravitational wave analysis.
        This function is inspired from:
        asimov/pipelines/pe-configurator/peconfigurator/proc_samples.py#432

        Parameters
        ----------
        total_mass : float
            The total mass of the binary system in solar masses.
        q : float
            Mass ratio of the binary, defined as m_secondary / m_primary <= 1.
        spin1z : float
            Dimensionless spin component along the z-axis for the primary black hole.
        spin2z : float
            Dimensionless spin component along the z-axis for the secondary black hole.
        f_ref : float, optional
            Initial reference frequency, default is 20 Hz.
        f_start : float, optional
            Initial starting frequency, default is 10 Hz.
        fudge_factor : float, optional
            Factor to scale the MECO frequency for the reference frequency check, default is 0.97.

        Returns
        -------
        float
            The recommended reference frequency.

        Raises
        ------
        ValueError
            If the provided f_start is higher than the adjusted f_ref.
        """

        # Compute the gravitational wave frequency at the MECO
        f_meco = self.gw_frequency_at_meco(total_mass, q, spin1z, spin2z)
        scaled_meco = fudge_factor * f_meco
        f_ref = np.minimum(np.floor(scaled_meco), f_ref)
        if np.any(f_ref <= f_start):
            raise ValueError(
                "Provided f_start is higher than or equal to the recommended f_ref."
            )
        return f_ref

    def remnant_mass_and_spin_from_bbh_params(self, **params):
        """
        Returns the mass and spin of the final remnant
        based on initial binary configuration.

        Parameters
        ----------
        params : dict
            Dictionary containing the following keys:
                * mass_1 : float or numpy.array
                    The mass of the first component object in the binary (in solar masses).
                * mass_2 : float or numpy.array
                    The mass of the second component object in the binary (in solar masses).
                * a_1 : float, optional
                    The dimensionless spin magnitude of the first binary component. Default = 0.
                * a_2 : float, optional
                    The dimensionless spin magnitude of the second binary component. Default = 0.
                * tilt_1 : float, optional
                    Zenith angle between S1 and LNhat (rad). Default = 0.
                * tilt_2 : float, optional
                    Zenith angle between S2 and LNhat (rad). Default = 0.
                * phi_12 : float, optional
                    Difference in azimuthal angle between S1 and S2 (rad). Default = 0.

        Returns
        -------
        dict
            Dictionary containing the following keys:
                * m_f : float
                    Final mass of the remnant (in solar masses).
                * a_f : float
                    Final dimensionless spin of the remnant.
        """

        # Use the following final mass and spin fits to calculate fISCO
        Mf_fits = ["UIB2016", "HL2016"]
        af_fits = ["UIB2016", "HL2016", "HBR2016"]

        for k in ["mass_1", "mass_2", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12"]:
            if k not in params:
                params[k] = 0.0

        # Final mass computation does not use phi12, so we set it to zero
        Mf = nr.bbh_average_fits_precessing(
            m1=params["mass_1"],
            m2=params["mass_2"],
            chi1=params["a_1"],
            chi2=params["a_2"],
            tilt1=params["tilt_1"],
            tilt2=params["tilt_2"],
            phi12=np.array([0.0]),
            quantity="Mf",
            fits=Mf_fits,
        )

        af = nr.bbh_average_fits_precessing(
            m1=params["mass_1"],
            m2=params["mass_2"],
            chi1=params["a_1"],
            chi2=params["a_2"],
            tilt1=params["tilt_1"],
            tilt2=params["tilt_2"],
            phi12=params["phi_12"],
            quantity="af",
            fits=af_fits,
        )
        return {"m_f": float(Mf), "a_f": float(af)}

    def f_220_ringdown_dimensionless(self, a_f):
        """
        Return the dimensionless fundamental RingDown frequency.

        Parameters
        ----------
        a_f : float
            Final dimensionless spin magnitude of the remnant.

        Returns
        -------
        float
            Dimensionless fundamental RingDown frequency.
        """

        f1 = 1.5251
        f2 = -1.1568
        f3 = 0.1292
        FRD = f1 + f2 * (1 - a_f) ** f3
        return FRD

    def f_220_ringdown_from_bbh_params(self, **params):
        """
        Fundamental RingDown frequency calculated using table viii of
        Berti, Cardoso and Will (gr-qc/0512160) value for the omega_220
        QNM frequency: <https://arxiv.org/pdf/gr-qc/0512160.pdf>.

        Parameters
        ----------
        params : dict
            Dictionary containing the following keys:
                * mass_1 : float or numpy.array
                    The mass of the first component object in the binary (in solar masses).
                * mass_2 : float or numpy.array
                    The mass of the second component object in the binary (in solar masses).
                * a_1 : float, optional
                    The dimensionless spin magnitude of the first binary component. Default = 0.
                * a_2 : float, optional
                    The dimensionless spin magnitude of the second binary component. Default = 0.
                * tilt_1 : float, optional
                    Zenith angle between S1 and LNhat (rad). Default = 0.
                * tilt_2 : float, optional
                    Zenith angle between S2 and LNhat (rad). Default = 0.
                * phi_12 : float, optional
                    Difference in azimuthal angle between S1 and S2 (rad). Default = 0.

        Returns
        -------
        float
            Ringdown Frequency (in Hz).
        """

        remnant_params = self.remnant_mass_and_spin_from_bbh_params(**params)
        m_f = remnant_params["m_f"]
        a_f = remnant_params["a_f"]
        FRD_dimless = self.f_220_ringdown_dimensionless(a_f)
        fac = 1 / (2 * np.pi * m_f * self.solar_mass_to_seconds)
        FRD = fac * FRD_dimless
        return FRD

    def compute_pycbc_match(
        self,
        wf1,
        wf2,
        psd=None,
        f_low=20.0,
        f_high=None,
        subsample_interpolation=False,
        return_phase=False,
        is_asd_file=False,
    ):
        """
        Computes match (overlap maximised over phase and time) between
        two time domain WFs using PyCBC's match function.
        That is, match = max_{t_c, phi_c} overlap <wf1(f), wf2(f) *  exp(phi_c + 2*pi*f*t_c)>.
        Note: Since a constant phase factor is being used for the whole WF,
        this method of phase maximisation can only maximize for (l=2, m=2) mode.

        Parameters
        ----------
        wf1 : pycbc.types.timeseries.TimeSeries object
            PyCBC time domain Waveform.
        wf2 : pycbc.types.timeseries.TimeSeries object
            PyCBC time domain Waveform.
        psd: {None, str}
            PSD file to use for computing match. Default = None.
            Predefined_PSDs: {'aLIGOZeroDetHighPower'}
        f_low : {None, float}, optional
            The lower frequency cutoff for the match computation. Default = 20.
        f_high : {None, float}, optional
            The upper frequency cutoff for the match computation. Default = None.
            subsample_interpolation: ({False, bool}, optional)
            If True the peak will be interpolated between samples using a
            simple quadratic fit. This can be important if measuring matches
            very close to 1 and can cause discontinuities if you don’t use
            it as matches move between discrete samples.
            If True the index returned will be a float instead of int. Default = False.
        return_phase: ({True, bool}, optional)
            If True, also return the phase shift that gives the match.
        is_asd_file : bool, optional
            Is psd provided corresponds to an asd file? Default = False.

        Returns
        -------
        tuple
        A tuple containing the following elements:
        match_val : float
            match value Phase to rotate complex waveform to get the match, if desired.
        index_shift : float
            The number of samples to shift to get the match.
        phase_shift : float
            Phase to rotate complex waveform to get the match, if desired.
            (returned when `return_phase=True`)

        """

        arrlen = max(len(wf1), len(wf2))
        wf1.resize(arrlen)
        wf2.resize(arrlen)

        delta_f = min(wf1.delta_f, wf2.delta_f)
        if isinstance(wf1, pycbc.types.timeseries.TimeSeries):
            assert (
                wf1.sample_rate == wf2.sample_rate
            ), f"Inconsistent sampling rates.\
            wf1.sample_rate={wf1.sample_rate:.3f} does not match\
            wf2.sample_rate={wf2.sample_rate:.3f}."
            flen = arrlen // 2 + 1
        else:
            flen = arrlen
            assert (
                wf1.delta_f == wf2.delta_f
            ), f"Inconsistent frequency spacings. \
            wf1.delta_f={wf1.delta_f:.3f} does not match\
            wf2.delta_f={wf2.delta_f:.3f}."
        if psd is not None:
            if psd == "aLIGOZeroDetHighPower":
                psd = aLIGOZeroDetHighPower(flen, delta_f, f_low)
            else:
                psd = pycbc.psd.from_txt(
                    psd, flen, delta_f, f_low, is_asd_file=is_asd_file
                )
        pycbc_match_res = match(
            wf1,
            wf2,
            psd=psd,
            low_frequency_cutoff=f_low,
            high_frequency_cutoff=f_high,
            subsample_interpolation=subsample_interpolation,
            return_phase=return_phase,
        )
        return pycbc_match_res

    def cumulative_phase(self, phase_array, period=np.pi):
        """
        Returns the unwrapped phase.

        Parameters
        ----------
        phase_array : np.array of floats
            Array containing wrapped phase values.
        period : float, optional
            Period of the signal. Default = np.pi.

        Returns
        -------
        np.array
            Array containing the unwrapped phase values.

        Example
        -------
        theta_arr = np.linspace(0, 30*np.pi, 1000)
        zs = np.exp(1j*theta_arr)   # rotation of a unit vector in a \
        # complex plane (should result in a linear increase in phase)
        plt.plot(np.angle(zs))
        plt.title('wrapped')
        plt.show()
        plt.plot(cumulative_phase(np.angle(zs), period=2*np.pi)/np.pi)
        plt.title('unwrapped')
        plt.show()

        """
        unwrapped_phase = np.zeros(len(phase_array), dtype=complex)
        k = 0
        for i in range(len(phase_array) - 1):
            unwrapped_phase[i] = k * (period) + phase_array[i]
            diff = np.abs(phase_array[i] - phase_array[i + 1])
            if diff > period / 2:
                k += 1
        unwrapped_phase[-1] = k * (period) + phase_array[-1]
        return np.real(unwrapped_phase)

    def wf_phase(self, wf):
        """
        Returns the phase of a GW waveform.

        Parameters
        ----------
        wf : {complex np.array, pycbc.types.frequencyseries.FrequencySeries object}
            Array containing the complex amplitude of the signal,
            or an object of type pycbc.types.frequencyseries.

        Returns
        -------
        complex np.array
            Phase of the GW signal (in rads).

        """

        if isinstance(wf, pycbc.types.timeseries.TimeSeries):
            print(
                "Converting TimeSeires object to FrequencySeries before evaluating the phase."
            )
            wf = wf.to_frequencyseries(delta_f=wf.delta_f)
        wf = np.asarray(wf)
        phase = np.arctan2(np.imag(wf), np.real(wf))
        return phase

    # Ref.: FINDCHIRP (arXiv:0509116)
    def chirp_duration_2PN(self, mtot, eta, f_low=20):
        """
        Chirp duration of a GW signal assuming 2PN approximation.
        Reference: Bruce Allen, et al. -
        `FINDCHIRP <https://arxiv.org/abs/gr-qc/0509116>`

        Parameters
        ----------
        mtot : float
            _description_
        eta : float
            _description_
        f_low : int, optional
            _description_, by default 20

        Returns
        -------
        float
            Chirp time.

        """
        fac = mtot * self.solar_mass_to_seconds
        v_low = np.power(fac * np.pi * f_low, 1 / 3)
        tchirp = (
            fac
            * (5 / (256 * eta))
            * (
                v_low ** (-8)
                + ((743 / 252) + (11 / 3) * eta) * v_low ** (-6)
                - (32 * np.pi / 5) * v_low ** (-5)
                + ((3058673 / 508032) + (5429 / 504) * eta + (617 / 72) * eta**2)
                * v_low ** (-4)
            )
        )
        return tchirp

    # more accurate function than above
    def rough_wf_duration(self, m1, m2, f_low=20.0, threshold=1.0):
        """
        Returns the rough WF duration for a given binary component masses and f_low.
        Duration is computated via actual generation of the WF using
        the WF approximant IMRPhenomPv2, and defined as:
        Chirp_duraion = duration between the times where strain amplitude reached
        $threshold (%) of the peak amplitude around the time of peak amplitude.

        Parameters
        ----------
        m1 : float
            The mass of the first component object in the binary (in solar masses).
        m2 : float
            The mass of the second component object in the binary (in solar masses).
        f_low : float, optional
            The lower frequency cutoff for the generation of WF. Default = 20.
        threshold : float, optional
            Threshold (%) to defined the start and end of a WF as a fraction
            of the peak amplitude. It represents the ratio between the start
            f a WF to the peak of the WF (in percetage).
            Default = 1 (i.e., the duration will be defined as the time when
            the amplitude first became 1% of the peak amplitude untill the
            time after which it again went below that).
            Caution: Very low threshold values will result in error,
            so values below 0.1% are NOT recommended.
            Values such as 0.5% is ideal, while 1% (default) works decently well.
        Returns
        -------
        dict:
            Dictionary of:

            * hp : pycbc.types.timeseries.TimeSeries
                Pure polarised time-domain WF.
            * hc : pycbc.types.timeseries.TimeSeries
                Pure polarised time-domain WF.
            * insp_to_merger_duration : float
                Rough WF duration in the Inspiral-Merger Phase.
            * post_merger_duration : float
                Rough WF duration in the Post-Merger Phase.
            * chirp_duration : float
                Rough WF duration for whole WF in the IMR Phase.
            * rough_duration_for_PE : int
                Rough WF duration to use for PE.

        """

        # this assumes trigger is at t=0 (true for pycbc WFs)
        hp, hc = get_td_waveform(
            approximant="IMRPhenomXP",
            mass1=m1,
            mass2=m2,
            delta_t=1.0 / 2048,
            f_lower=f_low,
        )

        hp = hp.cyclic_time_shift(-0.2)

        sr = 1.0 / hp.delta_t
        max_hp = max(hp)
        len_hp = len(hp)

        # duration from Inspiral till Merger
        ind = 0
        d_ind = round(sr / 128)  # sampling every (1/128)th of a second
        while hp[ind] / max_hp < threshold * 1e-2 and ind < len_hp:
            ind += d_ind

        insp_to_merger_duration = -1 * hp.sample_times[ind - d_ind]

        # duration of Post-Merger Signal
        ind = -1
        d_ind = round(sr / 512)  # sampling every (1/512)th of a second
        while hp[ind] / max_hp < threshold * 1e-2 and ind < len_hp:
            ind -= d_ind

        post_merger_duration = hp.sample_times[ind + d_ind]

        # total Duration of the WF
        chirp_duration = insp_to_merger_duration + post_merger_duration

        # duration of the WF for PE
        if insp_to_merger_duration < 0.5:
            rough_dur = 2
        else:
            rough_dur = np.power(
                2, np.ceil(np.log2(chirp_duration))
            )  # 2*round(chirp_duration/2+0.5) + 2

        # storing and returning all results as dictionary
        res = {
            "hp": hp,
            "hc": hc,
            "insp_to_merger_duration": insp_to_merger_duration,
            "post_merger_duration": post_merger_duration,
            "chirp_duration": chirp_duration,
            "rough_duration_for_PE": rough_dur,
        }
        return res

    def jframe_to_l0frame(
        self,
        mass_1,
        mass_2,
        f_ref,
        phi_ref=0.0,
        theta_jn=0.0,
        phi_jl=0.0,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
    ):
        """
        [This function is inherited from PyCBC and lalsimulation.]
        Function to convert J-frame coordinates (which Bilby uses for PE)
        to L0-frame coordinates (that Pycbc uses for waveform generation).
        J stands for the total angular momentum while L0 stands for
        the orbital angular momentum.

        Parameters
        ----------
        mass_1 : float
            The mass of the first component object in the binary (in solar masses).
        mass_2 : float
            The mass of the second component object in the binary (in solar masses).
        f_ref : float
            The reference frequency (in Hz).
        phi_ref : float, optional
            The orbital phase at ``f_ref``. Default = 0.
        theta_jn : float, optional
            Angle between the line of sight and the total
            angular momentume J. Default = 0.
        phi_jl : float, optional
            Azimuthal angle of L on its cone about J. Default = 0.
        a_1 : float, optional
            The dimensionless spin magnitude. Default = 0.
        a_2 : float, optional
            The dimensionless spin magnitude. Default = 0.
        tilt_1 : float, optional
            Angle between L and the spin magnitude of object 1. Default = 0.
        tilt_2 : float, optional
            Angle between L and the spin magnitude of object 2. Default = 0.
        phi_12 : float, optional
            Difference between the azimuthal angles of the spin of the
        object 1 and 2. Default = 0.

        Returns
        -------
        dict :
            Dictionary of:

            * inclination : float
                Inclination (rad), defined as the angle between the orbital
                angular momentum L and the line-of-sight at the reference frequency.
            * spin1x : float
                The x component of the first binary component's dimensionless spin.
            * spin1y : float
                The y component of the first binary component's dimensionless spin.
            * spin1z : float
                The z component of the first binary component's dimensionless spin.
            * spin2x : float
                The x component of the second binary component's dimensionless spin.
            * spin2y : float
                The y component of the second binary component's dimensionless spin.
            * spin2z : float
                The z component of the second binary component's dimensionless spin.

        """

        inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = (
            lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
                theta_jn,
                phi_jl,
                tilt_1,
                tilt_2,
                phi_12,
                a_1,
                a_2,
                mass_1 * MSUN_SI,
                mass_2 * MSUN_SI,
                f_ref,
                phi_ref,
            )
        )
        out_dict = {
            "inclination": inclination,
            "spin1x": spin1x,
            "spin1y": spin1y,
            "spin1z": spin1z,
            "spin2x": spin2x,
            "spin2y": spin2y,
            "spin2z": spin2z,
        }
        return out_dict

    def l0frame_to_jframe(
        self,
        mass_1,
        mass_2,
        f_ref,
        phi_ref=0.0,
        inclination=0.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.0,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.0,
    ):
        """
        [This function is inherited from PyCBC and lalsimulation.]
        Function to convert L-frame (that Pycbc uses for waveform generation)
        coordinates to J-frame coordinates (which Bilby uses for PE).
        J stands for the total angular momentum while L0 stands for
        the orbital angular momentum.

        Parameters
        ----------
        mass_1 : float
            The mass of the first component object in the binary (in solar masses).
        mass_2 : float
            The mass of the second component object in the binary (in solar masses).
        f_ref : float
            The reference frequency (in Hz).
        phiref : float
            The orbital phase at ``f_ref``.
        inclination : float
            Inclination (rad), defined as the angle between the
            orbital angular momentum L and the line-of-sight at
            the reference frequency. Default = 0.
        spin1x : float
            The x component of the first binary component's. Default = 0.
            dimensionless spin.
        spin1y : float
            The y component of the first binary component's. Default = 0.
            dimensionless spin.
        spin1z : float
            The z component of the first binary component's. Default = 0.
            dimensionless spin.
        spin2x : float
            The x component of the second binary component's. Default = 0.
            dimensionless spin.
        spin2y : float
            The y component of the second binary component's. Default = 0.
            dimensionless spin.
        spin2z : float
            The z component of the second binary component's. Default = 0.
            dimensionless spin.

        Returns
        -------
        dict :
            Dictionary of:

            * theta_jn : float, optional
                Angle between the line of sight and
                the total angular momentume J.
            * phi_jl : float, optional
                Azimuthal angle of L on its cone about J.
            * a_1 : float, optional
                The dimensionless spin magnitude.
            * a_2 : float, optional
                The dimensionless spin magnitude.
            * tilt_1 : float, optional
                Angle between L and the spin magnitude of object 1.
            * tilt_2 : float, optional
                Angle between L and the spin magnitude of object 2.
            * phi_12 : float, optional
                Difference between the azimuthal angles of
                the spin of the object 1 and 2.

        """

        thetajn, phijl, s1pol, s2pol, s12_deltaphi, spin1_a, spin2_a = (
            lalsimulation.SimInspiralTransformPrecessingWvf2PE(
                inclination,
                spin1x,
                spin1y,
                spin1z,
                spin2x,
                spin2y,
                spin2z,
                mass_1,
                mass_2,
                f_ref,
                phi_ref,
            )
        )
        out = {
            "theta_jn": thetajn,
            "phi_jl": phijl,
            "a_1": spin1_a,
            "a_2": spin2_a,
            "tilt_1": s1pol,
            "tilt_2": s2pol,
            "phi_12": s12_deltaphi,
        }
        return out

    def gw_wf_approximants_in_lal(self):
        """
        Prints available WF TD and FD approximants in LAL.

        Returns
        -------
        dict:
            Dictionary of:

            *  fd_approximants : list
                List of available FD approximants.
            *  td_approximants : list
                List of available TD approximants.

        """

        total_apxs = lalsimulation.NumApproximants
        list_of_fd_apxs = []
        list_of_td_apxs = []
        for i in range(total_apxs):
            if lalsimulation.SimInspiralImplementedFDApproximants(i):
                list_of_fd_apxs.append(lalsimulation.GetStringFromApproximant(i))

            if lalsimulation.SimInspiralImplementedTDApproximants(i):
                list_of_td_apxs.append(lalsimulation.GetStringFromApproximant(i))

        approximants = {
            "fd_approximants": list_of_fd_apxs,
            "td_approximants": list_of_td_apxs,
        }
        return approximants

    def cyclic_time_shift_of_wf(self, wf, rwrap=0.2):
        """
        Inspired by PyCBC's function pycbc.types.TimeSeries.cyclic_time_shift(),
        it shifts the data and timestamps in the time domain by a
        given number of seconds (rwrap).
        Difference between this and PyCBCs function is that this function
        preserves the sample rate of the WFs while cyclically rotating,
        but the time shift cannot be smaller than the intrinsic sample rate
        of the data, unlike PyCBc's function.
        To just change the time stamps, do ts.start_time += dt.
        Note that data will be cyclically rotated, so if you shift by 2
        seconds, the final 2 seconds of your data will now be at the
        beginning of the data set.

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
        # It is similar to PYCBC's "cyclic_time_shift" except for the
        # fact that it also preserves the Sample Rate of the original WF.
        if rwrap is not None and rwrap != 0:
            sn = abs(int(rwrap / wf.delta_t))  # number of elements to be shifted
            cycles = int(sn / len(wf))

            cyclic_shifted_wf = wf.copy()

            sn_new = sn - int(cycles * len(wf))

            if rwrap > 0:
                epoch = wf.sample_times[0] - sn_new * wf.delta_t
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[-sn_new:]
                    tmp_wf_p2 = wf_arr[:-sn_new]
                    shft_wf_arr = np.concatenate((tmp_wf_p1, tmp_wf_p2))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(
                        shft_wf_arr, delta_t=wf.delta_t, epoch=epoch
                    )
            else:
                epoch = wf.sample_times[sn_new]
                if sn_new != 0:
                    wf_arr = np.array(wf).copy()
                    tmp_wf_p1 = wf_arr[sn_new:]
                    tmp_wf_p2 = wf_arr[:sn_new]
                    shft_wf_arr = np.concatenate((tmp_wf_p1, tmp_wf_p2))
                    cyclic_shifted_wf = pycbc.types.TimeSeries(
                        shft_wf_arr, delta_t=wf.delta_t, epoch=epoch
                    )

            for _ in range(cycles):
                epoch = epoch - np.sign(rwrap) * wf.duration
                wf_arr = np.array(cyclic_shifted_wf)[:]
                cyclic_shifted_wf = pycbc.types.TimeSeries(
                    wf_arr, delta_t=wf.delta_t, epoch=epoch
                )

            assert len(cyclic_shifted_wf) == len(
                wf
            ), "Length mismatch: cyclic time shift added extra length to WF."
            return cyclic_shifted_wf
        return wf

    def set_delta_f_based_on_total_mass(self, total_mass):
        """
        Function to set `delta_f` based on the total binary mass.
        """
        if total_mass < 10:
            delta_f = 1 / 256
        elif 10 <= total_mass < 20:
            delta_f = 1 / 128
        elif 20 <= total_mass < 35:
            delta_f = 1 / 64
        else:
            delta_f = 1 / 32
        return delta_f

    def set_sample_rate_based_on_total_mass(self, total_mass):
        """
        Function to set sample_rate based on the total binary mass.
        Refer to notebook:
        `git.ligo.org/cbc-lensing/o4/planning/o4a_background_study/
        sample_rate_optimizer_based_on_fRD.ipynb`,
        or the plot:
        `git.ligo.org/cbc-lensing/o4/planning/o4a_background_study/
        data/misc/srate_optimizer_based_on_fRD_220.png`,
        for more details.
        """
        if total_mass <= 14:
            srate = 8192
        elif 14 < total_mass < 27:
            srate = 4096
        elif 27 < total_mass < 53:
            srate = 2048
        elif 53 < total_mass < 105:
            srate = 1024
        else:
            srate = 512
        return srate

    def hertz_to_mass_rescaled_frequency(self, f, total_mass):
        """
        Converts frequency from Hertz to a mass-rescaled dimensionless frequency.
        """
        return f * total_mass * self.solar_mass_to_seconds

    def mass_rescaled_frequency_to_hertz(self, fgeo, total_mass):
        """
        Converts mass-rescaled dimensionless frequency to frequency in Hertz.
        """
        return fgeo / total_mass / self.solar_mass_to_seconds

    def seconds_to_mass_rescaled_time(self, t, total_mass):
        """
        Converts time in seconds to mass-rescaled dimensionless time.
        """
        return t / total_mass / self.solar_mass_to_seconds

    def mass_rescaled_time_to_seconds(self, tgeo, total_mass):
        """
        Converts mass-rescaled dimensionless time to time in seconds.
        """
        return tgeo * total_mass * self.solar_mass_to_seconds

    def list_available_psds(self):
        """
        Return a list of all available PSDs within the gwmat framework.
        These PSD filenames can be used directly for simulating injections
        using the keys `psd_H1`, `psd_L1`, `psd_V1`.
        """
        psd_directory_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "detector_PSDs/"
        )
        psd_files = [os.path.basename(f) for f in glob(psd_directory_path + "*")]
        return sorted(psd_files)

    def sync_bilby_and_pycbc_keys(self, **params):
        # Sync bilby and pycbc keys.
        key_pairs_with_defaults = [
            ("mass_1", "mass1", None),
            ("mass_2", "mass2", None),
            ("spin_1x", "spin1x", None),
            ("spin_1y", "spin1y", None),
            ("spin_1z", "spin1z", None),
            ("spin_2x", "spin2x", None),
            ("spin_2y", "spin2y", None),
            ("spin_2z", "spin2z", None),
            ("iota", "inclination", None),
            ("luminosity_distance", "distance", None),
            ("phase", "coa_phase", 0.0),
            ("psi", "polarization", 0.0),
            ("geocent_time", "trigger_time", 0.0),
            ("approximant", "wf_approximant", "IMRPhenomXO4a"),
            ("f_start", "f_lower", 20.0),
        ]

        for key1, key2, default in key_pairs_with_defaults:
            params.update(self.sync_keys_in_dict(key1, key2, default, **params))

        return params

    def find_peak_time_from_posterior(
        self,
        posterior_data,
        ifo="H1",
        method="median",
        sample_num=1,
        approximant="IMRPhenomXO4a",
        sample_rate=16384,
        mode_array=[[2, 2]],
        f_ref=20,
    ):

        required_keys = []  # keys having float values
        for k in posterior_data.keys():
            if isinstance(posterior_data[k].values[0], float):
                required_keys.append(k)

        if method == "median":
            params = {}
            for k in required_keys:
                params[k] = np.median(posterior_data[k].values)
            params_df = pd.DataFrame([params])

        elif method == "maxL":
            maxL_index = posterior_data["log_likelihood"].idxmax()
            params = posterior_data.loc[maxL_index][required_keys]
            params_df = pd.DataFrame([params])

        elif method == "maxP":
            posterior_data["log_posterior"] = (
                posterior_data["log_likelihood"] + posterior_data["log_prior"]
            )
            maxP_index = posterior_data["log_posterior"].idxmax()
            params = posterior_data.loc[maxP_index][required_keys]
            params_df = pd.DataFrame([params])

        elif method == "random-sample":
            if len(posterior_data) < sample_num:
                sample_num = len(posterior_data)
                print(
                    "Warning: Provided `sample_num` is larger than the number of samples in the posterior."
                )
            params_df = posterior_data[::-1][: int(sample_num)][required_keys]
            # posterior_data.sample(n=int(sample_num), random_state=127)[required_keys]
        else:
            raise ValueError(
                "`method` can only take one of these values: ['median', 'maxL', 'maxP', 'random-sample']"
            )

        ifo_peak_time_array = []
        for i in tqdm(range(len(params_df))):
            params = params_df.iloc[i].to_dict()
            params.update(
                delta_t=1 / sample_rate,
                f_ref=f_ref,
                approximant=approximant,
                mode_array=mode_array,
                f_lower=20,
            )
            try:
                params = self.sync_bilby_and_pycbc_keys(**params)
                hp, hc = pycbc.waveform.get_td_waveform(**params)
                h_norm = np.sqrt(hp.data**2 + hc.data**2)
                # The waveform peaks around 0, but not exactly at 0, so we need to take into account this small shift.
                h_norm_peak_time = hp.sample_times[np.argmax(h_norm)]

                # Calculate total time shift for the peak of Sqrt(hp**2 + hc**2) to align with the detector's frame
                detector = Detector(ifo)
                ifo_timedelay = detector.time_delay_from_earth_center(
                    params["ra"], params["dec"], params["geocent_time"]
                )
                tmp_ifo_peak_time = (
                    params["geocent_time"] + ifo_timedelay + h_norm_peak_time
                )
                ifo_peak_time_array.append(tmp_ifo_peak_time)
            except:
                print(f"Caught waveform error. Ignoring iteration # {i}.")
                pass

        ifo_peak_time_array = np.asarray(ifo_peak_time_array)
        ifo_peak_time = np.median(ifo_peak_time_array)

        if method == "random-sample":
            ifo_peak_time_std = np.std(ifo_peak_time_array, ddof=1)
            print(f"len_of_tpeak_array = {len(ifo_peak_time_array)}")
            print(f"ifo_peak_time = {ifo_peak_time:.5f} \pm {ifo_peak_time_std:.5f}")
            return ifo_peak_time_array
        else:
            print(f"ifo_peak_time = {ifo_peak_time:.5f}")
            return ifo_peak_time

        # # Computing projected strain onto ifo
        # hphc_time_shift = params["geocent_time"] + ifo_timedelay
        # hp.start_time += hphc_time_shift
        # hc.start_time += hphc_time_shift
        # Fp, Fc = detector.antenna_pattern(params["ra"], params["dec"], params["polarization"], params["geocent_time"])
        # proj_hp = Fp * hp
        # proj_hc = Fc * hc
        # projected_signal = proj_hp + proj_hc

    def find_peaktime_of_injected_data(
        self, ifo_data, ifo, ra, dec, geocent_time, delta_t=0.03
    ):
        detector = Detector(ifo)
        ifo_timedelay = detector.time_delay_from_earth_center(ra, dec, geocent_time)
        sample_times = ifo_data.sample_times.data
        indices = np.where(
            ((geocent_time + ifo_timedelay - delta_t / 2) < sample_times)
            & (sample_times < (geocent_time + ifo_timedelay + delta_t / 2))
        )[0]
        tpeak = sample_times[indices[np.argmax(abs(ifo_data.data[indices]))]]
        return tpeak
