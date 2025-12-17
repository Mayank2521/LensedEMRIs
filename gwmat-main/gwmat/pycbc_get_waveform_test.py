import matplotlib.pyplot as pp
from pycbc.waveform import get_td_waveform

for apx in ['SEOBNRv4', 'IMRPhenomD']:
    cbc_prms = dict(
        approximant=apx,
        mass1=10,
        mass2=10,
        a_1=0,
        a_2=0,
        tilt_1=0,
        tilt_2=0,
        phi_12=0,
        phi_jl=0,
        luminosity_distance=100,
        spin1z=0,
        spin2z=0,
        inclination=0,
        distance=1000,
        polarization=0,
        coa_phase=0,
        trig_time=1242529720,
        f_lower=20,
        delta_t=1.0/4096,
    )

    ## lensed waveform generation
    lens_prms = dict(m_lens=1000, y_lens=1, z_lens=0)
    prms = {**lens_prms, **cbc_prms}

    hp, hc = get_td_waveform({**prms, "rwrap": 0.0})
    print(type(hp), type(hc))
    pp.plot(hp.sample_times, hp, label=apx)

pp.ylabel('Strain')
pp.xlabel('Time (s)')
pp.legend()
pp.show()