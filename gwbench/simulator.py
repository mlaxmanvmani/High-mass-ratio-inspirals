# Copyright (C) 2020  Ssohrab Borhanian
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


import gwbench.antenna_pattern_np as ant_pat_np

try:
    import gwbench.antenna_pattern_jx as ant_pat_jx
except ModuleNotFoundError:
    pass

def calc_wf_polarizations(net, f, inj_params):
    '''
    Calculate waveform polarizations directly from the passed frequency f and injection parameters
    inj_params and disregard whatever values are stored inside the Network object.

    Parameters
    ----------
    net : Network
        Network object.
    f : np.ndarray or jnp.ndarray
        Frequency array [Hz].
    inj_params : dict
        Injection parameters.

    Returns
    -------
    hfp : np.ndarray or jnp.ndarray
        Plus polarization.
    hfc : np.ndarray or jnp.ndarray
        Cross polarization.
    '''
    return net.wf.calc_wf_polarizations(f, inj_params)

def calc_det_responses(net, f, inj_params, trunc_f):
    '''
    Calculate detector responses.

    Calculate detector responses directly from the passed frequency f and injection parameters
    inj_params and disregard whatever values are stored inside the Network and Detector objects.

    Parameters
    ----------
    net : Network
        Network object.
    f : np.ndarray or jnp.ndarray
        Frequency array [Hz].
    inj_params : dict
        Injection parameters.
    trunc_f : bool
        Whether to truncate the frequency array of each detector according to its PSD setting.

    Returns
    -------
    hfs : list
        List of detector responses (either np.ndarray or jnp.ndarray).
    '''
    if net.wf.deriv_mod == 'jax': ant_pat = ant_pat_jx
    else:                         ant_pat = ant_pat_np

    if trunc_f: f_masks = [ (f >= det.f[0]) & (f <= det.f[-1]) for det in net.detectors ]
    else:       f_masks = len(net.detectors) * [ slice(None) ]

    hfp, hfc = calc_wf_polarizations(net, f, inj_params)

    return [ ant_pat.detector_response(hfp[f_mask], hfc[f_mask], f[f_mask],
                                       inj_params['Mc'], inj_params['eta'], inj_params['tc'],
                                       inj_params['ra'], inj_params['dec'], inj_params['psi'],
                                       det.loc, net.use_rot, user_locs=net.user_locs)
             for f_mask,det in zip(f_masks, net.detectors) ]
