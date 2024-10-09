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


import numpy as np

from gwbench.antenna_pattern_np import ant_pat_vectors, calc_gra, calc_rotating_time, calc_time_delay, det_ten_and_loc_vec, loc_phase_func
from gwbench.utils import REarth, time_to_rad_earth

cos = np.cos
sin = np.sin
pi  = np.pi

# detector response [DL, tc, phic, ra, dec, psi]
def detector_response_ana_derivs(f, params_dic, hf, hfp, hfc, Flp, loc, use_rot, user_locs=None):
    '''
    Return the paratial derivatives of the detector response with respect to the following parameters:
    - DL: the luminosity distance
    - tc: the time at coalescence
    - phic: the phase at coalescence
    - ra: the right ascension
    - dec: the declination
    - psi: the polarization angle

    Parameters
    ----------
    f : np.ndarray
        The frequency array.
    params_dic : dict
        The dictionary of the waveform parameters.
    hf : np.ndarray
        The detector response.
    hfp : np.ndarray
        The plus polarization of the waveform.
    hfc : np.ndarray
        The cross polarization of the waveform.
    Flp : np.ndarray
        The location phase factor.
    loc : np.ndarray
        The location of the detector.
    use_rot : bool
        Whether to use the rotation of the Earth or not.
    user_locs : dict, optional
        User-specified locations of the detectors.

    Returns
    -------
    dict
        The partial derivatives of the detector response.
    '''
    d_hf              = waveform_ana_derivs(f, params_dic, hf)
    d_Fp, d_Fc, d_Flp = antenna_pattern_and_loc_phase_fac_ana_derivs(f, params_dic, loc, use_rot, Flp=1, user_locs=user_locs)
    # d_Flp ~ Flp, hence in order to avoid unnecessary computations on arrays:
    # we pass Flp=1 and simply multiply with hf instead of hf/Flp

    return {
        'DL'    : d_hf['DL'],
        'phic'  : d_hf['phic'],
        'tc'    : d_hf['tc']  + Flp * (hfp * d_Fp['tc']  + hfc * d_Fc['tc'] ) + d_Flp['tc']  * hf,
        'ra'    :               Flp * (hfp * d_Fp['ra']  + hfc * d_Fc['ra'] ) + d_Flp['ra']  * hf,
        'dec'   :               Flp * (hfp * d_Fp['dec'] + hfc * d_Fc['dec']) + d_Flp['dec'] * hf,
        'psi'   :               Flp * (hfp * d_Fp['psi'] + hfc * d_Fc['psi']),
        }


# waveform (polarization or detector response) [DL, tc, phic]
def waveform_ana_derivs(f, params_dic, hf):
    '''
    Return the partial derivatives of the waveform (either a waveform polarization or the detector response)
    with respect to the following parameters:
    - DL: the luminosity distance
    - tc: the time at coalescence
    - phic: the phase at coalescence

    Parameters
    ----------
    f : np.ndarray
        The frequency array.
    params_dic : dict
        The dictionary of the waveform parameters.
    hf : np.ndarray
        Waveform (either a waveform polarization or the detector response).

    Returns
    -------
    dict
        The partial derivatives of the waveform.
    '''
    return {
        'DL'    : -hf / params_dic['DL'],
        'tc'    : (1j*2*pi) * f * hf,
        'phic'  : -1j * hf,
        }


# antenna pattern and location phase factor [ra, dec, psi, tc]
def antenna_pattern_and_loc_phase_fac_ana_derivs(f, params_dic, loc, use_rot, Flp=None, user_locs=None):
    '''
    Calculate the partial derivatives of the antenna pattern functions Fp, Fc and the location phase factor Flp
    with respect to the following parameters:
    - ra: the Greenwich right ascension
    - dec: the declination
    - psi: the polarization angle
    - tc: the time of coalescence

    Parameters
    ----------
    f : np.ndarray
        The frequency array [Hz].
    params_dic : dict
        The dictionary of the waveform parameters.
    loc : np.ndarray
        The location of the detector.
    use_rot : bool
        Whether to use the rotation of the Earth or not.
    Flp : np.ndarray, optional
        The location phase factor. Can be passed if available to avoid unnecessary computations.
    user_locs : dict
        User-specified locations of the detectors.

    Returns
    -------
    del_ra_dec_psi_tc_Fp : dict
        The partial derivatives of the plus polarization of the antenna pattern.
    del_ra_dec_psi_tc_Fc : dict
        The partial derivatives of the cross polarization of the antenna pattern.
    del_ra_dec_tc_Flp : dict
        The partial derivatives of the location phase factor.
    '''
    D, d                   = det_ten_and_loc_vec(loc, REarth, user_locs=user_locs)
    time                   = calc_rotating_time(params_dic['tc'], f, params_dic['Mc'], params_dic['eta'], use_rot)
    gra_Flp                = calc_gra(params_dic['ra'], time)
    time_delay             = calc_time_delay(gra_Flp, params_dic['dec'], d)
    gra_Fpc                = calc_gra(params_dic['ra'], time + time_delay)

    del_gra_td, del_dec_td = calc_d_time_delay(gra_Flp, params_dic['dec'], f, d)
    del_ra_graFpc          =  (1 - time_to_rad_earth * del_gra_td)
    del_dec_graFpc         = -time_to_rad_earth *      del_dec_td
    del_tc_graFpc          = -time_to_rad_earth * (1 + del_gra_td)

    del_gra_dec_psi_Fpc    = np.sum(np.matmul(
        np.einsum('ijkl->jlik', calc_d_ant_pat_funcs(D, *ant_pat_vectors(gra_Fpc, params_dic['dec'], params_dic['psi']))),
        np.einsum('ijkl->ilkj', calc_d_ant_pat_vectors(gra_Fpc, params_dic['dec'], params_dic['psi'])) ), axis=0)

    if Flp is None: Flp    = loc_phase_func(f, time_delay)
    d_Flp_fac              = (1j*2*pi) * f * Flp
    del_ra_Flp             = d_Flp_fac * del_gra_td

    # return dictionaries of the partial derivatives of Fp, Fc, Flp
    return  {   'ra'  : del_gra_dec_psi_Fpc[:,0,0] * del_ra_graFpc,
                'dec' : del_gra_dec_psi_Fpc[:,0,1] + del_gra_dec_psi_Fpc[:,0,0] * del_dec_graFpc,
                'psi' : del_gra_dec_psi_Fpc[:,0,2],
                'tc'  : del_gra_dec_psi_Fpc[:,0,0] * del_tc_graFpc,
                }, \
            {   'ra'  : del_gra_dec_psi_Fpc[:,1,0] * del_ra_graFpc,
                'dec' : del_gra_dec_psi_Fpc[:,1,1] + del_gra_dec_psi_Fpc[:,1,0] * del_dec_graFpc,
                'psi' : del_gra_dec_psi_Fpc[:,1,2],
                'tc'  : del_gra_dec_psi_Fpc[:,1,0] * del_tc_graFpc,
                }, \
            {   'ra' : del_ra_Flp,
                'dec': d_Flp_fac * del_dec_td,
                'tc' : -time_to_rad_earth * del_ra_Flp,
                }


# time delay [gra, dec]
def calc_d_time_delay(gra, dec, f, d):
    '''
    Calculate the partial derivatives of the time delay with respect to the following parameters:
    - gra: the Greenwich right ascension
    - dec: the declination

    Parameters
    ----------
    gra : np.ndarray
        The Greenwhich right ascension.
    dec : np.ndarray
        The declination.
    f : np.ndarray
        The frequency array.
    d : np.ndarray
        The location vector.

    Returns
    -------
    np.ndarray
        The partial derivatives of the time delay.
    '''
    # del_gra_time_delay, del_dec_time_delay
    return np.matmul(d, np.array([ -sin(gra)*cos(dec),  cos(gra)*cos(dec),          np.zeros_like(gra) ])), \
           np.matmul(d, np.array([ -cos(gra)*sin(dec), -sin(gra)*sin(dec), cos(dec)* np.ones_like(gra) ]))


# antenna patterns [XX, YY]
def calc_d_ant_pat_funcs(D, XX, YY):
    '''
    Calculate the partial derivatives of the plus and cross polarizations of the antenna pattern with respect to the following parameters:
    - XX [array_like]: x-arm antenna pattern vector
    - YY [array_like]: y-arm antenna pattern vector

    Parameters
    ----------
    D : np.ndarray
        The detector tensor.
    XX : np.ndarray
        The x-arm antenna pattern vector.
    YY : np.ndarray
        The y-arm antenna pattern vector.

    Returns
    -------
    dict
        The partial derivatives of the plus polarization of the antenna pattern.
    dict
        The partial derivatives of the cross polarization of the antenna pattern.
    '''
    # return [ [del_XX_Fp, del_YY_Fp], [del_XX_Fc, del_YY_Fc] ]
    return np.array([[np.matmul(D,XX), -np.matmul(D,YY)], [np.matmul(D,YY), np.matmul(D,XX)]])


# antenna pattern vectors
def calc_d_ant_pat_vectors(gra, dec, psi):
    '''
    Calculate the partial derivatives of the antenna pattern vectors with respect to the following parameters:
    - gra: the Greenwich right ascension
    - dec: the declination
    - psi: the polarization angle

    Parameters
    ----------
    gra : np.ndarray
        The Greenwich right ascension.
    dec : np.ndarray
        The declination.
    psi : np.ndarray
        The polarization angle.

    Returns
    -------
    np.ndarray
        The partial derivatives of the antenna pattern vector XX.
    np.ndarray
        The partial derivatives of the antenna pattern vector YY.
    '''
    # return [ [del_gra_XX, del_dec_XX, del_psi_XX],
    #          [del_gra_YY, del_dec_YY, del_psi_YY] ]
    return np.array([
        np.array([ [  cos(psi)*cos(gra) + sin(psi)*sin(gra)*sin(dec),  cos(psi)*sin(gra) - sin(psi)*cos(gra)*sin(dec),                   np.zeros_like(gra) ],
                   [                    - sin(psi)*cos(gra)*cos(dec),                    - sin(psi)*sin(gra)*cos(dec), -sin(psi)*sin(dec)*np.ones_like(gra) ],
                   [ -sin(psi)*sin(gra) - cos(psi)*cos(gra)*sin(dec),  sin(psi)*cos(gra) - cos(psi)*sin(gra)*sin(dec),  cos(psi)*cos(dec)*np.ones_like(gra) ]]),
       -np.array([ [ -sin(psi)*cos(gra) + cos(psi)*sin(gra)*sin(dec), -sin(psi)*sin(gra) - cos(psi)*cos(gra)*sin(dec),                   np.zeros_like(gra) ],
                   [                    - cos(psi)*cos(gra)*cos(dec),                    - cos(psi)*sin(gra)*cos(dec), -cos(psi)*sin(dec)*np.ones_like(gra) ],
                   [ -cos(psi)*sin(gra) + sin(psi)*cos(gra)*sin(dec),  cos(psi)*cos(gra) + sin(psi)*sin(gra)*sin(dec), -sin(psi)*cos(dec)*np.ones_like(gra) ]])])
