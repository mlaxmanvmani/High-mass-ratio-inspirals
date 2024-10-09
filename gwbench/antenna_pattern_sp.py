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
import sympy as sp

from gwbench.antenna_pattern_np import ap_symbs_string, det_angles_shape
from gwbench.basic_relations import tau_spa_PN35
from gwbench.utils import REarth, time_to_rad_earth, cLight

cos = sp.cos
sin = sp.sin
exp = sp.exp
log = sp.log
pi  = np.pi

f, Mc, eta, tc, ra, dec, psi = sp.symbols(ap_symbs_string, real=True)

def detector_response(hfp, hfc, loc, use_rot, user_locs=None):
    '''
    Calculate the sympy expression of the detector response for a given detector location and orientation.

    Parameters
    ----------
    hfp : sympy expression
        The plus polarization.
    hfc : sympy expression
        The cross polarization.
    loc : str
        The location (and implied orientation) of a detector.
    use_rot : bool
        Use frequency dependent time due to rotation of earth and SPA.
    user_locs : dict, optional
        User defined locations and orientations of detectors.

    Returns
    -------
    hf : sympy expression
        The detector response in the frequency domain.
    '''
    Fp, Fc, Flp = antenna_pattern_and_loc_phase_fac(loc, use_rot, user_locs=user_locs)
    return Flp * (Fp * hfp + Fc * hfc)

def antenna_pattern_and_loc_phase_fac(loc, use_rot, user_locs=None):
    '''
    Calculate the sympy expression of the antenna pattern and location phase factor for a given detector location and orientation.

    Parameters
    ----------
    loc : str
        location (and implied orientation) of a detector
    use_rot : bool
        use frequency dependent time due to rotation of earth and SPA
    user_locs : dict, optional
        user defined locations and orientations of detectors

    Returns
    -------
    Fp : sympy expression
        The plus polarization antenna pattern.
    Fc : sympy expression
        The cross polarization antenna pattern.
    Flp : sympy expression
        The location phase factor.
    '''
    D, d       = det_ten_and_loc_vec(loc, REarth, user_locs=user_locs)
    time       = calc_rotating_time(tc, f, Mc, eta, use_rot)
    time_delay = calc_time_delay(calc_gra(ra, time), dec, d)

    return *ant_pat_funcs(D, *ant_pat_vectors(calc_gra(ra, time + time_delay), dec, psi)), loc_phase_func(f, time_delay)

def calc_rotating_time(tc, f, Mc, eta, use_rot):
    '''
    Calculate the sympy expression of the time including the effects of the rotation of the earth
    using the stationary phase approximation, if use_rot is True.

    Parameters
    ----------
    tc : symbol
        The time of coalescence [s].
    f : symbol
        The frequency domain [Hz].
    Mc : symbol
        The chirp Mass [solar mass
    use_rot : bool
        Include rotation effects.

    Returns
    -------
    time : sympy expression
        The time including rotation effects [s].
    '''
    if use_rot: return tc - tau_spa_PN35(f, Mc, eta, log=log)
    else:       return tc

def calc_gra(ra, time):
    '''
    Calculate the sympy expression of the Greenwich Right Ascension (GRA) for a given detector location and orientation.

    Parameters
    ----------
    ra : symbol
        The right ascencsion [rad].
    time : sympy expression
        The corrected time [s].

    Returns
    -------
    gra : sympy expression
        The Greenwich Right Ascension [rad].
    '''
    return ra - time_to_rad_earth * time

def calc_time_delay(gra, dec, d):
    '''
    Calculate the sympy expression of the time delay from the geocenter for a given detector location and orientation.

    Parameters
    ----------
    gra : sympy expression
        Greenwich Right Ascension [rad].
    dec : symbol
        Declination [rad].
    d : sympy expression
        Detector location vector.

    Returns
    -------
    time_delay : sympy expression
        Time delay from the geocenter [s].
    '''
    # using cos/sin(dec) instead of cos/sin(theta) with polar angle (theta = pi/2 - dec)
    return (d.T*sp.Matrix([cos(gra)*cos(dec), sin(gra)*cos(dec), sin(dec)]))[0,0]

def loc_phase_func(f, time_delay):
    '''
    Calculate the sympy expression of the location phase factor encoding the phase difference between the signal at the
    detector and the signal at the geocenter.

    Parameters
    ----------
    f : symbol
        Frequency domain [Hz].
    time_delay : sympy expression
        The corrected time [s].

    Returns
    -------
    Flp : sympy expression
        Location phase factor
    '''
    return exp(1j * 2*pi*f * time_delay)

def ant_pat_funcs(D, XX, YY):
    '''
    Calculate the sympy expression of the antenna pattern for a given detector location and orientation.

    Parameters
    ----------
    D : sympy expression
        The detector tensor.
    XX : sympy expression
        The x-arm antenna pattern vector.
    YY : sympy expression
        The y-arm antenna pattern vector.

    Returns
    -------
    Fp : sympy expression
        The plus polarization antenna pattern.
    Fc : sympy expression
        The cross polarization antenna pattern.
    '''
    return (0.5 * (XX.T*D*XX - YY.T*D*YY))[0,0], \
           (0.5 * (XX.T*D*YY + YY.T*D*XX))[0,0]

def ant_pat_vectors(gra, dec, psi):
    '''
    Calculate the sympy expression of the antenna pattern vectors for a given detector location and orientation.

    Parameters
    ----------
    gra : sympy expression
        Greenwich Right Ascension [rad]
    dec : symbol
        Declination [rad]
    psi : symbol
        Polarization angle [rad]

    Returns
    -------
    XX : sympy expression
        x-arm antenna pattern vector
    YY : sympy expression
        y-arm antenna pattern vector
    '''
    return sp.Matrix([  cos(psi)*sin(gra) - sin(psi)*cos(gra)*sin(dec),
                       -cos(psi)*cos(gra) - sin(psi)*sin(gra)*sin(dec),
                                                     sin(psi)*cos(dec) ]), \
          -sp.Matrix([ -sin(psi)*sin(gra) - cos(psi)*cos(gra)*sin(dec),
                        sin(psi)*cos(gra) - cos(psi)*sin(gra)*sin(dec),
                                                     cos(psi)*cos(dec) ])

def det_ten_and_loc_vec(loc, R, user_locs=None):
    '''
    Calculate the sympy expression of the detector tensor and location vector for a given detector location and orientation.

    Parameters
    ----------
    loc : str
        Location (and implied orientation) of a detector.
    R : sympy expression
        Radius of earth [m].
    user_locs : dict, optional
        User defined locations and orientations of detectors.

    Returns
    -------
    D : sympy expression
        Detector tensor.
    d : sympy expression
        Detector location vector.
    '''
    i_vec = sp.Matrix([1,0,0])
    j_vec = sp.Matrix([0,1,0])
    k_vec = sp.Matrix([0,0,1])

    et_vec2 = ( i_vec + np.sqrt(3.)*j_vec)/2.
    et_vec3 = (-i_vec + np.sqrt(3.)*j_vec)/2.

    alpha, beta, gamma, shape = det_angles_shape(loc, user_locs=user_locs)
    # insert polar angle theta = pi/2 - beta instead of latitude beta
    EulerD1 = rot_mat(alpha, 2) * rot_mat(pi/2 - beta, 1) * rot_mat(gamma, 2)

    if   shape == 'V3':
        eDArm1 = -1 * EulerD1*et_vec2
        eDArm2 = -1 * EulerD1*et_vec3
    elif shape == 'V2':
        eDArm1 =      EulerD1*et_vec3
        eDArm2 = -1 * EulerD1*i_vec
    elif shape == 'V1':
        eDArm1 =      EulerD1*i_vec
        eDArm2 =      EulerD1*et_vec2
    elif shape == 'L':
        eDArm1 =      EulerD1*i_vec
        eDArm2 =      EulerD1*j_vec

    return eDArm1*eDArm1.T - eDArm2*eDArm2.T, -R/cLight * EulerD1*k_vec

def rot_mat(angle, axis):
    '''
    Calculate the sympy expression of the rotation matrix for a given angle and axis.

    Parameters
    ----------
    angle : float
        Rotation angle [rad]
    axis : int
        Rotation axis (0, 1, or 2)

    Returns
    -------
    rot : sympy matrix
        Rotation matrix
    '''
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 0: return sp.Matrix( [ [1,0,0], [0,c,-s], [0,s,c] ] )
    if axis == 1: return sp.Matrix( [ [c,0,s], [0,1,0], [-s,0,c] ] )
    if axis == 2: return sp.Matrix( [ [c,-s,0], [s,c,0], [0,0,1] ] )
