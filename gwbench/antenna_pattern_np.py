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

from gwbench.basic_relations import tau_spa_PN35
from gwbench.utils import REarth, time_to_rad_earth, cLight, log_msg, mod_1

cos = np.cos
sin = np.sin
exp = np.exp
log = np.log
pi  = np.pi

ap_symbs_string = 'f Mc eta tc ra dec psi'

def detector_response(hfp, hfc, f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=None):
    '''
    Calculate the detector response for a given detector location and orientation.

    Parameters
    ----------
    hfp : np.ndarray
        The plus polarization.
    hfc : np.ndarray
        The cross polarization.
    f : np.ndarray
        The frequency domain array [Hz].
    Mc : float
        The chirp Mass [solar mass].
    eta: float
        The symmetric mass ratio.
    tc : float
        The time of coalescence [s].
    dec : float
        The declination [rad].
    ra : float
        The right ascension [rad].
    psi : float
        The polarization angle [rad].
    loc : str
        The location (and implied orientation) of a detector.
    use_rot : bool
        Use frequency dependent time due to rotation of earth and SPA.
    user_locs : dict, optional
        User defined locations and orientations of detectors.

    Returns
    -------
    hf : np.ndarray
        The detector response in the frequency domain.
    '''
    Fp, Fc, Flp = antenna_pattern_and_loc_phase_fac(f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=user_locs)
    return Flp * (Fp * hfp + Fc * hfc)

def antenna_pattern_and_loc_phase_fac(f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=None):
    '''
    Calculate the antenna pattern and location phase factor for a given detector location and orientation.

    Parameters
    ----------
    f : np.ndarray
        The frequency domain [Hz].
    Mc : float
        The chirp Mass [solar mass].
    eta: float
        The symmetric mass ratio.
    tc : float
        The time of coalescence [s].
    dec : float
        The declination [rad].
    ra : float
        The right ascencsion [rad].
    psi : float
        The polarization angle [rad].
    loc : str
        The location (and implied orientation) of a detector.
    use_rot : bool
        Use frequency dependent time due to rotation of earth and SPA.
    user_locs : dict, optional
        User defined locations and orientations of detectors.

    Returns
    -------
    Fp : np.ndarray
        The plus polarization antenna pattern.
    Fc : np.ndarray
        The cross polarization antenna pattern.
    Flp : np.ndarray
        The location phase factor.
    '''
    D, d       = det_ten_and_loc_vec(loc, REarth, user_locs=user_locs)
    time       = calc_rotating_time(tc, f, Mc, eta, use_rot)
    time_delay = calc_time_delay(calc_gra(ra, time), dec, d)

    return *ant_pat_funcs(D, *ant_pat_vectors(calc_gra(ra, time + time_delay), dec, psi)), loc_phase_func(f, time_delay)

def calc_rotating_time(tc, f, Mc, eta, use_rot):
    '''
    Calculate the time including the effects of the rotation of the earth
    using the stationary phase approximation, if use_rot is True.

    Parameters
    ----------
    tc : float
        The time of coalescence [s].
    f : np.ndarray
        The frequency domain [Hz].
    Mc : float
        The chirp Mass [solar mass
    use_rot : bool
        Include rotation effects.

    Returns
    -------
    time : np.ndarray
        The time including rotation effects [s].
    '''
    if use_rot: return tc - tau_spa_PN35(f, Mc, eta, log=log)
    else:       return np.array([tc])

def calc_gra(ra, time):
    '''
    Calculate the Greenwich Right Ascension (GRA) for a given detector location and orientation.

    Parameters
    ----------
    ra : float
        The right ascencsion [rad].
    time : np.ndarray
        The corrected time [s].

    Returns
    -------
    gra : np.ndarray
        The Greenwich Right Ascension [rad].
    '''
    return ra - time_to_rad_earth * time

def calc_time_delay(gra, dec, d):
    '''
    Calculate the time delay from the geocenter for a given detector location and orientation.

    Parameters
    ----------
    gra : np.ndarray
        Greenwich Right Ascension [rad].
    dec : float
        Declination [rad].
    d : np.ndarray
        Detector location vector.

    Returns
    -------
    time_delay : np.ndarray
        Time delay from the geocenter [s].
    '''
    # using cos/sin(dec) instead of cos/sin(theta) with polar angle (theta = pi/2 - dec)
    return np.matmul(d, np.array([cos(gra)*cos(dec), sin(gra)*cos(dec), sin(dec)*np.ones_like(gra)]))

def loc_phase_func(f, time_delay):
    '''
    Calculate the location phase factor encoding the phase difference between the signal at the
    detector and the signal at the geocenter.

    Parameters
    ----------
    f : np.ndarray
        Frequency domain [Hz].
    time_delay : np.ndarray
        The corrected time [s].

    Returns
    -------
    Flp : np.ndarray
        Location phase factor
    '''
    return exp(1j * 2*pi * mod_1(f * time_delay, np=np))

def ant_pat_funcs(D, XX, YY):
    '''
    Calculate the antenna pattern for a given detector location and orientation.

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
    Fp : np.ndarray
        The plus polarization antenna pattern.
    Fc : np.ndarray
        The cross polarization antenna pattern.
    '''
    return (0.5 * (np.matmul(D,XX) * XX - np.matmul(D,YY) * YY)).sum(axis=0), \
           (0.5 * (np.matmul(D,XX) * YY + np.matmul(D,YY) * XX)).sum(axis=0)

def ant_pat_vectors(gra, dec, psi):
    '''
    Calculate the antenna pattern vectors for a given detector location and orientation.

    Parameters
    ----------
    gra : np.ndarray
        Greenwich Right Ascension [rad]
    dec : float
        Declination [rad]
    psi : float
        Polarization angle [rad]

    Returns
    -------
    XX : np.ndarray
        x-arm antenna pattern vector
    YY : np.ndarray
        y-arm antenna pattern vector
    '''
    return np.array([  cos(psi)*sin(gra) - sin(psi)*cos(gra)*sin(dec),
                      -cos(psi)*cos(gra) - sin(psi)*sin(gra)*sin(dec),
                                np.ones_like(gra) * sin(psi)*cos(dec) ]), \
          -np.array([ -sin(psi)*sin(gra) - cos(psi)*cos(gra)*sin(dec),
                       sin(psi)*cos(gra) - cos(psi)*sin(gra)*sin(dec),
                                np.ones_like(gra) * cos(psi)*cos(dec) ])

def det_ten_and_loc_vec(loc, R, user_locs=None):
    '''
    Calculate the detector tensor and location vector for a given detector location and orientation.

    Parameters
    ----------
    loc : str
        Location (and implied orientation) of a detector.
    R : float
        Radius of earth [m].
    user_locs : dict, optional
        User defined locations and orientations of detectors.

    Returns
    -------
    D : np.ndarray
        Detector tensor.
    d : np.ndarray
        Detector location vector.
    '''
    i_vec = np.array([1,0,0])
    j_vec = np.array([0,1,0])
    k_vec = np.array([0,0,1])

    et_vec2 = ( i_vec + np.sqrt(3.)*j_vec)/2.
    et_vec3 = (-i_vec + np.sqrt(3.)*j_vec)/2.

    alpha, beta, gamma, shape = det_angles_shape(loc, user_locs=user_locs)
    # insert polar angle theta = pi/2 - beta instead of latitude beta
    EulerD1 = np.matmul(np.matmul(rot_mat(alpha, 2), rot_mat(pi/2 - beta, 1)), rot_mat(gamma, 2))

    if   shape == 'V3':
        eDArm1 = -1 * np.matmul(EulerD1,et_vec2)
        eDArm2 = -1 * np.matmul(EulerD1,et_vec3)
    elif shape == 'V2':
        eDArm1 =      np.matmul(EulerD1,et_vec3)
        eDArm2 = -1 * np.matmul(EulerD1,i_vec)
    elif shape == 'V1':
        eDArm1 =      np.matmul(EulerD1,i_vec)
        eDArm2 =      np.matmul(EulerD1,et_vec2)
    elif shape == 'L':
        eDArm1 =      np.matmul(EulerD1,i_vec)
        eDArm2 =      np.matmul(EulerD1,j_vec)

    return np.outer(eDArm1,eDArm1) - np.outer(eDArm2,eDArm2), -R/cLight * np.matmul(EulerD1,k_vec)

def rot_mat(angle, axis):
    '''
    Calculate the rotation matrix for a given angle and axis.

    Parameters
    ----------
    angle : float
        Rotation angle [rad]
    axis : int
        Rotation axis (0, 1, or 2)

    Returns
    -------
    rot : np.ndarray
        Rotation matrix
    '''
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 0: return np.array( [ [1,0,0], [0,c,-s], [0,s,c] ] )
    if axis == 1: return np.array( [ [c,0,s], [0,1,0], [-s,0,c] ] )
    if axis == 2: return np.array( [ [c,-s,0], [s,c,0], [0,0,1] ] )

def correct_arm_azimuth(arm_azimuth, which_arm):
    '''
    Correct the azimuth of the interferometer arm.

    Parameters
    ----------
    arm_azimuth : float
        Azimuth of the interferometer arm [rad]
    which_arm : str
        Interferometer arm ('x', 'y', or 'bisector')

    Returns
    -------
    corr_arm_azimuth : float
        Corrected azimuth of the interferometer arm [rad]
    '''
    if   which_arm == 'x':        return pi/2 + arm_azimuth
    elif which_arm == 'y':        return        arm_azimuth
    elif which_arm == 'bisector': return pi/4 + arm_azimuth
    else: log_msg(f'correct_arm_azimuth: Unknown interferometer arm specifier {which_arm}, available options are: "x", "y", or "bisect"!', level='ERROR')

def det_angles_shape(loc, user_locs=None):
    '''
    Calculate the detector angles and shape for a given detector location.

    Parameters
    ----------
    loc : str
        Location (and implied orientation) of a detector.
    user_locs : dict
        User defined locations and orientations of detectors.

    Returns
    -------
    alpha : float
        Longitude [rad]
    beta : float
        Latitude [rad]
    gamma : float
        Angle from 'Due East' to y-arm [rad] (for in-built locations)
    shape : str
        Shape of interferometer (e.g. 'L', 'V1', ...)

    Note
    ----
    Switched to more precise angles for H, L, V, K, I, ET1, ET2, ET3 on 2022_03_29, found at:
    https://lscsoft.docs.ligo.org/lalsuite/lal/_l_a_l_detectors_8h_source.html
    '''
    if user_locs is not None and loc in user_locs:
        user_loc = user_locs[loc]
        return user_loc['longitude'], user_loc['latitude'], correct_arm_azimuth(user_loc['arm_azimuth'], user_loc['which_arm']), user_loc['shape']
    # 2G sites
    elif loc == 'H':     return -2.08405676917,  0.81079526383, correct_arm_azimuth(pi-5.65487724844, 'y'), 'L'
    elif loc == 'L':     return -1.58430937078,  0.53342313506, correct_arm_azimuth(pi-4.40317772346, 'y'), 'L'
    elif loc == 'V':     return  0.18333805213,  0.76151183984, correct_arm_azimuth(pi-0.33916285222, 'y'), 'L'
    elif loc == 'K':     return  2.396441015,    0.6355068497,  correct_arm_azimuth(pi-1.054113,      'y'), 'L'
    elif loc == 'I':     return  1.33401332494,  0.24841853020, correct_arm_azimuth(pi-1.57079637051, 'y'), 'L'
    elif loc == 'LHO':   return -2.06982474503,  0.81079270401, correct_arm_azimuth(2.19910516123923, 'x'), 'L'
    elif loc == 'LLO':   return -1.5572845695,   0.53342110078, correct_arm_azimuth(3.45080197126464, 'x'), 'L'
    elif loc == 'LIO':   return  1.3444416672,   0.34231239582, correct_arm_azimuth(2.05248780779809, 'x'), 'L'
    # fiducial ET sites
    elif loc == 'ET1':   return  0.18333805213,  0.76151183984, correct_arm_azimuth(pi-0.33916285222, 'y'), 'V1'
    elif loc == 'ET2':   return  0.18333805213,  0.76151183984, correct_arm_azimuth(pi-0.33916285222, 'y'), 'V2'
    elif loc == 'ET3':   return  0.18333805213,  0.76151183984, correct_arm_azimuth(pi-0.33916285222, 'y'), 'V3'
    elif loc == 'ETS1':  return  0.1643518379,   0.70714923527, correct_arm_azimuth(1.5707963267949,  'x'), 'V1'
    elif loc == 'ETS2':  return  0.1643518379,   0.70714923527, correct_arm_azimuth(1.5707963267949,  'x'), 'V2'
    elif loc == 'ETS3':  return  0.1643518379,   0.70714923527, correct_arm_azimuth(1.5707963267949,  'x'), 'V3'
    # fiducial CE sites
    elif loc == 'C':     return -1.9691740,      0.764918,      correct_arm_azimuth(0.,               'y'), 'L'
    elif loc == 'N':     return -1.8584265,      0.578751,      correct_arm_azimuth(-pi/3.,           'y'), 'L'
    elif loc == 'S':     return  2.5307270,     -0.593412,      correct_arm_azimuth(pi/4.,            'y'), 'L'
    elif loc == 'CEA':   return -2.18166156499,  0.80285145592, correct_arm_azimuth(4.53785605518526, 'x'), 'L'
    elif loc == 'CEB':   return -1.64060949687,  0.50614548308, correct_arm_azimuth(3.49065850398866, 'x'), 'L'
    elif loc == 'CES':   return  2.5307270,     -0.593412,      correct_arm_azimuth(pi/4.,            'y'), 'L'
    else: log_msg(f'det_angles_shape: Provided location {loc} is neither among the implemented locations nor in provided user_locs.', level='ERROR')
