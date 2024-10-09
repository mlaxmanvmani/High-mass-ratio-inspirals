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


from functools import partial

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np

from gwbench.antenna_pattern_np import det_angles_shape
from gwbench.basic_relations import tau_spa_PN35
from gwbench.utils import REarth, time_to_rad_earth, cLight, log_msg, mod_1

cos = np.cos
sin = np.sin
exp = np.exp
log = np.log
pi  = np.pi

def detector_response(hfp, hfc, f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=None):
    '''
    Calculate the detector response for a given detector location and orientation.

    Parameters
    ----------
    hfp : jnp.ndarray
        The plus polarization.
    hfc : jnp.ndarray
        The cross polarization.
    f : jnp.ndarray
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
    hf : jnp.ndarray
        The detector response in the frequency domain.
    '''
    Fp, Fc, Flp = antenna_pattern_and_loc_phase_fac(f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=user_locs)
    return Flp * (Fp * hfp + Fc * hfc)

def antenna_pattern_and_loc_phase_fac(f, Mc, eta, tc, ra, dec, psi, loc, use_rot, user_locs=None):
    '''
    Calculate the antenna pattern and location phase factor for a given detector location and orientation.

    Parameters
    ----------
    f : jnp.ndarray
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
    Fp : jnp.ndarray
        The plus polarization antenna pattern.
    Fc : jnp.ndarray
        The cross polarization antenna pattern.
    Flp : jnp.ndarray
        The location phase factor.
    '''
    D, d       = det_ten_and_loc_vec(loc, REarth, user_locs=user_locs)
    time       = calc_rotating_time(tc, f, Mc, eta, use_rot)
    time_delay = calc_time_delay(calc_gra(ra, time), dec, d)

    return *ant_pat_funcs(D, *ant_pat_vectors(calc_gra(ra, time + time_delay), dec, psi)), loc_phase_func(f, time_delay)

@partial(jax.jit, static_argnames=['use_rot'])
def calc_rotating_time(tc, f, Mc, eta, use_rot):
    '''
    Calculate the time including the effects of the rotation of the earth
    using the stationary phase approximation, if use_rot is True.

    Parameters
    ----------
    tc : float
        The time of coalescence [s].
    f : jnp.ndarray
        The frequency domain [Hz].
    Mc : float
        The chirp Mass [solar mass
    use_rot : bool
        Include rotation effects.

    Returns
    -------
    time : jnp.ndarray
        The time including rotation effects [s].
    '''
    if use_rot: return tc - tau_spa_PN35(f, Mc, eta, log=log)
    else:       return np.array([tc])

@jax.jit
def calc_gra(ra, time):
    '''
    Calculate the Greenwich Right Ascension (GRA) for a given detector location and orientation.

    Parameters
    ----------
    ra : float
        The right ascencsion [rad].
    time : jnp.ndarray
        The corrected time [s].

    Returns
    -------
    gra : jnp.ndarray
        The Greenwich Right Ascension [rad].
    '''
    return ra - time_to_rad_earth * time

@jax.jit
def calc_time_delay(gra, dec, d):
    '''
    Calculate the time delay from the geocenter for a given detector location and orientation.

    Parameters
    ----------
    gra : jnp.ndarray
        Greenwich Right Ascension [rad].
    dec : float
        Declination [rad].
    d : jnp.ndarray
        Detector location vector.

    Returns
    -------
    time_delay : jnp.ndarray
        Time delay from the geocenter [s].
    '''
    # using cos/sin(dec) instead of cos/sin(theta) with polar angle (theta = pi/2 - dec)
    return np.matmul(d, np.array([cos(gra)*cos(dec), sin(gra)*cos(dec), sin(dec)*np.ones_like(gra)]))

@jax.jit
def loc_phase_func(f, time_delay):
    '''
    Calculate the location phase factor encoding the phase difference between the signal at the
    detector and the signal at the geocenter.

    Parameters
    ----------
    f : jnp.ndarray
        Frequency domain [Hz].
    time_delay : jnp.ndarray
        The corrected time [s].

    Returns
    -------
    Flp : jnp.ndarray
        Location phase factor
    '''
    return exp(1j * 2*pi * mod_1(f * time_delay, np=np))

@jax.jit
def ant_pat_funcs(D, XX, YY):
    '''
    Calculate the antenna pattern for a given detector location and orientation.

    Parameters
    ----------
    D : jnp.ndarray
        The detector tensor.
    XX : jnp.ndarray
        The x-arm antenna pattern vector.
    YY : jnp.ndarray
        The y-arm antenna pattern vector.

    Returns
    -------
    Fp : jnp.ndarray
        The plus polarization antenna pattern.
    Fc : jnp.ndarray
        The cross polarization antenna pattern.
    '''
    return (0.5 * (np.matmul(D,XX) * XX - np.matmul(D,YY) * YY)).sum(axis=0), \
           (0.5 * (np.matmul(D,XX) * YY + np.matmul(D,YY) * XX)).sum(axis=0)

@jax.jit
def ant_pat_vectors(gra, dec, psi):
    '''
    Calculate the antenna pattern vectors for a given detector location and orientation.

    Parameters
    ----------
    gra : jnp.ndarray
        Greenwich Right Ascension [rad]
    dec : float
        Declination [rad]
    psi : float
        Polarization angle [rad]

    Returns
    -------
    XX : jnp.ndarray
        x-arm antenna pattern vector
    YY : jnp.ndarray
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
    D : jnp.ndarray
        Detector tensor.
    d : jnp.ndarray
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

@partial(jax.jit, static_argnames=['axis'])
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
    rot : jnp.ndarray
        Rotation matrix
    '''
    c = np.cos(angle)
    s = np.sin(angle)

    if axis == 0: return np.array( [ [1,0,0], [0,c,-s], [0,s,c] ] )
    if axis == 1: return np.array( [ [c,0,s], [0,1,0], [-s,0,c] ] )
    if axis == 2: return np.array( [ [c,-s,0], [s,c,0], [0,0,1] ] )
