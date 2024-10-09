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


import os
import warnings
from copy import copy

import astropy.cosmology as apcosm
import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

import gwbench.basic_relations as brs
from gwbench.utils import min_max_mask

###
#-----standard values for reference-----
std_vals = [{'source':'BBH', 'dist':'power_uniform', 'mmin':5, 'mmax':100, 'alpha':-1.6},
            {'source':'BNS', 'dist':'gaussian', 'mean':1.35, 'sigma':0.15},
            {'source':'BNS', 'dist':'uniform', 'mmin':0.8, 'mmax':3}]

###
#-----CBC parameters sampler-----
def injections_CBC_params_redshift(cosmo_dict, mass_dict, spin_dict, redshifted, num_injs=10, seed=None, file_path=None):
    '''
    Generate parameters [(redshifted) masses, spins, luminosity distance, redshift, inclination, right ascension, declination, polarization angle
    for compact binary coalescence (CBC) injections.

    Parameters
    ---------
    cosmo_dict : dict
        Dictionary containing cosmological parameters.
    mass_dict : dict
        Dictionary containing mass parameters.
    spin_dict : dict
        Dictionary containing spin parameters.
    redshifted : bool
        Flag indicating whether to apply redshift to the mass parameters.
    num_injs : int
        Number of injections to generate (default: 10).
    seed : int
        Seed for the random number generator (default: None).
    file_path : str
        File path to save the generated injections (default: None).

    Returns
    -------
    Mc_vec : np.ndarray
        Chirp mass values.
    eta_vec : np.ndarray
        Symmetric mass ratio values.
    chi1x_vec : np.ndarray
        Spin x-component of the primary black hole.
    chi1y_vec : np.ndarray
        Spin y-component of the primary black hole.
    chi1z_vec : np.ndarray
        Spin z-component of the primary black hole.
    chi2x_vec : np.ndarray
        Spin x-component of the secondary black hole.
    chi2y_vec : np.ndarray
        Spin y-component of the secondary black hole.
    chi2z_vec : np.ndarray
        Spin z-component of the secondary black hole.
    DL_vec : np.ndarray
        Luminosity distance values.
    iota_vec : np.ndarray
        Inclination angle values.
    ra_vec : np.ndarray
        Right ascension values.
    dec_vec : np.ndarray
        Declination values.
    psi_vec : np.ndarray
        Polarization angle values.
    z_vec : np.ndarray
        Redshift values.
    '''
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(100000, size=4)

    m1_vec, m2_vec                                                   = mass_sampler(mass_dict, num_injs, seeds[0])
    chi1x_vec, chi1y_vec, chi1z_vec, chi2x_vec, chi2y_vec, chi2z_vec = spin_sampler(spin_dict, num_injs, seeds[1])
    z_vec, DL_vec                                                    = redshift_lum_distance_sampler(cosmo_dict, num_injs, seeds[2])
    iota_vec, ra_vec, dec_vec, psi_vec                               = angle_sampler(num_injs, seeds[3])

    Mc_vec, eta_vec = get_Mc_eta(m1_vec, m2_vec)
    if redshifted: Mc_vec *= (1. + z_vec)

    params = [Mc_vec, eta_vec, chi1x_vec, chi1y_vec, chi1z_vec, chi2x_vec, chi2y_vec, chi2z_vec, DL_vec, iota_vec, ra_vec, dec_vec, psi_vec, z_vec]
    if file_path is not None: save_injections(params, file_path)
    return params

###
#-----IO functions-----
def load_injections(file_path):
    '''
    Load injection parameters from a file.

    Parameters
    ---------
    file_path : str
        File path to load the injections from.

    Returns
    -------
    params : list
        List of injection parameters (see injections.params_CBC_params_redshift for details)
    '''
    return np.transpose(np.loadtxt(file_path))

def save_injections(params,file_path):
    '''
    Save injection parameters to a file.

    Parameters
    ---------
    params : list
        List of injection parameters (see injections.params_CBC_params_redshift for details)
    file_path : str
        File path to save the injections to
    '''
    np.savetxt(os.path.join(file_path), np.transpose(np.array([params][0])), delimiter = ' ')


###
#-----angle samplers-----
def angle_sampler(num_injs, seed):
    '''
    Generate random values for the inclination, right ascension, declination, and polarization angle.
    The inclination and declination are sampled from a uniform distribution in cos(iota) and cos(dec), respectively.

    Parameters
    ----------
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    iota_vec : array
        Inclination angle values.
    ra_vec : array
        Right ascension values.
    dec_vec : array
        Declination values.
    psi_vec : array
        Polarization angle values.
    '''
    rngs = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=4)]
    iota_vec = np.arccos(rngs[0].uniform(low=-1, high=1, size=num_injs))
    ra_vec   = rngs[1].uniform(low=0., high=2.*np.pi, size=num_injs)
    dec_vec  = np.arccos(rngs[2].uniform(low=-1, high=1, size=num_injs)) - np.pi/2.
    psi_vec  = rngs[3].uniform(low=0., high=2.*np.pi, size=num_injs)
    return iota_vec, ra_vec, dec_vec, psi_vec

###
#-----spin samplers-----
def spin_sampler(spin_dict, num_injs, seed):
    '''
    Generate random values for the spins of the primary and secondary black holes.

    Parameters
    ----------
    spin_dict : dict
        Dictionary containing spin parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    chi1x_vec : np.ndarray
        Spin x-component of the primary black hole.
    chi1y_vec : np.ndarray
        Spin y-component of the primary black hole.
    chi1z_vec : np.ndarray
        Spin z-component of the primary black hole.
    chi2x_vec : np.ndarray
        Spin x-component of the secondary black hole.
    chi2y_vec : np.ndarray
        Spin y-component of the secondary black hole.
    chi2z_vec : np.ndarray
        Spin z-component of the secondary black hole.

    Note
    ----
    If the 'dist' key is not present in the spin_dict, the spins are sampled from either a 1D or 3D distribution specified
    by the 'dim' key. In the 1D case, the spins are aligned with the orbital angular momentum and the x and y components are
    set to zero while the z component is sampled from a uniform distribution in the range [chi_lo, chi_hi]. In the 3D case,
    the geometry has to be specified in the 'geom' key. If the geometry is 'cartesian', the spins are sampled from a uniform
    distribution in the range [chi_lo, chi_hi] for each component. If the geometry is 'spherical', the spins are sampled from
    a uniform distribution in the positive range [chi_lo, chi_hi] and the azimuthal and polar angles are sampled unfiromly on
    the sphere.

    If the 'dist' key is present, there are three possible distributions: 'gaussian_uniform', 'beta', and 'beta_gaussian_uniform'.
    Please check the code for the details of each distribution and additional parameters that need to be specified in the spin_dict.
    '''
    if 'dist' not in spin_dict:
        rngs   = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=6)]
        chi_lo = spin_dict['chi_lo']
        chi_hi = spin_dict['chi_hi']
        dim    = spin_dict['dim']

        if     dim == 1:
            chiz_vecs = [rngs[i].uniform(low=chi_lo, high=chi_hi, size=num_injs) for i in (2,5)]
            return [np.zeros(num_injs), np.zeros(num_injs), chiz_vecs[0], np.zeros(num_injs), np.zeros(num_injs), chiz_vecs[1]]
        elif dim == 3:
            if     spin_dict['geom'] == 'cartesian':
                return [rngs[i].uniform(low=chi_lo, high=chi_hi, size=num_injs) for i in range(6)]
            elif spin_dict['geom'] == 'spherical':
                # chi1
                chi_vec   = (rngs[0].uniform(low=chi_lo**3., high=chi_hi**3., size=num_injs))**(1./3.)
                theta_vec = np.arccos(rngs[1].uniform(low=-1, high=1, size=num_injs))
                phi_vec   = rngs[2].uniform(low=0., high=2.*np.pi, size=num_injs)
                chi1x_vec, chi1y_vec, chi1z_vec = get_cartesian_from_spherical(chi_vec,theta_vec,phi_vec)
                # chi2
                chi_vec   = (rngs[3].uniform(low=chi_lo**3., high=chi_hi**3., size=num_injs))**(1./3.)
                theta_vec = np.arccos(rngs[4].uniform(low=-1, high=1, size=num_injs))
                phi_vec   = rngs[5].uniform(low=0., high=2.*np.pi, size=num_injs)
                chi2x_vec, chi2y_vec, chi2z_vec = get_cartesian_from_spherical(chi_vec,theta_vec,phi_vec)
                return [chi1x_vec, chi1y_vec, chi1z_vec, chi2x_vec, chi2y_vec, chi2z_vec]

    elif spin_dict['dist'] == 'gaussian_uniform':
        rngs     = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=2)]
        mean     = spin_dict['mean']
        sigma    = spin_dict['sigma']
        chi1_min = spin_dict['chi1_min']
        chi1_max = spin_dict['chi1_max']
        chi2_min = spin_dict['chi2_min']
        chi2_max = spin_dict['chi2_max']

        return [np.zeros(num_injs), np.zeros(num_injs), truncated_gaussian(mean, sigma, chi1_min, chi1_max, num_injs, rng=rngs[0]),
                np.zeros(num_injs), np.zeros(num_injs), rngs[1].uniform(low=chi2_min, high=chi2_max, size=num_injs)]

    elif spin_dict['dist'] == 'beta':
        rngs  = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=2)]
        alpha = spin_dict['alpha']
        beta  = spin_dict['beta']

        return [np.zeros(num_injs), np.zeros(num_injs), rngs[0].beta(alpha, beta, num_injs),
                np.zeros(num_injs), np.zeros(num_injs), rngs[1].beta(alpha, beta, num_injs)]

    elif spin_dict['dist'] == 'beta_gaussian_uniform':
        rngs         = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=10)]
        alpha        = spin_dict['alpha']
        beta         = spin_dict['beta']
        mean         = spin_dict['mean']
        sigma        = spin_dict['sigma']
        weight       = spin_dict['weight']
        cos_min      = spin_dict['cos_min']
        cos_max      = spin_dict['cos_max']

        chi1_vec     = rngs[0].beta(alpha, beta, num_injs)
        chi2_vec     = rngs[1].beta(alpha, beta, num_injs)

        azimuth1_vec = rngs[2].uniform(low=0., high=2.*np.pi, size=num_injs)
        azimuth2_vec = rngs[3].uniform(low=0., high=2.*np.pi, size=num_injs)

        N            = int(weight * num_injs)
        M            = num_injs - N
        tilt1_vec    = np.arccos(np.concatenate((truncated_gaussian(mean, sigma, cos_min, cos_max, N, rng=rngs[4]),
                                                 rngs[5].uniform(low=cos_min, high=cos_max, size=M))))
        tilt2_vec    = np.arccos(np.concatenate((truncated_gaussian(mean, sigma, cos_min, cos_max, N, rng=rngs[6]),
                                                 rngs[7].uniform(low=cos_min, high=cos_max, size=M))))
        tilt1_vec    = tilt1_vec[rngs[8].permutation(num_injs)]
        tilt2_vec    = tilt2_vec[rngs[9].permutation(num_injs)]

        chi1xy_vec   = chi1_vec * np.sin(tilt1_vec)
        chi2xy_vec   = chi2_vec * np.sin(tilt2_vec)

        return [chi1xy_vec * np.cos(azimuth1_vec), chi1xy_vec * np.sin(azimuth1_vec), chi1_vec * np.cos(tilt1_vec),
                chi2xy_vec * np.cos(azimuth2_vec), chi2xy_vec * np.sin(azimuth2_vec), chi2_vec * np.cos(tilt2_vec)]

def get_cartesian_from_spherical(r,theta,phi):
    return r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)


###
#-----mass samplers-----
def mass_sampler(mass_dict,num_injs,seed):
    '''
    Generate random values for the masses of the primary and secondary black holes.

    Parameters
    ----------
    mass_dict : dict
        Dictionary containing mass parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    m1_vec : np.ndarray
        Mass of the primary black hole.
    m2_vec : np.ndarray
        Mass of the secondary black hole.

    Note
    ----
    The available distributions are 'gaussian', 'double_gaussian', 'lognormal', 'power', 'power_peak', 'power_peak_uniform',
    'nsbh_power_peak_uniform', 'power_uniform', and 'uniform', specified by the 'dist' key in the mass_dict. Please refer
    to the code for the details of each distribution and additional parameters that need to be specified in the mass_dict.
    '''
    rngs = [np.random.default_rng(seeed) for seeed in np.random.default_rng(seed).integers(100000,size=6)]
    if mass_dict['dist'] == 'gaussian':
        mmin  = mass_dict['mmin']
        mmax  = mass_dict['mmax']
        mean  = mass_dict['mean']
        sigma = mass_dict['sigma']
        m1_m2 = 0

        m1_vec = truncated_gaussian(mean, sigma, mmin, mmax, num_injs, rng=rngs[0])
        m2_vec = truncated_gaussian(mean, sigma, mmin, mmax, num_injs, rng=rngs[1])

    elif mass_dict['dist'] == 'double_gaussian':
        mmin   = mass_dict['mmin']
        mmax   = mass_dict['mmax']
        mean1  = mass_dict['mean1']
        sigma1 = mass_dict['sigma1']
        mean2  = mass_dict['mean2']
        sigma2 = mass_dict['sigma2']
        weight = mass_dict['weight']
        m1_m2  = 0

        N = int(weight * num_injs)
        M = num_injs - N

        m1_vec = np.concatenate((truncated_gaussian(mean1, sigma1, mmin, mmax, N, rng=rngs[0]),
                                 truncated_gaussian(mean2, sigma2, mmin, mmax, M, rng=rngs[1])))
        m2_vec = np.concatenate((truncated_gaussian(mean1, sigma1, mmin, mmax, N, rng=rngs[2]),
                                 truncated_gaussian(mean2, sigma2, mmin, mmax, M, rng=rngs[3])))
        m1_vec = m1_vec[rngs[4].permutation(num_injs)]
        m2_vec = m2_vec[rngs[5].permutation(num_injs)]

    elif mass_dict['dist'] == 'lognormal':
        mass_scale = mass_dict['mass_scale']
        mean       = mass_dict['mean']
        sigma      = mass_dict['sigma']
        m1_m2      = 1

        m1_vec = mass_scale * rngs[0].lognormal(mean, sigma, num_injs)
        m2_vec = mass_scale * rngs[1].lognormal(mean, sigma, num_injs)

    elif mass_dict['dist'] == 'power':
        mmin  = mass_dict['mmin']
        mmax  = mass_dict['mmax']
        alpha = mass_dict['alpha'] + 1
        m1_m2 = 0

        m1_vec = (mmin**alpha + (mmax**alpha - mmin**alpha)*rngs[0].random(num_injs))**(1./alpha)
        m2_vec = (mmin**alpha + (mmax**alpha - mmin**alpha)*rngs[1].random(num_injs))**(1./alpha)

    elif mass_dict['dist'] == 'power_peak':
                                                # standard power+peak parameters from GTWC-2 populations paper:
                                                # https://arxiv.org/abs/2010.14533
        mmin       = mass_dict['mmin']          # mmin       = 4.59
        mmax       = mass_dict['mmax']          # mmax       = 86.22
        m1_alpha   = mass_dict['m1_alpha']      # m1_alpha   = 2.63
        peak_frac  = mass_dict['peak_frac']     # peak_frac  = 0.1
        peak_mean  = mass_dict['peak_mean']     # peak_mu    = 33.07
        peak_sigma = mass_dict['peak_sigma']    # peak_sigma = 5.69
        delta_m    = mass_dict['delta_m']       # delta_m    = 4.82
        q_beta     = mass_dict['q_beta']        # q_beta     = 1.26
        m1_m2      = 1

        m1_vec     = power_peak(mmin, mmax, m1_alpha, peak_frac, peak_mean, peak_sigma, delta_m, num_injs, nm1s=5001, rng=rngs[0])

        q_vec      = rngs[1].power(q_beta + 1, num_injs)
        m2_vec     = q_vec * m1_vec
        m2_mask    = m2_vec < mmin

        while m2_mask.sum():
            q_vec[m2_mask] = rngs[1].power(q_beta + 1, m2_mask.sum())
            m2_vec         = q_vec * m1_vec
            m2_mask        = m2_vec < mmin

    elif mass_dict['dist'] == 'power_peak_uniform':
        mmin       = mass_dict['mmin']
        mmax       = mass_dict['mmax']
        m1_alpha   = mass_dict['m1_alpha']
        peak_frac  = mass_dict['peak_frac']
        peak_mean  = mass_dict['peak_mean']
        peak_sigma = mass_dict['peak_sigma']
        delta_m    = mass_dict['delta_m']
        m1_m2      = 1

        m1_vec     = power_peak(mmin, mmax, m1_alpha, peak_frac, peak_mean, peak_sigma, delta_m, num_injs, nm1s=5001, rng=rngs[0])
        m2_vec     = rngs[1].uniform(mmin, m1_vec)

    elif mass_dict['dist'] == 'nsbh_power_peak_uniform':
        m1_min     = mass_dict['m1_min']
        m1_max     = mass_dict['m1_max']
        m1_alpha   = mass_dict['m1_alpha']
        peak_frac  = mass_dict['peak_frac']
        peak_mean  = mass_dict['peak_mean']
        peak_sigma = mass_dict['peak_sigma']
        delta_m    = mass_dict['delta_m']
        m2_min     = mass_dict['m2_min']
        m2_max     = mass_dict['m2_max']
        m1_m2      = 1

        m1_vec     = power_peak(m1_min, m1_max, m1_alpha, peak_frac, peak_mean, peak_sigma, delta_m, num_injs, nm1s=5001, rng=rngs[0])
        m2_vec     = rngs[1].uniform(low=m2_min, high=m2_max, size=num_injs)

    elif mass_dict['dist'] == 'power_uniform':
        mmin  = mass_dict['mmin']
        mmax  = mass_dict['mmax']
        alpha = mass_dict['alpha'] + 1
        m1_m2 = 1

        m1_vec = (mmin**alpha + (mmax**alpha - mmin**alpha)*rngs[0].random(num_injs))**(1./alpha)
        m2_vec = rngs[1].uniform(mmin,m1_vec)

    elif mass_dict['dist'] == 'uniform':
        mmin  = mass_dict['mmin']
        mmax  = mass_dict['mmax']
        m1_m2 = 0

        m1_vec = rngs[0].uniform(low=mmin, high=mmax, size=num_injs)
        m2_vec = rngs[1].uniform(low=mmin, high=mmax, size=num_injs)

    elif mass_dict['dist'] == 'fixed_m1_q':
        m1    = mass_dict['m1']
        if mass_dict['q'] > 1: m2 = m1 / mass_dict['q']
        else:                  m2 = m1 * mass_dict['q']
        m1_m2 = 1

        m1_vec = m1 * np.ones(num_injs)
        m2_vec = m2 * np.ones(num_injs)

    if m1_m2: return m1_vec, m2_vec
    else:     return make_m1_m2(m1_vec,m2_vec,num_injs!=1)

#-----mass handling functions-----
def get_Mc_eta(m1_vec,m2_vec):
    '''
    Calculate the chirp mass and symmetric mass ratio from the primary and secondary black hole masses.

    Parameters
    ----------
    m1_vec : np.ndarray
        Mass of the primary black hole.
    m2_vec : np.ndarray
        Mass of the secondary black hole.

    Returns
    -------
    Mc_vec : np.ndarray
        Chirp mass values.
    eta_vec : np.ndarray
        Symmetric mass ratio values.
    '''
    eta_vec = brs.eta_of_q(m1_vec/m2_vec)
    Mc_vec  = brs.Mc_of_M_eta(m1_vec+m2_vec,eta_vec)
    return Mc_vec, eta_vec

def make_m1_m2(m1,m2,vec=True):
    '''
    Ensure that m1 >= m2.

    Parameters
    ----------
    m1 : np.ndarray
        Mass of the primary black hole.
    m2 : np.ndarray
        Mass of the secondary black hole.
    vec : bool
        Flag indicating whether to apply the operation to arrays (default: True).

    Returns
    -------
    m1 : np.ndarray
        Mass of the primary black hole.
    m2 : np.ndarray
        Mass of the secondary black hole.
    '''
    if vec:
        mt      = copy(m1)
        ids     = np.where(m1<m2)
        m1[ids] = m2[ids]
        m2[ids] = mt[ids]
        return m1, m2
    else:
        if m1 < m2: return m2, m1
        else:       return m1, m2

#-----power-peak helpers-----
def power(m, alpha):
    '''
    Power-law distribution.

    Parameters
    ----------
    m : np.ndarray
        Mass values.
    alpha : float
        Power-law exponent.

    Returns
    -------
    np.ndarray
        Power-law distribution.
    '''
    return m**(alpha)

def gaussian(m, mean, sigma):
    '''
    Gaussian distribution.

    Parameters
    ----------
    m : np.ndarray
        Mass values.
    mean : np.ndarray
        Mean value.
    sigma : np.ndarray
        Standard deviation.

    Returns
    -------
    np.ndarray
        Gaussian distribution.
    '''
    return np.exp(-((m - mean) / sigma)**2 / 2)

def smoothing(m, mmin, delta_m):
    '''
    Smoothing function.

    Parameters
    ----------
    m : np.ndarray
        Mass values.
    mmin : float
        Minimum mass value.
    delta_m : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Smoothed array.
    '''
    m_arr  = np.array(m)
    res    = np.zeros_like(m_arr)
    res[np.nonzero(m_arr >= mmin + delta_m)[0]] = 1
    ids    = np.nonzero(min_max_mask(m_arr, mmin, mmin + delta_m, strict_max=True))[0]
    m_arr -= mmin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res[ids] = 1 / (1 + np.exp(delta_m/m_arr[ids] + delta_m/(m_arr[ids] - delta_m)))
    return res

def power_peak(mmin, mmax, m1_alpha, peak_frac, peak_mean, peak_sigma, delta_m, num_injs, nm1s=5001, rng=None, seed=None):
    '''
    Power-law distribution with a Gaussian peak.

    Parameters
    ----------
    mmin : float
        Minimum mass value.
    mmax : float
        Maximum mass value.
    m1_alpha : float
        Power-law exponent.
    peak_frac : float
        Fraction of the peak.
    peak_mean : float
        Mean of the peak.
    peak_sigma : float
        Standard deviation of the peak.
    delta_m : float
        Smoothing parameter.
    num_injs : int
        Number of injections to generate.
    nm1s : int
        Number of mass values to sample (default: 5001).
    rng : np.random.Generator
        Random number generator (default: None).
    seed : int
        Seed for the random number generator (default: None).

    Returns
    -------
    np.ndarray
        Mass values.
    '''
    if rng is None: rng = np.random.default_rng(seed)
    m1s            = np.linspace(mmin, mmax, nm1s)
    power_part     = (1 - peak_frac) * power(m1s, -m1_alpha) / simpson(power(m1s, -m1_alpha), x=m1s)
    gauss_part     = peak_frac * gaussian(m1s, peak_mean, peak_sigma) / simpson(gaussian(m1s, peak_mean, peak_sigma), x=m1s)
    m1_dist        = (power_part + gauss_part) * smoothing(m1s, mmin, delta_m)
    window_cdf     = np.array([simpson(m1_dist[:i], x=m1s[:i]) for i in range(1, len(m1s)+1)]) / simpson(m1_dist, x=m1s)
    inv_window_cdf = interp1d(window_cdf, m1s)
    return inv_window_cdf(rng.random(num_injs))

#-----general helpers-----
def truncated_gaussian(mean, sigma, minv, maxv, num_injs, rng=None, seed=None):
    '''
    Truncated Gaussian distribution.

    Parameters
    ----------
    mean : float
        Mean value.
    sigma : float
        Standard deviation.
    minv : float
        Minimum value.
    maxv : float
        Maximum value.
    num_injs : int
        Number of injections to generate.
    rng : np.random.Generator
        Random number generator (default: None).
    seed : int
        Seed for the random number generator (default: None).

    Returns
    -------
    np.ndarray
        Truncated Gaussian distribution.
    '''
    if rng is None: rng = np.random.default_rng(seed)
    sample_vec = np.zeros(num_injs)
    ids        = np.arange(num_injs)
    while ids.size > 0:
        sample_vec[ids] = rng.normal(loc=mean, scale=sigma, size=ids.size)
        ids             = np.nonzero(np.logical_not(min_max_mask(sample_vec, minv, maxv, strict_min=True, strict_max=True)))[0]
    return sample_vec


###
#-----redshift and lum distance samplers-----
def redshift_lum_distance_sampler(cosmo_dict, num_injs, seed):
    '''
    Generate random values for the redshift and luminosity distance.

    Parameters
    ----------
    cosmo_dict : dict
        Dictionary containing cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    z_vec : np.ndarray
        Redshift values.
    DL_vec : np.ndarray
        Luminosity distance values.

    Note
    ----
    The available distributions are 'uniform', 'uniform_in_bins', 'uniform_comoving_volume_inversion',
    'uniform_comoving_volume_rejection', 'mdbn_rate_inversion', 'bns_md_rate_inversion', and 'popIII_rate_inversion',
    specified by the 'sampler' key in the cosmo_dict. Please refer to the code for the details of each distribution
    and additional parameters that need to be specified in the cosmo_dict.
    '''
    zmin = cosmo_dict['zmin']
    zmax = cosmo_dict['zmax']

    keys = list(cosmo_dict.keys())
    if 'Om0' in keys:  Om0 = cosmo_dict['Om0']
    else:              Om0 = None
    if 'Ode0' in keys: Ode0 = cosmo_dict['Ode0']
    else:              Ode0 = None
    if 'H0' in keys:   H0 = cosmo_dict['H0']
    else:              H0 = None

    if None in (Om0,Ode0,H0): cosmo = apcosm.Planck18
    else:                     cosmo = apcosm.LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

    if   cosmo_dict['sampler'] == 'uniform':
        z_vec = np.random.default_rng(seed).uniform(low=zmin, high=zmax, size=num_injs)
    elif cosmo_dict['sampler'] == 'uniform_in_bins':
        z_vec = np.concatenate([
            np.random.default_rng((i+1)*seed).uniform(low=zbin[0], high=zbin[1], size=cosmo_dict['injs_per_bin'][i])
            for i,zbin in enumerate(cosmo_dict['bins']) ])
    elif cosmo_dict['sampler'] == 'uniform_comoving_volume_inversion':
        z_vec = uniform_comoving_volume_redshift_inversion_sampler(zmin,zmax,cosmo,num_injs,seed,nzs=None)
    elif cosmo_dict['sampler'] == 'uniform_comoving_volume_rejection':
        z_vec = uniform_comoving_volume_redshift_rejection_sampler(zmin,zmax,cosmo,num_injs,seed,nzs=40)
    elif cosmo_dict['sampler'] == 'mdbn_rate_inversion':
        z_vec = mdbn_merger_rate_uniform_comoving_volume_redshift_inversion_sampler(zmin,zmax,cosmo,num_injs,seed,nzs=None)
    elif cosmo_dict['sampler'] == 'bns_md_rate_inversion':
        z_vec = bns_md_merger_rate_uniform_comoving_volume_redshift_inversion_sampler(zmin,zmax,cosmo,num_injs,seed,nzs=None)
    elif cosmo_dict['sampler'] == 'popIII_rate_inversion':
        z_vec = popIII_merger_rate_uniform_comoving_volume_redshift_inversion_sampler(zmin, zmax, cosmo, num_injs, seed, nzs=None)
    elif cosmo_dict['sampler'] == 'pbh_power_age':
        z_vec = pbh_universe_age_power_law_sampler(zmin, zmax, cosmo, num_injs, seed)

    np.random.default_rng(seed).shuffle(z_vec)

    return z_vec, cosmo.luminosity_distance(z_vec).value

#-----redshift samplers-----
def uniform_comoving_volume_redshift_inversion_sampler(zmin, zmax, cosmo, num_injs, seed=None, nzs=None):
    '''
    Generate redshift values using inversion sampling for a uniform distribution in comoving volume.

    Parameters
    ----------
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    cosmo : astropy.cosmology
        Cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator (default: None).
    nzs : int
        Number of redshift bins to sample (default: None).

    Returns
    -------
    np.ndarray
        Redshift values.
    '''
    rng = np.random.default_rng(seed)
    if nzs is None: nzs = max(5001, 50 * int((zmax-zmin)) + 1)
    zs = np.linspace(zmin,zmax,nzs)
    dist = (lambda z: ((4.*np.pi*cosmo.differential_comoving_volume(z).value)/(1.+z)))(zs)
    window_cdf = np.array([simpson(dist[:i], x=zs[:i]) for i in range(1,nzs+1)]) / simpson(dist, x=zs)
    inv_window_cdf = interp1d(window_cdf, zs)
    return inv_window_cdf(rng.random(num_injs))

def uniform_comoving_volume_redshift_rejection_sampler(zmin, zmax, cosmo, num_injs, seed=None, nzs=None):
    '''
    Generate redshift values using rejection sampling for a uniform distribution in comoving volume.

    Parameters
    ----------
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    cosmo : astropy.cosmology
        Cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator (default: None).
    nzs : int
        Number of redshift bins to sample (default: None).

    Returns
    -------
    np.ndarray
        Redshift values.
    '''
    rng = np.random.default_rng(seed)
    dist = lambda z: ((4.*np.pi*cosmo.differential_comoving_volume(z).value)/(1.+z))
    window_norm = quad(dist, zmin, zmax)[0]
    flip_window_pdf = lambda z: -dist(z) / window_norm
    window_pdf_max = -minimize_scalar(flip_window_pdf,bounds=[zmin,zmax],method='bounded').fun

    if nzs is None or nzs < 2: nzs = 2
    zs = np.linspace(zmin,zmax,nzs)
    segment_nums = np.asarray((num_injs * np.array([-quad(flip_window_pdf,zs[i],zs[i+1])[0] for i in range(nzs-1)])), dtype=int)
    segment_pts = np.sum(segment_nums)
    segment_maxs = np.array([-minimize_scalar(flip_window_pdf,bounds=[zs[i],zs[i+1]],method='bounded').fun for i in range(nzs-1)])

    z_sample = np.zeros(num_injs)

    for j,num in enumerate(segment_nums):
        id_shift = np.sum(segment_nums[:j+1]) - num
        ids = np.arange(num) + id_shift
        while ids.size > 0:
            z_sample[ids] = rng.uniform(zs[j],zs[j+1],ids.size)
            ids = ids[np.nonzero(rng.uniform(0.,segment_maxs[j],ids.size) >= -flip_window_pdf(z_sample[ids]))[0]]

    if nzs > 2:
        ids = np.arange(num_injs - segment_pts) + segment_pts
        while ids.size > 0:
            z_sample[ids] = rng.uniform(zmin,zmax,ids.size)
            ids = ids[np.nonzero(rng.uniform(0.,window_pdf_max,ids.size) >= -flip_window_pdf(z_sample[ids]))[0]]
        rng.shuffle(z_sample)

    return z_sample

def mdbn_merger_rate_uniform_comoving_volume_redshift_inversion_sampler(zmin, zmax, cosmo, num_injs, seed=None, nzs=None):
    '''
    Generate redshift values using inversion sampling for the Madau-Dickinson-Belczynski-Ng field BBH volumetric merger rate.

    Parameters
    ----------
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    cosmo : astropy.cosmology
        Cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator (default: None).
    nzs : int
        Number of redshift bins to sample (default: None).

    Returns
    -------
    np.ndarray
        Redshift values.
    '''
    rng = np.random.default_rng(seed)
    if nzs is None: nzs = max(5001, 50 * int((zmax-zmin)) + 1)
    zs = np.linspace(zmin,zmax,nzs)
    dist = (lambda z: ((mdbn_merger_rate(z)*4.*np.pi*cosmo.differential_comoving_volume(z).value)/(1.+z)))(zs)
    window_cdf = np.array([simpson(dist[:i], x=zs[:i]) for i in range(1,nzs+1)]) / simpson(dist, x=zs)
    inv_window_cdf = interp1d(window_cdf, zs)
    return inv_window_cdf(rng.random(num_injs))

def bns_md_merger_rate_uniform_comoving_volume_redshift_inversion_sampler(zmin, zmax, cosmo, num_injs, seed=None, nzs=None):
    '''
    Generate redshift values using inversion sampling for the Madau-Dickinson star formation rate and 1/t time delay.

    Parameters
    ----------
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    cosmo : astropy.cosmology
        Cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator (default: None).
    nzs : int
        Number of redshift bins to sample (default: None).

    Returns
    -------
    np.ndarray
        Redshift values.
    '''
    rng = np.random.default_rng(seed)
    if nzs is None: nzs = max(5001, 50 * int((zmax-zmin)) + 1)
    zs = np.linspace(zmin,zmax,nzs)
    dist = (lambda z: ((bns_md_merger_rate(z)*4.*np.pi*cosmo.differential_comoving_volume(z).value)/(1.+z)))(zs)
    window_cdf = np.array([simpson(dist[:i], x=zs[:i]) for i in range(1,nzs+1)]) / simpson(dist, x=zs)
    inv_window_cdf = interp1d(window_cdf, zs)
    return inv_window_cdf(rng.random(num_injs))

def popIII_merger_rate_uniform_comoving_volume_redshift_inversion_sampler(zmin, zmax, cosmo, num_injs, seed=None, nzs=None):
    '''
    Generate redshift values using inversion sampling for the BBH from a popIII volumetric merger rate.

    Parameters
    ----------
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    cosmo : astropy.cosmology
        Cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator (default: None).
    nzs : int
        Number of redshift bins to sample (default: None).

    Returns
    -------
    np.ndarray
        Redshift values.
    '''
    rng            = np.random.default_rng(seed)
    if nzs is None: nzs = max(5001, 50 * int((zmax - zmin)) + 1)
    zs             = np.linspace(zmin, zmax, nzs)
    dist           = (lambda z: ((popIII_merger_rate(z) * 4. * np.pi * cosmo.differential_comoving_volume(z).value) / (1. + z)))(zs)
    window_cdf     = np.array([simpson(dist[:i], x=zs[:i]) for i in range(1, nzs + 1)]) / simpson(dist, x=zs)
    inv_window_cdf = interp1d(window_cdf, zs)
    return inv_window_cdf(rng.random(num_injs))

# PBH redshift distribution following a power law wrt the age of the universe (https://arxiv.org/pdf/2204.11864.pdf, Eq. (5))
def pbh_universe_age_power_law_sampler(zmin, zmax, cosmo, num_injs, seed):
    '''
    Generate redshift values for PBH using a power-law distribution with respect to the age of the universe.

    Parameters
    ----------
    zmin : float
        Minimum redshift value.
    zmax : float
        Maximum redshift value.
    cosmo : astropy.cosmology
        Cosmological parameters.
    num_injs : int
        Number of injections to generate.
    seed : int
        Seed for the random number generator (default: None).

    Returns
    -------
    np.ndarray
        Redshift values.
    '''
    rng    = np.random.default_rng(seed)
    tau_0  = cosmo.age(0)
    x_min  = cosmo.age(zmin) / tau_0
    x_max  = cosmo.age(zmax) / tau_0
    x_vec  = rng.power(-34/37 + 1, num_injs)
    x_mask = np.logical_or(x_vec < x_max, x_vec > x_min)

    while x_mask.sum():
        x_vec[x_mask] = rng.power(-34/37 + 1, x_mask.sum())
        x_mask        = np.logical_or(x_vec < x_max, x_vec > x_min)

    x_vec  *= tau_0.value
    tau_vec = np.geomspace(np.sort(x_vec)[0], np.sort(x_vec)[-1], 100)
    return np.interp(x_vec, tau_vec, apcosm.z_at_value(cosmo.age, tau_vec * tau_0.unit).value)

#-----merger rate functions-----
def mdbn_merger_rate(z, a0=2.57, b0=5.83, c0=3.36, phi0=1):
    '''
    Madau-Dickinson-Belczynski-Ng field BBH volumetric merger rate, see Eq. (C13) in https://arxiv.org/pdf/2012.09876.pdf
    with F-values from page 13 (v3 of the paper).

    Parameters
    ----------
    z : np.ndarray
        Redshift value.
    a0 : float
        Power-law exponent.
    b0 : float
        Power-law exponent.
    c0 : float
        Power-law exponent.
    phi0 : float
        Normalization factor.

    Returns
    -------
    np.ndarray
        Merger rate.
    '''
    return phi0 * (1+z)**a0 / (1+((1+z)/c0)**b0)

def bns_md_merger_rate(z, a0=1.803219571, b0=5.309821767, c0=2.837264101, phi0=8.765949529):
    '''
        Madau-Dickinson star formation rate and 1/t time delay with metalicity not taken into account.
        The fit is based on the data found in 'xtra_files/merger_rates/bns_n_dot_bns_md_merger_rate.txt'.

    Parameters
    ----------
    z : np.ndarray
        Redshift value.
    a0 : float
        Power-law exponent.
    b0 : float
        Power-law exponent.
    c0 : float
        Power-law exponent.
    phi0 : float
        Normalization factor.

    Returns
    -------
    np.ndarray
        Merger rate.
    '''
    return phi0 * (1+z)**a0 / (1+((1+z)/c0)**b0)

# BBH from popIII volumetric merger rate:
# https://arxiv.org/pdf/2012.09876.pdf, Eq. (C15) with III-values from page 13 (v3 of the paper)
def popIII_merger_rate(z, aIII=0.66, bIII=0.3, zIII=11.6):
    '''
    BBH from popIII volumetric merger rate, see Eq. (C15) in https://arxiv.org/pdf/2012.09876.pdf
    with III-values from page 13 (v3 of the paper).

    Parameters
    ----------
    z : np.ndarray
        Redshift value.
    aIII : float
        Power-law exponent.
    bIII : float
        Power-law exponent.
    zIII : float
        Redshift value.

    Returns
    -------
    np.ndarray
        Merger rate.
    '''
    return np.exp(aIII * (z - zIII)) / (bIII + aIII * np.exp((aIII + bIII) * (z - zIII)))
