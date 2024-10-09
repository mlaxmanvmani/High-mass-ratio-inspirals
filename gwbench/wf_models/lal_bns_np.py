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


import lal
import lalsimulation as lalsim
from numpy import exp, pi

from gwbench.basic_relations import m1_m2_of_M_eta, M_of_Mc_eta, lam_12_of_lam_ts_eta
from gwbench.utils import Mpc, Msun, mod_1

deriv_mod       = 'numdifftools'
wf_symbs_string = 'f Mc eta chi1x chi1y chi1z chi2x chi2y chi2z DL tc phic iota lam_t delta_lam_t'

def hfpc(f, Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, tc, phic, iota, lam_t, delta_lam_t, approximant, is_lam12=0, fRef=0.):
    '''
    Waveform wrapper for lalsimulation.SimInspiralChooseFDWaveformSequence geared to BNS systems that depend on the following parameters:
    f, Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, DL, tc, phic, iota, lam_t, delta_lam_t (see Parameters for a decrioption).

    Parameters
    ----------
    f: np.ndarray
        Frequency array
    Mc: float
        Chirp mass
    eta: float
        Symmetric mass ratio
    chi1x : float
        Dimensionless spin component of the primary BH along the x-axis
    chi1y : float
        Dimensionless spin component of the primary BH along the y-axis
    chi1z : float
        Dimensionless spin component of the primary BH along the z-axis
    chi2x : float
        Dimensionless spin component of the secondary BH along the x-axis
    chi2y : float
        Dimensionless spin component of the secondary BH along the y-axis
    chi2z : float
        Dimensionless spin component of the secondary BH along the z-axis
    DL: float
        Luminosity distance
    tc: float
        Coalescence time
    phic: float
        Coalescence phase
    iota: float
        Inclination angle
    lam_t: float
        Effective tidal deformability
    delta_lam_t: float
        Delta effective tidal deformability
    approximant: str
        Approximant to use
    fRef: float, optional
        Reference frequency
    phiRef: float, optional
        Reference phase

    Returns
    -------
    hfp: np.ndarray
        Plus polarization waveform
    hfc: np.ndarray
        Cross polarization waveform
    '''
    # set up the lal variables
    lal_dict = lal.CreateDict()
    f_lal    = lal.CreateREAL8Vector(len(f))

    # set the lal frequency array to the numpy frequency array
    f_lal.data = f

    # set the reference frequency to the lowest frequency if not provided
    if not fRef: fRef = f[0]

    # ensure that multibanding is *always* off when calling a custom (frequency) grid
    # see line 2726 in https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_i_m_r_phenom_x_p_h_m_8c_source.html
    if 'IMRPhenomX' in approximant: lalsim.SimInspiralWaveformParamsInsertPhenomXHMThresholdMband(lal_dict, 0)

    # convert to lalsim.SimIninspiralChooseFDWaveformSequence parametrization
    _m1, _m2 = m1_m2_of_M_eta(M_of_Mc_eta(Mc,eta),eta)

    # in this case: lam_t = lam1, delta_lam_t = lam2
    if is_lam12: lam1, lam2 = lam_t, delta_lam_t
    else:        lam1, lam2 = lam_12_of_lam_ts_eta(lam_t,delta_lam_t,eta)

    if 0 > lam1 or 0 > lam2: raise ValueError(f'Either lam1 ({lam1}) or lam2 ({lam2}) are smaller than 0.')

    lalsim.SimInspiralWaveformParamsInsertTidalLambda1(lal_dict, lam1)
    lalsim.SimInspiralWaveformParamsInsertTidalLambda2(lal_dict, lam2)

    # phase factor (modulo 2pi to avoid large phases in the evaluation of the exponential)
    pf = exp(1j * 2*pi * mod_1(f*tc - phic / (2*pi)))

    # the phase phiRef is set to 0. and added manually below
    hPlus, hCross = lalsim.SimInspiralChooseFDWaveformSequence(
        0., _m1 * Msun, _m2 * Msun,
        chi1x, chi1y, chi1z,
        chi2x, chi2y, chi2z,
        fRef, DL * Mpc, iota,
        lal_dict, lalsim.GetApproximantFromString(approximant), f_lal)

    return pf * hPlus.data.data, pf * hCross.data.data
