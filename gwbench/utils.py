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


import logging
import os
import sys
from logging import getLevelName

import numpy as np

################################################################################
# available detectors technologies and locations
################################################################################

#-----available detector technologies and locations-----
# available detector technologies
available_tecs = (
    'aLIGO', 'A+', 'V+', 'K+',
    'A#', 'Voyager-CBO', 'Voyager-PMO',
    'ET', 'ET-10-XYL', 'CEwb',
    'CE-40', 'CE-40-LF', 'CE-20', 'CE-20-PM',
    'CE1-10-CBO', 'CE1-20-CBO', 'CE1-30-CBO', 'CE1-40-CBO',
    'CE2-10-CBO', 'CE2-20-CBO', 'CE2-30-CBO', 'CE2-40-CBO',
    'CE1-10-PMO', 'CE1-20-PMO', 'CE1-30-PMO', 'CE1-40-PMO',
    'CE2-10-PMO', 'CE2-20-PMO', 'CE2-30-PMO', 'CE2-40-PMO',
    'LISA-17', 'LISA-Babak17', 'LISA-Robson18'
    )

# available detector locations
available_locs = (
    'H', 'L', 'V', 'K', 'I', 'LHO', 'LLO', 'LIO',
    'ET1', 'ET2', 'ET3', 'ETS1', 'ETS2', 'ETS3',
    'C', 'N', 'S', 'CEA', 'CEB', 'CES'
    )

################################################################################
# constants
################################################################################

#-----constants in SI units-----
GNewton    = 6.6743e-11
cLight     = 2.99792458e8
Msun       = 1.9884099021470415e+30
Mpc        = 3.085677581491367e+22
REarth     = 6.371e6
AU         = 1.4959787066e11
year       = 3.1536e7
hPlanck    = 6.62607015e-34
TEarthSol  = 86400      # solar day    (24h 60m 60s)
TEarthSid  = 86164.0905 # sidereal day (23h 56m  4.0905s, found root 86164.09053826332)
TEarth     = TEarthSol

#-----convert mass in solar masses to seconds-----
MTsun    = Msun * GNewton/cLight**3.
#-----convert mass/distance in solar mass/Mpc to dimensionless-----
strain_fac = GNewton/cLight**2.*Msun/Mpc
#-----convert seconds to radians with a periodicty of one day-----
time_to_rad_earth = 2.*np.pi/TEarth


################################################################################
# basic functions
################################################################################

def reduce_symbols_strings(string1, string2):
    '''
    Combine two sympy symbols strings without duplicates.

    Parameters
    ----------
    string1 : str
        First sympy symbols string.
    string2 : str
        Second sympy symbols string.

    Returns
    -------
    str
        Combined sympy symbols string without duplicates.
    '''
    # combine the two lists
    symbols_list = string1.split(' ') + string2.split(' ')
    # remove duplicates
    symbols_list = list(dict.fromkeys(symbols_list))

    # recreate a string of symbols and return it
    return ' '.join(symbols_list)

def remove_symbols(string1, string2, keep_same=True):
    '''
    Remove symbols from one sympy symbols string that are not present in the other.

    Parameters
    ----------
    string1 : str
        First sympy symbols string.
    string2 : str
        Second sympy symbols string.
    keep_same : bool
        If True, keep the symbols that are present in both strings.

    Returns
    -------
    str
        Combined sympy symbols string without duplicates.
    '''
    symbs_list1 = string1.split(' ')
    symbs_list2 = string2.split(' ')
    # remove unwanted symbols from 1
    if keep_same:
        symbols_list = [x for x in symbs_list1 if x in symbs_list2]
    else:
        symbols_list = [x for x in symbs_list1 if x not in symbs_list2]

    # recreate a string of symbols and return it
    return ' '.join(symbols_list)

def get_sub_array_ids(arr, sub_arr):
    '''
    Get the indices of the elements of a subarray in a larger array.

    Parameters
    ----------
    arr : np.ndarray
        Larger array.
    sub_arr : np.ndarray
        Subarray.

    Returns
    -------
    np.ndarray
        Indices of the elements of the subarray in the larger array.
    '''
    return min_max_mask(arr, sub_arr[0], sub_arr[-1])

def get_sub_dict(dic, key_list, keep_in_dict=True):
    '''
    Get a subset of a dictionary based on a list of keys.

    Parameters
    ----------
    dic : dict
        Dictionary.
    key_list : list
        List of keys.
    keep_in_dict : bool
        If True, keep the keys in the dict. If False, keep the complientary keys.

    Returns
    -------
    dict
        Subset of the dictionary.
    '''
    if type(key_list) == str: key_list = key_list.split(' ')
    if keep_in_dict: return {k:v for k,v in dic.items() if k     in key_list}
    else:            return {k:v for k,v in dic.items() if k not in key_list}

def is_subset_lists(sub, sup):
    '''
    Check if a list is a subset of another list.

    Parameters
    ----------
    sub : list
        Sublist.
    sup : list
        Superlist.

    Returns
    -------
    bool
        True if the sublist is a subset of the superlist, False otherwise.
    '''
    return all([el in sup for el in sub])

def min_max_mask(arr, min_val=-np.inf, max_val=np.inf, strict_min=False, strict_max=False):
    '''
    Create a mask for an array based on minimum and maximum values.

    Parameters
    ----------
    arr : np.ndarray
        Array.
    min_val : float
        Minimum value. Default is -np.inf.
    max_val : float
        Maximum value. Default is np.inf.
    strict_min : bool
        If True, the minimum value is excluded. Default is False.
    strict_max : bool
        If True, the maximum value is excluded. Default is False.

    Returns
    -------
    mask : np.ndarray
        Mask for the array.
    '''
    if strict_min: min_mask = arr >  min_val
    else:          min_mask = arr >= min_val

    if strict_max: max_mask = arr <  max_val
    else:          max_mask = arr <= max_val

    return np.logical_and(min_mask, max_mask)

################################################################################
# waveform manipluations
################################################################################

#-----stable evaluation of exponential phase factors-----
def mod_1(val, np=np):
    '''
    Return stable evaluation of mod(val, 1) for a value val to be inserted in exp(1j * 2pi * val).
    The returned value is always in the range [0, 1).

    Parameters
    ----------
    val : np.ndarray or jnp.ndarray
        Values

    Returns
    -------
    mod : np.ndarray or jnp.ndarray
        mod(val, 1)
    '''
    return val - np.floor(val)

#-----convert waveform polarizations to amplitude and phase-----
def transform_hfpc_to_amp_pha(hfpc, f, params_list, np=np):
    '''
    Calculate and transform the complex frequency domain waveform for the plus and cross polarizations to amplitude and phase.

    Parameters
    ----------
    hfpc : np.ndarray
        Complex frequency domain waveform for the plus and cross polarizations.
    f : np.ndarray
        Frequency array.
    params_list : list
        List of parameters.

    Returns
    -------
    np.ndarray
        Amplitude of the plus polarization.
    np.ndarray
        Phase of the plus polarization.
    np.ndarray
        Amplitude of the cross polarization.
    np.ndarray
        Phase of the cross polarization.
    '''
    hfp, hfc = hfpc(f, *params_list)
    return pl_cr_to_amp_pha(hfp, hfc, np=np)

def pl_cr_to_amp_pha(hfp, hfc, np=np):
    '''
    Transform the plus and cross polarizations to amplitude and phase.

    Parameters
    ----------
    hfp : np.ndarray
        Plus polarization.
    hfc : np.ndarray
        Cross polarization.

    Returns
    -------
    np.ndarray
        Amplitude of the plus polarization.
    np.ndarray
        Phase of the plus polarization.
    np.ndarray
        Amplitude of the cross polarization.
    np.ndarray
        Phase of the cross polarization.
    '''
    hfp_amp, hfp_pha = amp_pha_from_z(hfp, np=np)
    hfc_amp, hfc_pha = amp_pha_from_z(hfc, np=np)
    return hfp_amp, hfp_pha, hfc_amp, hfc_pha

#-----convert amp/phase derivatives to re/im ones-----
def z_deriv_from_amp_pha(amp, pha, del_amp, del_pha, np=np):
    '''
    Calculate the real and imaginary part of waveform derivatives from the amplitude and phase and their derivatives.

    Parameters
    ----------
    amp : np.ndarray
        Amplitude of the waveform.
    pha : np.ndarray
        Phase of the waveform.
    del_amp : np.ndarray
        Derivative of the amplitude.
    del_pha : np.ndarray
        Derivative of the phase.

    Returns
    -------
    np.ndarray
        Real part of the waveform derivative.
    np.ndarray
        Imaginary part of the waveform derivative.
    '''
    if len(del_amp.shape) == 2: return np.array([ del_amp[:,i] * np.exp(1j*pha) + amp * np.exp(1j*pha) * 1j * del_pha[:,i]
                                                  for i in range(del_amp.shape[1]) ], dtype=complex).T
    else:                       return del_amp * np.exp(1j*pha) + amp * np.exp(1j*pha) * 1j * del_pha

#-----re/im vs. amp/phase transformations-----
def re_im_from_amp_pha(amp, pha, np=np):
    '''
    Calculate the real and imaginary part of a complex number from its amplitude and phase.

    Parameters
    ----------
    amp : np.ndarray
        Amplitude.
    pha : np.ndarray
        Phase.

    Returns
    -------
    np.ndarray
        Real part of the complex number.
    np.ndarray
        Imaginary part of the complex number.
    '''
    return re_im_from_z(z_from_amp_pha(amp, pha), np=np)

def amp_pha_from_re_im(re, im, np=np):
    '''
    Calculate the amplitude and phase of a complex number from its real and imaginary part.

    Parameters
    ----------
    re : np.ndarray
        Real part.
    im : np.ndarray
        Imaginary part.

    Returns
    -------
    np.ndarray
        Amplitude.
    np.ndarray
        Phase.
    '''
    return amp_pha_from_z(z_from_re_im(re, im), np=np)

#-----re/im or amp/phase vs. complex number transformations-----
def re_im_from_z(z, np=np):
    '''
    Calculate the real and imaginary part of a complex number.

    Parameters
    ----------
    z : np.ndarray
        Complex number.

    Returns
    -------
    np.ndarray
        Real part of the complex number.
    np.ndarray
        Imaginary part of the complex number.
    '''
    return np.real(z), np.imag(z)

def z_from_re_im(re, im):
    '''
    Calculate the complex number from its real and imaginary part.

    Parameters:
        re (array): Real part.
        im (array): Imaginary part.

    Returns:
        array: Complex number.
    '''
    return re + 1j * im

def amp_pha_from_z(z, np=np):
    '''
    Calculate the amplitude and phase of a complex number.

    Parameters
    ----------
    z : np.ndarray
        Complex number.

    Returns
    -------
    np.ndarray
        Amplitude.
    np.ndarray
        Phase.
    '''
    return np.abs(z), np.unwrap(np.angle(z))

def z_from_amp_pha(amp, pha, np=np):
    '''
    Calculate the complex number from its amplitude and phase.

    Parameters
    ----------
    amp : np.ndarray
        Amplitude.
    pha : np.ndarray
        Phase.

    Returns
    -------
    np.ndarray
        Complex number.
    '''
    return amp * np.exp(1j*pha)


################################################################################
# IO functions
################################################################################

#-----Block and unblock printing-----
def block_print(active=True):
    '''
    Block printing to the standard output.

    Parameters
    ----------
    active : bool
        If True, block the printing. If False, do nothing.
    '''
    if active: sys.stdout = open(os.devnull, 'w')
    return

def unblock_print(active=True):
    '''
    Unblock printing to the standard output.

    Parameters
    ----------
    active : bool
        If True, unblock the printing. If False, do nothing.
    '''
    if active: sys.stdout = sys.__stdout__
    return

#-----sending warning or error message-----
def log_msg(message, logger=None, level='INFO'):
    '''
    Log a message to the standard output or a logger.

    Parameters
    ----------
    message : str
        Message to be logged.
    logger : logging.Logger
        Logger to log the message. If None, the message is printed to the standard output.
    level : str
        Level of the message, if a logger is used. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
    '''
    if logger is None: print(level + ': ' + message)
    else:              logger.log(getLevelName(level), message)
    if level in ['ERROR', 'CRITICAL']: sys.exit()

def get_logger(name, level='INFO', stdout=True, logfile=None):
    '''
    Get a logger.

    Parameters
    ----------
    name : str
        Name of the logger.
    level : str
        Level of the logger. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
    stdout : bool
        If True, log to the standard output.
    logfile : str
        If not None, log to a file with the given name.

    Returns
    -------
    logging.Logger
        Logger.
    '''
    logging.basicConfig(format = '%(asctime)s - %(name)s - %(levelname)s : %(message)s')
    if stdout: logging.basicConfig(stream = sys.stdout)
    if logfile is not None: logging.basicConfig(filename = logfile, filemode = 'w')
    logger = logging.getLogger(name)
    set_logger_level(logger, level)
    return logger

def set_logger_level(logger, level):
    '''
    Set the level of a logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger.
    level : str
        Level of the logger. Can be 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.
    '''
    logger.setLevel(level)


################################################################################
# network_spec handlers
################################################################################

def read_det_keys_from_label(network_label):
    '''
    Read the network label and find all detectors.

    Parameters
    ----------
    network_label : str
        Network label.

    Returns
    -------
    list
        list: List of detector keys.
    '''
    det_keys = []

    ##-----read the network label and find all detectors-----
    keys = list(network_label)

    # - in network list means that all 2G detectors up to that index are to be
    # taken at aLIGO sensitivity
    aLIGO = int('-' in keys)
    if aLIGO: aLIGO_id = keys.index('-')

    # + in network list means that all 2G detectors up to that index are to be
    # taken at A+ sensitivity
    a_pl = int('+' in keys)
    if a_pl: a_pl_id = keys.index('+')

    # v in network list means that all 2G detectors up to that index are to be
    # taken at Voyager sensitivity
    voy = int('v' in keys)
    if voy:
        voy_id = keys.index('v')
        tmp = int(keys[voy_id+1] == 'p')
        voy_pmo = tmp * 'PMO' + (1-tmp) * 'CBO'

    # find out which locations with which PSDs are in the network
    for loc in available_locs:
        if loc in keys:
            loc_id = keys.index(loc)

            if loc in ('H','L','V','K','I'):
                if aLIGO and loc_id < aLIGO_id:
                    name = 'aLIGO_'+loc
                elif a_pl and loc_id < a_pl_id:
                    if loc == 'V':
                        name = 'V+_'+loc
                    elif loc == 'K':
                        name = 'K+_'+loc
                    else:
                        name = 'A+_'+loc
                elif voy and loc_id < voy_id:
                    name = 'Voyager-{}_{}'.format(voy_pmo,loc)

            elif loc in ('C','N','S'):
                if keys[loc_id+1] == 'c':
                    name = f'CE-{keys[loc_id+2]}0'
                    if   keys[loc_id+2] == 'l': name += '-LF'
                    elif keys[loc_id+2] == 'p': name += '-PM'
                    name += f'_{loc}'
                else:
                    ce_a = int(keys[loc_id+1] == 'a') # 0 for i, 1 for a - CE1 as i, CE2 as a
                    ce_arm = int(keys[loc_id+2])*10  # arm length (n for n*10km)
                    tmp = int(keys[loc_id+3] == 'p')
                    ce_pmo = tmp * 'PMO' + (1-tmp) * 'CBO'
                    name = f'CE{ce_a+1}-{ce_arm}-{ce_pmo}_{loc}'

            det_keys.append(name)

    # add 3 ET detectors
    if 'E' in keys:
        for name in ['ET_ET1','ET_ET2','ET_ET3']:
            det_keys.append(name)

    return det_keys
