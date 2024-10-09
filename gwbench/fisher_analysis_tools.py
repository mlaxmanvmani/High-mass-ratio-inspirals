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


import mpmath as mp
import numpy as np

import gwbench.snr as snr_mod
from gwbench.utils import log_msg

def calc_fisher_cov_matrices(del_hf_list, psd, f, only_fisher=False, cond_sup=None, fac=101, logger=None, tag=''):
    '''
    Calculate the Fisher matrix and its inverse covariance matrix for a list of waveform derivatives.

    Parameters
    ----------
    del_hf_list : list
        list of waveform derivatives
    psd : np.ndarray
        power spectral density
    f : np.ndarray
        frequency array
    only_fisher : bool, optional
        if True, only the Fisher matrix is calculated, default is False.
    cond_sup : float, optional
        condition number threshold for using the high precision mpmath library, default is None.
    fac : float, optional
        factor for the condition number threshold, default is 101.
    logger : logging.Logger, optional
        logger object, default is None.
    tag : str, optional
        tag for the logger, default is an empty string.

    Returns
    -------
    fisher : np.ndarray
        Fisher matrix
    cov : np.ndarray
        covariance matrix
    cond_num : float
        condition number of the Fisher matrix
    inv_err : dict
        dictionary containing information regarding the quality of the inversion
    '''
    n = len(del_hf_list)
    fisher = np.zeros((n,n))

    for i in np.arange(n):
        fisher[i,i]     = snr_mod.scalar_product(del_hf_list[i], del_hf_list[i], psd, f)
        for j in np.arange(i+1,n):
            fisher[i,j] = snr_mod.scalar_product(del_hf_list[i], del_hf_list[j], psd, f)
            fisher[j,i] = fisher[i,j]

    if only_fisher: return fisher, None, None, None

    cond_num     = calc_cond_number(fisher, logger=logger, tag=tag)
    cov, inv_err = calc_cov_inv_err(fisher, cond_num, cond_sup, fac, logger=logger, tag=tag)
    return fisher, cov, cond_num, inv_err

def calc_cond_number(fisher, logger=None, tag=''):
    '''
    Calculate the condition number of the Fisher matrix.

    Parameters
    ----------
    fisher : np.ndarray
        Fisher matrix
    logger : logging.Logger, optional
        logger object, default is None.
    tag : str, optional
        tag for the logger, default is an empty string.

    Returns
    -------
    cond_num : float
        condition number of the Fisher matrix
    '''
    EWs,_ = np.linalg.eig(fisher)
    if not np.amin(np.abs(EWs)):
        log_msg(f'calc_cond_number: tag = {tag} - Fisher matrix is singular!', logger=logger, level='WARNING')
        return np.inf
    else:
        return np.amax(np.abs(EWs))/np.amin(np.abs(EWs))

def calc_cov_inv_err(fisher, cond_num=None, cond_sup=None, fac=101, logger=None, tag=''):
    '''
    Calculate the covariance matrix and the inversion error from the Fisher matrix.

    Parameters
    ----------
    fisher : np.ndarray
        Fisher matrix
    cond_num : float, optional
        condition number of the Fisher matrix, default is None. 
    cond_sup : float, optional
        condition number threshold for using the high precision mpmath library, default is None.
    fac : float, optional
        factor for the condition number threshold, default is 101.
    logger : logging.Logger, optional
        logger object, default is None.
    tag : str, optional
        tag for the logger, default is an empty string.

    Returns
    -------
    cov : np.ndarray
        covariance matrix
    inv_err : dict
        dictionary containing information regarding the quality of the inversion
    '''
    if cond_num is None: cond_num = calc_cond_number(fisher, logger=logger, tag=tag)
    if cond_num == np.inf: return None, None # Singular Fisher matrix

    try:
        if cond_sup is not None and cond_sup < 1e16 and cond_sup > cond_num:
            cov   = np.linalg.inv(fisher)
            cov   = (cov + cov.T) / 2
            ident = np.identity(len(fisher))
            return cov, inv_err_from_fisher_cov(fisher, cov, ident, fac, None)
        else:
            mp.mp.dps = max(int(np.ceil(np.log10(cond_num))) + 2, 20)
            fisher_mp = mp.matrix(fisher)
            cov_mp    = fisher_mp**-1
            cov_mp    = (cov_mp + cov_mp.T) / 2
            ident_mp  = mp.matrix(np.identity(len(fisher)))
            return conv_mp_to_np(cov_mp), inv_err_from_fisher_cov(fisher_mp, cov_mp, ident_mp, fac, mp.mp.dps)

    except (ZeroDivisionError, np.linalg.LinAlgError):
        log_msg(f'calc_cov_inv_err: tag = {tag} - There was an exception in the inversion of the Fisher matrix!',
                logger=logger, level='WARNING')
        return None, None

def get_errs_from_cov(cov, deriv_variables):
    '''
    Pull the standard deviations from the covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        covariance matrix
    deriv_variables : list
        list of variable names

    Returns
    -------
    errs : dict
        dictionary containing the standard deviations of the variables
    '''
    if cov is None:              return None
    if np.any(np.diag(cov) < 0): return { name : np.nan for name in deriv_variables }
    else:                        return { name : np.sqrt(cov[i,i]) for i,name in enumerate(deriv_variables) }

def conv_mp_to_np(matrix, dtype=np.float32):
    '''
    Convert a matrix from the mpmath library to a numpy array.

    Parameters
    ----------
    matrix : mpmath.matrix
        matrix to be converted
    dtype : type, optional
        data type of the numpy array, default is np.float32.

    Returns
    -------
    np.matrix
        numpy array
    '''
    return np.matrix(matrix.tolist(), dtype=dtype)

def inv_err_from_fisher_cov(fisher, cov, ident, fac, dps):
    '''
    Calculate the inversion error from the Fisher matrix and the covariance matrix.

    Parameters
    ----------
    fisher : np.ndarray
        Fisher matrix
    cov : np.ndarray
        covariance matrix
    ident : np.ndarray
        identity matrix
    fac : float
        factor for the condition number threshold
    dps : int
        decimal precision for the mpmath library

    Returns
    -------
    inv_err : dict
        dictionary containing information regarding the quality of the inversion
    '''
    cov_err, delta = calc_cov_err_inv_res(fisher, cov, ident)
    if dps is None:
        inv_err                = {'fisher_mp': None,   'cov_mp': None, 'dps_mp':None}
        inv_err['inv_err_mat'] = delta
        inv_err['inv_err']     = np.amax(np.abs(delta))
        inv_err['inv_cond']    = np.all(np.abs(fac * cov_err) < np.abs(cov))
        return inv_err
    else:
        inv_err                = {'fisher_mp': fisher, 'cov_mp': cov,  'dps_mp':dps}
        inv_err['inv_err_mat'] = delta
        inv_err['inv_err']     = np.amax(np.abs(conv_mp_to_np(delta)))
        inv_err['inv_cond']    = np.all(np.abs(fac * conv_mp_to_np(cov_err)) < np.abs(conv_mp_to_np(cov)))
        return inv_err

def calc_cov_err_inv_res(fisher, cov, ident):
    '''
    Calculate the covariance error and the inversion residual from the Fisher matrix and the covariance matrix.

    Parameters
    ----------
    fisher : np.ndarray
        Fisher matrix
    cov : np.ndarray
        covariance matrix
    ident : np.ndarray
        identity matrix

    Returns
    -------
    cov_err : np.ndarray
        covariance error
    delta : np.ndarray
        inversion residual

    Notes
    -----
    The inversion residual is calculated as follows:
    delta = fisher * cov - ident
    The covariance error is calculated as follows:
    cov_err = cov * delta * (ident + delta)**-1
    '''
    if isinstance(fisher, np.ndarray):
        delta = np.matmul(fisher, cov) - ident
        return np.matmul(np.matmul(cov, delta), np.linalg.inv(ident + delta)), delta
    else:
        delta = fisher * cov - ident
        return cov * delta * (ident + delta)**-1, delta
