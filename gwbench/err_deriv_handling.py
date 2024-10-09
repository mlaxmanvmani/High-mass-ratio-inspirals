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
from scipy.integrate import simpson
from scipy.optimize import toms748

#-----derivative and error manipulations-----
def convert_to_cos_derivative(deriv, param_key, param_val):
    '''
    Convert the partial derivative [del_X(f)] of a function [f] with respect to a parameter [X]
    to the derivative [del_cos_X(f)] with respect to the cosine of the parameter.

    Parameters
    ----------
    deriv : float
        the partial derivative [del_X(f)] of the function [f] with respect to the parameter [X].
    param_key : str
        the name of the parameter [X].
    param_val : float
        the value of the parameter [X].

    Returns
    -------
    c_deriv : float
        the derivative [del_cos_X(f)] with respect to the cosine of the parameter.
    c_param_key : str
        the name of the converted parameter [cos_X].
    c_param_val : float
        the value of the parameter [cos_X].
    '''
    c_deriv     = (-1./np.sin(param_val)) * deriv
    c_param_key = 'cos_'+param_key
    c_param_val = np.cos(param_val)
    return c_deriv, c_param_key, c_param_val

def convert_to_log_derivative(deriv, param_key, param_val):
    '''
    Convert the partial derivative [del_X(f)] of a function [f] with respect to a parameter [X]
    to the derivative [del_log_X(f)] with respect to the logarithm of the parameter.

    Parameters
    ----------
    deriv : float
        the partial derivative [del_X(f)] of the function [f] with respect to the parameter [X].
    param_key : str
        the name of the parameter [X].
    param_val : float
        the value of the parameter [X].

    Returns
    -------
    c_deriv : float
        the derivative [del_log_X(f)] with respect to the logarithm of the parameter.
    c_param_key : str
        the name of the converted parameter [log_X].
    c_param_val : float
        the value of the parameter [log_X].
    '''
    c_deriv     = param_val * deriv
    c_param_key = 'log_'+param_key
    c_param_val = np.log(param_val)
    return c_deriv, c_param_key, c_param_val

def dim_err_to_rel_err(err, param_val, param_key=None, param_kind=None):
    '''
    Convert the error [err] of a dimensional parameter [X] to the relative error [err_rel] with respect to the parameter [X].

    Parameters
    ----------
    err : float
        the error of the parameter [X].
    param_val : float
        the value of the parameter [X].
    param_key : str, optional
        the name of the parameter [X].
    param_kind : str, optional
        the kind of the parameter [X].

    Returns
    -------
    err_rel : float
        the relative error of the parameter [X].
    '''
    if param_key in ('M','Mc','Dl','DL','tc') or param_kind == 'dim': return np.abs(err/param_val)
    else:                                                             return err

def one_sigma_to_percent_error(percent, sigma):
    '''
    Convert the one-sigma error [sigma] to the percent error [percent].

    Parameters
    ----------
    percent : float
        the percentage desired as a fraction.
    sigma : float
        the one-sigma error.

    Returns
    -------
    percent : float
        the percent error.
    '''
    sigma_orders = [0, 1, 2, 3, 4, 5, 6]
    sigma_percents = [0., 68.2689492137, 95.4499736104, 99.7300203937, 99.9936657516, 99.9999426697, 99.9999998027]
    for i,sigma_order in enumerate(sigma_orders):
        if percent < sigma_percents[i]: break

    def func(error,percent,sigma,sigma_order):
        x = np.linspace(0,error,sigma_order*100)
        return percent/100/2 - simpson(np.exp(-x**2/2/sigma**2)/np.sqrt(2*np.pi)/sigma, x=x)

    if sigma_order == 1: return toms748(func,1e-2*sigma,sigma_order*sigma,args=(percent,sigma,sigma_order))
    else:                return toms748(func,(sigma_order-1)*sigma,sigma_order*sigma,args=(percent,sigma,sigma_order))

#-----Convert the evaluated derivatives according to conv_cos, conv_log-----
def get_conv_inj_params_deriv_variables(c_quants, inj_params, deriv_variables):
    '''
    Convert the injected parameters and the differentiated variables string according to the conversion dictionary [c_quants].

    Parameters
    ----------
    c_quants : dict
        the conversion dictionary.
    inj_params : dict
        the injected parameters.
    deriv_variables : list
        the differentiation variables.

    Returns
    -------
    inj_params : dict
        the converted injected parameters.
    deriv_variables : list
        the converted differentiation variables.
    '''
    if c_quants == {}: return inj_params, deriv_variables
    else:
        for o_key in c_quants:
            c_key = c_quants[o_key][0]
            c_val = c_quants[o_key][1]

            if c_key not in deriv_variables: deriv_variables[deriv_variables.index(o_key)] = c_key
            inj_params[c_key] = c_val

        return inj_params, deriv_variables

def get_conv_del_eval_dic(del_eval_dic, params_dic, conv_cos, conv_log):
    '''
    Convert the evaluated derivatives according to the conversion dictionaries [conv_cos, conv_log].

    Parameters
    ----------
    del_eval_dic : dict
        the evaluated derivatives.
    params_dic : dict
        the parameters dictionary.
    conv_cos : dict
        the conversion dictionary for the cosine of the parameters.
    conv_log : dict
        the conversion dictionary for the logarithm of the parameters.
    deriv_symbs_string : str
        the string of the differentiation variables.

    Returns
    -------
    conv_dic : dict
        the converted evaluated derivatives.
    c_quants : dict
        the converted parameters and their values.
    '''
    if conv_cos is None and conv_log is None: return del_eval_dic, {}
    else:
        conv_dic = {}
        c_quants = {}

        for deriv in del_eval_dic:
            key = '_'.join(deriv.split('_')[1:-1])
            c_key = None

            if   conv_cos is not None and key in conv_cos: c_deriv, c_key, c_val = convert_to_cos_derivative(del_eval_dic[deriv],key,params_dic[key])
            elif conv_log is not None and key in conv_log: c_deriv, c_key, c_val = convert_to_log_derivative(del_eval_dic[deriv],key,params_dic[key])
            else:                                          conv_dic[deriv]       = del_eval_dic[deriv]

            if c_key is not None:
                c_quants[key]   = (c_key, c_val)
                prefix,suffix   = deriv.split(key)
                n_key           = prefix + c_key + suffix
                conv_dic[n_key] = c_deriv

        return conv_dic, c_quants

#-----sky area calculations-----
def sky_area_90(ra_err, dec_err, cov_ra_dec, dec_val, is_cos_dec):
    '''
    Calculate the 90% credible sky area in square degrees.

    Parameters
    ----------
    ra_err : float
        the error of the right ascension in radians.
    dec_err : float
        the error of the declination in radians.
    cov_ra_dec : float
        the covariance between the right ascension and the declination.
    dec_val : float
        the value of the declination in radians.
    is_cos_dec : bool
        if True, the declination is in cosine.

    Returns
    -------
    sky_area : float
        the 90% credible sky area in square degrees.

    References
    ----------
    [1] arXiv:1403.6915
    [2] arXiv:gr-qc/0310125v3
    '''
    if is_cos_dec: trig_fac = np.abs( 1/np.tan(dec_val) )
    else:          trig_fac = np.abs(   np.cos(dec_val) )
    if (ra_err*dec_err)**2 > cov_ra_dec**2: return trig_fac * np.sqrt( (ra_err * dec_err)**2 - cov_ra_dec**2 ) * 2*np.pi * (180./np.pi)**2 * np.log(10)
    else:                                   return None
