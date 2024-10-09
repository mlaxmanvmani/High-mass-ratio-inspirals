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


'''This module contains two methods that calculate numerical derivatives.
'''

import numdifftools as nd
import numpy
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ModuleNotFoundError:
    pass

import gwbench.analytic_derivatives as ad
import gwbench.utils as utils

def part_deriv_hf_func(hf, symbols_list, deriv_symbs_list, f, params_dic,
                       deriv_mod, pl_cr=True, compl=True, label='hf',
                       ana_deriv_symbs_list=[], ana_deriv_aux={},
                       step=1e-9, method='central', order=2, d_order_n=1):
    '''
    Calculate the partial derivatives of the waveform function numerically. The function can have one (hf) or two outputs (hfp, hfc), in the
    latter case the derivatives are calculated for the plus and cross polarizations the parameter [pl_cr=True] has to be passed. The derivatives
    are calculated with respect to the variables in [deriv_symbs_list] and evaluated at the values of the parameters in [params_dic].
    The numerical derivatives are calculated with the [numdifftools] package, while the diffentiation method and order can be set with the [step],
    [method], and [order] parameters. The order of the derivative can be set with the [d_order_n] parameter. Certain variables can be excluded from
    the derivatives by passing a list of their names to the [ana_deriv_symbs_list] parameter. The derivatives of the excluded variables are calculated
    analytically and the results are added to the numerical derivatives.

    Parameters
    ----------
    hf : function
        the waveform function. If it has two outputs, the first output is the plus polarization and the second output is the cross polarization.
    symbols_list : list
        the list of variables that the function depends on.
    deriv_symbs_list : list
        the list of variables for which the derivatives are calculated.
    f : float
        the frequency at which the waveform is evaluated.
    params_dic : dict
        the dictionary of parameters and their values at which the derivatives are evaluated.
    deriv_mod : str
        the method of numerical differentiation, choices are ['numdifftools', 'jax'].
    pl_cr : bool, optional
        if True, the derivatives are calculated for the plus and cross polarizations separately, default is True.
    compl : bool, optional
        if True, the function has two outputs and the derivatives are calculated for the real and imaginary parts separately, default is True.
    label : str, optional
        the label of waveform (hf, hfp, hfc), default is 'hf'.
    ana_deriv_symbs_list : list, optional
        the list of variables that are excluded from the numerical derivatives, default is None.
    ana_deriv_aux : dict, optional
        the dictionary of auxiliary variables that are used in the analytical derivatives, default is None.
    step : float, optional
        the step size for the numerical differentiation, default is 1e-9.
    method : str, optional
        the method of numerical differentiation, default is 'central'.
    order : int, optional
        the order of numerical differentiation, default is 2.
    d_order_n : int, optional
        the order of the derivative, default is 1.

    Returns
    -------
    del_hf_dic : dict
        dictionary that contains the numerical and analytical partial derivatives.
    '''

    label_keys            = int(pl_cr) * [label + 'p', label + 'c'] + int(not pl_cr) * [label]
    del_hf_dic_oder       = [ f'del_{name}_{label_key}' for name in deriv_symbs_list for label_key in label_keys ]
    _ana_deriv_symbs_list = [ el for el in deriv_symbs_list if el     in ana_deriv_symbs_list ]
    _num_deriv_symbs_list = [ el for el in deriv_symbs_list if el not in ana_deriv_symbs_list ]
    params_list           = [ params_dic[param] for param in symbols_list ]
    num_deriv_params_list = [ params_dic[param] for param in _num_deriv_symbs_list ]
    hf_ids                = { el : _num_deriv_symbs_list.index(el) for el in _num_deriv_symbs_list }

    if _ana_deriv_symbs_list:
        if pl_cr: del_hf_dic_ana = { f'del_{name}_{label_key}' : deriv for label_key in label_keys
                                     for name,deriv in ad.waveform_ana_derivs(f, params_dic, ana_deriv_aux[label_keys]).items() }
        else:     del_hf_dic_ana = { f'del_{name}_{label_key}' : deriv for label_key in label_keys
                                     for name,deriv in ad.detector_response_ana_derivs(f, params_dic, **ana_deriv_aux).items() }
    else:         del_hf_dic_ana = {}

    if not _num_deriv_symbs_list: return { key : del_hf_dic_ana[key] for key in del_hf_dic_oder }

    def hf_of_deriv_params(f, *deriv_params_list):
        return hf(f, *[deriv_params_list[hf_ids[el]] if el in _num_deriv_symbs_list else params_list[i] for i,el in enumerate(symbols_list)])

    del_hf = part_deriv(hf_of_deriv_params, f, num_deriv_params_list, deriv_mod, pl_cr=pl_cr, compl=compl, step=step, method=method, order=order, d_order_n=d_order_n)

    if not pl_cr: del_hf = [del_hf]

    del_hf_dic_num = { f'del_{name}_{label_key}' : _del_hf[:,i] for i,name in enumerate(_num_deriv_symbs_list) for label_key,_del_hf in zip(label_keys, del_hf) }

    return { key : { **del_hf_dic_num, **del_hf_dic_ana }[key] for key in del_hf_dic_oder }


def part_deriv(func, f, params_list, deriv_mod, pl_cr=0, compl=None, step=1e-9, method='central', order=2, d_order_n=1):
    '''
    Calculate the partial derivatives of a function numerically, see [part_deriv_gradient] for more details.

    Parameters
    ----------
    func : function
        The function for which the derivatives are calculated.
    f : float
        The frequency at which the function is evaluated.
    params_list : list, optional
        The list of parameters at which the derivatives are evaluated.
    deriv_mod : str
        The method of numerical differentiation. Choices are ['numdifftools', 'jax'].
    pl_cr : bool, optional
        If True, the derivatives are calculated for the two outputs of the function separately.
    compl : bool, optional
        If True, the function has two outputs and the derivatives are calculated for the real and imaginary parts separately.
    step : float, optional
        The step size for the numerical differentiation.
    method : str, optional
        The method of numerical differentiation.
    order : int, optional
        The order of numerical differentiation.
    d_order_n : int, optional
        The order of the derivative.

    Returns
    -------
    del_hf : array
        The numerical partial derivatives of the function.
    '''

    if deriv_mod == 'jax': np = jax.numpy
    else:                  np = numpy

    if pl_cr:
        if compl:
            def amp_pha_of_hfpc_compl(f, *params_list):
                return utils.pl_cr_to_amp_pha(*func(f, *params_list), np=np)

            amp_pl, pha_pl, amp_cr, pha_cr = amp_pha_of_hfpc_compl(f, *params_list)

            del_amp_pl = part_deriv_gradient(amp_pha_of_hfpc_compl, f, params_list, deriv_mod, 0, step, method, order, d_order_n)
            del_pha_pl = part_deriv_gradient(amp_pha_of_hfpc_compl, f, params_list, deriv_mod, 1, step, method, order, d_order_n)
            del_amp_cr = part_deriv_gradient(amp_pha_of_hfpc_compl, f, params_list, deriv_mod, 2, step, method, order, d_order_n)
            del_pha_cr = part_deriv_gradient(amp_pha_of_hfpc_compl, f, params_list, deriv_mod, 3, step, method, order, d_order_n)

            del_hfp = utils.z_deriv_from_amp_pha(amp_pl, pha_pl, del_amp_pl, del_pha_pl, np=np)
            del_hfc = utils.z_deriv_from_amp_pha(amp_cr, pha_cr, del_amp_cr, del_pha_cr, np=np)

            return del_hfp, del_hfc

        else:
            def amp_pha_of_hfpc_real(f, *params_list):
                return utils.amp_pha_from_re_im(*func(f, *params_list), np=np)

            amp, pha = amp_pha_of_hfpc_real(f, *params_list)
            del_amp = part_deriv_gradient(amp_pha_of_hfpc_real, f, params_list, deriv_mod, 0, step, method, order, d_order_n)
            del_pha = part_deriv_gradient(amp_pha_of_hfpc_real, f, params_list, deriv_mod, 1, step, method, order, d_order_n)

            return utils.re_im_from_z(utils.z_deriv_from_amp_pha(amp, pha, del_amp, del_pha, np=np), np=np)

    else:
        if compl:
            def amp_pha_of_hf(f, *params_list):
                return utils.amp_pha_from_z(func(f, *params_list), np=np)

            amp, pha = amp_pha_of_hf(f, *params_list)
            del_amp = part_deriv_gradient(amp_pha_of_hf, f, params_list, deriv_mod, 0, step, method, order, d_order_n)
            del_pha = part_deriv_gradient(amp_pha_of_hf, f, params_list, deriv_mod, 1, step, method, order, d_order_n)

            return utils.z_deriv_from_amp_pha(amp, pha, del_amp, del_pha, np=np)

        else:
            return part_deriv_gradient(func, f, params_list, deriv_mod, None, step, method, order, d_order_n)


def part_deriv_gradient(func, f, params_list, deriv_mod, funcid=None, step=1e-9, method='central', order=2, d_order_n=1):
    '''
    A wrapper for various gradient derivative methods:
      - [numdifftools.Gradient] - can be used with lalsimulation and tf2 waveforms.
      - [jax.jacrev] - ca be used with jax waveforms (tf2_jx, tf2_tidal_jx).
        The packages ripple and wf4py could be used, but require wrappers to be written.

    Parameters
    ----------
    func : function
        The function for which the derivatives are calculated.
    f : float
        The frequency at which the function is evaluated.
    params_list : list
        The list of parameters at which the derivatives are evaluated.
    deriv_mod : str
        The method of numerical differentiation. Choices are ['numdifftools', 'jax'].
    funcid : int, optional
        The index of the output of the function for which the derivatives are calculated.
    step : float, optional
        The step size for the numerical differentiation.
    method : str, optional
        The method of numerical differentiation.
    order : int, optional
        The order of numerical differentiation.
    d_order_n : int, optional
        The order of the derivative.

    Returns
    -------
    del_hf : array
        The numerical partial derivatives of the function.
    '''

    if deriv_mod == 'numdifftools':

        if funcid is None:
            def wraps(x):
                return func(f, *x)
        else:
            def wraps(x):
                return func(f, *x)[funcid]

        if len(params_list) == 1: return nd.Gradient(wraps, step=step, method=method, order=order, n=d_order_n)(params_list)[:,None]
        else:                     return nd.Gradient(wraps, step=step, method=method, order=order, n=d_order_n)(params_list)

    elif deriv_mod == 'jax':

        if funcid is None:
            def wraps(*x):
                return func(f, *x)
        else:
            def wraps(*x):
                return func(f, *x)[funcid]

        return numpy.array(jax.jacrev(wraps, argnums=range(len(params_list)))(*params_list)).T

    else: raise ValueError(f'part_deriv_gradient: The derivative module {deriv_mod} is not supported.')
