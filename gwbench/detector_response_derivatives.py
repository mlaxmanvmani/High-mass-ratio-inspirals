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

import dill

import gwbench.antenna_pattern_np as ant_pat_np
import gwbench.antenna_pattern_sp as ant_pat_sp
import gwbench.waveform as wfc
import gwbench.wf_derivatives_num as wfd_num
import gwbench.wf_derivatives_sym as wfd_sym
import gwbench.utils as utils

try:
    import gwbench.antenna_pattern_jx as ant_pat_jx
except ModuleNotFoundError:
    pass

lambdified_functions_path = os.path.join(os.getcwd(), 'lambdified_functions')

def calc_det_responses_derivs_num(loc, wf, deriv_symbs_string, f_arr,
                                  params_dic, use_rot=1, label='hf',
                                  step=1e-9, method='central', order=2, d_order_n=1,
                                  user_locs=None, ana_deriv_symbs_string=None,
                                  ana_deriv_aux=None):
    '''
    Calculate the partial derivatives of the detector response numerically with respect to the
    specified variables [deriv_symbs_string] at the specified location [loc] using the waveform
    model [wf]. The partial derivatives are evaluated at params_dic and calculated using the
    numerical method specified by [method] and [order] and the step size [step]. The derivative
    order is specified by [d_order_n]. The user can also specify the analytic derivatives using
    [ana_deriv_symbs_string] and the corresponding auxiliary functions [ana_deriv_aux].

    Parameters
    ----------
    loc : str
        The location at which the detector response is evaluated.
    wf : Waveform
        The waveform model object.
    deriv_symbs_string : str
        The string of the derivative variables.
    f_arr : np.ndarray
        The array of frequencies at which the detector response is evaluated.
    params_dic : dict
        The dictionary of the waveform model parameters.
    use_rot : bool
        The flag to use the rotation of the detector response.
    label : str
        The label of the partial derivative.
    step : float
        The step size for the numerical derivative.
    method : str
        The method for the numerical derivative.
    order : int
        The order of the numerical derivative.
    d_order_n : int
        The derivative order.

    Returns
    -------
    dict
        The partial derivatives of the detector response or waveform polarizatons in the form of a dictionary.

    Note:
    If the location [loc] is None, the partial derivatives of the waveform polarizations [hfp, hfc]
    are calculated instead of the detector response.
    The returned dictionary has the following keys: 'del_var_hf{p,c} where var is indicates the
    variable with respect to which the derivative is calculated, and {p,c} indicates the plus or cross
    polarization. The dictionary also contains the keys 'variables' and 'deriv_variables' which
    indicate the variables and the derivative variables, respectively.
    '''

    deriv_symbs_list = deriv_symbs_string.split(' ')
    wf_symbs_list    = wf.wf_symbs_string.split(' ')
    ap_symbs_list    = ant_pat_np.ap_symbs_string.split(' ')
    dr_symbs_list    = utils.reduce_symbols_strings(wf.wf_symbs_string,
                                                    ant_pat_np.ap_symbs_string).split(' ')

    if ana_deriv_symbs_string is None: ana_deriv_symbs_list = []
    else:                              ana_deriv_symbs_list = ana_deriv_symbs_string.split(' ')

    if 'f' in deriv_symbs_list:     deriv_symbs_list.remove('f')
    if 'f' in ana_deriv_symbs_list: ana_deriv_symbs_list.remove('f')
    if 'f' in wf_symbs_list:        wf_symbs_list.remove('f')
    if 'f' in ap_symbs_list:        ap_symbs_list.remove('f')
    if 'f' in dr_symbs_list:        dr_symbs_list.remove('f')

    if wf.deriv_mod == 'jax': ant_pat = ant_pat_jx
    else:                     ant_pat = ant_pat_np

    if loc is None:
        def pc_func(f_arr, *wf_params_list):
            return wf.calc_wf_polarizations(f_arr, wf_params_list)

        return wfd_num.part_deriv_hf_func(pc_func, wf_symbs_list, deriv_symbs_list, f_arr, params_dic,
                                          wf.deriv_mod, pl_cr=True, compl=True, label=label,
                                          ana_deriv_symbs_list=ana_deriv_symbs_list,
                                          ana_deriv_aux=ana_deriv_aux,
                                          step=step, method=method, order=order, d_order_n=d_order_n)

    else:
        wf_ids = [ dr_symbs_list.index(el) for el in wf_symbs_list ]
        ap_ids = [ dr_symbs_list.index(el) for el in ap_symbs_list ]

        def dr_func(f_arr, *dr_params_list):
            return ant_pat.detector_response(
                *wf.calc_wf_polarizations(f_arr, [dr_params_list[idx] for idx in wf_ids]),
                f_arr, *[dr_params_list[idx] for idx in ap_ids ], loc, use_rot, user_locs=user_locs)

        return wfd_num.part_deriv_hf_func(dr_func, dr_symbs_list, deriv_symbs_list, f_arr, params_dic,
                                          wf.deriv_mod, pl_cr=False, compl=True, label=label,
                                          ana_deriv_symbs_list=ana_deriv_symbs_list,
                                          ana_deriv_aux=ana_deriv_aux,
                                          step=step, method=method, order=order, d_order_n=d_order_n)



def generate_det_responses_derivs_sym(wf_model_name, wf_other_var_dic, deriv_symbs_string, use_rot,
                                      locs=None, user_locs=None, pl_cr=True, user_waveform=None,
                                      user_lambdified_functions_path=None, logger=None):
    '''
    Generate the lambdified partial derivatives of the detector response with respect to the
    specified variables [deriv_symbs_string] at the specified locations [locs] using the waveform
    model [wf_model_name]. The user can also specify the locations [user_locs] and the waveform
    model [user_waveform]. The lambdified functions are stored in the directory specified by
    [user_lambdified_functions_path].

    Parameters
    ----------
    wf_model_name : str
        The name of the waveform model.
    wf_other_var_dic : dict
        The dictionary of the waveform model variables.
    deriv_symbs_string : str
        The string of the derivative variables.
    use_rot : bool
        The flag to use the rotation of the detector response.
    locs : list
        The list of locations at which the detector response is evaluated.
    user_locs : list
        The list of user specified locations.
    pl_cr : bool
        The flag to generate the plus/cross polarizations.
    user_waveform : Waveform
        The user specified waveform model.
    user_lambdified_functions_path : str
        The path to the directory to store the lambdified functions.
    logger : Logger
        The logger object.
    '''

    # initialize waveform object
    wf = wfc.Waveform(wf_model_name, wf_other_var_dic, user_waveform=user_waveform)

    # check that derivative variables are a subset of the detector response variables
    full_set = set( wf.wf_symbs_string.split(' ') + ant_pat_np.ap_symbs_string.split(' ') )
    sub_set  = set( deriv_symbs_string.split(' ') )
    if not sub_set <= full_set:
        utils.log_msg('The choice of derivative variables is not a subset of the waveform ' +
                      'and antenna pattern variables!', logger=logger, level='ERROR')

    utils.log_msg( 'Generating lambdified derivatives via sympy with the following settings:',
                  logger=logger, level='DEBUG')
    utils.log_msg(f'    wf_model_name      = {wf.wf_model_name}', logger=logger, level='DEBUG')
    utils.log_msg(f'    deriv_symbs_string = {deriv_symbs_string}', logger=logger, level='DEBUG')
    utils.log_msg(f'    use_rot            = {bool(use_rot)}', logger=logger, level='DEBUG')
    utils.log_msg( '    Use these settings for a network loading these derivatives.',
                  logger=logger, level='DEBUG')

    # make sure specified locations are known
    if locs is None: locs = utils.available_locs
    else:
        for loc in locs:
            if loc not in utils.available_locs and loc not in user_locs:
                utils.log_msg(f'generate_det_responses_derivs_sym: Specified location {loc} not ' +
                               'known in antenna pattern module and was not provided in user_locs.',
                                logger=logger, level='ERROR')

    if user_lambdified_functions_path is None: output_path = lambdified_functions_path
    else:
        output_path = os.path.join(user_lambdified_functions_path, 'lambdified_functions')
    if not os.path.exists(output_path): os.makedirs(output_path)

    responses = { loc : None for loc in locs }
    hfpc      = wf.calc_wf_polarizations_expr()
    if pl_cr: responses['pl_cr'] = hfpc

    # compute sympy expressions of the detector responses
    # compute derivatives
    for key in responses.keys():
        if key == 'pl_cr':
            utils.log_msg(f'Generating lambdified derivatives for the plus/cross polarizations.',
                          logger=logger, level='INFO')
            utils.log_msg('    Calculating derivatives of the plus/cross polarizations.',
                          logger=logger, level='INFO')
            _deriv_symbs_string = utils.remove_symbols(deriv_symbs_string, wf.wf_symbs_string)
            symbs_string        = wf.wf_symbs_string
        else:
            utils.log_msg(f'Generating lambdified derivatives for {key}.',
                          logger=logger, level='INFO')
            utils.log_msg(f'    Loading the detector response expression for {key}.',
                          logger=logger, level='INFO')
            responses[key]      = ant_pat_sp.detector_response(hfpc[0], hfpc[1], loc,
                                                               use_rot, user_locs=user_locs)
            utils.log_msg(f'    Calculating derivatives of the detector responses for: {key}.',
                          logger=logger, level='INFO')
            _deriv_symbs_string = deriv_symbs_string
            symbs_string        = utils.reduce_symbols_strings(wf.wf_symbs_string,
                                                               ant_pat_np.ap_symbs_string)

        if not _deriv_symbs_string:
            utils.log_msg( '    The plus and cross polarizations of the waveform model ' +
                          f'{wf.wf_model_name}   do not depend on the derivative variables:  ' +
                          f'{deriv_symbs_string}\n' +
                          f'         Did not generate lambdified functions file!',
                          logger=logger, level='WARNING')
            continue

        deriv_dic                    = wfd_sym.part_deriv_hf_expr(responses[key], symbs_string,
                                                                  _deriv_symbs_string,
                                                                  pl_cr=(key=='pl_cr'))
        deriv_dic['variables']       = symbs_string
        deriv_dic['deriv_variables'] = _deriv_symbs_string

        file_name = os.path.join(
            output_path,
            f'par_deriv_WFM_{wf.wf_model_name}_' +
            f'DVAR_{_deriv_symbs_string.replace(" ", "_")}_ROT_{int(use_rot)}_DET_{key}.dat')
        utils.log_msg(f'    Stored at: {file_name}', logger=logger, level='INFO')
        with open(file_name, "wb") as fi:
            dill.dump(deriv_dic, fi, recurse=True)


def load_det_responses_derivs_sym(loc, wf_model_name, deriv_symbs_string, use_rot, gen_derivs=None,
                                  return_bin=0, user_lambdified_functions_path=None, logger=None):
    '''
    Load the lambdified partial derivatives of the detector response with respect to the specified
    variables [deriv_symbs_string] at the specified location [loc] using the waveform model [wf_model_name].
    The lambdified functions are loaded from the directory specified by [user_lambdified_functions_path].

    Parameters
    ----------
    loc : str
        The location at which the detector response is evaluated.
    wf_model_name : str
        The name of the waveform model.
    deriv_symbs_string : str
        The string of the derivative variables.
    use_rot : bool
        The flag to use the rotation of the detector response.
    gen_derivs : dict
        The dictionary of the generated derivatives.
    return_bin : bool
        The flag to return the binary file.
    user_lambdified_functions_path : str
        The path to the directory to store the lambdified functions.
    logger : Logger
        The logger object.

    Returns
    -------
    dict:
        The partial derivatives of the detector response in the form of a dictionary.

    Note:
    If the location [loc] is None, the partial derivatives of the waveform polarizations [hfp, hfc]
    are loaded instead of the detector response.
    The returned dictionary has the following keys: 'del_var_hf{p,c} where var is indicates the variable
    with respect to which the derivative is calculated, and {p,c} indicates the plus or cross polarization.
    The dictionary also contains the keys 'variables' and 'deriv_variables' which indicate the variables
    and the derivative variables, respectively.
    '''

    if user_lambdified_functions_path is None: _user_lambdified_functions_path = lambdified_functions_path
    file_name = f'par_deriv_WFM_{wf_model_name}_' + \
                f'DVAR_{deriv_symbs_string.replace(" ", "_")}_ROT_{int(use_rot)}_DET_{loc}.dat'

    try:
        with open(os.path.join(_user_lambdified_functions_path, file_name), "rb") as fi:
            if return_bin: return fi.read()
            else:          return dill.load(fi)
    except FileNotFoundError:
        if gen_derivs is None: utils.log_msg(f'Could not find the lambdified function file: {file_name}',
                                             logger=logger, level='ERROR')
        else:
            if loc == 'pl_cr': locs = []
            else:              locs = [loc]
            generate_det_responses_derivs_sym(wf_model_name, gen_derivs['wf_other_var_dic'],
                                              deriv_symbs_string, use_rot, locs=locs,
                                              user_locs=gen_derivs['user_locs'], pl_cr=gen_derivs['pl_cr'],
                                              user_waveform=gen_derivs['user_waveform'], logger=logger,
                                              user_lambdified_functions_path=user_lambdified_functions_path)
            return load_det_responses_derivs_sym(loc, wf_model_name, deriv_symbs_string, use_rot,
                                                 gen_derivs=None, return_bin=return_bin, logger=logger,
                                                 user_lambdified_functions_path=user_lambdified_functions_path)
