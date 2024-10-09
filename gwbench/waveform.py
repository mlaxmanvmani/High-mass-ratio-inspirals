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

import importlib.util
import os

import sympy as sp

from gwbench.utils import log_msg, get_sub_dict

from gwbench.wf_models import lal_bbh_np
from gwbench.wf_models import lal_bbh_np_ecc
from gwbench.wf_models import lal_bns_np
from gwbench.wf_models import tf2_np
from gwbench.wf_models import tf2_sp
from gwbench.wf_models import tf2_tidal_np
from gwbench.wf_models import tf2_tidal_sp

try:
    from gwbench.wf_models import tf2_jx
    from gwbench.wf_models import tf2_tidal_jx
except ModuleNotFoundError:
    tf2_jx       = None
    tf2_tidal_jx = None

###
#-----Get waveform functions for np, sp and the symbols string based on the model name-----
def select_wf_model_quants(wf_model_name, user_waveform=None, logger=None):
    '''
    Selects the waveform model and returns the numpy and sympy functions and the symbols string.

    Parameters:
        wf_model_name [str]: The name of the waveform model.
        user_waveform [dict]: A dictionary containing the paths to the numpy and sympy versions of a user-specified waveform model.
        logger [Logger]: An instance of the Logger class.

    Returns:
        wf_symbs_string [str]: The string containing the symbols of the waveform model.
        hfpc    [function]: The numpy / jax version of the waveform model.
        hfpc_sp [function]: The sympy version of the waveform model.
    '''
    if wf_model_name is None: return None, None, None, None

    if user_waveform is not None:             wf_mod = load_module_from_file(user_waveform)
    else:
        if   wf_model_name == 'lal_bbh':      wf_mod = lal_bbh_np
        elif wf_model_name == 'lal_bbh_ecc':  wf_mod = lal_bbh_np_ecc
        elif wf_model_name == 'lal_bns':      wf_mod = lal_bns_np
        elif wf_model_name == 'tf2':          wf_mod = tf2_sp
        elif wf_model_name == 'tf2_jx':       wf_mod = tf2_jx
        elif wf_model_name == 'tf2_np':       wf_mod = tf2_np
        elif wf_model_name == 'tf2_sp':       wf_mod = tf2_sp
        elif wf_model_name == 'tf2_tidal':    wf_mod = tf2_tidal_sp
        elif wf_model_name == 'tf2_tidal_jx': wf_mod = tf2_tidal_jx
        elif wf_model_name == 'tf2_tidal_np': wf_mod = tf2_tidal_np
        elif wf_model_name == 'tf2_tidal_sp': wf_mod = tf2_tidal_sp
        else: log_msg(f'select_wf_model_quants: wf_model_name {wf_model_name} is not known!', logger=logger, level='ERROR')

    if wf_mod is None: log_msg('waveform.py: JAX is not installed; JAX-based waveform models cannot be used!', level='ERROR')
    return wf_mod.wf_symbs_string, wf_mod.deriv_mod, wf_mod.hfpc, getattr(wf_mod, 'hfpc_sp', None)

def load_module_from_file(file_path):
    '''
    Loads a module from a file.

    Parameters
    ----------
    file_path : str
        The path to the file containing the module.

    Returns
    -------
    module : module
        The module loaded from the file.
    '''
    spec   = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(file_path))[0], file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Waveform(object):
    '''
    A class to store the waveform model and its parameters.

    Attributes
    ----------
    wf_model_name : str
        The name of the waveform model.
    wf_other_var_dic : dict
        A dictionary containing the other variables of the waveform model.
    user_waveform : dict
        A dictionary containing the paths to the numpy and sympy versions of a user-specified waveform model.
    wf_symbs_string : str
        The string containing the symbols of the waveform model.
    hfpc : function
        The numpy / jax version of the waveform model.
    hfpc_sp : function
        The sympy version of the waveform model.
    deriv_mod : str
        The name of the derivative module for numerical differentiation: ['numdifftools', 'jax'].

    Methods
    -------
    calc_wf_polarizations(f, inj_params)
        Calculates the waveform polarizations using the numpy version of the waveform model.
    calc_wf_polarizations_expr()
        Calculates the waveform polarizations using the sympy version of the waveform model.
    get_sp_expr()
        Returns the sympy expression of the waveform model.
    eval_np_func(f, inj_params)
        Evaluates the numpy version of the waveform model.
    print_waveform()
        Prints the waveform model and its parameters.

    Example
    -------
    wf = Waveform(wf_model_name='lal_bbh', wf_other_var_dic={'approximant':'IMRPhenomXHM'})
    wf.print_waveform()

    Note
    ----
    The numpy / jax and sympy versions of the waveform model are stored in the hfpc and hfpc_sp attributes, respectively.
    '''

    ###
    #-----Init methods-----
    def __init__(self, wf_model_name=None, wf_other_var_dic=None, user_waveform=None, logger=None):
        '''
        Initializes the Waveform class.

        Parameters
        ----------
        wf_model_name : str
            The name of the waveform model.
        wf_other_var_dic : dict
            A dictionary containing the other variables of the waveform model.
        user_waveform : dict
            A dictionary containing the paths to the numpy and sympy versions of a user-specified waveform model.
        logger : Logger
            An instance of the Logger class.
        '''

        if wf_other_var_dic is None: wf_other_var_dic = {}

        self.wf_model_name    = wf_model_name
        self.wf_other_var_dic = wf_other_var_dic
        self.user_waveform    = user_waveform

        self.wf_symbs_string, self.deriv_mod, self.hfpc, self.hfpc_sp = \
                select_wf_model_quants(wf_model_name, user_waveform=user_waveform, logger=logger)


    ###
    #-----Getter methods-----
    def calc_wf_polarizations(self, f, inj_params):
        '''
        Calculates the waveform polarizations using the numpy version of the waveform model.

        Parameters
        ----------
        f : float
            The frequency.
        inj_params : dict or list
            A dictionary or list containing the injection parameters.

        Returns
        -------
        hfpc : list
            A list containing the waveform polarizations.
        '''
        if isinstance(inj_params, dict): return self.hfpc(f, **get_sub_dict(inj_params, self.wf_symbs_string), **self.wf_other_var_dic)
        else:                            return self.hfpc(f, *inj_params, **self.wf_other_var_dic)

    def calc_wf_polarizations_expr(self):
        '''
        Calculates the waveform polarizations expression using the sympy version of the waveform model.

        Returns
        -------
        hfpc : sympy expression
            The sympy expression of the waveform polarizations.
        '''
        if self.hfpc_sp is None: log_msg('get_sp_expr: Waveform does not have a sympy expression!', level='ERROR')
        return self.hfpc_sp(*[sp.symbols(name, real=True) for name in self.wf_symbs_string.split(' ')], **self.wf_other_var_dic)

    def get_sp_expr(self):
        '''
        Returns the sympy expression of the waveform model.

        Returns
        -------
        hfpc : sympy expression
            The sympy expression of the waveform model.
        '''
        return self.calc_wf_polarizations_expr()

    def eval_np_func(self, f, inj_params): # for legacy use
        '''
        Evaluates the numpy version of the waveform model.

        Parameters
        ----------
        f : np.ndarray
            The frequency.
        inj_params : dict or list
            A dictionary or list containing the injection parameters.

        Returns
        -------
        hfpc : np.ndarray
            The waveform polarizations.
        '''
        return self.calc_wf_polarizations(f, inj_params)


    ###
    #-----IO methods-----
    def print_waveform(self):
        '''
        Prints the waveform model and its parameters.
        '''
        for key,value in vars(self).items():
            print(key.ljust(16,' '),'  ',value)
            print()
