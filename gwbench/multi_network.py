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


'''This module handles the benchmarking of graviational waveforms observed by multiple networks of detectors.

'''

import logging
from copy import copy, deepcopy

import dill
import numpy as np

import gwbench.network as nc
import gwbench.utils as utils

# logger
glob_logger = logging.getLogger('multi_network_module')
glob_logger.setLevel('INFO')

class MultiNetwork:
    '''
    Class for handling multiple networks of detectors and the benchmarking of gravitational waveforms observed by them.

    Attributes
    ----------
    network_specs : list
        list of network specifications, each of which is a list of detector keys.
    networks : list
        list of Network objects.
    loc_net : Network
        Network object for unique locations.

    Methods
    -------
    __init__(network_specs=None, logger_name='MultiNetwork', logger_level='WARNING', logger_level_network='WARNING')
        Initialize a MultiNetwork object.
    set_logger_level(level)
        Set the level of the logger.
    set_multi_network_from_specs(network_specs, logger_level_network='WARNING')
        Set the MultiNetwork object from the given network specifications.
    set_wf_vars(wf_model_name, wf_other_var_dic=None, user_waveform=None)
        Set the waveform variables for all networks.
    set_net_vars(wf_model_name=None, wf_other_var_dic=None, user_waveform=None,
                    f=None, inj_params=None, deriv_symbs_string=None, conv_cos=None, conv_log=None,
                    use_rot=None, user_locs=None, user_psds=None, user_lambdified_functions_path=None, ana_deriv_symbs_string=None)
        Set network variables for all networks (including the waveform variables).
    reset_ant_pat_lpf_psds()
        Reset the PSDs, antenna patterns, and LPFs for all networks.
    reset_wf_polarizations()
        Reset the waveform polarizations for all networks.
    reset_det_responses()
        Reset the detector responses for all networks.
    reset_snrs()
        Reset the SNRs for all networks.
    reset_errors()
        Reset the errors for all networks.
    check_none_vars_det(labels, ret_bool, tag='')
        Check for variables that are None in the networks.
    setup_ant_pat_lpf_psds(f_lo=-np.inf, f_hi=np.inf)
        Setup the PSDs, antenna patterns, and LPFs for all networks.
    calc_wf_polarizations()
        Calculate the waveform polarizations for the unique locations network.
    calc_wf_polarizations_derivs_num(step=1e-9, method='central', order=2, d_order_n=1)
        Calculate partial derivatives of the waveform polarizations numerically.
    load_wf_polarizations_derivs_sym(gen_derivs=None, return_bin=0)
        Load lambdified partial derivatives of the waveform polarizations for the unique locations network.
    calc_wf_polarizations_derivs_sym(gen_derivs=None)
        Calculate partial derivatives of the waveform polarizations using lambdified functions for the unique locations network.
    dist_wf_polarizations()
        Distribute the waveform polarizations and partial derivatives from the unique locations network to all networks.
    calc_det_responses()
        Calculate the detector responses for the unique locations network.
    calc_det_responses_derivs_num(step=1e-9, method='central', order=2, d_order_n=1)
        Calculate the partial derivatives of the detector responses numerically for the unique locations network.
    load_det_responses_derivs_sym(gen_derivs=None, return_bin=0)
        Load lambdified partial derivatives of the detector responses for the unique locations network.
    calc_det_responses_derivs_sym(gen_derivs=None, num_cores=None)
        Calculate the partial derivatives of the detector responses using lambdified functions for the unique locations network.
    dist_det_responses()
        Distribute the detector responses and partial derivatives from the unique locations network to all networks.
    calc_snrs(only_net=0)
        Calculate the SNRs for all networks.
    calc_snr_sq_integrand()
        Calculate the integrands of the SNR squared for all networks.
    calc_errors(cond_sup=None, only_net=0, derivs=None, step=None, method=None, order=None, gen_derivs=None, num_cores=None)
        Calculate the Fisher and covariance matrices and the errors for all networks.
    save_multi_network(filename_path)
        Save the network under the given path using *dill*.
    load_multi_network(filename_path)
        Loading the network from the given path using *dill*.
    '''

    ###
    #-----Init methods-----
    def __init__(self, network_specs=None, logger_name='MultiNetwork', logger_level='WARNING', logger_level_network='WARNING'):
        '''
        Initialize a MultiNetwork object.

        Parameters
        ----------
        network_specs : list
            list of network specifications, each of which is a list of detector keys.
        logger_name : str
            name of the logger.
        logger_level : str
            level of the logger.
        logger_level_network : str
            level of the logger for the networks.
        '''
        ##-----logger-----
        self.logger = utils.get_logger(name=logger_name, level=logger_level)

        ##-----initialize network object-----
        if network_specs is None:
            # network_specs list and list of networks
            self.network_specs = None
            self.networks = None
            # dummy networks for unique locations
            self.loc_net  = None
        elif isinstance(network_specs, list):
            self.set_multi_network_from_specs(network_specs, logger_level_network=logger_level_network)

        if self.networks is None: self.logger.debug('Empty MultiNetwork initialized.')
        else:                     self.logger.debug('MultiNetwork initialized.')

    ###
    #-----Setter methods-----
    #
    # it is best practice to always change the instance variables using these setter methods
    #
    def set_logger_level(self, level):
        '''
        Set the level of the logger.

        Parameters
        ----------
        level : str
            level of the logger.
        '''
        utils.set_logger_level(self.logger, level)

    def set_multi_network_from_specs(self, network_specs, logger_level_network='WARNING'):
        '''
        Set the MultiNetwork object from the given network specifications.

        Parameters
        ----------
        network_specs : list
            list of network specifications, each of which is a list of detector keys.
        logger_level_network : str
            level of the logger for the networks.
        '''
        # copy network_specs
        self.network_specs = copy(network_specs)
        # prepare list of all networks specified in network_specs
        self.networks = [ nc.Network(network_spec, logger_name='..'.join(network_spec), logger_level=logger_level_network)
                          for network_spec in network_specs ]
        # find unique locations and initialize Network for these with dummy technologies
        self.loc_net = nc.Network(
            list(dict.fromkeys([ f'tec_{det_key.split("_")[1]}' for network_spec in network_specs for det_key in network_spec ])),
            logger_name='loc_net', logger_level=logger_level_network)

    def set_wf_vars(self, wf_model_name, wf_other_var_dic=None, user_waveform=None):
        '''
        Set the waveform variables for all networks.

        Parameters
        ----------
        wf_model_name : str
            name of the waveform model.
        wf_other_var_dic : dict
            dictionary of other waveform variables.
        user_waveform : function
            user-defined waveform function.
        '''
        for net in [self.loc_net, *self.networks]:
            net.set_wf_vars(wf_model_name, wf_other_var_dic=wf_other_var_dic, user_waveform=user_waveform)

    def set_net_vars(self, wf_model_name=None, wf_other_var_dic=None, user_waveform=None,
                     f=None, inj_params=None, deriv_symbs_string=None, conv_cos=None, conv_log=None,
                     use_rot=None, user_locs=None, user_psds=None, user_lambdified_functions_path=None, ana_deriv_symbs_string=None):
        '''
        Set network variables for all networks (including the waveform variables). All variables are optional, with a default value of None.
        This is done to allow for a more flexible use of the method when some variables are already set.
        The variables that are strictly necessary are:

        Parameters
        ----------
        wf_model_name : str
            Name of the waveform model.
        wf_other_var_dic : dict, optional
            Dictionary of other waveform variables.
        user_waveform : function, optional
            User-defined waveform function.
        f : np.ndarray, optional
            Frequency array.
        inj_params : dict, optional
            Dictionary of injection parameters.
        deriv_symbs_string : str, optional
            String of derivative symbols.
        conv_cos : list, optional
            Flag for cosine conversion.
        conv_log : list, optional
            Flag for logarithmic conversion.
        use_rot : bool, optional
            Flag for using rotation.
        user_locs : dict, optional
            Dictionary of user-defined locations.
        user_psds : dict, optional
            Dictionary of user-defined PSDs.
        ana_deriv_symbs_string : str, optional
            String of analytical derivative symbols.
        user_lambdified_functions_path : str, optional
            Path to user-defined lambdified functions.
        '''
        for net in [self.loc_net, *self.networks]:
            net.set_net_vars(wf_model_name=wf_model_name, wf_other_var_dic=wf_other_var_dic, user_waveform=user_waveform,
                             f=f, inj_params=inj_params, deriv_symbs_string=deriv_symbs_string, conv_cos=conv_cos, conv_log=conv_log,
                             use_rot=use_rot, user_locs=user_locs, user_psds=user_psds, ana_deriv_symbs_string=ana_deriv_symbs_string,
                             user_lambdified_functions_path=user_lambdified_functions_path)


    ###
    #-----Resetter methods for instance variables-----
    def reset_ant_pat_lpf_psds(self):
        '''
        Reset the PSDs, antenna patterns, and LPFs for all networks.
        '''
        for net in [self.loc_net, *self.networks]:
            net.reset_ant_pat_lpf_psds()

    def reset_wf_polarizations(self):
        '''
        Reset the waveform polarizations for all networks.
        '''
        for net in [self.loc_net, *self.networks]:
            net.reset_wf_polarizations()

    def reset_det_responses(self):
        '''
        Reset the detector responses for all networks.
        '''
        for net in [self.loc_net, *self.networks]:
            net.reset_det_responses()

    def reset_snrs(self):
        '''
        Reset the SNRs for all networks.
        '''
        for net in [self.loc_net, *self.networks]:
            net.reset_snrs()

    def reset_errors(self):
        '''
        Reset the errors for all networks.
        '''
        for net in [self.loc_net, *self.networks]:
            net.reset_errors()


    ###
    #-----Helper-----
    def check_none_vars_det(self, labels, ret_bool, tag=''):
        '''
        Check for variables that are None in the networks.

        Parameters
        ----------
        labels : list
            List of variable names.
        ret_bool : bool
            Return boolean, if False log an error message, if True simply return True, if triggered.
        tag : str
            Tag for the log message.

        Returns
        -------
        bool
            Boolean, False if all variables are set, True if at least one variable is None and ret_bool is True.
        '''
        for net in [self.loc_net, *self.networks]:
            if net.check_none_vars_det(labels, ret_bool, tag=''): return True
        return False


    ###
    #-----PSDs and antenna patterns-----
    def setup_ant_pat_lpf_psds(self, f_lo=-np.inf, f_hi=np.inf):
        '''
        Setup the PSDs, antenna patterns, and LPFs for all networks.

        Parameters
        ----------
        f_lo : float
            Lower frequency limit.
        f_hi : float
            Upper frequency limit.
        '''
        for net in [self.loc_net, *self.networks]:
            net.setup_ant_pat_lpf_psds(f_lo=f_lo, f_hi=f_hi)
        self.logger.info('PSDs, antenna patterns, and LPFs loaded.')


    ###
    #-----Waveform polarizations-----
    def calc_wf_polarizations(self):
        '''
        Calculate the waveform polarizations for the unique locations network.
        '''
        self.loc_net.calc_wf_polarizations()

    def calc_wf_polarizations_derivs_num(self, step=1e-9, method='central', order=2, d_order_n=1):
        '''
        Calculate partial derivatives of the waveform polarizations numerically. The function also calculates the waveform polarizations.

        Parameters
        ----------
        step : float
            Step size for numerical differentiation.
        method : str
            Method for numerical differentiation.
        order : int
            Order of the numerical differentiation.
        d_order_n : int
            Number of derivatives.
        '''
        self.loc_net.calc_wf_polarizations_derivs_num(step=step, method=method, order=order, d_order_n=d_order_n)

    def load_wf_polarizations_derivs_sym(self, gen_derivs=None, return_bin=0):
        '''
        Load lambdified partial derivatives of the waveform polarizations for the unique locations network.

        Parameters
        ----------
        gen_derivs : bool
            Generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        return_bin : int
            Return binary flag.
        '''
        self.loc_net.load_wf_polarizations_derivs_sym(gen_derivs=gen_derivs, return_bin=return_bin)

    def calc_wf_polarizations_derivs_sym(self, gen_derivs=None):
        '''
        Calculate partial derivatives of the waveform polarizations using lambdified functions for the unique locations network.
        The function also calculates the waveform polarizations.

        Parameters
        ----------
        gen_derivs : bool
            Generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        '''
        self.loc_net.calc_wf_polarizations_derivs_sym(gen_derivs=gen_derivs)

    def dist_wf_polarizations(self):
        '''
        Distribute the waveform polarizations and partial derivatives from the unique locations network to all networks.
        '''
        for net in self.networks:
            net.hfp             =     copy(self.loc_net.hfp)
            net.hfc             =     copy(self.loc_net.hfc)
            net.del_hfpc        = deepcopy(self.loc_net.del_hfpc)
            net.del_hfpc_expr   = deepcopy(self.loc_net.del_hfpc_expr)
            net.inj_params      = deepcopy(self.loc_net.inj_params)
            net.deriv_variables =     copy(self.loc_net.deriv_variables)
        self.logger.info('Polarizations distributed among all networks.')


    ###
    #-----Detector responses-----
    def calc_det_responses(self):
        '''
        Calculate the detector responses for the unique locations network.
        '''
        self.loc_net.calc_det_responses()

    def calc_det_responses_derivs_num(self, step=1e-9, method='central', order=2, d_order_n=1, num_cores=None):
        '''
        Calculate the partial derivatives of the detector responses numerically for the unique locations network.

        Parameters
        ----------
        step : float
            Step size for numerical differentiation.
        method : str
            Method for numerical differentiation.
        order : int
            Order of the numerical differentiation.
        d_order_n : int
            Number of derivatives.
        num_cores : int
            Number of cores to use for parallel computation.
        '''
        self.loc_net.calc_det_responses_derivs_num(step=step, method=method, order=order, d_order_n=d_order_n, num_cores=num_cores)

    def load_det_responses_derivs_sym(self, gen_derivs=None, return_bin=0):
        '''
        Load lambdified partial derivatives of the detector responses for the unique locations network.

        Parameters
        ----------
        gen_derivs : bool
            Generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        return_bin : int
            Return binary flag.
        '''
        self.loc_net.load_det_responses_derivs_sym(gen_derivs=gen_derivs, return_bin=return_bin)

    def calc_det_responses_derivs_sym(self, gen_derivs=None, num_cores=None):
        '''
        Calculate the partial derivatives of the detector responses using lambdified functions for the unique locations network.
        The function also calculates the detector responses.

        Parameters
        ----------
        gen_derivs : bool
            Generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        num_cores : int
            Number of cores to use for parallel computation.
        '''
        self.loc_net.calc_det_responses_derivs_sym(gen_derivs=gen_derivs, num_cores=num_cores)

    def dist_det_responses(self):
        '''
        Distribute the detector responses and partial derivatives from the unique locations network to all networks.
        '''
        if self.check_none_vars_det(['psd'], True): self.setup_ant_pat_lpf_psds()
        for net in self.networks:
            net.inj_params          = deepcopy(self.loc_net.inj_params)
            net.deriv_variables     =     copy(self.loc_net.deriv_variables)
            for det in net.detectors:
                loc_det             = self.loc_net.get_detector(f'tec_{det.loc}')
                mask                = utils.min_max_mask(loc_det.f, det.f[0], det.f[-1])
                det.hf              = copy(loc_det.hf[mask])
                if loc_det.del_hf is not None:
                    det.del_hf = { deriv : copy(del_hf_deriv[mask]) for deriv,del_hf_deriv in loc_det.del_hf.items() }
                det.del_hf_expr     = deepcopy(loc_det.del_hf_expr)
        self.logger.info('Detector responses distributed among all networks.')


    ###
    #-----SNR calculations-----
    def calc_snrs(self, only_net=False, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculate the SNRs for all networks.

        Parameters
        ----------
        only_net : bool
            Calculate the SNRs for the networks only, default is False.
        '''
        if self.check_none_vars_det(['hf'], True):
            self.calc_det_responses()
            self.dist_det_responses()
        for net in self.networks:
            net.calc_snrs(only_net=only_net, f_lo=f_lo, f_hi=f_hi)
        self.logger.info('SNRs calculated.')

    def calc_snr_sq_integrand(self, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculate the integrands of the SNR squared for all networks.

        Parameters
        ----------
        only_net : bool
            Calculate the SNRs for the networks only, default is False.
        '''
        if self.check_none_vars_det(['hf'], True): self.calc_det_responses()
        for net in self.networks:
            net.calc_snr_sq_integrand(f_lo=f_lo, f_hi=f_hi)
        self.logger.info('SNR integrands calculated.')


    ###
    #-----Error calculation and Fisher analysis-----
    def calc_errors(self, cond_sup=None, only_net=0, derivs=None, step=None, method=None, order=None, gen_derivs=None, num_cores=None, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculate the Fisher and covariance matrices and the errors for all networks. If the derivatives are not pre-calculated,
        the method will calculate and distribute them using the specified method (derivs, step, method, order, gen_derivs, num_cores).

        Parameters
        ----------
        cond_sup : float
            Condition number suppression for the Fisher matrix.
        only_net : bool, optional
            Calculate the errors for the networks only, default is False.
        derivs : str, optional
            Type of derivatives, default is None. Valid options are: 'num' (numerical), 'sym' (symbolic).
        step : float, optional
            Step size for numerical differentiation, default is None.
        method : str, optional
            Method for numerical differentiation, default is None.
        order : int, optional
            Order of the numerical differentiation, default is None.
        gen_derivs : bool, optional
            Generate derivatives, if True, default is None.
        num_cores : int, optional
            Number of cores to use for parallel computation, default is None.
        '''
        if self.check_none_vars_det(['del_hf'], True):
            if not self.loc_net.check_none_vars_det(['del_hf'], True):
                self.dist_det_responses()
            elif derivs == 'num' and None not in [step, method, order]:
                self.calc_det_responses_derivs_num(step=step, method=method, order=order, d_order_n=1, num_cores=num_cores)
                self.dist_det_responses()
            elif derivs == 'sym':
                self.load_det_responses_derivs_sym(gen_derivs=gen_derivs)
                self.calc_det_responses_derivs_sym(num_cores=num_cores)
                self.dist_det_responses()
            else: utils.log_msg('calc_errors: Neither detector response derivatives have been pre-calculated, nor was ' +
                                'the differentiation method specified. Calculate the derivatives beforehand ' +
                                'or specify the derivative type (derivs=[num, sym]) and [step, method, order] for ' +
                                'the numerical differentiation.', logger=self.logger, level='ERROR')
        self.logger.info('Calculate errors (Fisher & cov matrices).')
        for net in self.networks:
            net.calc_errors(cond_sup=cond_sup, only_net=only_net, f_lo=f_lo, f_hi=f_hi)
        self.logger.info('Errors calculated.')


    ###
    #-----IO methods-----
    def save_multi_network(self, filename_path):
        '''Save the network under the given path using *dill*.'''
        with open(filename_path, "wb") as fi:
            dill.dump(self, fi, recurse=True)
        self.logger.info('MultiNetwork pickled.')
        return

    def load_multi_network(filename_path):
        '''Loading the network from the given path using *dill*.'''
        with open(filename_path, "rb") as fi:
            mul_net = dill.load(fi)
        mul_net.logger.info('MultiNetwork loaded.')
        return mul_net
