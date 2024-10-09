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


'''This module handles the benchmarking of gravitaional waveforms observed by a network of detectors.

'''

import logging
from copy import copy, deepcopy

import dill
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import find_peaks

import gwbench.detector as dc
import gwbench.detector_response_derivatives as drd
import gwbench.err_deriv_handling as edh
import gwbench.fisher_analysis_tools as fat
import gwbench.waveform as wfc
import gwbench.utils as utils

# logger
glob_logger = logging.getLogger('network_module')
glob_logger.setLevel('INFO')

class Network:
    '''
    Class for handling a network of detectors and the benchmarking of gravitational waveforms.

    Attributes
    ----------
    label : str
        label of the network
    det_keys : list
        list of detector labels
    detectors : list
        list of detectors in the network
    f : array
        frequency array
    inj_params : dict
        dictionary of injection parameters
    deriv_symbs_string : str
        derivative variables
    deriv_variables : list
        list of derivative variables
    ana_deriv_symbs_string : str
        derivative variables for which analytical derivatives should be used
    ana_deriv_symbs_list : list
        list of derivative variables for which analytical derivatives should be used
    conv_cos : list
        list of inj_params to convert to cos versions
    conv_log : list
        list of inj_params to convert to log versions
    use_rot : bool
        use f-dependent gmst (SPA) in antenna patterns
    user_locs : dict
        user defined detector locations
    user_psds : dict
        user defined PSDs
    user_lambdified_functions_path : str
        path to user defined lambdified functions
    hfp : array
        plus polarization
    hfc : array
        cross polarization
    del_hfpc : dict
        derivative dictionary for polarizations
    del_hfpc_expr : dict
        sympy expressions of derivative dictionary for polarizations
    snr : float
        SNR
    snr_sq : float
        SNR^2
    fisher : array
        Fisher matrix
    cond_num : float
        condition number of Fisher matrix
    cov : array
        covariance matrix
    inv_err : dict
        dictionary containing information about the inversion error between the two matrices
    errs : dict
        dictionary of errors for given derivative variables

    Methods
    -------
    __init__(network_spec=None, logger_name='Network', logger_level='INFO')
        Initialize the network object
    set_logger_level(level)
        Set the logger level
    set_network_and_detectors_from_label(network_label)
        Set network and detectors from label
    set_network_and_detectors_from_key_list(det_key_list)
        Set network and detectors from key list
    set_wf_vars(wf_model_name, wf_other_var_dic=None, user_waveform=None)
        Set waveform variables
    set_net_vars(wf_model_name=None, wf_other_var_dic=None, user_waveform=None, f=None, inj_params=None, deriv_symbs_string=None, conv_cos=None, conv_log=None, use_rot=None, user_locs=None, user_psds=None, user_lambdified_functions_path=None, ana_deriv_symbs_string=None)
        Set network variables
    reset_ant_pat_lpf_psds()
        Reset PSDs, antenna patterns, and LPFs
    reset_wf_polarizations()
        Reset waveform polarizations
    reset_det_responses()
        Reset detector responses
    reset_snrs()
        Reset SNRs
    reset_errors()
        Reset Fisher and covariance matrices and errors
    get_detector(det_key)
        Get detector
    get_snrs_errs_cov_fisher_inv_err_for_key(key='network')
        Get SNR, errors, covariance, Fisher, and inversion error for given key
    pool_calc_func(calc_func, num_cores)
        Pool calculation function
    check_none_vars_net(labels, ret_bool, tag='')
        Check none variables for network
    check_none_vars_det(labels, ret_bool, tag='')
        Check none variables for detectors
    setup_ant_pat_lpf_psds(f_lo=-np.inf, f_hi=np.inf)
        Setup antenna patterns, LPFs, and PSDs
    calc_wf_polarizations()
        Calculate waveform polarizations
    calc_wf_polarizations_derivs_num(step=1e-9, method='central', order=2, d_order_n=1)
        Calculate numeric derivatives of polarizations
    load_wf_polarizations_derivs_sym(gen_derivs=None, return_bin=False)
        Load lambdified polarizations
    calc_wf_polarizations_derivs_sym(gen_derivs=None)
        Calculate lambdified polarizations
    calc_det_responses()
        Calculate detector responses
    calc_det_responses_derivs_num(step=1e-9, method='central', order=2, d_order_n=1, num_cores=None)
        Calculate numeric derivatives of detector responses
    load_det_responses_derivs_sym(gen_derivs=None, return_bin=False)
        Load lambdified detector responses
    calc_det_responses_derivs_sym(gen_derivs=None, num_cores=None)
        Calculate lambdified detector responses
    calc_snrs(only_net=0)
        Calculate SNRs
    calc_snr_sq_integrand()
        Calculate squared SNR integrands
    calc_errors(cond_sup=None, only_net=0, derivs=None, step=None, method=None, order=None, gen_derivs=None, num_cores=None)
        Calculate Fisher and covariance matrices and errors
    save_network(filename_path)
        Save network
    load_network(filename_path)
        Load network
    print_network()
        Print network
    print_detectors()
        Print detectors within network
    '''

    ###
    #-----Init methods-----
    def __init__(self, network_spec=None, logger_name='Network', logger_level='INFO'):
        '''
        Initialize the network object.

        Parameters
        ----------
        network_spec : str, list, tuple, None
            Network specification, default is None.
        logger_name : str
            Name of the logger, default is 'Network'.
        logger_level : str
            Level of the logger, default is 'INFO'.
        '''
        ##-----logger-----
        self.logger = utils.get_logger(name=logger_name, level=logger_level)

        ##-----initialize network object-----
        if network_spec is None:
            #-----network and detectors
            # network label
            self.label = None
            # detector labels list
            self.det_keys = None
            # list of detectors in network
            self.detectors = None
        elif isinstance(network_spec, str):
            #-----network and detectors
            self.set_network_and_detectors_from_label(network_spec)
        elif isinstance(network_spec, list) or isinstance(network_spec, tuple):
            #-----network and detectors
            self.set_network_and_detectors_from_key_list(network_spec)

        #-----injection and waveform quantities-----
        # frequency array
        self.f = None
        # dictionary of injection parameters
        self.inj_params = None
        # derivative variables - symbs_string and list
        self.deriv_symbs_string = None
        self.deriv_variables = None
        # derivative variables for which analytical derivatives should be used
        self.ana_deriv_symbs_string = None
        self.ana_deriv_symbs_list = None
        # waveform
        self.wf = None

        #-----analysis settings-----
        # list of inj_params to convert to cos, ln versions
        self.conv_cos = None
        self.conv_log = None
        # use f-dependent gmst (SPA) in antenna patterns
        self.use_rot = None

        #-----user defined detector locations and PSDs-----
        self.user_locs = None
        self.user_psds = None
        self.user_lambdified_functions_path = None

        #-----waveform polarizations-----
        # plus/cross polarizations
        self.hfp = None
        self.hfc = None
        # derivative dictionary for polarizations
        self.del_hfpc = None
        # sympy expressions of derivative dictionary for polarizations
        self.del_hfpc_expr = None

        #-----network SNR-----
        # SNR, SNR^2 calculated from hf
        self.snr = None
        self.snr_sq = None

        #-----network errors-----
        # Fisher matrix
        self.fisher = None
        # condition number of Fisher matrix
        self.cond_num = None
        # covariance matrix
        self.cov = None
        # dictionary containing information about the inversion error between the two matrices
        self.inv_err = None
        # dictionary of errors for given derivative variables
        self.errs = None

        if self.label is None: self.logger.debug('Empty network initialized.')
        else:                  self.logger.debug('Network initialized.')


    ###
    #-----Setter methods-----
    #
    # it is best practice to always change the instance variables using these setter methods
    #
    def set_logger_level(self, level):
        '''
        Set the logger level.

        Parameters
        ----------
        level : str
            level of the logger
        '''
        utils.set_logger_level(self.logger, level)

    def set_network_and_detectors_from_label(self, network_label):
        '''
        Set network and detectors from a label.

        Parameters
        ----------
        network_label : str
            label of the network
        '''
        #-----network and detectors
        self.label     = network_label
        self.det_keys  = utils.read_det_keys_from_label(network_label)
        self.detectors = [ dc.Detector(det_key) for det_key in self.det_keys ]

    def set_network_and_detectors_from_key_list(self, det_key_list):
        '''
        Set network and detectors from a list of detector keys.

        Parameters
        ----------
        det_key_list : list, tuple
            list of detector keys
        '''
        #-----network and detectors
        if isinstance(det_key_list, tuple): self.det_keys = [ tec + '_' + loc for tec,loc in zip(det_key_list[0], det_key_list[1]) ]
        else:                               self.det_keys = copy(det_key_list)
        self.label     = '..'.join(self.det_keys)
        self.detectors = [ dc.Detector(det_key) for det_key in self.det_keys ]

    def set_wf_vars(self, wf_model_name, wf_other_var_dic=None, user_waveform=None):
        '''
        Set waveform variables.

        Parameters
        ----------
        wf_model_name : str
            name of the waveform model
        wf_other_var_dic : dict
            dictionary of other waveform variables, default is None.
        user_waveform : function
            user defined waveform, default is None.
        '''
        self.wf = wfc.Waveform(wf_model_name, wf_other_var_dic=wf_other_var_dic, user_waveform=user_waveform, logger=self.logger)

    def set_net_vars(self, wf_model_name=None, wf_other_var_dic=None, user_waveform=None,
                     f=None, inj_params=None, deriv_symbs_string=None, conv_cos=None, conv_log=None,
                     use_rot=None, user_locs=None, user_psds=None, user_lambdified_functions_path=None, ana_deriv_symbs_string=None):
        '''
        Set network variables (iclude waveform variables). All variables are optional, with a default value of None. This is done to allow
        for a more flexible use of the method when some variables are already set.

        Parameters
        ----------
        wf_model_name : str, optional
            Name of the waveform model, default is None.
        wf_other_var_dic : dict, optional
            Dictionary of other waveform variables, default is None.
        user_waveform : function, optional
            User defined waveform, default is None.
        f : array, optional
            Frequency array, default is None.
        inj_params : dict, optional
            Dictionary of injection parameters, default is None.
        deriv_symbs_string : str, optional
            Derivative variables, default is None.
        conv_cos : list, optional
            List of inj_params to convert to cos versions, default is None.
        conv_log : list, optional
            List of inj_params to convert to log versions, default is None.
        use_rot : bool, optional
            Use f-dependent gmst (SPA) in antenna patterns, default is None.
        user_locs : dict, optional
            User defined detector locations, default is None.
        user_psds : dict, optional
            User defined PSDs, default is None.
        user_lambdified_functions_path : str, optional
            Path to user defined lambdified functions, default is None.
        ana_deriv_symbs_string : str, optional
            Derivative variables for which analytical derivatives should be used, default is None.
        '''
        if wf_model_name is not None:
            self.set_wf_vars(wf_model_name, wf_other_var_dic=wf_other_var_dic, user_waveform=user_waveform)
        if f is not None:
            self.f = copy(f)
            if self.detectors is not None:
                for det in self.detectors:
                    det.set_f(self.f)
        if inj_params is not None:
            self.inj_params = deepcopy(inj_params)
        if deriv_symbs_string is not None:
            self.deriv_symbs_string = copy(deriv_symbs_string)
            self.deriv_variables = deriv_symbs_string.split(' ')
        if ana_deriv_symbs_string is not None:
            self.ana_deriv_symbs_string = copy(ana_deriv_symbs_string)
            self.ana_deriv_symbs_list = ana_deriv_symbs_string.split(' ')
        if conv_cos is not None:
            self.conv_cos = copy(conv_cos)
        if conv_log is not None:
            self.conv_log = copy(conv_log)
        if use_rot is not None:
            self.use_rot = copy(use_rot)
        if user_locs is not None:
            self.user_locs = deepcopy(user_locs)
        if user_psds is not None:
            self.user_psds = deepcopy(user_psds)
        if user_lambdified_functions_path is not None:
            self.user_lambdified_functions_path = copy(user_lambdified_functions_path)

    ##
    #-----Resetter methods for instance variables-----
    def reset_ant_pat_lpf_psds(self):
        '''
        Reset PSDs, antenna patterns, and LPFs.
        '''
        for det in self.detectors:
            det.Fp = None
            det.Fc = None
            det.Flp = None
            det.psd = None
        self.logger.info('Reset PSDs, antenna patterns, and LPFs.')

    def reset_wf_polarizations(self):
        '''
        Reset waveform polarizations and their derivatives.
        '''
        self.hfp = None
        self.hfc = None
        self.del_hfpc = None
        self.del_hfpc_expr = None
        self.logger.info('Reset waveform polarizations and derivatives.')

    def reset_det_responses(self):
        '''
        Reset detector responses and their derivatives.
        '''
        for det in self.detectors:
            det.hf = None
            det.del_hf = None
            det.del_hf_expr = None
        self.logger.info('Reset detector responses and derivatives.')

    def reset_snrs(self):
        '''
        Reset SNRs and related quantities.
        '''
        self.snr = None
        self.snr_sq = None
        for det in self.detectors:
            det.snr = None
            det.snr_sq = None
            det.d_snr_sq = None
        self.logger.info('Reset SNRs.')

    def reset_errors(self):
        '''
        Reset Fisher and covariance matrices, errors and related quantities.
        '''
        self.fisher = None
        self.cond_num = None
        self.cov = None
        self.inv_err = None
        self.errs = None
        for det in self.detectors:
            det.fisher = None
            det.cond_num = None
            det.cov = None
            det.inv_err = None
            det.errs = None
        self.logger.info('Reset Fisher and covariance matrices and errors.')


    ###
    #-----Getters-----
    def get_detector(self, det_key):
        '''
        Get detector specified by the det_key.
        '''
        return self.detectors[self.det_keys.index(det_key)]

    def get_snrs_errs_cov_fisher_inv_err_for_key(self, key='network'):
        '''
        Get SNR, errors, covariance, Fisher, and inversion error for given key.
        '''
        if key == 'network': out_obj = self
        else:                out_obj = self.detectors[self.det_keys.index(key)]
        return out_obj.snr, out_obj.errs, out_obj.cov, out_obj.fisher, out_obj.inv_err


    ###
    #----Helper-----
    def pool_calc_func(self, calc_func, num_cores):
        '''
        Function to be used for parallel calculation of detector responses and their derivatives.

        Parameters
        ----------
        calc_func : function
            function to be parallelized
        num_cores : int
            number of cores
        '''
        self.detectors, self.inj_params, self.deriv_variables = list(zip(*Parallel(n_jobs=num_cores)(delayed(calc_func)(det) for det in self.detectors)))
        self.inj_params                                       = self.inj_params[0]
        self.deriv_variables                                  = self.deriv_variables[0]

    def check_none_vars_net(self, labels, ret_bool, tag=''):
        '''
        Check for variables that are None in the Network.

        Parameters
        ----------
        labels : list
            list of variable names
        ret_bool : bool
            return boolean, if False log an error message, if True simly return True, if triggered.
        tag : str
            tag for the log message

        Returns
        -------
        bool
            boolean, False if all variables are set, True if at least one variable is None and ret_bool is True.
        '''
        if tag: tag += ': '
        for label in labels:
            if self.__getattribute__(label) is None:
                if ret_bool: return True
                else:        utils.log_msg(f'{tag}{label} is not set!', logger=self.logger, level='ERROR')
        return False

    def check_none_vars_det(self, labels, ret_bool, tag=''):
        '''
        Check for variables that are None in the detectors.

        Parameters
        ----------
        labels : list
            List of variable names
        ret_bool : bool
            If tiggered, log an error message if False, else return True (to handle the trigger externally)
        tag : str
            Tag for the log message

        Returns
        -------
        bool
            Boolean, False if all variables are set, True if at least one variable is None and ret_bool is True.
        '''
        if tag: tag += ': '
        for label in labels:
            for det in self.detectors:
                if det.__getattribute__(label) is None:
                    if ret_bool: return True
                    else:        utils.log_msg(f'{tag}{det.det_key} - {label} is not set!', logger=self.logger, level='ERROR')
        return False


    ###
    #-----PSDs and antenna patterns-----
    def setup_ant_pat_lpf_psds(self, f_lo=-np.inf, f_hi=np.inf):
        '''
        Setup antenna patterns, LPFs, and PSDs.

        Parameters
        ----------
        f_lo : float
            Low frequency cutoff, default is -np.inf.
        f_hi : float
            High frequency cutoff, default is np.inf.
        '''
        self.check_none_vars_net(['inj_params'], False, tag='setup_ant_pat_lpf_psds')
        self.check_none_vars_det(['f'], False, tag='setup_ant_pat_lpf_psds')
        for det in self.detectors:
            # the psd has to be set first to avoid inconsistent array lengths
            det.setup_psds(f_lo, f_hi, user_psds=self.user_psds)
            det.setup_ant_pat_lpf(self.inj_params, self.use_rot, user_locs=self.user_locs)
        self.logger.info('PSDs, antenna patterns, and LPFs loaded.')

    def setup_psds(self, f_lo=-np.inf, f_hi=np.inf):
        '''
        Setup antenna patterns, LPFs, and PSDs. Meant to be used when preparing the nettwork
        to be used with the simulator module.

        Parameters
        ----------
        f_lo : float
            Low frequency cutoff, default is -np.inf.
        f_hi : float
            High frequency cutoff, default is np.inf.
        '''
        self.check_none_vars_det(['f'], False, tag='setup_ant_pat_lpf_psds')
        for det in self.detectors:
            det.setup_psds(f_lo, f_hi, user_psds=self.user_psds)
        self.logger.info('PSDs loaded.')


    ###
    #-----Waveform polarizations-----
    def calc_wf_polarizations(self):
        '''
        Calculate waveform polarizations.
        '''
        self.check_none_vars_net(['wf', 'f', 'inj_params'], False, tag='calc_wf_polarizations')
        self.hfp, self.hfc = self.wf.calc_wf_polarizations(self.f, self.inj_params)
        self.logger.info('Polarizations calculated.')

    def calc_wf_polarizations_derivs_num(self, step=1e-9, method='central', order=2, d_order_n=1):
        '''
        Calculate partial derivatives of the waveform polarizations numerically. The function also calculates the waveform polarizations.

        Parameters
        ----------
        step : float, optional
            Step size for numerical differentiation, default is 1e-9.
        method : str, optional
            Method for numerical differentiation, default is 'central'.
        order : int, optional
            Order of the method for numerical differentiation, default is 2.
        d_order_n : int, optional
            Number of derivatives, default is 1.
        '''
        self.logger.info('Calculate numeric derivatives of polarizations.')
        self.check_none_vars_net(['deriv_symbs_string', 'deriv_variables'], False, tag='calc_wf_polarizations_derivs_num')
        self.calc_wf_polarizations()
        self.del_hfpc = drd.calc_det_responses_derivs_num(None, self.wf, utils.remove_symbols(self.deriv_symbs_string,self.wf.wf_symbs_string),
                                                          self.f, self.inj_params, use_rot=self.use_rot, label='hf',
                                                          step=step, method=method, order=order, d_order_n=d_order_n, user_locs=self.user_locs,
                                                          ana_deriv_symbs_string=self.ana_deriv_symbs_string, ana_deriv_aux={'hfp':self.hfp,'hfc':self.hfc})
        self.del_hfpc, c_quants = edh.get_conv_del_eval_dic(self.del_hfpc, self.inj_params, self.conv_cos, self.conv_log)
        self.inj_params, self.deriv_variables = edh.get_conv_inj_params_deriv_variables(c_quants, self.inj_params, self.deriv_variables)
        self.logger.info('Numeric derivatives of polarizations calculated.')

    def load_wf_polarizations_derivs_sym(self, gen_derivs=None, return_bin=False):
        '''
        Load lambdified partial derivatives of the waveform polarizations.

        Parameters
        ----------
        gen_derivs : bool, optional
            Generate derivatives, if True. If False or None, the derivatives are loaded from file. Default is None.
        return_bin : bool, optional
            Return binary. Default is False. Might be needed for parallel calculations.
        '''
        self.check_none_vars_net(['wf', 'deriv_symbs_string'], False, tag='load_wf_polarizations_derivs_sym')
        if gen_derivs is not None and gen_derivs:
            _gen_derivs = {'wf_other_var_dic': self.wf.wf_other_var_dic, 'user_waveform':self.wf.user_waveform, 'pl_cr':True, 'user_locs':self.user_locs}
        else: _gen_derivs = None
        self.del_hfpc_expr = drd.load_det_responses_derivs_sym('pl_cr', self.wf.wf_model_name,
                                                               utils.remove_symbols(self.deriv_symbs_string, self.wf.wf_symbs_string),
                                                               self.use_rot, gen_derivs=_gen_derivs, return_bin=return_bin, logger=self.logger,
                                                               user_lambdified_functions_path=self.user_lambdified_functions_path)
        self.logger.info('Lambdified polarizations loaded.')

    def calc_wf_polarizations_derivs_sym(self, gen_derivs=None):
        '''
        Calculate partial derivatives of the waveform polarizations using lambdified functions.

        Parameters
        ---------
        gen_derivs : bool
            generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        '''
        self.logger.info('Evaluate polarizations.')
        self.check_none_vars_net(['deriv_symbs_string', 'deriv_variables'], False, tag='calc_wf_polarizations_derivs_sym')
        if self.check_none_vars_net(['del_hfpc_expr'], True): self.load_wf_polarizations_derivs_sym(gen_derivs=gen_derivs)
        self.calc_wf_polarizations()
        self.del_hfpc = { deriv : self.del_hfpc_expr[deriv](self.f, **utils.get_sub_dict(self.inj_params, self.del_hfpc_expr['variables']))
                          for deriv in self.del_hfpc_expr if deriv not in ('variables','deriv_variables')}
        self.del_hfpc, c_quants = edh.get_conv_del_eval_dic(self.del_hfpc, self.inj_params, self.conv_cos, self.conv_log)
        self.inj_params, self.deriv_variables = edh.get_conv_inj_params_deriv_variables(c_quants, self.inj_params, self.deriv_variables)
        self.logger.info('Lambdified polarizations evaluated.')


    ###
    #-----Detector responses-----
    def calc_det_responses(self):
        '''
        Calculate detector responses.
        '''
        self.check_none_vars_net(['wf', 'inj_params'], False, tag='calc_det_responses')
        self.check_none_vars_det(['f'], False, tag='calc_det_responses')
        if self.check_none_vars_det(['psd', 'Fp', 'Fc', 'Flp'], True): self.setup_ant_pat_lpf_psds()
        for det in self.detectors:
            det.calc_det_responses(self.wf, self.inj_params)
        self.logger.info('Detector responses calculated.')

    def calc_det_responses_derivs_num(self, step=1e-9, method='central', order=2, d_order_n=1, num_cores=None):
        '''
        Calculate partial derivatives of the detector responses numerically. The function also calculates the detector responses.

        Parameters
        ----------
        step : float, optional
            Step size for numerical differentiation, default is 1e-9.
        method : str, optional
            Method for numerical differentiation, default is 'central'.
        order : int, optional
            Order of the method for numerical differentiation, default is 2.
        d_order_n : int, optional
            Number of derivatives, default is 1.
        num_cores : int, optional
            Number of cores for parallel calculation, default is None.
        '''
        self.logger.info('Calculate numeric derivatives of detector responses.')
        self.check_none_vars_net(['wf', 'inj_params', 'deriv_symbs_string', 'deriv_variables'], False, tag='calc_det_responses_derivs_num')
        self.check_none_vars_det(['f'], False, tag='calc_det_responses_derivs_num')
        if self.check_none_vars_det(['psd', 'Fp', 'Fc', 'Flp'], True): self.setup_ant_pat_lpf_psds()
        def calc_func(det):
            self.logger.info(f'   {det.det_key}')
            det.calc_det_responses_derivs_num(self.inj_params, self.wf, self.deriv_symbs_string, self.deriv_variables, self.conv_cos, self.conv_log,
                                              self.use_rot, step, method, order, d_order_n, self.user_locs, self.ana_deriv_symbs_string)
            return det, self.inj_params, self.deriv_variables
        self.pool_calc_func(calc_func, num_cores)
        self.logger.info('Numeric derivatives of detector responses calculated.')

    def load_det_responses_derivs_sym(self, gen_derivs=None, return_bin=False):
        '''
        Load lambdified partial derivatives of the detector responses.

        Parameters
        ----------
        gen_derivs : bool
            generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        return_bin : bool
            return binary, default is False. Might be needed for parallel calculations.
        '''
        self.check_none_vars_net(['wf', 'deriv_symbs_string', 'deriv_variables'], False, tag='load_det_responses_derivs_sym')
        if gen_derivs is not None and gen_derivs:
            _gen_derivs = {'wf_other_var_dic': self.wf.wf_other_var_dic, 'user_waveform':self.wf.user_waveform, 'pl_cr':False, 'user_locs':self.user_locs}
        else: _gen_derivs = None
        for det in self.detectors:
            det.load_det_responses_derivs_sym(self.wf.wf_model_name, self.deriv_symbs_string, self.use_rot, gen_derivs=_gen_derivs, return_bin=return_bin,
                                              user_lambdified_functions_path=self.user_lambdified_functions_path, logger=self.logger)
        self.logger.info('Lambdified detector responses loaded.')

    def calc_det_responses_derivs_sym(self, gen_derivs=None, num_cores=None):
        '''
        Calculate partial derivatives of the detector responses using lambdified functions.

        Parameters
        ----------
        gen_derivs : bool
            generate derivatives, if True. If False or None the derivatives are loaded from file, default is None.
        num_cores : int
            number of cores for parallel calculation, default is None.
        '''
        self.logger.info('Evaluate lambdified detector responses.')
        self.check_none_vars_net(['wf', 'inj_params', 'deriv_symbs_string', 'deriv_variables'], False, tag='calc_det_responses_derivs_sym')
        self.check_none_vars_det(['f'], False, tag='calc_det_responses_derivs_sym')
        if self.check_none_vars_det(['psd', 'Fp', 'Fc', 'Flp'], True): self.setup_ant_pat_lpf_psds()
        if self.check_none_vars_det(['del_hf_expr'], True): self.load_det_responses_derivs_sym(gen_derivs=gen_derivs)
        def calc_func(det):
            self.logger.info(f'   {det.det_key}')
            det.calc_det_responses_derivs_sym(self.wf, self.inj_params, self.deriv_symbs_string, self.deriv_variables, self.conv_cos, self.conv_log)
            return det, self.inj_params, self.deriv_variables
        self.pool_calc_func(calc_func, num_cores)
        self.logger.info('Lambdified detector responses evaluated.')


    ###
    #-----SNR calculations-----
    def calc_snrs(self, only_net=False, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculate SNRs.

        Parameters
        ----------
        only_net : bool
            calculate only network SNR, default is False.
        '''
        self.check_none_vars_det(['f'], False, tag='calc_snrs')
        if self.check_none_vars_det(['psd'], True): self.setup_ant_pat_lpf_psds()
        if self.check_none_vars_det(['hf'], True): self.calc_det_responses()
        self.snr_sq = 0
        for det in self.detectors:
             self.snr_sq += det.calc_snrs(only_net, f_lo=f_lo, f_hi=f_hi)
        self.snr = np.sqrt(self.snr_sq)
        self.logger.info('SNRs calculated.')

    def calc_snr_sq_integrand(self, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculate the squared SNR integrands.
        '''
        self.check_none_vars_det(['f'], False, tag='calc_snr_sq_integrand')
        if self.check_none_vars_det(['psd'], True): self.setup_ant_pat_lpf_psds()
        if self.check_none_vars_det(['hf'], True): self.calc_det_responses()
        for det in self.detectors:
            det.calc_snr_sq_integrand(f_lo=f_lo, f_hi=f_hi)
        self.logger.info('SNR integrands calculated.')


    ###
    #-----Error calculation and Fisher analysis-----
    def calc_errors(self, cond_sup=None, only_net=False, derivs=None, step=1e-9, method='central', order=2, gen_derivs=None, num_cores=None, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculate Fisher and covariance matrices and errors. If the derivatives are not pre-calculated, the method will calculate them using
        the specified method (derivs, step, method, order, gen_derivs, num_cores).

        Parameters
        ----------
        cond_sup : float
            condition number supremum, under which numpy is used for inversion, default is None (use mpmath for inversions).
        only_net : bool
            calculate only network errors, default is False.
        derivs : str
            type of derivatives, default is None. Valid options are: 'num' (numerical), 'sym' (symbolic).
        step : float
            step size for numerical differentiation, default is None.
        method : str
            method for numerical differentiation, default is None.
        order : int
            order of the method for numerical differentiation, default is None.
        gen_derivs : bool
            generate symbolic derivatives, if True. If False or None the derivatives are loaded from files. The default is None.
        num_cores : int
            number of cores for parallel calculation, default is None.
        '''
        self.check_none_vars_net(['deriv_variables'], False, tag='calc_errors')
        self.check_none_vars_det(['f'], False, tag='calc_errors')
        if self.check_none_vars_det(['psd'], True): self.setup_ant_pat_lpf_psds()
        if self.check_none_vars_det(['del_hf'], True):
            if   derivs == 'num': self.calc_det_responses_derivs_num(step=step, method=method, order=order, d_order_n=1, num_cores=num_cores)
            elif derivs == 'sym': self.calc_det_responses_derivs_sym(gen_derivs=gen_derivs, num_cores=num_cores)
            else: utils.log_msg('calc_errors: Neither detector response derivatives have been pre-calculated, nor was ' +
                                'the differentiation method specified. Calculate the derivatives beforehand ' +
                                'or specify the derivative type (derivs=[num, sym]) and [step, method, order] for ' +
                                'the numerical differentiation.', logger=self.logger, level='ERROR')
        if self.check_none_vars_net(['snr'], True): self.calc_snrs(only_net=only_net, f_lo=f_lo, f_hi=f_hi)
        self.logger.info('Calculate errors (Fisher & cov matrices).')
        #-----calculate the error matrices: Fisher and Cov-----
        self.fisher = 0
        for det in self.detectors:
            self.logger.info(f'   {det.det_key}')
            self.fisher += det.calc_fisher_cov_matrices(only_net, cond_sup, f_lo=f_lo, f_hi=f_hi, logger=self.logger)
        self.cond_num = fat.calc_cond_number(self.fisher, logger=self.logger, tag='network')
        self.cov, self.inv_err = fat.calc_cov_inv_err(self.fisher, self.cond_num, cond_sup=cond_sup, logger=self.logger, tag='network')
        #-----calculate the absolute errors of the various variables-----
        self.errs = fat.get_errs_from_cov(self.cov, self.deriv_variables)
        if not only_net:
            for det in self.detectors:
                det.calc_errs(self.deriv_variables)
        #-----calculate 90%-credible sky area
        if 'ra' in self.deriv_variables and ('cos_dec' in self.deriv_variables or 'dec' in self.deriv_variables):
            if 'cos_dec' in self.deriv_variables: dec_str = 'cos_dec'
            else:                                 dec_str = 'dec'
            ra_id      = self.deriv_variables.index('ra')
            dec_id     = self.deriv_variables.index(dec_str)
            is_cos_dec = (dec_str == 'cos_dec')
            if self.cov is None or self.errs is None: self.logger.warning('calc_errors: tag = network - 90%-credible sky area not calculated due to None-valued cov or errs.')
            else: self.errs['sky_area_90'] = edh.sky_area_90(self.errs['ra'], self.errs[dec_str], self.cov[ra_id,dec_id], self.inj_params['dec'], is_cos_dec)
            if not only_net:
                for det in self.detectors:
                    det.calc_sky_area_90_network(ra_id, dec_id, self.inj_params['dec'], is_cos_dec, dec_str, logger=self.logger)
            self.logger.info('Sky areas calculated.')
        else: self.logger.warning('calc_errors: tag = network - 90%-credible sky area not calculated due to missing RA or DEC (COS_DEC) errors.')
        self.logger.info('Errors calculated.')


    ###
    #-----methods used to compute quantities for applications outside the Network-----
    def comp_peak_freq(self, return_max=True, **kwargs):
        '''
        Get peak frequency of the gravitational-wave strain h = hfp - 1j * hfc for the
        set frequency range set in the Network.

        Parameters
        ----------
        return_max : bool
            Whether to return the peak frequency with the maximum amplitude or all found peak frequencies. Default is True.
        kwargs : dict
            Keyword arguments, check `scipy.signal.find_peaks` for more details.

        Returns
        -------
        f_pm : None or float or np.ndarray
            Peak frequencies [Hz].
        '''
        self.calc_wf_polarizations()
        amp_norm = np.abs(self.hfp - 1j * self.hfc) / self.f**(-7./6)
        peaks    = find_peaks(amp_norm, **kwargs)[0]

        if not len(peaks): return None
        elif return_max:   return self.f[peaks[np.argmax(amp_norm[peaks])]]
        else:              return self.f[peaks]


    ###
    #-----IO methods-----
    def save_network(self,filename_path):
        '''Save the network under the given path using *dill*.'''
        with open(filename_path, "wb") as fi:
            dill.dump(self, fi, recurse=True)
        self.logger.info('Network pickled.')
        return

    def load_network(filename_path):
        '''Loading the network from the given path using *dill*.'''
        with open(filename_path, "rb") as fi:
            net = dill.load(fi)
        net.logger.info('Network loaded.')
        return net

    def print_network(self):
        '''Print network.'''
        sepl='--------------------------------------------------------------------------------------'
        print()
        print(sepl)
        print('Printing network.')
        print(sepl)
        print()
        for key,value in vars(self).items():
            if type(value) == dict:
                print('Key: ',key)
                for kkey in value.keys():
                    print('',kkey)
                    print('',value[kkey])
                print()
            elif value is not None:
                if key == 'wf':
                    print('Key: ',key)
                    for kkey,vvalue in vars(value).items():
                        print('',kkey.ljust(16,' '),'  ',vvalue)
                    print()
                else:
                    print('Key: ',key)
                    print(value)
                    print()
        print(sepl)
        print('Printing network done.')
        print(sepl)
        print()

    def print_detectors(self):
        '''Print detectors within network.'''
        sepl='--------------------------------------------------------------------------------------'
        sepl1='-------------------------------------------'
        print()
        print(sepl)
        print('Printing detectors.')
        print(sepl)
        for det in self.detectors:
            print(sepl1)
            print(det.det_key)
            print(sepl1)
            det.print_detector(0)
        print(sepl)
        print('Printing detectors done.')
        print(sepl)
        print()
