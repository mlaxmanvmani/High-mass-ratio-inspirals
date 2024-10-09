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


'''This module handles calculations for a single gravitational wave detector.
'''

from copy import copy

import numpy as np

import gwbench.antenna_pattern_np as ant_pat_np
import gwbench.detector_response_derivatives as drd
import gwbench.err_deriv_handling as edh
import gwbench.fisher_analysis_tools as fat
import gwbench.psd as psd
import gwbench.snr as snr_mod
import gwbench.utils as utils

class Detector:
    """
    This class handles calculations for a single gravitational wave detector.

    Attributes
    ----------
    det_key : str
        full detector specification, specifying technology and location, e.g. CE2-40-CBO_C
    tec : str
        detector technology
    loc : str
        detector location
    f : np.ndarray
        frequency array
    psd : np.ndarray
        detector PSD
    Fp : np.ndarray
        plus polarization antenna pattern
    Fc : np.ndarray
        cross polarization antenna pattern
    Flp : np.ndarray
        location phase factor
    hf : np.ndarray
        detector response
    del_hf : dict
        derivative dictionary for detector responses
    del_hf_expr : dict
        sympy expression of derivative dictionary for detector responses
    snr : float
        SNR
    snr_sq : float
        SNR^2
    d_snr_sq : np.ndarray
        d(SNR^2) calculated from self.hf
    fisher : np.ndarray
        Fisher matrix
    cond_num : float
        condition number of Fisher matrix
    cov : np.ndarray
        covariance matrix
    inv_err : dict
        dictionary containing information about the inversion error between the two matrices
    errs : dict
        dictionary of errors for given derivative variables

    Methods
    -------
    __init__(det_key)
        Class constructor.
    set_f(f)
        Setter for frequency array.
    setup_psds(f_lo=-np.inf, f_hi=np.inf, user_psds=None)
        Sets up the detector PSD.
    setup_ant_pat_lpf(inj_params, use_rot, user_locs=None)
        Sets up the antenna pattern.
    calc_det_responses(wf, inj_params)
        Calculates the detector response.
    calc_det_responses_derivs_num(inj_params, wf, deriv_symbs_string, deriv_variables, conv_cos, conv_log, use_rot, step, method, order, d_order_n, user_locs, ana_deriv_symbs_string)
        Calculates the derivative dictionary for detector responses using numerical methods.
    load_det_responses_derivs_sym(wf_model_name, deriv_symbs_string, use_rot, gen_derivs=None, return_bin=0, user_lambdified_functions_path=None, logger=None)
        Loads the derivative dictionary for detector responses from a file.
    calc_det_responses_derivs_sym(wf, inj_params, deriv_symbs_string, deriv_variables, conv_cos, conv_log)
        Calculates the derivative dictionary for detector responses using sympy.
    calc_snrs(only_net, f_lo=-np.inf, f_hi=np.inf)
        Calculates the SNR and SNR^2.
    calc_snr_sq_integrand(f_lo=-np.inf, f_hi=np.inf)
        Calculates the integrand for SNR^2.
    calc_fisher_cov_matrices(only_net, cond_sup, f_lo=-np.inf, f_hi=np.inf, logger=None)
        Calculates the Fisher matrix and covariance matrix.
    calc_inv_err()
        Calculates the inversion error between the Fisher matrix and the covariance matrix.
    calc_errs(deriv_variables)
        Calculates the errors for given derivative variables.
    calc_sky_area_90(deriv_variables, logger=None)
        Calculates the 90% credible sky area.
    calc_sky_area_90_network(ra_id, dec_id, dec_val, is_cos_dec, dec_str)
        Calculates the 90% credible sky area for network.
    print_detector(print_format=1)
        Prints the detector.
    """

    ###
    #-----Init methods-----
    def __init__(self, det_key):
        '''
        Class constructor.

        Parameters
        ----------
            det_key : str
                full detector specification, specifying technology and location, e.g. CE2-40-CBO_C
        '''
        #-----detector specification-----
        # full detector specification, specifying technology and location, e.g. CE2-40-CBO_C
        self.det_key = det_key
        # detector technology and locations
        self.tec = det_key.split('_')[0]
        self.loc = det_key.split('_')[1]

        #-----waveform and injection based quantities-----
        # frequency array
        self.f = None

        #-----technology and location based quantities-----
        # detector PSD
        self.psd = None
        # antenna pattern
        self.Fp = None
        self.Fc = None
        # location phase factor
        self.Flp = None

        #-----detector reponses-----
        # detector repsonse
        self.hf = None
        # derivative dictionary for detector responses
        self.del_hf = None
        # sympy expression of derivative dictionary for detector responses
        self.del_hf_expr = None

        #-----SNR-----
        # SNR, SNR^2 and d(SNR^2) calculated from self.hf
        self.snr = None
        self.snr_sq = None
        self.d_snr_sq = None

        #-----errors-----
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


    ###
    #-----Setter methods-----
    def set_f(self, f):
        '''
        Setter for frequency array.

        Parameters
        ----------
        f : np.ndarray
            frequency array
        '''
        self.f = copy(f)


    ###
    #-----PSDs and antenna patterns-----
    def setup_psds(self, f_lo=-np.inf, f_hi=np.inf, user_psds=None):
        '''
        Sets up the detector PSD.

        Parameters
        ----------
        f_lo : float
            lower frequency bound
        f_hi : float
            upper frequency bound
        user_psds : dict
            user-defined PSDs
        '''
        if user_psds is None or (self.det_key not in user_psds and self.tec not in user_psds):
            psd_file = None
            is_asd   = None
        elif self.det_key in user_psds:
            psd_file = user_psds[self.det_key]['psd_file']
            is_asd   = user_psds[self.det_key]['is_asd']
        else:
            psd_file = user_psds[self.tec]['psd_file']
            is_asd   = user_psds[self.tec]['is_asd']
        self.psd, self.f = psd.psd(self.tec,self.f,f_lo,f_hi,psd_file,is_asd)

    def setup_ant_pat_lpf(self, inj_params, use_rot, user_locs=None):
        '''
        Sets up the antenna pattern.

        Parameters
        ----------
        inj_params : dict
            injection parameters
        use_rot : bool
            use rotation
        user_locs : dict
            user-defined locations
        '''
        self.Fp, self.Fc, self.Flp = ant_pat_np.antenna_pattern_and_loc_phase_fac(self.f, inj_params.get('Mc'), inj_params.get('eta'),
            inj_params['tc'], inj_params['ra'], inj_params['dec'], inj_params['psi'], self.loc, use_rot, user_locs=user_locs)


    ###
    #-----Detector responses-----
    def calc_det_responses(self, wf, inj_params):
        '''
        Calculates the detector response.

        Parameters
        ----------
        wf : Waveform
            Waveform class instance
        inj_params : dict
            injection parameters

        Returns
        -------
        hf : np.ndarray
            detector response
        hfp : np.ndarray
            plus polarization of the waveform
        hfc : np.ndarray
            cross polarization of the waveform
        '''
        hfp, hfc = wf.calc_wf_polarizations(self.f, inj_params)
        self.hf = self.Flp * (hfp * self.Fp + hfc * self.Fc)
        return self.hf, hfp, hfc

    def calc_det_responses_derivs_num(self, inj_params, wf, deriv_symbs_string, deriv_variables, conv_cos, conv_log, use_rot,
                                      step, method, order, d_order_n, user_locs, ana_deriv_symbs_string):
        '''
        Calculates the partial derivatives of the detector response using numerical methods. The function also calculates
        the detector response.

        Parameters
        ----------
        inj_params : dict
            injection parameters
        wf : Waveform
            waveform
        deriv_symbs_string : str
            string of derivative symbols
        deriv_variables : list
            list of derivative variables
        conv_cos : bool
            convert to derivatives with respect to cosine of parameters
        conv_log : bool
            convert to derivatives with respect to logarithm of parameters
        use_rot : bool
            incorporate Earth's rotation
        step : float
            The step size for the numerical derivative.
        method : str
            The method for the numerical derivative.
        order : int
            The order of the numerical derivative.
        d_order_n : int
            The derivative order.
        user_locs : dict
            user-defined locations
        ana_deriv_symbs_string : str
            string of analytical derivative symbols
        '''
        _, hfp, hfc = self.calc_det_responses(wf, inj_params)
        self.del_hf = drd.calc_det_responses_derivs_num(self.loc, wf, deriv_symbs_string, self.f, inj_params, use_rot=use_rot, label='hf',
                                                        step=step, method=method, order=order, d_order_n=d_order_n, user_locs=user_locs,
                                                        ana_deriv_symbs_string=ana_deriv_symbs_string,
                                                        ana_deriv_aux={'hf':self.hf, 'hfp':hfp, 'hfc':hfc, 'Flp':self.Flp,
                                                                       'loc':self.loc, 'use_rot':use_rot, 'user_locs':user_locs})
        self.del_hf, c_quants = edh.get_conv_del_eval_dic(self.del_hf, inj_params, conv_cos, conv_log)
        inj_params, deriv_variables = edh.get_conv_inj_params_deriv_variables(c_quants, inj_params, deriv_variables)

    def load_det_responses_derivs_sym(self, wf_model_name, deriv_symbs_string, use_rot, gen_derivs=None, return_bin=0,
                                      user_lambdified_functions_path=None, logger=None):
        '''
        Loads the partial derivatives of the detector response.

        Parameters
        ----------
        wf_model_name : str
            waveform model name
        deriv_symbs_string : str
            string of derivative symbols
        use_rot : bool
            incorporate Earth's rotation
        gen_derivs : bool
            generate derivatives
        return_bin : int
            return binary
        user_lambdified_functions_path : str
            user lambdified functions path
        logger : Logger
            Python.logging logger instance
        '''
        self.del_hf_expr = drd.load_det_responses_derivs_sym(self.loc, wf_model_name, deriv_symbs_string, use_rot,
                                                             gen_derivs=gen_derivs, return_bin=return_bin, logger=logger,
                                                             user_lambdified_functions_path=user_lambdified_functions_path)

    def calc_det_responses_derivs_sym(self, wf, inj_params, deriv_symbs_string, deriv_variables, conv_cos, conv_log):
        '''
        Calculates the partial derivatives of the detector response using sympy. The function also calculates the detector response.

        Parameters
        ----------
        wf : Waveform
            waveform
        inj_params : dict
            injection parameters
        deriv_symbs_string : str
            string of derivative symbols
        deriv_variables : list
            list of derivative variables
        conv_cos : bool
            convert to derivatives with respect to cosine of parameters
        conv_log : bool
            convert to derivatives with respect to logarithm of parameters
        '''
        self.calc_det_responses(wf, inj_params)
        self.del_hf = {}
        for deriv in self.del_hf_expr:
            if deriv in ('variables', 'deriv_variables'): continue
            self.del_hf[deriv] = self.del_hf_expr[deriv](self.f, **utils.get_sub_dict(inj_params, self.del_hf_expr['variables']))

        self.del_hf, c_quants = edh.get_conv_del_eval_dic(self.del_hf, inj_params, conv_cos, conv_log)
        inj_params, deriv_variables = edh.get_conv_inj_params_deriv_variables(c_quants, inj_params, deriv_variables)


    ###
    #-----SNR calculations-----
    def calc_snrs(self, only_net, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculates the SNR and SNR^2.

        Parameters
        ----------
        only_net : bool
            if True only returns the SNR^2 value and does not store the SNR and SNR^2 values of the detector
        f_lo : float
            lower frequency bound, default is -np.inf
        f_hi : float
            upper frequency bound, default is np.inf

        Returns
        -------
        snr_sq : float
            SNR^2
        '''
        mask       = utils.min_max_mask(self.f, f_lo, f_hi)
        snr,snr_sq = snr_mod.snr_snr_square(self.hf[mask], self.psd[mask], self.f[mask])
        if not only_net:
            self.snr = snr
            self.snr_sq = snr_sq
        return snr_sq

    def calc_snr_sq_integrand(self, f_lo=-np.inf, f_hi=np.inf):
        '''
        Calculates the integrand for SNR^2.

        Parameters
        ----------
        f_lo : float
            lower frequency bound, default is -np.inf
        f_hi : float
            upper frequency bound, default is np.inf
        '''
        mask          = utils.min_max_mask(self.f, f_lo, f_hi)
        self.d_snr_sq = snr_mod.snr_square_integrand(self.hf[mask], self.psd[mask])


    ###
    #-----Error calculation and Fisher analysis-----
    def calc_fisher_cov_matrices(self, only_net, cond_sup, f_lo=-np.inf, f_hi=np.inf, logger=None):
        '''
        Calculates the Fisher matrix and covariance matrices.

        Parameters
        ----------
        only_net : bool
            if True only returns the Fisher matrix and does not store the Fisher matrix, covariance matrix, and inversion error of the detector
        cond_sup : float
            condition number support
        f_lo : float
            lower frequency bound, default is -np.inf
        f_hi : float
            upper frequency bound, default is np.inf
        logger : Logger
            Python.logging logger instance

        Returns
        -------
        fisher : np.ndarray
            Fisher matrix
        '''
        mask            = utils.min_max_mask(self.f, f_lo, f_hi)
        del_hf_sub_list = [ el[mask] for el in utils.get_sub_dict(self.del_hf, ['hf'], keep_in_dict=False).values() ]
        if not only_net:
            self.fisher, self.cov, self.cond_num, self.inv_err = fat.calc_fisher_cov_matrices(del_hf_sub_list, self.psd[mask], self.f[mask], only_fisher=0, cond_sup=cond_sup, logger=logger, tag=self.det_key)
            return self.fisher
        else:
            fisher,_,_,_ = fat.calc_fisher_cov_matrices(del_hf_sub_list, self.psd[mask], self.f[mask], only_fisher=1, cond_sup=cond_sup, logger=logger, tag=self.det_key)
            return fisher

    def calc_inv_err(self):
        '''
        Calculates the inversion error between the Fisher matrix and the covariance matrix.
        '''
        self.inv_err = fat.inv_err_from_fisher_cov(self.fisher, self.cov)

    def calc_errs(self, deriv_variables):
        '''
        Calculates the errors for given derivative variables.

        Parameters
        ----------
        deriv_variables : list
            list of derivative variables
        '''
        self.errs = fat.get_errs_from_cov(self.cov, deriv_variables)

    def calc_sky_area_90(self, deriv_variables, logger=None):
        '''
        Calculates the 90% credible sky area.

        Parameters
        ----------
        deriv_variables : list
            list of derivative variables
        logger : Logger
            Python.logging logger instance
        '''
        if self.cov is None or self.errs is None: return
        if 'ra' in deriv_variables and ('cos_dec' in deriv_variables or 'dec' in deriv_variables):
            if 'cos_dec' in deriv_variables: dec_str = 'cos_dec'
            else:                            dec_str = 'dec'
            ra_id      = deriv_variables.index('ra')
            dec_id     = deriv_variables.index(dec_str)
            is_cos_dec = (dec_str == 'cos_dec')
            self.errs['sky_area_90'] = edh.sky_area_90(self.errs['ra'],self.errs[dec_str],self.cov[ra_id,dec_id],self.inj_params['dec'],is_cos_dec)
        else: utils.log_msg(f'calc_sky_area_90: tag = {self.det_key} - Nothing done due to missing of either RA or COS_DEC (DEC) errors.', logger=logger, level='WARNING')

    def calc_sky_area_90_network(self, ra_id, dec_id, dec_val, is_cos_dec, dec_str, logger=None):
        '''
        Calculates the 90% credible sky area when called from an instance of Network (otherwise use calc_sky_area_90).

        Parameters
        ----------
        ra_id : int
            index of RA in the derivative variables
        dec_id : int
            index of DEC in the derivative variables
        dec_val : float
            declination value
        is_cos_dec : bool
            if True the declination is in cosine
        dec_str : str
            declination string
        '''
        if self.cov is None or self.errs is None: logger.warning(f'calc_sky_area_90_network: tag = {self.det_key} - 90%-credible sky area not calculated due to None-valued cov or errs.')
        else: self.errs['sky_area_90'] = edh.sky_area_90(self.errs['ra'], self.errs[dec_str], self.cov[ra_id,dec_id], dec_val, is_cos_dec)


    ###
    #-----IO methods-----
    def print_detector(self,print_format=1):
        '''Prints the detector.'''
        if print_format:
            sepl='-----------------------------------------------------------------------------------'
            print()
            print(sepl)
            print('Printing detector.')
            print(sepl)
            print()
        for key,value in vars(self).items():
            if type(value) == dict:
                print('Key: ',key)
                for key in value.keys():
                    print('',key)
                    print('',value[key])
                print()
            elif value is not None:
                print('Key: ',key)
                print(value)
                print()
        if print_format:
            print(sepl)
            print('Printing detector done.')
            print(sepl)
            print()
