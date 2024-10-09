from copy import deepcopy

import dill
import numpy as np

import gwbench.detector as dc
import gwbench.detector_response_derivatives as drd
import gwbench.err_deriv_handling as edh
import gwbench.network as nc
import gwbench.utils as utils

###
#-----Dealing with several networks-----
def get_det_responses_psds_from_locs_tecs(net,loc_net,tec_net,F_lo=-np.inf,F_hi=np.inf,sym_derivs=0,keep_variables=None):

    net.inj_params = loc_net.inj_params
    net.deriv_variables = loc_net.deriv_variables
    net.f = loc_net.f

    for i,det in enumerate(net.detectors):
        tec_det = tec_net.get_detector(det.tec+'_loc')

        f_lo = np.maximum(tec_det.f[0], F_lo)
        f_hi = np.minimum(tec_det.f[-1], F_hi)
        ids_det_f = np.logical_and(tec_det.f>=f_lo,tec_det.f<=f_hi)
        ids_net_f = np.logical_and(net.f>=f_lo,net.f<=f_hi)

        det.f = tec_det.f[ids_det_f]
        det.psd = tec_det.psd[ids_det_f]

        loc_det = loc_net.get_detector('tec_'+det.loc)
        det.hf = deepcopy(loc_det.hf[ids_net_f])
        det.del_hf = deepcopy(loc_det.del_hf)
        for deriv in det.del_hf:
            det.del_hf[deriv] = det.del_hf[deriv][ids_net_f]
        if sym_derivs:
            det.del_hf_expr = deepcopy(loc_det.del_hf_expr)

    # only keep derivs defined by the the variables in keep_variables
    if keep_variables is not None:
        keep_derivs = []
        keep_deriv_variables = []

        for keep_variable in keep_variables:
            k = 0
            for i,deriv in enumerate(list(net.detectors[0].del_hf.keys())):
                if keep_variable in deriv:
                    keep_derivs.append(deriv)
                    keep_deriv_variables.append(net.deriv_variables[i])
                    k = 1
            if not k:
                net.logger.warning(f'get_det_responses_psds_from_locs_tecs: {keep_variable} not among the derivatives.')

        net.deriv_variables = keep_deriv_variables
        for det in net.detectors:
            det.del_hf = utils.get_sub_dict(det.del_hf,keep_derivs,keep_in_list=1)

    net.logger.info('Detector responses transferred.')


###
#-----Dealing with several networks-----
def unique_tecs(network_specs, f, F_lo=-np.inf, F_hi=np.inf, user_psds=None, logger_level='WARNING'):
    # initialize empty network
    tec_net = nc.Network(logger_name='unique_tecs', logger_level=logger_level)
    # get the detector keys
    tec_net.det_keys = []

    tec_net.logger.info('Calculate PSDs for unique detector technologies.')

    # find unique technologies
    for network_spec in network_specs:
        network = nc.Network(network_spec, logger_name='unique_tecs', logger_level=logger_level)
        for det in network.detectors:
            tec_net.det_keys.append(det.tec)
    tec_net.det_keys = list(dict.fromkeys(tec_net.det_keys))

    # make them into fake detector keys
    for i,tec in enumerate(tec_net.det_keys):
        tec_net.det_keys[i] = tec+'_loc'
    # initialize fake detectors
    tec_net.detectors = []
    for det_key in tec_net.det_keys:
        tec_net.detectors.append(dc.Detector(det_key))

    # get PSDs
    tec_net.set_net_vars(f=f, user_psds=user_psds)
    tec_net.setup_psds(F_lo, F_hi)

    tec_net.logger.info('PSDs for unique detector technologies calculated.')
    return tec_net


def unique_locs_det_responses(network_specs, f, inj_params, deriv_symbs_string, wf_model_name,
                              wf_other_var_dic=None, conv_cos=None, conv_log=None, use_rot=1,
                              user_waveform=None, user_locs=None, user_lambdified_functions_path=None,
                              ana_deriv_symbs_string=None, cosmo=None, logger_level='WARNING', num_cores=None,
                              step=None, method=None, order=None, d_order_n=None):
    # initialize empty network
    loc_net = nc.Network(logger_name='unique_locs_det_responses', logger_level=logger_level)
    # get the detector keys
    loc_net.det_keys = []

    # find unique locations
    for network_spec in network_specs:
        network = nc.Network(network_spec, logger_name='unique_locs_det_responses', logger_level=logger_level)
        for det in network.detectors:
            loc_net.det_keys.append(det.loc)
    loc_net.det_keys = list(dict.fromkeys(loc_net.det_keys))

    # make them into fake detectpr keys
    for i,loc in enumerate(loc_net.det_keys):
        loc_net.det_keys[i] = 'tec_'+loc
    # initialize fake detectors
    loc_net.detectors = []
    for det_key in loc_net.det_keys:
        loc_net.detectors.append(dc.Detector(det_key))
    # set all the other necessary variables
    loc_net.set_wf_vars(wf_model_name, wf_other_var_dic=wf_other_var_dic, user_waveform=user_waveform, cosmo=cosmo)
    loc_net.set_net_vars(f=f, inj_params=inj_params, deriv_symbs_string=deriv_symbs_string, conv_cos=conv_cos, conv_log=conv_log,
                         use_rot=use_rot, user_locs=user_locs, user_lambdified_functions_path=user_lambdified_functions_path)

    # setup Fp, Fc, and Flp and calculate the detector responses
    loc_net.setup_ant_pat_lpf()
    loc_net.calc_det_responses()

    if step is None:
        loc_net.logger.info('Loading the lamdified functions.')
        loc_net.load_det_responses_derivs_sym(return_bin = 1)
        loc_net.logger.info('Loading done.')

    loc_net.logger.info('Evaluate lambdified detector responses for unique locations.')
    if num_cores is None:
        if step is None:
            for det in loc_net.detectors:
                det.del_hf, c_quants = eval_loc_sym(det.loc,det.del_hf_expr,deriv_symbs_string,f,inj_params,conv_cos,conv_log,logger=loc_net.logger)
        else:
            for det in loc_net.detectors:
                det.del_hf, c_quants = eval_loc_num(det.loc,loc_net.wf,deriv_symbs_string,f,inj_params,conv_cos,conv_log,use_rot,
                                                    step,method,order,d_order_n,user_locs,ana_deriv_symbs_string,logger=loc_net.logger)
        loc_net.inj_params, loc_net.deriv_variables = edh.get_conv_inj_params_deriv_variables(c_quants, loc_net.inj_params, loc_net.deriv_variables)

    else:
        from multiprocessing import Pool
        pool = Pool(num_cores)
        if step is None:
            arg_tuple_list = [(det.loc,det.del_hf_expr,deriv_symbs_string,f,inj_params,conv_cos,conv_log,loc_net.logger) for det in loc_net.detectors]
            result = pool.starmap_async(eval_loc_sym, arg_tuple_list)
            result.wait()
        else:
            arg_tuple_list = [(det.loc,loc_net.wf,deriv_symbs_string,f,inj_params,conv_cos,conv_log,use_rot,
                               step,method,order,d_order_n,user_locs,ana_deriv_symbs_string,loc_net.logger) for det in loc_net.detectors]
            result = pool.starmap_async(eval_loc_num, arg_tuple_list)
            result.wait()

        for det, (del_hf,c_quants) in zip(loc_net.detectors, result.get()):
            det.del_hf = del_hf

        loc_net.inj_params, loc_net.deriv_variables = edh.get_conv_inj_params_deriv_variables(c_quants, loc_net.inj_params, loc_net.deriv_variables)

    if step is None:
        for det in loc_net.detectors:
            det.del_hf_expr = dill.loads(det.del_hf_expr)

    loc_net.logger.info('Lambdified detector responses for unique locations evaluated.')
    return loc_net


def eval_loc_sym(loc ,del_hf_expr, deriv_symbs_string, f, inj_params, conv_cos, conv_log, logger=None):
    if logger is None: glob_logger.info(f'   {loc}')
    else:              logger.info(f'   {loc}')
    del_hf = {}
    del_hf_expr = dill.loads(del_hf_expr)
    for deriv in del_hf_expr:
        if deriv in ('variables','deriv_variables'): continue
        del_hf[deriv] = del_hf_expr[deriv](f,**utils.get_sub_dict(inj_params,del_hf_expr['variables']))
    return edh.get_conv_del_eval_dic(del_hf,inj_params,conv_cos,conv_log)


def eval_loc_num(loc, wf, deriv_symbs_string, f, inj_params, conv_cos, conv_log, use_rot,
                 step, method, order, d_order_n, user_locs, ana_deriv_symbs_string, logger=None):
    if logger is None: glob_logger.info(f'   {loc}')
    else:              logger.info(f'   {loc}')
    del_hf = drd.calc_det_responses_derivs_num(loc, wf, deriv_symbs_string, f, inj_params, use_rot=use_rot, label='hf',
                                               step=step, method=method, order=order, d_order_n=d_order_n, user_locs=user_locs,
                                               ana_deriv_symbs_string=ana_deriv_symbs_string)
    return edh.get_conv_del_eval_dic(del_hf,inj_params,conv_cos,conv_log)
