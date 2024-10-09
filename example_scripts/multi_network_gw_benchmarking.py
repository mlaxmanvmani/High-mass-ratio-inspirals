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


from argparse import ArgumentParser

import numpy as np

from gwbench import MultiNetwork, injections_CBC_params_redshift, M_of_Mc_eta, f_isco_Msolar

np.set_printoptions(linewidth=200)

parser = ArgumentParser()
parser.add_argument('--inj', type = int, help = 'Injection ID in [0,100).', default = 0)
parser.add_argument('--derivs', type = str, help = 'Specify wich differentiation method to use: [num, sym].', default = 'num')

############################################################################
### User Choices
############################################################################

# choose between numeric or symbolic derivatives ['num', 'sym']
derivs = parser.parse_args().derivs

# choose injection id
inj_id = parser.parse_args().inj

# user's choice: waveform to use
wf_model_name = 'tf2'
#wf_model_name = 'tf2_tidal'
#wf_model_name = 'lal_bbh'
#wf_model_name = 'lal_bns'

# the lal_... waveform models are wrappers to call frequency domain waveform from lalsuite
# and thus need further specification of the approximant to use
if   wf_model_name == 'tf2':       wf_other_var_dic = None
elif wf_model_name == 'tf2_tidal': wf_other_var_dic = None
elif wf_model_name == 'lal_bbh':   wf_other_var_dic = {'approximant':'IMRPhenomD'}
elif wf_model_name == 'lal_bns':   wf_other_var_dic = {'approximant':'IMRPhenomD_NRTidalv2'}

# user defined waveform model, defined by the files specified in the dictionary
# only the 'np' one is needed when using numeric derivatives
if 1: user_waveform = None
else:
    wf_model_name    = 'tf2_user'
    wf_other_var_dic = None
    user_waveform   = {'np': '../gwbench/wf_models/tf2_np.py', 'sp':'../gwbench/wf_models/tf2_sp.py'}

# example detector location defined by user
user_locs = {'user-loc':{'longitude': 3.2, 'latitude': 0.4, 'arm_azimuth':0.3, 'which_arm':'y', 'shape':'L'}}
# example detector psd defined by the user
user_psds = {'user-tec':{'psd_file':'../gwbench/noise_curves/ce2_40km_cb.txt', 'is_asd':True}}

# user's choice: with respect to which parameters to take derivatives for the Fisher analysis
if 'tidal' in wf_model_name or 'bns' in wf_model_name: deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota lam_t ra dec psi'
else: deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi'

# user's choice: convert derivatives to cos or log for specific variables
conv_cos = ('dec','iota')
conv_log = ('Mc','DL','lam_t')

# if symbolic derivatives, take from generate_lambdified_functions.py
# if numeric  derivatives, user's decision
use_rot = 1
# 1 for True, 0 for False
# calculate SNRs, error matrices, and errors only for the network
only_net = 1

# number of cores to use for parallelize of the calc_det_responses_derivs
# = None for no parallelization, = 2,3,4,... to allocate N cores (even numbers preferred)
num_cores = None

# options for numeric derivative calculation
if derivs == 'num':
    # user's choice: switch particular partial derivatives to be analytical, options = [DL,tc,phic,ra,dec,psi]
    # otherwise set to None
    ana_deriv_symbs_string = 'DL tc phic ra dec psi'

    # choose numdifftools parameters for numerical derivatives
    step      = 1e-9
    method    = 'central'
    order     = 2

    # only relevant for symbolic derivatives
    gen_derivs = None

# options for symbolic derivative calculation
elif derivs == 'sym':

    # user's choice: switch particular partial derivatives to be analytical, options = [DL,tc,phic,ra,dec,psi]
    # otherwise set to None
    ana_deriv_symbs_string = None

    # choose numdifftools parameters for numerical derivatives
    step      = None
    method    = None
    order     = None

    # tell the code to generate symbolic derivatives as needed (turned on this tutorial)
    # the recommendation is to precompute them externally and load them for large-scale runs
    gen_derivs = True

# user's choice to generate injection parameters
if 'tidal' in wf_model_name or 'bns' in wf_model_name:
    mmin      = 0.8
    mmax      = 3
    chi_lo    = -0.05
    chi_hi    = 0.05
else:
    mmin      = 5
    mmax      = 100
    chi_lo    = -0.75
    chi_hi    = 0.75

cosmo_dict = {'zmin':0, 'zmax':0.2, 'sampler':'uniform_comoving_volume_inversion'}
mass_dict  = {'dist':'uniform', 'mmin':mmin, 'mmax':mmax}
# the default waveforms above are non-precessing, hence dim=1, set dim=3 for precessing waveforms like 'IMRPhenomPv2' or 'IMRPhenomPv2_NRTidalv2'
spin_dict  = {'dim':1, 'geom':'cartesian', 'chi_lo':chi_lo, 'chi_hi':chi_hi}

redshifted = 1
num_injs   = 100
seed       = 29378
file_path  = None

injections_data = injections_CBC_params_redshift(cosmo_dict,mass_dict,spin_dict,redshifted,num_injs,seed,file_path)

############################################################################
### injection parameters
############################################################################

inj_params = {
    'Mc'    : injections_data[0][inj_id],
    'eta'   : injections_data[1][inj_id],
    'chi1x' : injections_data[2][inj_id],
    'chi1y' : injections_data[3][inj_id],
    'chi1z' : injections_data[4][inj_id],
    'chi2x' : injections_data[5][inj_id],
    'chi2y' : injections_data[6][inj_id],
    'chi2z' : injections_data[7][inj_id],
    'DL'    : injections_data[8][inj_id],
    'tc'    : 0.,
    'phic'  : 0.,
    'iota'  : injections_data[9][inj_id],
    'ra'    : injections_data[10][inj_id],
    'dec'   : injections_data[11][inj_id],
    'psi'   : injections_data[12][inj_id],
    'z'     : injections_data[13][inj_id],
    }

if 'tidal' in wf_model_name or 'bns' in wf_model_name:
    inj_params['lam_t']       = 600.
    inj_params['delta_lam_t'] = 0.

print('injections parameter: ', inj_params)
print()

############################################################################
### Network specification
############################################################################

network_specs = [ ['CE2-40-CBO_H','CE2-40-CBO_L','CE2-40-CBO_V']        ,
                  ['CE2-40-CBO_H','CE2-40-CBO_L','CE2-40-CBO_user-loc'] ,
                  ['CE2-40-CBO_H','CE2-40-CBO_V','user-tec_user-loc']   ]
print('network specs: ', network_specs)
print()

f_lo = 1.
f_hi = f_isco_Msolar(M_of_Mc_eta(inj_params['Mc'],inj_params['eta']))
df   = 2.**-4
f    = np.arange(f_lo,f_hi+df,df)

print('f_lo:', f_lo, '   f_hi:', f_hi, '   df:', df)
print()

#############################################################################
### Multi Network GW Benchmarking
#############################################################################

# initialize MultiNetwork and do general setup
net = MultiNetwork(network_specs, logger_level='INFO', logger_level_network='WARNING')
# pass all the needed and optional variables
net.set_net_vars(wf_model_name=wf_model_name, wf_other_var_dic=wf_other_var_dic, user_waveform=user_waveform,
                 f=f, inj_params=inj_params, deriv_symbs_string=deriv_symbs_string,
                 conv_cos=conv_cos, conv_log=conv_log, use_rot=use_rot,
                 user_locs=user_locs, user_psds=user_psds, ana_deriv_symbs_string=ana_deriv_symbs_string)
# start the actual analysis
net.calc_errors(only_net=only_net, derivs=derivs, step=step, method=method, order=order, gen_derivs=gen_derivs, num_cores=num_cores)

### for completeness here are the steps involved, but since gwbench-0.7.4 the code will perform the necessary
### internally as needed
'''
net.setup_ant_pat_lpf_psds()
if derivs == 'num':
    net.calc_det_responses_derivs_num(step=step, method=method, order=order, num_cores=num_cores)
elif derivs = 'sym':
    # generation of lambdified derivatives used to be handles outside and is still recommended for speed reasons
    # refer to example_script/generate_lambdified_functions.py
    net.load_det_responses_derivs_sym(gen_derivs=True)
    net.calc_det_responses_derivs_sym(num_cores=num_cores)
# need to distribute the detector responses and derivatives within the MultiNetwork
net.dist_det_responses()
net.calc_snrs(only_net=only_net)
net.calc_errors(only_net=only_net)
'''

############################################################################
### Print results
############################################################################

print()
for subnet in net.networks:
    print(f'network = {subnet.label}')
    print(f'    snr = {subnet.snr}')
    print(f'   errs = {subnet.errs}')
    print()

print('Fisher analysis done.')
