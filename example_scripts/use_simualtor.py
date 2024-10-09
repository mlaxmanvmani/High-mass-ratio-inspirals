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
from gwbench import Network, M_of_Mc_eta, f_isco_Msolar, simulator

############################################################################
### settings
############################################################################

# choose the desired detectors
network_spec        = ['V+_V','CE-40_C','ET_ET1']
# choose the desired waveform
wf_model_name       = 'tf2'
# auxillary waveform variables (not needed for tf2)
wf_other_var_dic    = None
# choose which parameters to take partial derivatives for Fisher analysis
deriv_symbs_string  = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi'
# choose which parameters to convert to cos (None) or log versions
conv_log            = ('Mc','DL')
# choose whether to take Earth's rotation into account
use_rot             = True
# choose whether to perform the Fisher analysis only at the network level
only_net            = True
# choose settings for numeric differentiation
step                = 1e-9
method              = 'central'
order               = 2
df                  = 2.**-4

############################################################################
### prepare the network
############################################################################

# initialize the network with the desired detectors
net = Network(network_spec, logger_name='3_40km_CE', logger_level='INFO')
# pass all variables to the network
net.set_net_vars(wf_model_name=wf_model_name, wf_other_var_dic=wf_other_var_dic,
                 deriv_symbs_string=deriv_symbs_string, conv_log=conv_log, use_rot=use_rot)

############################################################################
### injection parameters (which have not been passed to the network)
############################################################################

# set the injection parameters
inj_params = {
    'Mc'    : 42.92123785996995,
    'eta'   : 0.2179560548922493,
    'chi1x' : 0.0,
    'chi1y' : 0.0,
    'chi1z' : -0.5291825652680437,
    'chi2x' : 0.0,
    'chi2y' : 0.0,
    'chi2z' : -0.5533273077110299,
    'DL'    : 595.845227743203,
    'tc'    : 0.0,
    'phic'  : 0.0,
    'iota'  : 0.2367522187410996,
    'ra'    : 4.043179805492967,
    'dec'   : 1.1829476432948405,
    'psi'   : 3.277322349164817,
    }

# pick the desired frequency range
f_lo = 1.
f_hi = f_isco_Msolar(M_of_Mc_eta(inj_params['Mc'],inj_params['eta']))
df   = 2.**-4
f    = np.arange(f_lo,f_hi+df,df)

############################################################################
### simulate the waveform polarizations and detector responses
############################################################################

# simulate the waveform polarizations
hfp, hfc = simulator.calc_wf_polarizations(net, f, inj_params)

# simulate the detector responses directly for f (trunc_f = False)
# the responses are return as a list and DO NOT take the detector's PSD into account
trunc_f  = False
hfs      = simulator.calc_det_responses(net, f, inj_params, trunc_f)

# simulate the detector responses taking the PSDs into account (trunc_f = True)
# the responses are returned as a list and DO take the detector's PSD into account
#
# requires the PSDs to be set up and thus also the frequency array to be set;
# to avoid conflicts with frequency arrays passed for detector response simulation
# a wide range dummy frequency array (with the same df) can be passed
net.set_net_vars(f=np.arange(1., 2048.+df, df))
net.setup_psds()

trunc_f  = True
hfs_trun = simulator.calc_det_responses(net, f, inj_params, trunc_f)

print(f'Shape of frequency array f      = {f.shape}')
print(f'Shape of hfp, hfc               = {hfp.shape}, {hfc.shape}')
print(f'Shape of hfs without truncation = {[hf.shape for hf in hfs]}')
print(f'Shape of hfs with    truncation = {[hf.shape for hf in hfs_trun]}')
