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


"""This script calculates the lamdified derivatives of the waveform polarizations, antenna patterns, and detector responses.

The first lines of output are needed for the main benchmarking methods and should be copied to the main script.

"""

import os

from gwbench import antenna_pattern_np as ant_pat_np
from gwbench import detector_response_derivatives as drd
from gwbench import waveform as wfc

############################################################################
### User Choices
############################################################################

#-----choose waveform model - see below for available waveforms-----
# uncomment chosen waveform model from the list of currently available
if 1:
    wf_model_name = 'tf2'
    #wf_model_name = 'tf2_tidal'

    user_waveform = None

    #-----choose partial derivative variables for the chose waveform model-----
    if wf_model_name == 'tf2':
        wf_other_var_dic = None
        deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota ra dec psi'

    elif wf_model_name == 'tf2_tidal':
        wf_other_var_dic = None
        deriv_symbs_string = 'Mc eta chi1z chi2z DL tc phic iota lam_t ra dec psi'

else:
    wf_model_name    = 'tf2_user'
    wf_other_var_dic = None
    user_waveform   = {'np': '../gwbench/wf_models/tf2_np.py', 'sp':'../gwbench/wf_models/tf2_sp.py'}

#-----choose locations whose detector response derivatives should be calculated-----
# pass a list or None (if all locations, available in gwbench, are supposed to be used)
locs = ['H', 'L', 'V', 'K', 'I', 'ET1', 'ET2', 'ET3', 'C', 'N', 'S', 'user-loc']

#-----choose whether antenna patterns and location phase factors should use frquency dependent gra, thus incorporating the rotation of earth-----
# 1 for True, 0 for False - use 0 for speed or when comparability to code not incorporating the rotation of earth
use_rot = 1

#-----choose where to save the lambidified functions files-----
output_path = None

#-----user defined locations-----
user_locs = {'user-loc':{'longitude': 3.2, 'latitude': 0.4, 'arm_azimuth':0.3, 'which_arm':'y', 'shape':'L'}}

############################################################################
### Calculation of Lambdified Functions
############################################################################

drd.generate_det_responses_derivs_sym(wf_model_name, wf_other_var_dic, deriv_symbs_string, use_rot,
                                      locs=locs, user_locs=user_locs, user_waveform=user_waveform,
                                      user_lambdified_functions_path=output_path)
