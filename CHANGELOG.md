# Changelog

## [0.8.4] - 2024-07-04

### Added/Changed

* fixed bug in `injections.py` regarding the `beta_gaussian_uniform` function
* fixed depcreated name `scipy.integrate.simps` to `scipy.integrate.simpson`
* added GPS time to GMST conversion in `basic_relations.gmst_of_gps_time`
* added `TEarthSid` and `TEarthSol` to `utils` for Earth's sidereal and solar days (`TEarth=TEarthSol` as before)

## [0.8.3] - 2024-06-12

### Added/Changed

* **IMPORTANT**: modifcation to `get_errs_from_cov` in **fisher_analysis_tools.py** 
  * previous behavior: calculate the square root of the diagonal elements of the covariance matrix regardless of their sign as the user was expected to perform appropriate checks
  * new behavior: calculate the square root of the diagonal elements of the covariance matrix only if they are positive, otherwise return all errors as `np.nan`
    * this implementation discards a covariance matrix with negative elements on the diagonal as faulty (the user can still access the full covariance matrix via the `cov` attribute)
* **IMPORTANT**: all instances of `F_lo` and `F_hi` in the code have been replaced by `f_lo` and `f_hi` for consistency
* added the option to specify `f_lo` and `f_hi` for snr and errors calculations in **multi_network.py**, **network.py**, and **detector.py**
  * this allows the user to obtain the SNR or perform Fisher analysis only in a specific frequency range without recalculating the detector responses and their derivatives
* added a new function `Network.comp_peak_freq` to the `Network` class in **network.py** to calculate the frequency at the peak of the gravitational-wave strain amplitude (`abs(hfp - 1j*hfc)`)
* added **example_scripts/use_simulator.py** example script to demonstrate the use of the new simulator functions
* cleaned the code in **psd.py**

## [0.8.2] - 2024-06-07

### Added/Changed

* **simulator.py** -- **NEW** module:
  * allows for faster and more efficient waveform generation (e.g. when using `gwbench` as a waveform generator for other codes)
  * 2 functions: `simulator.calc_wf_polarizations` and `simulator.calc_det_responses`
  * to ensure consistency a `Network` object needs to be initialized, set up, and passed
  * the simulator functions then use the `Network` to generate waveforms but
    * they disregard `f` and `inj_params` within the `Network`
    * computed waveform polarizations are not stored in the `Network` object
    * computed detector responses and antenna patterns are not stored in the network's `Detector` objects
    * if the PSDs are setup, `simulator.det_responses` can trauncate the detector responses to the PSD's frequency rangewwwwww
  * not incorporated in the `Network` class due to clear distinction in the philosophy: the Network uses intrinsic variables and functions or those from its `Detector` objects while the simulator functions simply exploit the rigid structure but are tuned to speed instead (particularly visible in `simulator.calc_det_responses`)
* switched from `pathos` to `joblib` for multiprocessing in **network.py**
* broadened the frequency range of the dummy PSD `tec` to be `1e-7` to `1e5` Hz in **psd.py**

## [0.8.1] - 2024-06-04

### Added/Changed

* softened the `jax` dependency to be optional
  * this requires the user to install `jax` and `jaxlib` themselves (e.g. via `pip install jax jaxlib`) as the `pip install gwbench` will not
  * respective imports are now wrapped in `try`-`except`-blocks
  * `jax` remains in the `requirements.txt` files and needs to be removed by the user when not needed and installing from source

## [0.8.0] - 2024-06-03

### Added/Changed

* all changes are internal and did ont affect the usability when used through the `Network` class --> no changes to the **example_scripts**
  * at the user level extra options are available (such as using automatic differentiation via `jax`)
  * various functions in the modules have been renamed, removed, or their arguments modified
    * particularly **network.py**, **multi_network.py**, **detector.py**, **detector_response_derivatives.py**, and **fisher_analysis_tools.py** have only edits to incorporate changes in other modules
* **IMPORTANT: changes to `tc` and `gmst0` handling: `gmst0` is NOT used anymore**
  * previous seperation of these variables was an error and is now corrected
  * instead of separating these variables all occurences of `gmst0` are now replaced appropriately by `tc`
  * the antenna pattern handling tested against `gwfast` code (https://github.com/CosmoStatGW/gwfast)
  * changes were necesssary in **antenna_pattern_{np,sp}.py**, **analytic_derivatives.py**, **basic_relations.py** (see below)
* **IMPORTANT: added automatic differentiation via `jax`**
  * modified **wf_derivatives_num.py** to enable `jax.jacrev` for numerical derivatives besides the existing `numdifftools.Gradient`
  * added **antenna_pattern_jx.py** to handle the differentiation of the antenna pattern functions with `jax`
  * the current implementation works well and gives access to `jax`-based waveform wrappers, but is preliminary and does not exploit the full `jax` capabilites
* **analytic_derivatives.py**:
  * incorporated the new 'tc' dependence of the antenna pattern functions
  * improved the speed of the calculation of the derivatives
* **antenna_pattern_{jx,np,sp}.py**:
  * added the new `tc` dependence to the antenna pattern functions 
  * modularized the main computation of the antenna pattern functions and location phase factor
* **basic_relations.py**:
  * added Keplerian velocity formula
  * added two functions to calculate the frequency-dependent time in the stationary-phase-approximation to account for Earth's rotation.
    1.`tau_spa_PN0` is at 0th post-Newtonian order and was implemented directly in **antenna_pattern_...py** before
    1. `tau_spa_PN35` is at 3.5th post-Newtonian order and the default when incorporating Earth's rotation.
* **snr.py**:
  * changed names to be cleaner and simpler
  * modified scalar_product_integrand to be more efficient
  * removed unnecessary imports
* **utils.py**:
  * constants: added (`time_to_rad_earth`), removed (`halfTEarth`, `time_fac` since MTsun is the same and clearer)
  * added `mod_1` function to have a fast and reliable modulo evaluation for exponential functions avoiding large arguments
  * due to the implementation of `jax`, relevant functions contain a `np` keyword argument to allow switching to `jax.numpy` instead of `numpy`
* **waveform.py**:
  * improvements to the handling of `wf_other_var_dic` in `calc_wf_polarizations` and `calc_wf_polarizations_expr`
  * cleaned up the `select_wf_model_quants` function to initialize the `Waveform` in order to allow for `jax` waveforms
  * removed the unused `cosmo` attribute
  * new waveform models:
    * `tf2_np`, `tf2_sp`, `tf2_jx` for numerical, symbolic, and `jax`-based waveform generation
    * `tf2_tidal_np`, `tf2_tidal_sp`, `tf2_tidal_jx` for numerical, symbolic, and `jax`-based waveform generation
    * `tf2` and `tf2_tidal` still exist and give the same functionality as before (corrsponding to `tf2_sp` and `tf2_tidal_sp`)
  * there is a new attribute `Waveform.deriv_mod` which is either `numdifftools` or `jax` depending on the chosen numerical differentiation method
    * `sympy` waveforms need the `np` version regardless for fast polarizations calculation and hence still allow for numerical differentiation via `numdifftools` (hence `tf2` == `tf2_sp`, ...)
* **wf_models.lal...**:
  * both wrappers have been cleaned up and use `lalsimulation.SimInspiralChooseFDWaveformSequence` for waveform generation to avoid freqency array inconsistencies
* **wf_models.tf2...**:
  * `tf2` models have been restructured: the main difference in the `np`, `sp` and `jax` implementations is the handling of the `cos`, `sin`, `exp`, and `log` functions
  * the new versions are split into one raw/root model (**tf2.py**, **tf2_tidal.py**) with `hfpc_raw` functions and `wf_symbs_string` and three wrappers for numerical, symbolic, and `jax`-based waveform generation
  * the `np` and `jax` wrappers contain the `hfpc` function used for waveform generation by the rest of the code
  * the `sp` version also imports the `hfpc` from the `np` version and adds a `hfpc_sp` wrapper to generate `sympy` expressions
* switched to GNU General Public License v2.0 or later
* added the gwbench logo under **xtra_files/gwbench_logo.png**
* updated README.md

## [0.7.5] - 2024-03-22

### Added/Changed

* added documentation via docstrings to all modules
* replaced instance of np.complex with complex due to deprecation in numpy

## [0.7.5] - 2024-02-16

### Added/Changed

* pre-calculation of lambdified derivatives for symbolic derivatives using sympy is not strictly needed anymore (but for consistency and speed reasons recommended)
* changes in **detector_response_derivatives.py**, **detector.py**, **multi_network.py**, and **network.py** to reflect that
* change for the user: to use the in-built generation of derivatives pass gen_derivs=True in the following (Multi)Network functions:
  * load_wf_polarizations_derivs_sym
  * calc_wf_polarizations_derivs_sym
  * load_det_responses_derivs_sym
  * calc_det_responses_derivs_sym
  * calc_errors
* **example_scripts**
  * combined **num_gw_benchmarking.py** and **sym_gw_benchmarking.py** into one script **single_network_gw_bencharking.py**
  * renamed **multi_network_example.py** to **multi_network_gw_benchmarking.py**
  * removed **quick_start.py** and **quick_start.ipynb**

## [0.7.4] - 2024-02-10

### Added/Changed

* a None-value in use_rot is handled equivalently to a False-value (i.e. do not take Earth's rotation into account)
* Network, MultiNetwork, Detector, Waveform as well as injections and basic_relations functions can be directly imported from gwbench without specifying the respective modules
* clean-up in **detector.py**
* bug-fix in **injections.py** regarding double_gaussian mass_sampler and the theta_vec sampling in beta_gaussian_uniform spin_sampler
  * in both cases vec1 adnd vec2 were permutated the same way and thus only matched parameters from the same sub-population (e.g. from the same Gaussian in the double_gaussian sampler)
* bug-fix in **waveform.py** regarding error messages (did not affect functionality or correctness of the waveform wrapper)
* changes to **network.py** and **multi_network.py**:
  * removed setup_psds, setup_ant_pat_lpf to avoid inconsistencies
  * incorporated set_wf_vars functionality into set_net_vars, set_wf_vars still works
  * changed the (Multi)Network such that called functions explicitly state which necessary variables have not been set yet
  * changed the (Multi)Network such that called functions calculate missing components as needed
  * BEWARE: incorporated calc_sky_area_90 into calc_errors and removed calc_sky_area_90
  * BEWARE: calc_errors is treated specially, since the choice of differentiation scheme is very important
    * if the detector response derivatives are not pre-calculated, they will be calculated according to the passed arguments (new set of keyword arguments)

## [0.7.3] - 2024-01-20

### Added/Changed

* switched to Ver 0.7.3
* added **multi_network.py**:
  * to handle treatment of multiple networks in a single class MultiNetwork
  * used to be handled via routines in **network.py** which have been moved to **legacy.py**
  * the old methods do not handle the analytical derivatives (see below)
  * the Network class and the MultiNetwork class support multiprocessing via `pathos` (simplified and improved over legacy implementation)
* improved the **analytical derivative**s implementation introduced in Ver 0.7.1
  * incorporated analytical derivatives for DL, tc, phic, ra, dec, psi, including effects from Earth's rotation
  * added **analytic_derivatives.py** to handle analytical derivative computations
  * removed **wf_derivatives_ana.py** and moved all functionality to analytic_derivatives.py
  * improvements to the handling of analytical derivatives in **detector.py**, **detector_response_derivatives.py**, **network.py**, and **wf_derivatives_num.py**
* **general changes**:
  * **BEWARE:** flipped the overall phase of the **wf_models/tf2_...py** models to be the same as in the lal waveform wrappers
  * **BEWARE:** renamed the derivative order `n` for numerical derivatives to `d_order_n` for clarity
  * sped up **wf_derivatives_num.py** via improved loop-handling
  * simplified **example_scripts/generate_lambdified_functions.py** by including some of the functionality into **detector_response_derivatives.generate_det_responses_derivs_sym**
  * modularized parts of the code to make it clearer, removed unnecessary variable assignments to improve efficiency in **antenna_pattern_np.py** and **antenna_pattern_sp.py**
  * clean up of **snr.py**
  * added **requirements_pip.txt** for pip installation

## [0.7.1] - 2023-07-21

### Added/Changed

* switched to Ver 0.7.1
* added the ability to choose analytical derivatives for DL, tc, phic (changes in wf_derivatives_num + where needed, added of wf_derivatives_ana)
* improved example_scripts to reflect this possibility
* fixed bug in injections.py
* cleaned up some code

## [0.7.0] - 2023-06-19

### Added/Changed

* switched to Ver 0.7.0
* **noise curves**
  * added 4 most recent, official CE curves from https://dcc.cosmicexplorer.org/CE-T2000017/public
    *  `'CE-40', 'CE-40-LF', 'CE-20', 'CE-20-PM'`
  * added ET 10km xylophone curve from https://apps.et-gw.eu/tds/?content=3&r=18213
    *  `'ET-10-XYL'`
  * added A_sharp curve from https://dcc.ligo.org/LIGO-T2300041-v1/public
    *  `'A#'`
* **general changes and cleanup of code for readability**
  * renamed 2 modules: wf_class.py -> waveform.py and detector_class.py -> detector.py
  * switched from copy to deepcopy in network.py where needed
  * added a logger that prints the previous "verbose" logs at "INFO"-level, but not at "WARNING"-level and above
  * added the available detector technologies and locations to utils.py (and removed them from psd.py and antenna_pattern_np.py, respectively)
  * removed the `df`-option from the functions in snr.py and respective calls
* **antenna_pattern_np.py / antenna_pattern_sp.py**
  * added new detector locations for the MPSAC project: CEA, CEB, CES, ETS, LLO, LHO, LIO
  * using the more precise detector angles for H, L, V, K, I, ET1, ET2, ET3
  * changed the output named `beta` of function `det_angles` in antenna_pattern_np.py and antenna_pattern_sp.py from polar angle to latitude for consistency with the waveform parameter `dec`
  * added function `det_shape` to antenna_pattern_np.py to streamline the definition of which locations are L- or V-shaped
  * removed function `check_loc_gen` from antenna_pattern_np.py and adjusted the only call in network.py
  * removed `ap_symbs_string, det_shape, det_angles` from antenna_pattern_sp.py and import these from antenna_pattern_np.py instead (for code consistency)
* **fisher_analysis_tools.py**
  * added `mpmath` for arbitrary precision matrix inversion of Fisher matrices and removed well-conditioning checks and variables as these are not needed anymore
  * `cond_sup` is now used to set the level up to which `np.linalg.inv` is used (beyond that the code is uses the `mpmath` inversion
    * when set to `None` (*default*) the code will always use `mpmath`
  * removed `np.abs` inside `get_errs_from_cov`
    * if the covariance matrix has negative elements on the diagonal something went bad and this should not be hidden
  * switched `inv_err` into a dictionary containing information about the quality of the inversion of the Fisher matrix
  * removed `by_element` option from inversion error calculation
* **utils.py**
  * united basic_constants.py, basic_functions.py, io_mod.py, and wf_manipulations.py  in utils.py

## [0.65] - 2021-11-05

### Added/Changed

*  fixed a bug in gwbench/detector_response_derivatives.py when attempting to calculate derivatives of wf polarizations wrt ra, dec, psi

## [0.65] - 2021-10-12

### Added/Changed

*  set default step-size for numerical differentiation to 1e-9
*  change the passing of psd_file_dict to allow the keys to be either det_key or psd tags (tec_loc or just tec) in detector_class.py

## [0.65] - 2021-09-21

### Added/Changed

*  updated requirements_conda.txt to use more modern packages
*  added Network to be called directly from gwbench (from gwbench import Network)
*  made detector_class.py less verbose
*  small code clean-ups

## [0.65] - 2021-08-18

### Added/Changed

*  switched to Ver 0.65
*  corrected (original formula was for theta not dec) and improved sky_area_90 calculations in err_deriv_handling.py, network.py, and detector_class.py

## [0.6] - 2021-07-07

### Added/Changed

*  fixed a bug for cos and log conversion of derivatives in detector_class.py
*  fixed mass sampling methods power_peak and power_peak_uniform in injections.py
*  cleaned up numerical detector response differentiation calls in detector_class.py and network.py

## [0.6] - 2021-07-07

### Added/Changed

*  added Planck's constant to basic_constants.py
*  added early warning frequency functions to basic_relations.py
*  added the functionality to specify user definied path for the lambdified sympy functions in detector_response_derivatives.py
*  added new mass samplers to injections.py
*  added new redshift/distance samplers to injections.py (in a previous commit)
*  added the option to limit the frequency range and which variables to use when using get_det_responses_psds_from_locs_tecs in network.py
*  changed function names for better understanding in detector_response_derivatives.py, detector_calls.py, fisher_analysis_tools.py, and network.py
*  changed sampling to maintain ordering when changing the number of injections in injections.py
*  changed example_scripts as necessary
*  renamed detector_responses.py to detector_response_derivatives.py

## [0.6] - 2021-02-15

### Added/Changed

*  attempted fix for segmentation fault: specify specific version of dependencies in requirements_conda.txt

## [0.6] - 2020-10-24

### Added/Changed

*  Initial Release
