## Examples:

- GW benchmarking example for a single network with numeric or symbolic derivatives:  
`python single_network_gw_bencharking.py`  

- GW benchmarking example for multiple network with numeric or symbolic derivatives:  
`python multi_network_gw_benchmarking.py`  

- GW benchmarking test (against previously calculated values + symbolic vs numeric):
  - BEWARE: differences in CPU architecture or package versions can affect Fisher error estimates at the 1%-level
`python test_run.py`  

- Recommendation: Precalculate the symbolic derivatives if needed. This example shows how to
`python generate_lambdified_functions.py`  

- Basic script to calculate antenna patterns, location phase factors, and PSDs:  
`python compute_ant_pat_lpf_psd.py`  

- Legacy way to perform GW benchmarking for multiple networks (not maintained and might generate errors):
`python legacy_multi_network_example.py`


**Available detector locations:**  
- standard sites:  
'H', 'L', 'V', 'K', 'I', 'LHO', 'LLO', 'LIO'  

- fiducial sites:  
'ET1', 'ET2', 'ET3', 'U', 'A', 'W', 'B', 'C', 'N', 'S', 'ETS1', 'ETS2', 'ETS3', 'CEA', 'CEB', 'CES'  

**Available detector technologies (PSDs):**  
- 2G/2G+/2G#:  
'aLIGO', 'A+', 'V+', 'K+', 'A#'  

- Voyager:  
'Voyager-CBO', 'Voyager-PMO'  

- 3G:  
'ET', 'ET-10-XYL', 'CEwb',  
'CE-40', 'CE-40-LF', 'CE-20', 'CE-20-PM',  
'CE1-10-CBO', 'CE1-20-CBO', 'CE1-30-CBO', 'CE1-40-CBO'  
'CE1-10-PMO', 'CE1-20-PMO', 'CE1-30-PMO', 'CE1-40-PMO'  
'CE2-10-CBO', 'CE2-20-CBO', 'CE2-30-CBO', 'CE2-40-CBO'  
'CE2-10-PMO', 'CE2-20-PMO', 'CE2-30-PMO', 'CE2-40-PMO'  

- LISA:  
'LISA-17', 'LISA-Babak17', 'LISA-Robson18'  
