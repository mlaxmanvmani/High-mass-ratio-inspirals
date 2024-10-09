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


from numpy import pi, euler_gamma

import gwbench.basic_relations as brs
from gwbench.utils import MTsun, strain_fac

wf_symbs_string = 'f Mc eta chi1z chi2z DL tc phic iota'

def hfpc_raw(f, Mc, eta, chi1z, chi2z, DL, tc, phic, iota, cos, sin, exp, log):
     '''
     Implementation of TaylorF2 waveform model for BBH systems that depend on the following parameters:
     - f: frequency array
     - Mc: chirp mass
     - eta: symmetric mass ratio
     - chi1x, chi1y, chi1z: dimensionless spin components of the primary BH
     - chi2x, chi2y, chi2z: dimensionless spin components of the secondary BH
     - DL: luminosity distance
     - tc: coalescence time
     - phic: coalescence phase
     - iota: inclination angle

     Parameters
     ----------
     f: np.ndarray or jnp.ndarray or sympy.Symbol
          Frequency array
     Mc: float or sympy.Symbol
          Chirp mass [solar mass]
     eta: float or sympy.Symbol
          Symmetric mass ratio
     chi1x: float or sympy.Symbol
          Dimensionless spin component of the primary BH along the x-axis
     chi1y: float or sympy.Symbol
          Dimensionless spin component of the primary BH along the y-axis
     chi1z: float or sympy.Symbol
          Dimensionless spin component of the primary BH along the z-axis
     chi2x : float or sympy.Symbol
          Dimensionless spin component of the secondary BH along the x-axis
     chi2y: float or sympy.Symbol
          Dimensionless spin component of the secondary BH along the y-axis
     chi2z: float or sympy.Symbol
          Dimensionless spin component of the secondary BH along the z-axis
     DL: float or sympy.Symbol
          Luminosity distance [Mpc]
     tc: float or sympy.Symbol
          Coalescence time [s]
     phic: float or sympy.Symbol
          Coalescence phase [rad]
     iota: float or sympy.Symbol
          Inclination angle [rad]
     cos: np.cos or jnp.cos or sympy.cos
          cosine function
     sin: np.sin or jnp.sin or sympy.sin
          sine function
     exp: np.exp or jnp.exp or sympy.exp
          exponential function
     log: np.log or jnp.log or sympy.log
          natural logarithm function

     Returns
     -------
     hfp: np.ndarray or jnp.ndarray or sympy expression
          Plus polarization waveform
     hfc: np.ndarray or jnp.ndarray or sympy expression
          Cross polarization waveform
     '''

     # Mc ... in solar mass
     # DL ... in mega parsec
     # convert to sec
     Mc = Mc * MTsun
     DL = DL * MTsun/strain_fac

     # get sym and asym chi combinations
     chi_s = brs.chi_s(chi1z,chi2z)
     chi_a = brs.chi_a(chi1z,chi2z)

     # Mc is in sec, e.g., Mc = 10*MTSUN_SI (for 10 solar mass)
     # DL is in sec, e.g., DL = 100*1e6*PC_SI/C_SI (for 100 Mpc)
     M = Mc/eta**(3./5.)
     delta = (1.-4.*eta)**0.5
     v  = (pi*M*f)**(1./3.)
     flso = brs.f_isco(M)
     vlso = (pi*M*flso)**(1./3.)
     A =((5./24.)**0.5/pi**(2./3.))*(Mc**(5./6.)/DL)

     # 3.5PN phasing (point particle limit)
     p0 = 1.

     p1 = 0

     p2 = (3715./756. + (55.*eta)/9.)

     p3 = (-16.*pi + (113.*delta*chi_a)/3. + (113./3. - (76.*eta)/3.)*chi_s)

     p4 = (15293365./508032. + (27145.*eta)/504.+ (3085.*eta**2)/72. + (-405./8. + 200.*eta)*chi_a**2 - (405.*delta*chi_a*chi_s)/4. + (-405./8. + (5.*eta)/2.)*chi_s**2)

     gamma = (732985./2268. - 24260.*eta/81. - 340.*eta**2/9.)*chi_s + (732985./2268. + 140.*eta/9.)*delta*chi_a

     p5 = (38645.*pi/756. - 65.*pi*eta/9. - gamma)

     p5L = (38645.*pi/756. - 65.*pi*eta/9. - gamma)*3*log(v/vlso)

     p6 = (11583231236531./4694215680. - 640./3.*pi**2 - 6848./21.*euler_gamma + eta*(-15737765635./3048192. + 2255./12.*pi**2) + eta*eta*76055./1728. - eta*eta*eta*127825./1296. \
          - (6848./21.)*log(4.) + pi*(2270.*delta*chi_a/3. + (2270./3. - 520.*eta)*chi_s) + (75515./144. - 8225.*eta/18.)*delta*chi_a*chi_s \
          + (75515./288. - 263245.*eta/252. - 480.*eta**2)*chi_a**2 + (75515./288. - 232415.*eta/504. + 1255.*eta**2/9.)*chi_s**2)

     p6L = -(6848./21.)*log(v)

     p7 = (((77096675.*pi)/254016. + (378515.*pi*eta)/1512.- (74045.*pi*eta**2)/756. + (-25150083775./3048192. + (10566655595.*eta)/762048. - (1042165.*eta**2)/3024. + (5345.*eta**3)/36.
          + (14585./8. - 7270.*eta + 80.*eta**2)*chi_a**2)*chi_s + (14585./24. - (475.*eta)/6. + (100.*eta**2)/3.)*chi_s**3 + delta*((-25150083775./3048192.
          + (26804935.*eta)/6048. - (1985.*eta**2)/48.)*chi_a + (14585./24. - 2380.*eta)*chi_a**3 + (14585./8. - (215.*eta)/2.)*chi_a*chi_s**2)))

     phase = 2*f*pi*tc - phic - pi/4. + (3./(128.*v**5*eta))*(p0 + v**2*p2 + v**3*p3+ v**4*p4 + v**5*(p5+p5L) + v**6*(p6+p6L) + v**7*p7)

     hpc   = A * f**(-7./6.) * exp(1j * phase)

     return 0.5 * (1 + cos(iota)**2) * hpc, 1j * cos(iota) * hpc
