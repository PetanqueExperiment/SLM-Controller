'''
This file was obtained from http://opticspy.org/
'''

#from __future__ import division as __division__
import numpy as __np__
#import interferometer_zenike as __interferometer__
from numpy import cos as __cos__
from numpy import sin as __sin__
from numpy import sqrt as __sqrt__
from numpy import arctan2 as __arctan2__
import numpy as np
#import matplotlib.pyplot as __plt__
#import matplotlib.pyplot as plt
##from mplot3d import Axes3D as __Axes3D__
#from matplotlib import cm 
#from matplotlib.ticker import LinearLocator as __LinearLocator__
#from matplotlib.ticker import FormatStrFormatter as __FormatStrFormatter__




def __zernikepolynomials__(r,u):
	"""
	------------------------------------------------
	__zernikepolar__(coefficient,r,u):

	Return the aberration polynomials
    
	Zernike Polynomials Caculation in polar coordinates

	coefficient: Zernike Polynomials Coefficient from input
	r: rho in polar coordinates
	u: theta in polar coordinates
     
      WARNING BY SYLVAIN : The polynomials have been reordered to match the 
      ordering of the Shack Hartman we were using...
	------------------------------------------------
	"""
#	Z1  = 1*(__cos__(u)**2+__sin__(u)**2)                                 
	Z2  = 2*r*__cos__(u)
	Z3  = 2*r*__sin__(u)
	Z4  = __sqrt__(3)*(2*r**2-1)
	Z5  = __sqrt__(6)*r**2*__cos__(2*u)
	Z6  = __sqrt__(6)*r**2*__sin__(2*u)
	Z7  = __sqrt__(8)*(3*r**2-2)*r*__cos__(u)
	Z8  = __sqrt__(8)*(3*r**2-2)*r*__sin__(u)
	Z9  = __sqrt__(5)*(1-6*r**2+6*r**4)
	Z10 = __sqrt__(8)*r**3*__cos__(3*u)
	Z11 = __sqrt__(8)*r**3*__sin__(3*u)
	Z12 = __sqrt__(10)*(4*r**2-3)*r**2*__cos__(2*u)
	Z13 = __sqrt__(10)*(4*r**2-3)*r**2*__sin__(2*u)
	Z14 = __sqrt__(12)*(10*r**4-12*r**2+3)*r*__cos__(u)   
	Z15 = __sqrt__(12)*(10*r**4-12*r**2+3)*r*__sin__(u)   
	Z16 = __sqrt__(7)*(20*r**6-30*r**4+12*r**2-1)
	Z17 = __sqrt__(10)*r**4*__cos__(4*u)
	Z18 = __sqrt__(10)*r**4*__sin__(4*u)     
	Z19 = __sqrt__(12)*(5*r**2-4)*r**3*__cos__(3*u)
	Z20 = __sqrt__(12)*(5*r**2-4)*r**3*__sin__(3*u)    
	Z21 = __sqrt__(14)*(15*r**4-20*r**2+6)*r**2*__cos__(2*u)      
	Z22 = __sqrt__(14)*(15*r**4-20*r**2+6)*r**2*__sin__(2*u)
	Z23 = 4*(35*r**6-60*r**4+30*r**2-4)*r*__cos__(u)
	Z24 = 4*(35*r**6-60*r**4+30*r**2-4)*r*__sin__(u)
	Z25 = 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)        
	Z26 = __sqrt__(12)*r**5*__cos__(5*u)           
	Z27 = __sqrt__(12)*r**5*__sin__(5*u)          
	Z28 = __sqrt__(14)*(6*r**2-5)*r**4*__cos__(4*u)
	Z29 = __sqrt__(14)*(6*r**2-5)*r**4*__sin__(4*u)
	Z30 = 4*(21*r**4-30*r**2+10)*r**3*__cos__(3*u)
	Z31 = 4*(21*r**4-30*r**2+10)*r**3*__sin__(3*u)
	Z32 = __sqrt__(18)*(56*r**6-105*r**4+60*r**2-10)*r**2*__cos__(2*u)
	Z33 = __sqrt__(18)*(56*r**6-105*r**4+60*r**2-10)*r**2*__sin__(2*u)
#	Z33 =  Z[33] * 4*(7*r**2-6)*r**5*__sin__(5*u)
#	Z34 =  Z[34] * 4*(7*r**2-6)*r**5*__cos__(5*u)
#	Z35 =  Z[35] * 4*r**7*__sin__(7*u)
#	Z36 =  Z[36] * 4*r**7*__cos__(7*u)
#	Z37 =  Z[37] *  __sqrt__(14)*r**6*__sin__(6*u)
#	Z28 =  Z[28] * __sqrt__(14)*r**6*__cos__(6*u)

	return [Z2,Z3,Z4,Z5,Z6,Z7,Z8,Z9,Z10,Z11,Z12,Z13,Z14,Z15,Z16,Z17,Z18,Z19,Z20,Z21,Z22,Z23,Z24,Z25,Z26,Z27,Z28,Z29,Z30,Z31,Z32,Z33]

def __zernikepolar__(coefficient,r,u):
	"""
	------------------------------------------------
	__zernikepolar__(coefficient,r,u):

	Return combined aberration

	Zernike Polynomials Caculation in polar coordinates

	coefficient: Zernike Polynomials Coefficient from input
	r: rho in polar coordinates
	u: theta in polar coordinates
     
      WARNING BY SYLVAIN : The polynomials have been reordered to match the 
      ordering of the Shack Hartman we were using...
	------------------------------------------------
	"""
	Z = [0] + [0] + coefficient
	Z1  =  Z[1]  * 1*(__cos__(u)**2+__sin__(u)**2)                                 
	Z2  =  Z[2]  * 2*r*__cos__(u)
	Z3  =  Z[3]  * 2*r*__sin__(u)
	Z4  =  Z[4]  * __sqrt__(3)*(2*r**2-1)
	Z5  =  Z[5]  * __sqrt__(6)*r**2*__cos__(2*u)
	Z6  =  Z[6]  * __sqrt__(6)*r**2*__sin__(2*u)
	Z7  =  Z[7]  * __sqrt__(8)*(3*r**2-2)*r*__cos__(u)
	Z8  =  Z[8]  * __sqrt__(8)*(3*r**2-2)*r*__sin__(u)
	Z9  =  Z[9]  * __sqrt__(5)*(1-6*r**2+6*r**4)
	Z10 =  Z[10] * __sqrt__(8)*r**3*__cos__(3*u)
	Z11 =  Z[11] * __sqrt__(8)*r**3*__sin__(3*u)
	Z12 =  Z[12] * __sqrt__(10)*(4*r**2-3)*r**2*__cos__(2*u)
	Z13 =  Z[13] * __sqrt__(10)*(4*r**2-3)*r**2*__sin__(2*u)
	Z14 =  Z[14] * __sqrt__(12)*(10*r**4-12*r**2+3)*r*__cos__(u)   
	Z15 =  Z[15] * __sqrt__(12)*(10*r**4-12*r**2+3)*r*__sin__(u)   
	Z16 =  Z[16] * __sqrt__(7)*(20*r**6-30*r**4+12*r**2-1)
	Z17 =  Z[17] * __sqrt__(10)*r**4*__cos__(4*u)
	Z18 =  Z[18] * __sqrt__(10)*r**4*__sin__(4*u)     
	Z19 =  Z[19] * __sqrt__(12)*(5*r**2-4)*r**3*__cos__(3*u)
	Z20 =  Z[20] * __sqrt__(12)*(5*r**2-4)*r**3*__sin__(3*u)    
	Z21 =  Z[21] * __sqrt__(14)*(15*r**4-20*r**2+6)*r**2*__cos__(2*u)      
	Z22 =  Z[22] * __sqrt__(14)*(15*r**4-20*r**2+6)*r**2*__sin__(2*u)
	Z23 =  Z[23] * 4*(35*r**6-60*r**4+30*r**2-4)*r*__cos__(u)
	Z24 =  Z[24] * 4*(35*r**6-60*r**4+30*r**2-4)*r*__sin__(u)
	Z25 =  Z[25] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)        
	Z26 =  Z[26] * __sqrt__(12)*r**5*__cos__(5*u)           
	Z27 =  Z[27] * __sqrt__(12)*r**5*__sin__(5*u)          
	Z28 =  Z[28] * __sqrt__(14)*(6*r**2-5)*r**4*__cos__(4*u)
	Z29 =  Z[29] * __sqrt__(14)*(6*r**2-5)*r**4*__sin__(4*u)
	Z30 =  Z[30] * 4*(21*r**4-30*r**2+10)*r**3*__cos__(3*u)
	Z31 =  Z[31] * 4*(21*r**4-30*r**2+10)*r**3*__sin__(3*u)
	Z32 =  Z[32] * __sqrt__(18)*(56*r**6-105*r**4+60*r**2-10)*r**2*__cos__(2*u)
	Z33 =  Z[33] * __sqrt__(18)*(56*r**6-105*r**4+60*r**2-10)*r**2*__sin__(2*u)
#	Z33 =  Z[33] * 4*(7*r**2-6)*r**5*__sin__(5*u)
#	Z34 =  Z[34] * 4*(7*r**2-6)*r**5*__cos__(5*u)
#	Z35 =  Z[35] * 4*r**7*__sin__(7*u)
#	Z36 =  Z[36] * 4*r**7*__cos__(7*u)
#	Z37 =  Z[37] *  __sqrt__(14)*r**6*__sin__(6*u)
#	Z28 =  Z[28] * __sqrt__(14)*r**6*__cos__(6*u)


	Z =  Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ \
		Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ \
		Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ \
		Z30+ Z31+ Z32+ Z33#+ Z34+ Z35+ Z36+ Z37
	return Z
