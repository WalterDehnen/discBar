#
# file       discBar.py
#
# author     Walter Dehnen
#
# date       2022
#
# copyright  Walter Dehnen (2022)
#
# license    GNU GENERAL PUBLIC LICENSE version 3.0
#            see file LICENSE for details
#
# version 0.0       jun-2022 WD  initial hack as jupyter notebook
# version 0.1.1  29-jun-2022 WD  added axisymmetric model T0
# version 0.1.2  30-jun-2022 WD  changed model name system {k,l} -> Tk,Vk
# version 0.1.3  01-jul-2022 WD  added axisymmetric model V0
# version 0.1.4  06-jul-2022 WD  surf. density for rotated horizontal projection
# version 0.1.5  06-jul-2022 WD  crossedModel (tested & debugged)
# version 0.1.6  07-jul-2022 WD  abandon razor_thin parameter to surface_density
# version 0.1.7  19-jul-2022 WD  k=0 models barred potential; some refactoring
# version 0.1.8  19-jul-2022 WD  k=0 models barred surface density; refactoring
# version 0.1.9  31-jul-2022 WD  some re-naming, minor debugging
# version 0.1.10 26-aug-2022 WD  correcting error for α=-1; refactoring a -> γ
# version 0.2.0  29-aug-2022 WD  licensed (GPLv3) and published at github
# version 0.2.1  05-sep-2022 WD  changed 'rod' --> 'needle'
# version 0.2.2  14-sep-2022 WD  makeBulgeBar(), makeThinBar(), rescale_size()
# version 0.2.3  17-oct-2022 WD  collectionModel.mass()
#

version = '0.2.3'

"""
debugging level, defaults to 0
"""
debug   = 0

import numpy as np
import math
import copy

from scipy import integrate
from inspect import currentframe

# print debug information
def debugInfo(debug_level, *args):
    if debug_level <= debug:
        cf = currentframe()
        print("discBar debug info: discPar.py line " + \
              str(cf.f_back.f_lineno)+": ", args)

# class Convolve
class Convolve:
    """
    implement convolution
        Fn(r,args;L,γ) = M/(2L) int{(1+γ-2γ|x-t|/L) fn(t,args), t=x-l…x+l}
                       = c1 [A(r+LX) - A(r-LX)] + c2 [C(r+LX) + C(r-LX) - 2C[r)]
        with       c1  = M(1-γ)/(2L)
                   c2  = Mγ/L²
    """

    def order(self):
        """
        Return current order n of function fn
        """
        return self.__Fm.order()

    def increment_order(self):
        """
        increment n to n+2
        """
        self.__Fm.increment_order()
        self.__Fp.increment_order()
        if self.__C2 != 0:
            self.__F0.increment_order()

    def increment_order_to(self,n):
        """
        increment n to target, if n < target
        """
        while self.__n < n:
            self.increment_order()        
        
    def __init__(self, funcAC, M,x,L,g, args=()):
        """
        Parameters
        funcAC : class for computing convolution integrals An and Cn
            An = ∫ fn(x,args) dx
            Cn = x A - ∫ fn(x,args) x dx
        M : float
            total mass M
        x : float or array of floats
            coordinate x
        L : float
            parameter L = needle half-length
        g : float
            parameter γ controlling the slope of the needle
        args : list of arguments
            further arguments passed to constructor of funcAC (after x) 
        """
        if L <= 0:
            raise Exception("L="+str(L)+" ≤ 0 is not supported")
        self.__C1 = 0.5*M*(1-g)/L
        self.__C2 = M*g/(L*L)
        self.__Fm =     funcAC(x-L,*args)
        if self.__C2 != 0:
            self.__F0 = funcAC(x  ,*args)
        self.__Fp =     funcAC(x+L,*args)

    def __call__(self):
        """
        Return F(x,args;L,γ)
        """
        result = self.__C1 * (self.__Fp.A() - self.__Fm.A())
        if self.__C2 != 0:
            result += self.__C2 * (self.__Fp.C() + self.__Fm.C() - \
                                   2*self.__F0.C())
        return result

    def dx(self):
        """
        Return ∂F/∂x
        """
        result = self.__C1 * (self.__Fp.dAx() - self.__Fm.dAx())
        if self.__C2 != 0:
            result += self.__C2 * (self.__Fp.dCx() + self.__Fm.dCx() - \
                                   2 * self.__F0.dCx())
        return result
    
    def dy(self):
        """
        Return ∂F/∂y
        """
        result = self.__C1 * (self.__Fp.dAy() - self.__Fm.dAy())
        if self.__C2 != 0:
            result += self.__C2 * (self.__Fp.dCy() + self.__Fm.dCy() - \
                                   2 * self.__F0.dCy())
        return result

# class RFuncAxiN
class RFuncAxiN:
    """
    compute Rn(x,y,z) = M/r^n for n>0 for n increasing in steps of 2
    argument for Convolve to obtain In(x,y,z)
    """
    
    def order(self):
        """
        Return current order n
        """
        return self.__n
    
    def increment_order(self):
        """
        increment n to n+2
        """
        self.__F *= self.__iR2
        self.__n += 2
        
    def __init__(self, M, x,y,z, n):
        """
        Parameters
        M : float
            mass
        x : float or array of floats
            x  coordinate, 
        y : float or array of floats
            y  coordinate, 
        z : float or array of floats
            z  coordinate, 
        n : int
            initial order n > 0
        """
        if n < 1:
            raise Exception("n="+str(n)+" < 1 is not supported")
        self.__iR2 = 1.0/(x*x+y*y+z*z)
        self.__xR2 = x * self.__iR2
        if (n&1) == 1:
            self.__n = 1
            self.__F = M * np.sqrt(self.__iR2)
        else:
            self.__n = 2
            self.__F = M * self.__iR2
        while self.__n < n:
            self.increment_order()

    def F(self):
        return np.copy(self.__F)
            
    def __call__(self):
        """
        Return Rn(x,u)
        """
        return np.copy(self.__F)      # np.copy() is essential here!
    
    def dx(self):
        """
        Return ∂Rn/∂x
        """
        return - self.__n * self.__xR2 * self.__F

# class functionAC
class functionACRn:
    """
    integrals needed for density, potential, and surface density of all models
        An(x,u) = int{1/(t²+u²)^(n/2),t=x} and 
        Bn(x,u) = int{t/(t²+u²)^(n/2),t=x} and
        Cn(x,u) = x An - Bn
    for n increasing in steps of 2
    """

    def order(self):
        """
        Return current order n
        """
        return self.__n

    def increment_order(self):
        """
        increment n to n+2
        """
        #  B[n+2]    =-∂An/∂x / n
        #  A[n+2]    = ((n-1) An + x ∂An/∂x) / nu²
        # ∂A[n+2]/∂x = ∂An/∂x / r²
        self.__B   =-self.__Ax / self.__n
        self.__A  *= self.__n - 1
        self.__A  += self.__x * self.__Ax
        self.__A  *= self.__iU2 / self.__n
        self.__Ax *= self.__iR2
        self.__n  += 2
        
    def __init__(self, x,y,z, n):
        """
        Parameters
        x : float or array of floats
            coordinate x
        y : float or array of floats
            coordinate y
        z : float or array of floats
            coordinate z
        n : int
            initial order n > 0
        """
        if n < 1:
            raise Exception("n="+str(n)+" < 1 is not supported")
        self.__x   = x
        u2         = y*y + z*z
        self.__iU2 = 1.0 / u2
        R2         = x*x + u2
        self.__iR2 = 1.0 / R2
        if (n&1) :
            sR2 = np.sqrt(R2)
            self.__Ax = 1.0 / sR2
            if n==1:
                self.__n  = 1
                self.__A  = np.log(x+sR2)
                self.__B  = sR2
            else:
                self.__n  = 3
                self.__A  = x * self.__Ax * self.__iU2
                self.__B  =-self.__Ax
                self.__Ax*= self.__iR2
        else:
            self.__n  = 2
            self.__Ax = np.copy(self.__iR2)
            iU        = np.sqrt(self.__iU2)
            self.__A  = iU * np.arctan(iU*x)
            if n == 2:
                self.__B  = 0.5 * np.log(R2)
        while self.__n < n:
            self.increment_order()

    def A(self):
        return self.__A

    def dAx(self):
        return self.__Ax

    def C(self):
        return self.__x * self.__A - self.__B

    def dCx(self):
        return self.__A

# function RFuncBarN
def RFuncBarN(M, x,y,z, n, L,g):
    """
    needed for the horizontal projection of k=0 models:
    return object for computing the functions
        L0(x,z;L,γ) = M/(2L) int{(1+γ-2γ|x-t|/L)) l0(t,z), t=x-L…x+L}
    with
            l0(x,z) = - ln(x²+z²)
    For axisymmetric models
        L0(x,z;0,γ) = l0(x,z)

    Parameters
    M : float
        mass
    x : float or array of floats
        coordinate x
    z : float or array of floats
        coordinate z
    n : int
        initial order
    L : float > 0
        parameter L = needle half-length
    g : float
        parameter γ controlling the slope of needle
    """
    debugInfo(2,"RFuncBarN: M="+str(M)+" L="+str(L)+" γ="+str(g)+" n="+str(n))
    return Convolve(functionACRn, M=M, x=x, L=L, g=g, args=[y,z,n])

# class LFuncAxiZ
class LFuncAxiZ:
    """
    compute
        M ln((r1+z1)/(r1+z2))
    which is needed for the potential of axisymmetric k=0 models
    """
    class rho:
        def __init__(self,Rq,z):
            self.r  = np.sqrt(Rq + z*z)
            self.rz = self.r + z
        def dR(self,M):
            return M/(self.rz*self.r)
        
    def __init__(self, M, x,y, z1,z2):
        self.__x  = x
        self.__y  = y
        Rq        = x*x + y*y
        self.__R1 = self.rho(Rq,z1)
        self.__R2 = self.rho(Rq,z2)
        self.__M  = M
        self.__F  = M*np.log(self.__R1.rz/self.__R2.rz)
        self.__D  = None
    def __call__(self):
        return np.copy(self.__F)
    def __setD(self):
        if self.__D is None:
            self.__D = self.__R1.dR(self.__M) - self.__R2.dR(self.__M)
    def dx(self):
        self.__setD()
        return self.__x * self.__D
    def dy(self):
        self.__setD()
        return self.__y * self.__D

# class functionACLZ
class functionACLZ:
    """
    integrals needed for the potential of the barred k=0 models
        A = ∫ ln[(r1+z1)/(r2+z2)] dx
        B = ∫ ln[(r1+z1)/(r2+z2)] x dx
    """
    class rho:
        def __init__(self,x,xq,yq,z):
            self.z  = z                      # z
            self.uq = yq + z*z               # u² = y²+z²
            self.r  = np.sqrt(self.uq + xq)  # r
            self.rx = self.r + x             # r+x
            self.rz = self.r + z             # r+z
        def dAu(self,x,Rq):
            #  ∂(z ln(r+x) + x ln(r+z) - atan(xz/yr) )/∂y
            return (self.z/self.rx + x/self.rz + \
                    x*self.z*(Rq+self.uq)/(Rq*self.uq) ) / self.r
        def dBu(self,Rq):
            return (self.z + Rq/self.rz ) / self.r
    
    def __init__(self,x,y,z1,z2):
        """
        Parameters:
        x  : float or array of floats
            x coordinate
        y  : float or array of floats
            y coordinate
        z1 : float or array of floats
            z coordinate in numerator of argument of logarithm
        z2 : float or array of floats
            z coordinate in denominator of argument of logarithm
        """
        self.__x   = x
        self.__y   = y
        xq = x*x
        yq = y*y
        self.__Rq  = xq + yq
        self.__R1  = self.rho(x,xq,yq,z1)
        self.__R2  = self.rho(x,xq,yq,z2)
        self.__llz = np.log(self.__R1.rz/self.__R2.rz)
        self.__att = np.arctan2(x*y*(self.__R1.r * self.__R2.z - \
                                     self.__R2.r * self.__R1.z), \
                                yq*self.__R1.r * self.__R2.r + \
                                xq*self.__R1.z * self.__R2.z)
    
    def A(self):
        return self.__y * self.__att + \
               self.__R1.z * np.log(self.__R1.rx) - \
               self.__R2.z * np.log(self.__R2.rx) + \
               self.__x * self.__llz
    
    def B(self):
        return 0.5*(self.__Rq  *self.__llz + \
                    self.__R1.z*self.__R1.r - \
                    self.__R2.z*self.__R2.r)
    
    def C(self):
        return self.__x * self.A() - self.B()
    
    def dAx(self):
        return self.__llz
    
    def dBx(self):
        return self.__x * self.__llz
    
    def dCx(self):
        return self.A()
    
    def dAy(self):
        # avoid NaN in case of x=y=0 by setting Rq=1
        Rq  = np.where(self.__Rq == 0.0, 1.0, self.__Rq)
        tmp = self.__R1.dAu(self.__x,Rq) - self.__R2.dAu(self.__x,Rq)
        return self.__att + self.__y * tmp

    def dBy(self):
        tmp = self.__R1.dBu(self.__Rq) - \
              self.__R2.dBu(self.__Rq)
        return self.__y * (self.__llz + 0.5 * tmp)

    def dCy(self):
        return self.__x * self.dAy() - self.dBy()

# function LFuncBarZ
def LFuncBarZ(M, x,y,z1,z2, L,g):
    """
    needed for the horizontal projection of k=0 models:
    return object for computing the functions
        L0(x,z;L,γ) = M/(2L) int{(1+γ-2γ|x-t|/L)) l0(t,z), t=x-L…x+L}
    with
            l0(x,z) = - ln(x²+z²)
    For axisymmetric models
        L0(x,z;0,γ) = l0(x,z)

    Parameters
    M : float
        mass
    x : float or array of floats
        coordinate x
    z1 : float or array of floats
        z coordinate in numerator of argument of logarithm
    z2 : float or array of floats
        z coordinate in denominator of argument of logarithm
    L : float > 0
        parameter L = needle half-length
    g : float
        parameter γ controlling the slope of needle
    """
    return Convolve(functionACLZ, M=M, x=x, L=L, g=g, args=[y,z1,z2])

# class LFuncAxi0
class LFuncAxi0:
    """
    - M * ln(x²+z²)
    """
    def __init__(self,M,x,z):
        self.__F = -M*np.log(x*x+z*z)
    def __call__(self):
        return np.copy(self.__F)

# class functionACL0
class functionACL0:
    """
    integrals needed for the horizontal projection of barred k=0 models
        A = - ∫ ln(x²+z²) dx   = 2 z atan(x/z) + x ln(x²+z²)
        B = - ∫ ln(x²+z²) x dx = s² ln(x²+z²) / 2
    where s²=x²+z²
    """
    def __init__(self,x,z):
        s2       = x*x + z*z
        lg       = np.log(s2)
        self.__a = - 2 * z * np.arctan2(x,z) - x * lg
        b        = - 0.5 * s2 * lg
        self.__c = x * self.__a - b
    def A(self):
        return self.__a
    
    def C(self):
        return self.__c

# function LFuncBar0
def LFuncBar0(M, x,z, L,g):
    """
    needed for the horizontal projection of k=0 models:
    return object for computing the functions
        L0(x,z;L,γ) = M/(2L) int{(1+γ-2γ|x-t|/L)) l0(t,z), t=x-L…x+L}
    with
            l0(x,z) = - ln(x²+z²)
    For axisymmetric models
        L0(x,z;0,γ) = l0(x,z)

    Parameters
    M : float
        mass
    x : float or array of floats
        coordinate x
    z : float or array of floats
        coordinate z
    L : float > 0
        parameter L = needle half-length
    g : float
        parameter γ controlling the slope of needle
    """
    return Convolve(functionACL0, M=M, x=x, L=L,g=g, args=[z])

# class ProjFunc
class ProjFunc:
    """
    for n > 0: B(1/2,n/2) * In(x,0,z,n,args), where B = Beta Function, 
    for n = 0:              L0(x,  z,  args)
    """
    def order(self):
        return 0 if self.__zero else self.__Fn.order()
    
    def increment_order(self):
        if self.__zero:
            self.__zero = False
        else:
            k         = self.__Fn.order()
            self.__B *= k/(k + 1.0)
            self.__Fn.increment_order()
        
    def __init__(self, n, F0, Fn, M, x,z, args=()):
        """
        Parameters:
        n : int
            initial order n > 0
        F0 : class/function
            providing/constructing object for L0(x,z,args)
        Fn : class/function
            providing/constructing object for In(x,y,z,n,args)
        M : float
            mass
        x : float or array of floats
            horizontal coordinate
        z : float or array of floats
            vertical coordinate
        args : additional arguments
            passed to initialisation of F0,Fn       
        """
        self.__zero = (n==0)
        if self.__zero:
            self.__F0 = F0(M, x,      z,    *args)
            self.__Fn = Fn(M, x, 0.0, z, 2, *args)
            self.__B  = 2.0
        else:
            nmin      = 1 if n==1 else 3 if (n&1)==1 else 2
            self.__Fn = Fn(M, x, 0.0, z, nmin, *args)
            if   nmin == 1:
                self.__B = np.pi
            elif nmin == 2:
                self.__B = 2.0
            else:
                self.__B = 0.5 * np.pi
            while self.order() < n:
                self.increment_order()

    def __call__(self):
        if self.__zero:
            return self.__F0()
        else:
            return self.__B * self.__Fn()

# class discbar.model
class model:
    """
    interface (abstract base class) for a disc model with potential, density
    and surface density (projected horizontally or vertically)
    """

    def is_razor_thin(self):
        """
        Return bool: is this model razor thin (density ρ=0 for z≠0)?
        """
        raise NotImplementedError("called for instance of abstract base class")
    
    def model_type(self):
        """
        Return str: type of single-component model or 'compound' for 
                    compound model
        """
        raise NotImplementedError("called for instance of abstract base class")
    
    def is_compound(self):
        """
        Return bool: is this a compound model 
        """
        return len(self.model_type()) > 2
    
    def is_elementary(self):
        """
        Return bool: is this an elementary or  compound model 
        """
        return not self.is_compound()
    
    def has_thick_component(self):
        """
        Return bool: does this model have a component that is not razor thin?
        """
        return not self.is_razor_thin()
    
    def is_barred(self):
        """
        Return bool: is this model barred (not axially symmetric)?
        """
        raise NotImplementedError("called for instance of abstract base class")
    
    def is_axisymmetric(self):
        """
        Return bool: is this model axially symmetric?
        """
        return not self.is_barred()
    
    def mass(self):
        """
        Return float: total mass (may be zero)
        """
        raise NotImplementedError("called for instance of abstract base class")
    
    def rescale_mass(self, factor):
        """
        re-scale the mass normalisation by factor factor
        """
        raise NotImplementedError("called for instance of abstract base class")

    def rescale_height(self, factor):
        """
        re-scale the scale height by factor factor
        """
        raise NotImplementedError("called for instance of abstract base class")

    def rescale_length(self, factor):
        """
        re-scale the scale length by factor factor
        Note: for barred models the ratio a/L is kept fixed
        """
        raise NotImplementedError("called for instance of abstract base class")

    def rescale_size(self, factor):
        """
        re-scale the size by factor factor
        """
        self.rescale_length(factor)
        self.rescale_height(factor)

    def scan_models(self, func):
        """
        call func for all individual models contained
        """
        raise NotImplementedError("called for instance of abstract base class")
    
    def density(self, x, y, z=0):
        """
        compute density

        Parameters:
        x : float or array of floats, must numpy broadcast with y,z
            x coordinate
        y : float or array of floats, must numpy broadcast with x,z
            y coordinate
        z : float or array of floats, must numpy broadcast with x,y
            z coordinate, ignored for razor-thin models

        Returns:
        float or array of floats
            for models with finite thickness: space density
            for razor-thin models: surface density (when z is ignored)       
        """
        raise NotImplementedError("called for instance of abstract base class")

    def surface_density(self, X, Y, proj='z', nz=32, alt=None):
        """
        compute projected (surface-) density

        Parameters:
        X : float or array of floats, must numpy broadcast with Y
            x coordinate on the sky: X=y if proj='x' else x or xsinφ+ycosφ
        Y : float or array of floats, must numpy broadcast with X
            y coordinate on the sky: Y=y if proj='z' else z
        proj : str or float
            if str:   'x','y','z': direction of line-of-sight (=projection)
            if float: |sinφ| in (0,1): horizontal projection along line of
                      sight at angle φ to the x-axis
        nz : int
            Number of points in Gauss-Legendre integration for proj='z' of
            thick models. The relative error is ~ 1e-6 for nz=32 and ~ 1e-8
            for nz=128, depending on the model and position.
            Has no effect for razor-thin models or if proj≠'z'.
            Default is 32
            
        Returns:
        float or array of floats
            surface density after projections along projection axis
        """
        raise NotImplementedError("called for instance of abstract base class")

    def potential(self, x, y, z=0, forces=True, twoDim=False):
        """
        compute gravitational potential Φ and forces f=-∇Φ

        Parameters:
        x : float or array of floats, must numpy broadcast with y,z
            x coordinate
        y : float or array of floats, must numpy broadcast with x,z
            y coordinate
        z : float or array of floats, must numpy broadcast with x,y
            z coordinate, ignored for razor-thin models
        forces : bool
            if True return the forces f=-∇Φ
        twoDim : bool
            if True and forces==True, only return fx,fy

        Returns: Φ, f=-∇Φ
        Φ : float or array of floats
            gravitational potential
        f : tupel of floats or of arrays of floats
            [fx,fy,fz] or [fx,fy]
        """
        raise NotImplementedError("called for instance of abstract base class")

# class crossedModel
class crossedModel(model):
    """
    combination of two identical bars rotated by angles ± φ around z-axis
    """
    def __init__(self,disc,phi):
        """
        Parameters:
        disc : model
            model to rotate, mirror, and add up
        phi : float
            angle by which to rotate input model around z-axis
        """
        # this doesn't work (I have no idea why)
        #if not isinstance(disc,model):
        #    raise Exception("disc must be a discBar.model")
        if disc.is_axisymmetric():
            raise Exception("using an axisymmetric disc is not sensible")
        if phi == 0.0:
            raise Exception("using φ = 0 is not sensible")
        self.__disc   = copy.deepcopy(disc)
        self.__sinPhi = abs(math.sin(phi))
        self.__cosPhi = abs(math.cos(phi))
        debugInfo(3,"crossedModel: φ={:}, sinφ={:}, cosφ={:}".
                  format(phi,self.__sinPhi,self.__cosPhi))

    def model_type(self):
        """
        Return str: type of model
        """
        return self.__disc.model_type()
    
    def is_razor_thin(self):
        """
        Return bool: is this model razor thin (density ρ=0 for z≠0)?
        """
        return self.__disc.is_razor_thin()
    
    def has_thick_component(self):
        """
        Return bool: does this model have a component that is not razor thin?
        """
        return not self.is_razor_thin()
    
    def is_barred(self):
        """
        Return bool: is this model barred (not axially symmetric)?
        """
        return True
    
    def is_axisymmetric(self):
        """
        Return bool: is this model axially symmetric?
        """
        return False
    
    def mass(self):
        """
        Return float: total mass (may be zero)
        """
        return self.__disc.mass()
    
    def rescale_mass(self, factor):
        """
        re-scale the mass normalisation by factor factor
        """
        self.__disc.rescale_mass(factor)

    def rescale_height(self, factor):
        """
        re-scale the scale height by factor factor
        """
        self.__disc.rescale_height(factor)

    def rescale_length(self, factor):
        """
        re-scale the scale length by factor factor
        Note: for barred models the ratio a/L is kept fixed
        """
        self.__disc.rescale_length(factor)

    def scan_models(self, func):
        func(self);
        
    def density(self, x, y, z=0):
        """
        compute density

        Parameters:
        x : float or array of floats, must numpy broadcast with y,z
            x coordinate
        y : float or array of floats, must numpy broadcast with x,z
            y coordinate
        z : float or array of floats, must numpy broadcast with x,y
            z coordinate, ignored for razor-thin models

        Returns:
        float or array of floats
            space density for models with finite thickness
            surface density for razor-thin models (when z is ignored)       
        """
        X,Y  = self.__rotateForward(x,y)
        rho  = self.__disc.density(X,Y,z)
        X,Y  = self.__rotateBackward(x,y)
        rho += self.__disc.density(X,Y,z)
        return 0.5*rho

    def surface_density(self, X, Y, proj='z', nz=32, alt=None):
        """
        compute projected (surface-) density

        Parameters:
        X : float or array of floats, must numpy broadcast together with Y
            x coordinate on the sky: y if proj='x' otherwise x
        Y : float or array of floats, must numpy broadcast together with X
            y coordinate on the sky: y if proj='z' otherwise z
        proj : str or float
            One of 'x','y','z': direction of line-of-sight (=projection)
            if a float: |sinφ| in (0,1): horizontal projection along line of
                        sight at angle φ to the x-axis
        nz : int
            Number of points in Gauss-Legendre integration for proj='z' of
            thick models. The relative error is ~ 1e-6 for nz=32 and ~ 1e-8
            for nz=128, depending on the model and position.
            Has no effect for razor-thin models or if proj≠'z'.
            Default is 32
                        
        Returns:
        float or array of floats
            surface density after projections along projection axis
        """
        if proj=='z':
            x,y  = self.__rotateForward(X,Y)
            SigF = self.__disc.surface_density(x,y,proj='z',nz=nz,alt=alt)
            if debug >= 3:
                debugInfo(3,"X,Y=[{:},{:}] --> [{:},{:}] with Σ={:}".
                          format(X,Y,x,y,SigF))
            x,y  = self.__rotateBackward(X,Y)
            SigB = self.__disc.surface_density(x,y,proj='z',nz=nz,alt=alt)
            if debug >= 3:
                debugInfo(3,"X,Y=[{:},{:}] --> [{:},{:}] with Σ={:}".
                          format(X,Y,x,y,SigB))
            return 0.5*(SigF + SigB)
        if proj=='x':
            return self.__disc.surface_density(X,Y,proj=self.__sinPhi)
        if proj=='y':
            return self.__disc.surface_density(X,Y,proj=self.__cosPhi)
        if type(proj) is float and proj>0 and proj<1:
            sinPsi = proj
            cosPsi = math.sqrt(1-sinPsi*sinPsi)
            sinLeft = abs(sinPsi*self.__cosPhi + cosPsi*self.__sinPhi)
            sinRght = abs(sinPsi*self.__cosPhi - cosPsi*self.__sinPhi)
            Sig = self.__disc.surface_density(X,Y, proj=sinLeft)
            Sig+= self.__disc.surface_density(X,Y, proj=sinRght)
            return 0.5*Sig
        raise Exception("unknown proj='"+proj+"'")

    def potential(self, x, y, z=0, forces=True, twoDim=False):
        """
        compute gravitational potential Φ and forces f=-∇Φ

        Parameters:
        x : float or array of floats, must numpy broadcast with y,z
            x coordinate
        y : float or array of floats, must numpy broadcast with x,z
            y coordinate
        z : float or array of floats, must numpy broadcast with x,y
            z coordinate, ignored for razor-thin models
        forces : bool
            if True return the forces f=-∇Φ
        twoDim : bool
            if True and forces==True, only return fx,fy

        Returns: Φ, f=-∇Φ
        Φ : float or array of floats
            gravitational potential
        f : tupel of floats or of arrays of floats
            [fx,fy,fz] or [fx,fy]
        """
        P = None
        F = None
        X,Y = self.__rotateForward(x,y)
        if not forces:
            P = self.__disc.potential(X,Y,z,forces=False)
        else:
            P,F = self.__disc.potential(X,Y,z,forces=True,twoDim=twoDim)
            if debug >= 3:
                debugInfo(3,"x,y=[{:},{:}] --> [{:},{:}]: Fx,Fy=[{:},{:}]".
                          format(x,y,X,Y,F[0],F[1]))
            F[0],F[1] = self.__rotateBackward(F[0],F[1])
            if debug >= 3:
                debugInfo(3,"Fx,Fy --> [{:},{:}]".format(F[0],F[1]))
        X,Y = self.__rotateBackward(x,y)
        if not forces:
            P += self.__disc.potential(X,Y,z,forces=False)
            P *= 0.5
            return P
        else:
            p,f = self.__disc.potential(X,Y,z,forces=True,twoDim=twoDim)
            P += p
            P *= 0.5
            if debug >= 3:
                debugInfo(3,"x,y=[{:},{:}] --> [{:},{:}]: Fx,Fy=[{:},{:}]".
                          format(x,y,X,Y,f[0],f[1]))
            f[0],f[1] = self.__rotateForward(f[0],f[1])
            if debug >= 3:
                debugInfo(3,"Fx,Fy --> [{:},{:}]".format(f[0],f[1]))
            F[0] += f[0]
            F[0] *= 0.5
            F[1] += f[1]
            F[1] *= 0.5
            if not twoDim:
                F[2] += f[2]
                F[2] *= 0.5
            return P,F
    
    def __rotateForward(self,x,y):
        return self.__cosPhi * x + self.__sinPhi * y, \
               self.__cosPhi * y - self.__sinPhi * x

    def __rotateBackward(self,x,y):
        return self.__cosPhi * x - self.__sinPhi * y, \
               self.__cosPhi * y + self.__sinPhi * x
                
# class collectionModel
class collectionModel(model):
    """
    sum of two or more disc models
    """

    def __init__(self, discA):
        self.__models = []
        self.__positive = []
        self.__thin = True
        self.__barred = False
        self.add(discA)
        
    def is_razor_thin(self):
        return self.__thin
    
    def is_barred(self):
        return self.__barred

    def model_type(self):
        """
        Return str: type of model or compount name for model
        """
        return "compound"
    
    def add(self,mod):
        """
        adds another model, which may be a combined model
        """
        if   isinstance(mod,collectionModel):
            self.__models.extend(mod.__models)
            self.__positive.extend(mod.__positive)
        elif isinstance(mod,model):
            self.__models.append(mod)
            self.__positive.append(True)
        else:
            raise RuntimeError("mod must be a  discBar.model")
        self.__thin   = self.__thin  and mod.is_razor_thin()
        self.__barred = self.__barred or mod.is_barred()
        
    def sub(self,mod):
        """
        subtracts another model, which may be a combined model
        """
        if   isinstance(mod,collectionModel):
            self.__models.extend(mod.__models)
            self.__positive.extend([not p for p in mod.__positive])
        elif isinstance(mod,model):
            self.__models.append(mod)
            self.__positive.append(False)
        else:
            raise RuntimeError("mod must be a discBar.model")
        self.__thin   = self.__thin  and mod.is_razor_thin()
        self.__barred = self.__barred or mod.is_barred()
        
    def __iadd__(self,other):
        """
        adding two disc models
        """
        self.add(other)
        return self
    
    def __add__(self,other):
        """
        adding two disc models
        """
        copied = copy.deepcopy(self)
        copied.add(other)
        return copied
    
    def __radd__(self,other):
        """
        adding two disc models
        """
        copied = copy.deepcopy(self)
        copied.add(other)
        return copied
    
    def __isub__(self,other):
        """
        subtracting a disc model from another
        """
        self.sub(other)
        return self
    
    def __sub__(self,other):
        """
        subtracting a disc model from another
        """
        copied = copy.deepcopy(self)
        copied.sub(other)
        return copied
    
    def __rsub__(self,other):
        """
        subtracting a disc model from another
        """        
        copied = copy.deepcopy(self)
        copied.__positive = [not p for p in copied.__positive]
        copied.add(other)
        return copied

    def mass(self):
        m = 0.0
        for model in self.__models:
            m += model.mass()
        return m
    
    def rescale_mass(self,factor):
        """
        re-scale the masses of all models
        Warning:
            since we don't keep deep copies, this will affect any handles
            of the component models
        """
        if   factor == -1:
            self.__positive = [not p for p in self.__positive]
        elif factor !=  1:
            for model in __models:
                model.rescale_mass(factor)

    def rescale_size(self,factor):
        """
        re-scale the sizes of all models
        Warning:
            since we don't keep deep copies, this will affect any handles
            of the component models
        """
        for model in __models:
            model.rescale_size(factor)

    def rescale_height(self, factor):
        """
        re-scale the scale heights of all models by factor factor
        Warning:
            since we don't keep deep copies, this will affect any handles
            of the component models
        """
        for model in __models:
            model.rescale_height(factor)

    def rescale_length(self, factor):
        """
        re-scale the scale lengths of all models by factor factor
        Note: for barred models the ratio a/L is kept fixed
        Warning:
            since we don't keep deep copies, this will affect any handles
            of the component models
        """
        for model in __models:
            model.rescale_length(factor)

    def __acc(self,result,i,contribution):
        if self.__positive[i]:
            result += contribution
        else:
            result -= contribution

    def __set(self,i,contribution):
        if self.__positive[i]:
            return  contribution
        else:
            return -contribution

    def scan_models(self, func):
        for mod in self.__models:
            func(mod);

    def density(self,x,y,z=0):
        """
        density of combined model
        """
        i = 0
        R = self.__set(i,self.__models[i].density(x,y,z))
        i+= 1
        while i < len(self.__models):
            self.__acc(R,i,self.__models[i].density(x,y,z))
            i+= 1
        return R

    def surface_density(self, X, Y, proj='z', nz=32, alt=None):
        """
        surface density of combined model
        """
        i = 0
        S = self.__set(i,self.__models[i].surface_density(X,Y,proj,nz,alt))
        i+= 1
        while i < len(self.__models):
            self.__acc(S,i,self.__models[i].surface_density(X,Y,proj,nz,alt))
            i+= 1
        return S

    def potential(self,x,y,z=0,forces=True,twoDim=False):
        """
        potential and forces of combined model
        """
        i = 0
        if not forces:
            P = self.__set(i,self.__models[i].potential(x,y,z,forces=False))
            i+= 1
            while i < len(self.__models):
                self.__acc(P,i,self.__models[i].potential(x,y,z,forces=False))
                i+= 1
            return P
        else:
            P,F = self.__models[i].potential(x,y,z,forces,twoDim)
            if not self.__positive[i]:
                P = -P
                F = [-f for f in F]
            i+= 1
            while i < len(self.__models):
                p,f = self.__models[i].potential(x,y,z,forces,twoDim)
                if self.__positive[i]:
                    P   += p;
                    F[0]+= f[0]
                    F[1]+= f[1]
                    if not twoDim:
                        F[2]+= f[2]
                else:
                    P   -= p;
                    F[0]-= f[0]
                    F[1]-= f[1]
                    if not twoDim:
                        F[2]-= f[2]
                i+= 1
            return P,F

# class singleModel
class singleModel(model):
    """
    A single barred disc model of Dehnen & Aly (2022)

    provides methods for computation of
    – density
    – surface density projected vertically or horizontally
    – gravitational potential
    – forces
    
    """
    def is_razor_thin(self):
        """
        Return bool: is this model razor thin?
        """
        return self.__B <= 0
    
    def is_barred(self):
        """
        Return bool: is this model barred (not axially symmetric)?
        """
        return self.__L > 0
    
    def model_type(self):
        """
        Return str: one of T0..T4 or V0..V4
        """
        return self.__T
    
    def needle_length(self):
        """
        Return float: length L of needle used for convolution
        """
        return self.__L
        
    def gamma(self):
        """
        Return float: needle parameter γ
        """
        return self.__G
    
    def scale_length(self):
        """
        Return float: a for the Toomre-Miyamoto-Nagai model
        """
        return self.__A
    
    def scale_radius(self):
        """
        Return float: s=a+b for the Toomre-Miyamoto-Nagai model
        """
        return self.__A + self.__B
    
    def scale_height(self):
        """
        Return float: b = q*s  for the Toomre-Miyamoto-Nagai model
        """
        return self.__B

    def flattening(self):
        """
        Return float: q = b/s  for the Toomre-Miyamoto-Nagai model
        """
        return self.__B / (self.__A+self.__B)
    
    def mass(self):
        """
        Return float: total mass
        """
        return self.__M
    
    def __add__(self,other):
        """
        adding two disc-bar models gives the combined model
        """
        combined = collectionModel(self)
        combined.add(other)
        return combined

    def __sub__(self,other):
        """
        subtracting two disc-bar models gives the difference model
        """
        combined = collectionModel(self)
        combined.sub(other)
        return combined    
    
    def __init__(self, M=1.0, s=1.0, q=0.0, L=0.0, gamma=0.0,\
                 mtype='T1', a=None, b=None):
        """
        don't use this constructor directly, but function makeSingleModel(),
        which allows for the additional parameter phi.
        """
        if np.isinf(q) or np.isnan(q):
            raise Exception("q = {:}".format(q))
        if np.isinf(s) or np.isnan(s):
            raise Exception("s = {:}".format(s))
        if q < 0:
            raise Exception("q = {:} < 0 is not supported".format(q))
        if q > 1:
            raise Exception("q = {:} > 1 is not supported".format(q))
        if not a is None and not b is None:
            a = float(a)
            b = float(b)
            if np.isinf(a) or np.isnan(a):
                raise Exception("a = {:}".format(a))
            if np.isinf(b) or np.isnan(b):
                raise Exception("b = {:}".foramt(b))
            if a < 0:
                raise Exception("a = {:} < 0 is not supported".format(a))
            if b < 0:
                raise Exception("b = {:} < o is not supported".format(b))
            self.__A = a
            self.__B = b
            self.__S = a+b
        else:
            self.__S = s
            self.__B = s * q
            self.__A = s - self.__B
        if self.__S <= 0:
            raise Exception("s = a + b = {:} ≤ 0 is not supported".\
                            format(self.__S))
        self.__hB2 = 0.5 * self.__B * self.__B
        if np.isinf(L) or np.isnan(L):
            raise Exception("L="+str(L))
        if L < 0:
            raise Exception("L = {:} < 0 is not supported".format(self.__L))
        self.__L = L
        if len(mtype) != 2 or 'TV'.find(mtype[0]) < 0 or \
            '01234'.find(mtype[1]) < 0:
            raise Exception("mtype='"+str(mtype)+"' unknown")
        order = mtype[1] if self.__A > 0 else '1'
        if mtype[0] == 'V' and self.__B <= 0:
            self.__T = 'T' + str(order)
        else:
            self.__T = mtype[0] + str(order)
        if M == 0.0:
            raise Exception("M = 0")
        self.__M = M
        self.__G = 0.0
        if self.__L > 0.0:
            if gamma < -1.0:
                raise Exception("γ = {:} < -1 not sensible".format(gamma))
            if gamma >  1.0:
                raise Exception("γ = {:} > 1 not supported".format(gamma))
            self.__G = gamma
        # data of self:
        # __T  string: model type
        # __M  float:  total mass
        # __A  float:  scale length
        # __B  float:  scale height
        # __L  float:  half-length (radius) of the needle
        # __G  float:  parameter γ controlling the slope of needle
                    
    def rescale_mass(self,factor):
        """
        re-scale mass normalisation by factor
        """
        if factor == 0.0:
            raise Exception("factor = 0")
        self.__M *= factor

    def rescale_height(self,factor):
        """
        re-scale scale height b by factor
        """
        if factor == 0.0:
            raise Exception("factor = 0")
        self.__A *= factor
        self.__B *= factor
        self.__L *= factor

    def rescale_length(self,factor):
        """
        re-scale scale length a and needle length L by factor
        """
        if factor == 0.0:
            raise Exception("factor = 0")
        self.__A *= factor
        self.__L *= factor

    def scan_models(self, func):
        func(self);
        
    def density(self,x,y,z=0.0):
        """
        compute density

        Parameters:
        x : float or array of floats
            x coordinate
        y : float or array of floats
            y coordinate
        z : float or array of floats
            z coordinate, ignored for razor-thin models

        Returns: float or array of floats
            space density for models with finite thickness
            surface density for razor-thin models (when z is ignored)       
        """
        if self.__B <= 0:
            return self.__sigma_razor(x,y)
        else:
            return self.__density(self.__rfunc(x,y),z)
        
    def surface_density(self, X, Y, proj='z', nz=32, alt=None):
        """
        compute projected (surface-) density

        Parameters:
        X : float or array of floats
            x coordinate on the sky: y if proj='x' otherwise xsinφ+ycosφ
        Y : float or array of floats
            y coordinate on the sky: y if proj='z' otherwise z
        proj : str or float
            if str:     'x','y','z': direction of line-of-sight (=projection)
            if a float: |sinφ| ∊ [0,1]: horizontal projection along line of
                        sight at angle φ to the x-axis
        nz : int
            Number of points in Gauss-Legendre integration for proj='z' 
            of thick models. The relative error is ~ 1e-4 for nz=32 and
            ~ 1e-8 for nz=128, depending on model type and position.
            Has no effect for razor-thin models or if proj ≠ 'z'.
            Default is 32
        alt : bool or None
            for proj='z' use an alternative numerical treatment
            Default: chose better option, which is alt = model_type() != 'V1'
            
        Returns:
        float or array of floats
            projected (surface-) density
            
        Raises Exception
            if proj is anything else but 'x','y','z', or float ∊ [0,1]
            if proj != 'z' and model is_razor_thin()
        """
        if proj == 'z':
            if self.__B <= 0:
                return self.__sigma_razor(X,Y)
            if alt is None:
                alt = self.__T != 'V1'
            if alt:
                return self.__sigma_thick_alt(X,Y,nz)
            else:
                return self.__sigma_thick(X,Y,nz)
        elif self.__B <= 0:
            raise Exception("horizontal projection for razor-thin model")

        if   proj == 'x' or self.__L == 0:
            proj = 0.0
        elif proj == 'y':
            proj = 1.0
        elif not type(proj) is float and not type(proj) is int:
            raise Exception("proj='" + str(proj) + \
                            "' but must be 'x','y','z', or float ∊ [0,1]")

        if   proj == 0.0:
            return self.__density(self.__pfunc_x(X),Y)
        elif proj == 1.0:
            return self.__density(self.__pfunc_y(X),Y)
        elif proj > 0 and proj < 1:
            keep__L   = self.__L
            self.__L *= proj
            surf = None
            try:
                surf = self.__density(self.__pfunc_y(X),Y)
            except:
                self.__L = keep__L
                raise
            self.__L = keep__L
            return surf
        else:
            raise Exception("proj='" + str(proj) + \
                            "' but must be 'x','y','z', or float ∊ [0,1]")

    def potential(self, x,y,z=0.0, forces=True, twoDim=False):
        """
        compute gravitational potential Φ and forces f=-∇Φ

        Parameters:
        x : float or array of floats, must numpy broadcast with y,z
            x coordinate
        y : float or array of floats, must numpy broadcast with x,z
            y coordinate
        z : float or array of floats, must numpy broadcast with x,y
            z coordinate
        forces : bool
            if True return the forces f=-∇Φ
        twoDim : bool
            if True and forces==True, only return fx,fy

        Returns: Φ, f=-∇Φ
        Φ : float or array of floats
            gravitational potential
        f : tupel of floats or of arrays of floats
            [fx,fy,fz] or [fx,fy]
        """
        ze = np.hypot(z, self.__B)
        Z  = ze + self.__A
        if self.__T[1] == '0':
            fac = 1.0/self.__A
            if not forces:
                return -self.__pot_modK0(self.__lfunc(x,y,fac),\
                                         self.__rfunc(x,y,fac),Z,ze)
            elif twoDim:
                ps,fx,fy = self.__pot_modK0_dxdy(self.__lfunc(x,y,fac),\
                                                 self.__rfunc(x,y,fac),y,Z,ze)
                return -ps,[fx,fy]
            else:
                ps,fx,fy,fz = self.__pot_modK0_dxdydZ(self.__lfunc(x,y,fac),\
                                                      self.__rfunc(x,y,fac),y,Z,ze)
                fz *= z/ze
                return -ps,[fx,fy,fz]
        else:
            if not forces:
                return -self.__pot_modKn(self.__rfunc(x,y),Z,ze)
            elif twoDim:
                ps,fx,fy = self.__pot_modKn_dxdu(self.__rfunc(x,y),Z,ze)
                fy *= y
                return -ps,[fx,fy]
            else:
                ps,fx,fy,fz = self.__pot_modKn_dxdudZ(self.__rfunc(x,y),Z,ze)
                fz += Z * fy
                fz *= z/ze      # ∂z = z/ζ [ (Z/u) ∂u + ∂Z ]
                fy *= y         # ∂y = y  (1/u) ∂u
                return -ps,[fx,fy,fz]

    def __reduced_density(self,func,ze,iz):
        # auxiliary for __density() and  __sigma_thick()
        # compute density times ζ³/b², i.e. w/o the factor b²/ζ³
        # ze=ζ, iz=1/ζ
        # func(n,z): returns an object for computing the replacements of M/r^n
        Z  = ze + self.__A
        Z2 = Z*Z
        # debugInfo(2,"ζ="+str(ze)+" Z="+str(Z))
        if self.__T[1] == '0' :
            # models T0 and V0
            F0 = func(1,ze)               # 1/r0
            Fn = func(1,Z)                # 1/r
            rh = F0() - Fn()
            # debugInfo(4," F0()="+str(F0())+" Fn()="+str(Fn())+" rh="+str(rh))
            F0.increment_order()          # 1/r0^3
            Fn.increment_order()          # 1/r^3
            f0 = F0()
            fn = Fn()
            rh += ze * (ze * f0 - Z * fn)
            # debugInfo(4," F0()="+str(f0)+" Fn()="+str(fn)+" rh="+str(rh))
            if self.__T[0] == 'T':
                return (0.25 / (np.pi*self.__A)) * rh
            ze2 = ze * ze
            rh *= 3 / ze2
            rh += (fn - f0)
            F0.increment_order()          # 1/r0^5
            Fn.increment_order()          # 1/r^5
            rh += 3 * (ze2 * F0() - Z2 * Fn())
            return (0.25 * self.__hB2 / (np.pi*self.__A)) * rh
        if self.__T == 'T1':
            # Kuzmin / Miyamoto-Nagai model = Toomre's model 1
            Fn = func(3,Z)                # 1/r^3
            rh = self.__A * Fn()
            Fn.increment_order()          # 1/r^5
            rh+= 3 * Z2 * ze * Fn()
            return (0.25 / np.pi) * rh
        AZ = self.__A * Z
        A2 = self.__A * self.__A
        if self.__T == 'T2':
            # Toomre's model 2
            Fn = func(5,Z)                # 1/r^5
            rh = (A2 * self.__A + ze*ze*ze) * Fn()
            Fn.increment_order()          # 1/r^7
            rh+= 5 * AZ * Z2 * ze * Fn()
            return (0.75 / np.pi) * rh
        if self.__T == 'T3':
            # Toomre's model 3
            Fn = func(5,Z)                # 1/r^5
            rh = 3 * ze*ze*ze * Fn()
            Fn.increment_order()          # 1/r^7
            rh+= 5 * self.__A * Z2*((3*ze-2*self.__A)*ze+A2) * Fn()
            Fn.increment_order()          # 1/r^9
            rh+= 35 * A2 * Z2*Z2 * ze * Fn()
            return (0.25 / np.pi) * rh
        if self.__T == 'T4':
            # Toomre's model 4
            Fn = func(5,Z)                # 1/r^5
            rh = 3 * Fn()
            Fn.increment_order()          # 1/r^7
            rh+= 15 * AZ * Fn()
            rh*= ze**3
            Fn.increment_order()          # 1/r^9
            rh+= 7 * A2 * Z*Z2 * ((6*ze-3*self.__A)*ze + A2) * Fn()
            Fn.increment_order()          # 1/r^11
            rh+= 63 * A2 * AZ * Z2*Z2 * ze * Fn()
            return (0.25 / np.pi) * rh
        if self.__T == 'V1':
            Fn = func(3,Z)                # 1/r^3
            rh = self.__A * iz * iz * Fn()
            Fn.increment_order()          # 1/r^5
            rh+= 3 * AZ * iz * Fn()
            Fn.increment_order()          # 1/r^7
            rh+= 5 * Z*Z2 * Fn()
            return (0.375 * self.__B*self.__B / np.pi) * rh
        if self.__T == 'V2':
            Fn = func(5,Z)                # 1/r^5
            rh = 3 * self.__A*A2 * iz*iz * Fn()
            Fn.increment_order()          # 1/r^7
            rh+= 5 * Z2 * (ze*ze-2*self.__A*ze+3*A2) * iz * Fn()
            Fn.increment_order()          # 1/r^9
            rh+= 35 * self.__A * Z2*Z2 * Fn()
            return (0.375 * self.__B*self.__B / np.pi) * rh
        if self.__T == 'V3':
            Fn = func(7,Z)                # 1/r^7
            rh = 3 * ze*ze*ze * (1 + (self.__A*iz)**5) * Fn()
            Fn.increment_order()          # 1/r^9
            rh+= 7 * AZ*Z2 * iz * ((3*ze-4*self.__A)*ze+3*A2) * Fn()
            Fn.increment_order()          # 1/r^11
            rh+= 63 * A2 * Z*Z2*Z2 * Fn()
            return (0.625 * self.__B*self.__B / np.pi) * rh
        if self.__T == 'V4':
            Fn = func(7,Z)                # 1/r^7
            rh = 5 * ze*ze*ze * Fn()
            Fn.increment_order()          # 1/r^9
            rh+= 7 * self.__A * Z2 * (5*ze*ze -4*self.__A*ze +3*A2 -\
                                      2*self.__A*A2*iz +A2*A2*iz*iz) * Fn()
            Fn.increment_order()          # 1/r^11
            rh+= 63 * A2 * Z2*Z2 * (ze+ze - 2*self.__A + A2*iz) * Fn()
            Fn.increment_order()          # 1/r^13
            rh+= 231 * A2*AZ*Z2*Z2*Z * Fn()
            return (0.375 * self.__B*self.__B / np.pi) * rh
        raise Exception("mtype '"+self.__T+"' not implemented")
    
    def __density(self,func,z):
        # func(n,z): returns an object for computing the replacements of M/r^n
        ze = np.hypot(z, self.__B)
        iz = 1/ze
        return ((self.__B*self.__B) * iz*iz*iz) * \
            self.__reduced_density(func,ze,iz)
    
    def __rfunc(self,x,y,factor=1):
        # return function returning object providing replacements of
        # M factor/r^n  in density and potential
        if self.__L > 0:
            return lambda n,z: RFuncBarN(M=self.__M*factor, x=x,y=y,z=z, n=n, \
                                         L=self.__L, g=self.__G)
        else:
            return lambda n,z: RFuncAxiN(M=self.__M*factor, x=x,y=y,z=z, n=n)

    def __lfunc(self,x,y,factor=1):
        # return function returning object providing replacements of
        # M factor ln[(r1+z1)/(r2+z2)] in potential
        if self.__L > 0:
            return lambda z1,z2: LFuncBarZ(M=self.__M*factor, x=x,y=y, \
                                           z1=z1, z2=z2, L=self.__L, g=self.__G)
        else:
            return lambda z1,z2: LFuncAxiZ(M=self.__M*factor, x=x,y=y, \
                                           z1=z1, z2=z2)

    def __pfunc_x(self,y):
        # return function returning object providing x-projection of
        # M / r^n  for  n ≥ 1
        return     lambda n,z: ProjFunc(n=n-1, F0=LFuncAxi0, Fn=RFuncAxiN, \
                                        M=self.__M, x=y, z=z)

    def __pfunc_y(self,x):
        # return function returning object providing y-projection of
        # M / r^n  for  n ≥ 1
        if self.__L > 0:
            return lambda n,z: ProjFunc(n=n-1, F0=LFuncBar0, Fn=RFuncBarN, \
                                        M=self.__M, x=x, z=z, \
                                        args=[self.__L, self.__G])
        else:
            return self.__pfunc_x(x)

    def __sigma_razor(self,x,y):
        # z-projected surface density of razor thin model
        s = self.__S
        f = self.__rfunc(x,y)
        if self.__T == 'T0':
            return (0.5 / (s*np.pi)) * (f(1,0.0)() - f(1,s)())
        n = 2 * int(self.__T[1]) - 1
        return (0.5 * n * s**n / np.pi) * f(n+2,s)()
    
    def __integrand_sigma_thick(self,x,y,t):
        zeta = self.__B / np.sqrt(1.0-t*t)
        return self.__reduced_density(self.__rfunc(x,y),zeta,1.0/zeta)
    
    def __sigma_thick(self,x,y,nz):
        # computing z-projected surface density via Gauss-Legendre quadrature
        # with nz points using substitution t=z/ζ when dt/dz=b²/ζ³ and
        # ζ=b/√(1-t²) hence int(ρ, z=-oo...oo) = 2 int(ρ ζ³/b² t=0...1)
        xe = x
        ye = y
        if type(xe) is np.ndarray or type(ye) is np.ndarray:
            xe,ye = np.broadcast_arrays(x,y)
            xe = np.expand_dims(xe,xe.ndim)
            ye = np.expand_dims(ye,ye.ndim)
        func = lambda t : self.__integrand_sigma_thick(xe,ye,t)
        return 2 * integrate.fixed_quad(func,0.0,1.0,n=nz)[0]
    
    def __surf_sigma(self,func,z):
        # compute the σ (see paper) for the surface-density integral
        # func(n,z): returns an object for computing the replacements of M/r^n
        ze = np.hypot(z,self.__B)
        Z  = self.__A + ze
        Z2 = Z*Z
        Fn = func(3,Z)
        sg = -Fn()
        if self.__T == 'T0':
            sg*= Z
            F0 = func(3,ze)
            sg+= ze * F0()
            return sg / self.__A
        if self.__T == 'V0':
            iz = 1/ze
            sg*= Z-self.__hB2*iz
            F0 = func(3,ze)
            sg+= (ze-self.__hB2*iz) * F0()
            F0.increment_order()
            Fn.increment_order()
            sg += 3*self.__hB2 * (ze*F0() - Z2*iz*Fn())
            return sg / self.__A            
        Fn.increment_order()
        if self.__T == 'T1':
            sg += 3 * Z2 * Fn()
            return sg
        if self.__T == 'T2':
            sg += 3 * Z * (Z-3*self.__A) * Fn()
            Fn.increment_order()
            sg += 15 * self.__A * Z2 * Z * Fn()
            return sg
        A2 = self.__A * self.__A
        if self.__T == 'T3':
            sg += 3*(Z2 - 3*self.__A*Z + A2)*Fn()
            Fn.increment_order()
            sg += 15 * self.__A * Z2 * (Z-2*self.__A)*Fn()
            Fn.increment_order()
            sg += 35 * A2 * Z2 * Z2 * Fn()
            return sg
        AZ = self.__A*Z
        if self.__T == 'T4':
            sg += 3 * (Z2-3*AZ+1.2*A2) * Fn()
            Fn.increment_order()
            sg += 3 * AZ*(5*(Z2+A2)-12*AZ) * Fn()
            Fn.increment_order()
            sg += 14 * A2 * Z2 * Z * (3*Z-5*self.__A) * Fn()
            Fn.increment_order()
            sg += 63 * A2*self.__A * Z2*Z2*Z * Fn()
            return sg
        iz = 1/ze
        if self.__T == 'V1':
            sg += (3 * Z2 - 9 * self.__hB2 * Z * iz) * Fn()
            Fn.increment_order()
            sg += 15 * self.__hB2 * Z2 * Z * iz * Fn()
            return sg
        if self.__T == 'V2':
            sg += 3*(Z*(Z-3*self.__A)-3*self.__hB2)*Fn()
            Fn.increment_order()
            sg += 15*Z2*(self.__A*Z+self.__hB2*(Z-6*self.__A)*iz)*Fn()
            Fn.increment_order()
            sg += 105*self.__hB2*self.__A*Z2*Z2*iz*Fn()
            return sg
        if self.__T == 'V3':
            sg += 3*(Z2-3*(AZ+self.__hB2)+A2)*Fn()
            Fn.increment_order()
            sg += 15*Z*(AZ*(Z-2*self.__A)+self.__hB2*(Z-5*self.__A))*Fn()
            Fn.increment_order()
            sg += 35*AZ*Z2*(AZ+self.__hB2*(3*Z-10*self.__A)*iz)*Fn()
            Fn.increment_order()
            sg += 315*self.__hB2*AZ*AZ*Z2*Z*iz*Fn()
            return sg
        if self.__T == 'V4':  
            sg += 3*(Z2-3*(AZ+self.__hB2)+1.2*A2)*Fn()
            Fn.increment_order()
            sg += (3*AZ*(5*(Z2+A2)-12*AZ) + 15*self.__hB2*(Z2-5*AZ+A2)) * Fn()
            Fn.increment_order()
            sg += (14*A2*Z2*Z*(3*Z-5*self.__A) + \
                   105*self.__hB2*self.__A*Z2*(Z-3*self.__A)) * Fn()
            Fn.increment_order()
            sg += (63*A2*self.__A*Z2*Z2*Z + \
                   189*self.__hB2*iz*A2*Z2*Z2*(2*Z-5*self.__A)) * Fn()
            Fn.increment_order()
            sg += 693*self.__hB2*iz*A2*self.__A*Z2*Z2*Z2 * Fn()
            return sg
        raise Exception("mtype '"+self.__T+"' not implemented")
                    
    def integrand_sigma_thick_alt_z(self,x,y,z):
        # for debugging           
        return self.__surf_sigma(self.__rfunc(x,y),z)
        
    def __integrand_sigma_thick_alt(self,x,y,q,t):
        u  = 1 / np.sqrt(1-t*t)
        dz = q * u*u*u
        z  = q*t*u    # note: cannot use *= (it may not broadcast)
        result = dz * self.__surf_sigma(self.__rfunc(x,y),z)
        return result
        
    def __sigma_thick_alt(self,x,y,nz):
        # computing z-projected surface density via Gauss-Legendre quadrature
        # with nz points using a special alternative approach (Appendix C)
        xe = x
        ye = y
        if type(xe) is np.ndarray or type(ye) is np.ndarray:
            xe,ye = np.broadcast_arrays(x,y)
            xe = np.expand_dims(xe,xe.ndim)
            ye = np.expand_dims(ye,ye.ndim)
        qe = np.sqrt(xe*xe + ye*ye + self.__S*self.__S)
        func = lambda t : self.__integrand_sigma_thick_alt(xe,ye,qe,t)
        return (0.5/np.pi) * integrate.fixed_quad(func,0.0,1.0,n=nz)[0]

    def __pot_modK0(self,lfunc,rfunc,Z,zeta):
        Lf  = lfunc(Z,zeta)
        ps  = Lf()
        if self.__T[0] == 'V' :
            F0  = rfunc(1,zeta)
            Fa  = rfunc(1,Z)
            ps  = ps + (self.__hB2/zeta) * (F0() - Fa())
        return ps
                
    def __pot_modK0_dxdy(self,lfunc,rfunc,y,Z,zeta):
        Lf  = lfunc(Z,zeta)
        ps  = Lf()
        px  = Lf.dx()
        py  = Lf.dy()
        if self.__T[0] == 'V' :
            F0  = rfunc(1,zeta)
            Fa  = rfunc(1,Z)
            fc  = self.__hB2/zeta
            ps  = ps + fc * (F0() - Fa())
            px  = px + fc * (F0.dx() - Fa.dx())
            F0.increment_order()
            Fa.increment_order()
            py  = py - y * fc * (F0() - Fa())
        return ps,px,py

    def __pot_modK0_dxdydZ(self,lfunc,rfunc,y,Z,zeta):
        Lf  = lfunc(Z,zeta)                 # Lf = ln(ra+Z)/(r0+ζ)
        ps  = Lf()
        px  = Lf.dx()
        py  = Lf.dy()
        F0  = rfunc(1,zeta)                 # F0 = 1/r0
        Fa  = rfunc(1,Z)                    # Fa = 1/ra
        F   = F0() - Fa()
        pZ  =-F
        if self.__T[0] == 'V' :
            fc  = self.__hB2/zeta
            ps  = ps + fc * F
            px  = px + fc * (F0.dx() - Fa.dx())
            pZ  = pZ - fc * F / zeta
            F0.increment_order()            # F0 = 1/r0^3
            Fa.increment_order()            # Fa = 1/ra^3
            py  = py - y * fc * (F0() - Fa())
            pZ  = pZ - fc * (zeta*F0() - Z*Fa())
        return ps,px,py,pZ
                
    def __pot_modKn(self,func,Z,zeta):
        # gravitational potential w/o forces
        # func(n,z): returns an object for computing the replacements of M/r^n
        Fn = func(1,Z)
        ph = Fn()
        if self.__T == 'T1':
            return ph
        Fn.increment_order()
        aZ = self.__A * Z
        if self.__T == 'T2':
            ph += aZ * Fn()
            return ph
        if self.__T == 'T3':
            ph += self.__A * (Z - self.__A / 3) * Fn()
            Fn.increment_order()
            ph += aZ**2 * Fn()
            return ph
        if self.__T == 'T4':
            ph += self.__A * (Z - 0.4 * self.__A) * Fn()
            Fn.increment_order()
            ph += 0.6 * self.__A * aZ * (Z + Z - self.__A) * Fn()
            Fn.increment_order()
            ph += aZ*aZ*aZ * Fn()
            return ph
        if self.__T == 'V1':
            ph += (self.__hB2 * Z / zeta) * Fn()
            return ph
        if self.__T == 'V2':
            ph += (aZ + self.__hB2) * Fn()
            Fn.increment_order()
            ph += (3 * self.__hB2 * aZ * Z / zeta) * Fn()
            return ph
        if self.__T == 'V3':
            ph += (self.__A * (Z - self.__A / 3) + self.__hB2) * Fn()
            Fn.increment_order()
            ph += aZ * (aZ + 3 * self.__hB2) * Fn()
            Fn.increment_order()
            ph += (5 * self.__hB2 * aZ * aZ * Z / zeta) * Fn()
            return ph
        if self.__T == 'V4':
            ph += (self.__A * (Z - 0.4 * self.__A) + self.__hB2) * Fn()
            Fn.increment_order()
            ph += 3 * self.__A * (0.2*aZ*(Z+Z-self.__A) + \
                                  self.__hB2*(Z-0.2*self.__A)) * Fn()
            Fn.increment_order()
            ph += aZ*aZ* (aZ + 6*self.__hB2) * Fn()
            Fn.increment_order()
            ph += (7*self.__hB2 * aZ * aZ * aZ * Z / zeta) * Fn()
            return ph
        raise Exception("mtype '"+self.__T+"' not implemented")

    def __pot_modKn_dxdu(self,func,Z,zeta):
        # gravitational potential and its ∂x and Du:=(1/u)∂u derivatives
        Fn = func(1,Z)              # 1/r
        ph = Fn()
        px = Fn.dx()
        Fn.increment_order()        # 1/r^3
        F  = Fn()
        pu =-F
        if self.__T == 'T1':
            return ph,px,pu
        aZ = self.__A * Z
        if self.__T == 'T2':
            ph += aZ * F
            px += aZ * Fn.dx()
            Fn.increment_order()    # 1/r^5
            pu -= 3 * aZ * Fn()
            return ph,px,pu
        if self.__T == 'T3':
            w   = self.__A * (Z - self.__A / 3)
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = aZ**2
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            pu -= 5 * w * Fn()
            return ph,px,pu
        if self.__T == 'T4':
            w   = self.__A * (Z - 0.4 * self.__A)
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = 0.6 * self.__A * aZ * (Z + Z - self.__A)
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            w   = aZ**3
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^9
            pu -= 7 * w * Fn()
            return ph,px,pu
        if self.__T == 'V1':
            w   = self.__hB2 * Z / zeta
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            return ph,px,pu
        if self.__T == 'V2':
            w   = aZ + self.__hB2
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = 3 * self.__hB2 * aZ * Z / zeta
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            return ph,px,pu
        if self.__T == 'V3':
            w   = self.__A * (Z - self.__A / 3) + self.__hB2
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = aZ * (aZ + 3 * self.__hB2)
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            w   = 5 * self.__hB2 * aZ * aZ * Z / zeta
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^9
            pu -= 7 * w * Fn()
            return ph,px,pu
        if self.__T == 'V4':
            w   = (self.__A * (Z - 0.4 * self.__A) + self.__hB2)
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = 3 * self.__A * (0.2*aZ*(Z+Z-self.__A) + \
                                  self.__hB2*(Z-0.2*self.__A)) 
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            w   = aZ*aZ* (aZ + 6*self.__hB2)
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^9
            F   = Fn()
            pu -= 7 * w * F
            w   = 7*self.__hB2 * aZ * aZ * aZ * Z / zeta
            ph += w * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^11
            pu -= 9 * w * Fn()
            return ph,px,pu
        raise Exception("mtype '"+self.__T+"' not implemented")
        
    def __pot_modKn_dxdudZ(self,func,Z,zeta):
        # gravitational potential and its ∂x, Du:=(1/u)∂u, and ∂ζ derivatives
        Fn = func(1,Z)              # 1/r
        ph = Fn()
        px = Fn.dx()
        Fn.increment_order()        # 1/r^3
        F  = Fn()
        pu =-F
        if self.__T == 'T1':
            return ph,px,pu,0.0
        aZ = self.__A * Z
        if self.__T == 'T2':
            ph += aZ * F
            pZ  = self.__A * F
            px += aZ * Fn.dx()
            Fn.increment_order()    # 1/r^5
            pu -= 3 * aZ * Fn()
            return ph,px,pu,pZ
        if self.__T == 'T3':
            w   = self.__A * (Z - self.__A / 3)
            ph += w * F
            pZ  = self.__A * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = aZ**2
            ph += w * F
            pZ += 2 * self.__A * aZ * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            pu -= 5 * w * Fn()
            return ph,px,pu,pZ
        if self.__T == 'T4':
            A2  = self.__A * self.__A
            w   = self.__A * (Z - 0.4 * self.__A)
            ph += w * F
            pZ  = self.__A * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = 0.6 * self.__A * aZ * (Z + Z - self.__A)
            ph += w * F
            pZ += 0.6 * A2 * (4 * Z - self.__A) * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            w   = aZ*aZ*aZ
            ph += w * F
            pZ += 3 * self.__A * aZ*aZ * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^9
            pu -= 7 * w * Fn()
            return ph,px,pu,pZ
        if self.__T == 'V1':
            w   = self.__hB2 * Z / zeta
            ph += w * F
            pZ  = w * (1/Z - 1/zeta) * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            return ph,px,pu,pZ
        if self.__T == 'V2':
            w   = aZ + self.__hB2
            ph += w * F
            px += w * Fn.dx()
            pZ  = self.__A * F
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = 3 * self.__hB2 * aZ * Z / zeta
            ph += w * F
            px += w * Fn.dx()
            pZ += w * (2/Z  - 1/zeta) * F
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 5 * w * F
            return ph,px,pu,pZ
        if self.__T == 'V3':
            hB2 = 0.5 * self.__B * self.__B
            w   = self.__A * (Z - self.__A / 3) + hB2
            ph += w * F
            pZ  = self.__A * F
            px += w * Fn.dx()
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = aZ * (aZ + 3 * hB2)
            ph += w * F
            px += w * Fn.dx()
            pZ += self.__A * (2 * aZ + 3 * hB2) * F
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            tm  = 5 * hB2 * aZ * aZ / zeta
            w   = tm * Z
            ph += w * F
            px += w * Fn.dx()
            pZ += tm * (3 - Z/zeta) * F
            Fn.increment_order()    # 1/r^9
            pu -= 7 * w * Fn()
            return ph,px,pu,pZ
        if self.__T == 'V4':
            A2  = self.__A * self.__A
            w   = (self.__A * (Z - 0.4 * self.__A) + self.__hB2)
            ph += w * F
            px += w * Fn.dx()
            pZ  = self.__A * F
            Fn.increment_order()    # 1/r^5
            F   = Fn()
            pu -= 3 * w * F
            w   = 3 * self.__A * (0.2*aZ*(Z+Z-self.__A) + \
                                  self.__hB2*(Z-0.2*self.__A)) 
            ph += w * F
            px += w * Fn.dx()
            pZ += self.__A * (2.4*aZ - 0.6*A2 + 3 * self.__hB2) * F
            Fn.increment_order()    # 1/r^7
            F   = Fn()
            pu -= 5 * w * F
            w   = aZ * aZ * (aZ + 6*self.__hB2)
            ph += w * F
            px += w * Fn.dx()
            pZ += 3 * self.__A * aZ * (aZ + 4*self.__hB2) * F
            Fn.increment_order()    # 1/r^9
            F   = Fn()
            pu -= 7 * w * F
            tm  = 7 * self.__hB2 * aZ * aZ * aZ / zeta
            w   = tm * Z
            ph += w * F
            px += w * Fn.dx()
            pZ += tm * (4 - Z/zeta) * F
            Fn.increment_order()    # 1/r^11
            pu -= 9 * w * Fn()
            return ph,px,pu,pZ
        raise Exception("mtype '"+self.__T+"' not implemented")

# function makeSingle
def makeSingleModel(mtype='T1', M=1.0, s=1.0, q=0.0, L=0.0, gamma=0.0, \
                    phi=0.0, a=None, b=None):
    """

    create a single bar model of Dehnen & Aly (2022)

    Parameters:
    mtype : str
        one of 'Tk','Vk' with k in [0..4] or 'Dk','Wk' with k in [1..4]
        mtype T1 gives the Kuzmin (q=0 or b=0) or Miyamoto-Nagai disc (q,b>0)
        Default is 'T1'
    M : float
        total mass of the model, M ≠ 0 is required
        Default is 1
    s : float
        scale radius s = a+b; s > 0 is required
        Default is 1
    q : float
        ratio q = b/s, if q=0 the model is razor thin, 0 ≤ q ≤ 1 is required
        Default is 0
    L : float
        half-length L ≥ 0 of needle with which disc model is convolved
        Default is L=0
    gamma : float
        parameter γ controlling the slope of needle
        no effect if L=0. -1 ≤ γ ≤ 1 is required (for non-negative density)
        Default is γ=0, when the needle density is constant
    phi : float
        angle φ of bar axes with x-axis.
        If ≠ 0 two bars at angles  ± φ are generated
    a : float, optional
        scale length a; if given, b must also be given, they override s,q
        Default is None
    b : float, optional
        scale height b; if given, a must also be given, they override s,q
        Default is None
        """
    disc = None
    if mtype[0] == 'D':
        if   mtype[1] == '0' or mtype[1] == '1':
            disc = singleModel(M,s,q,L,gamma,'T0',a,b)
        elif mtype[1] == '2':
            dsc0 = singleModel(2*M,s,q,L,gamma,'T0',a,b)
            dsc1 = singleModel(  M,s,q,L,gamma,'T1',a,b)
            disc = dsc0 - dsc1
            disc.set_type('D2')
        elif mtype[1] == '3':
            dsc0 = singleModel(8*M/3,s,q,L,gamma,'T0',a,b)
            dsc1 = singleModel(4*M/3,s,q,L,gamma,'T1',a,b)
            dsc2 = singleModel(  M/3,s,q,L,gamma,'T2',a,b)
            disc = dsc0 - dsc1 - dsc2
            disc.set_type('D3')
        elif mtype[1] == '4':
            dsc0 = singleModel(3.2*M,s,q,L,gamma,'T0',a,b)
            dsc1 = singleModel(1.6*M,s,q,L,gamma,'T1',a,b)
            dsc2 = singleModel(0.4*M,s,q,L,gamma,'T2',a,b)
            dsc3 = singleModel(0.2*M,s,q,L,gamma,'T3',a,b)
            disc = dsc0 - dsc1 - dsc2 - dsc3
            disc.set_type('D4')
        else:
            raise Exception("unknown mtype '"+str(mtype)+"'")
    elif mtype[0] == 'W':
        if   mtype[1] == '0' or mtype[1] == '1':
            disc = singleModel(M,s,q,L,gamma,'V0',a,b)
        elif mtype[1] == '2':
            dsc0 = singleModel(2*M,s,q,L,gamma,'V0',a,b)
            dsc1 = singleModel(  M,s,q,L,gamma,'V1',a,b)
            disc = dsc0 - dsc1
            disc.set_type('W2')
        elif mtype[1] == '3':
            dsc0 = singleModel(8*M/3,s,q,L,gamma,'V0',a,b)
            dsc1 = singleModel(4*M/3,s,q,L,gamma,'V1',a,b)
            dsc2 = singleModel(  M/3,s,q,L,gamma,'V2',a,b)
            disc = dsc0 - dsc1 - dsc2
            disc.set_type('W3')
        elif mtype[1] == '4':
            dsc0 = singleModel(3.2*M,s,q,L,gamma,'V0',a,b)
            dsc1 = singleModel(1.6*M,s,q,L,gamma,'V1',a,b)
            dsc2 = singleModel(0.4*M,s,q,L,gamma,'V2',a,b)
            dsc3 = singleModel(0.2*M,s,q,L,gamma,'V3',a,b)
            disc = dsc0 - dsc1 - dsc2 - dsc3
            disc.set_type('W4')
        else:
            raise Exception("unknown mtype '"+str(mtype)+"'")
    else:
        disc = singleModel(M,s,q,L,gamma,mtype,a,b)
    if L > 0 and phi != 0.0:
        disc = crossedModel(disc,phi)
    return disc
    
# function makeHoledDisc
def makeHoledDisc(M,s,q,rh,mtype="T1",dk=0,a=None,b=None):
    """
    create a simple axisymmetric disc model with an inner hole:
    the difference between two axisymmetric disc models
    
    Parameters:
    M : float
        mass of the disc if it had no hole
    s : float
        scale radius s = a+b; s > 0 is required
        Default is 1
    q : float
        ratio q = b/s, if q=0 the model is razor thin, 0 ≤ q ≤ 1 is required
        Default is 0
    a : float, optional
        scale length a; if given, b must also be given, they override s,q
        Default is None
    b : float, optional
        scale height b; if given, a must also be given, they override s,q
        Default is None
    mtype : string
        model type to use; must not be 'T0', 'V0', or any 'Dk' or 'Wk' model
    dk : int
        use model X[k+dk] to subtract; model must exist
    rh : float
        radius of the hole, must be < s
    """
    if mtype[1]=='0' or (mtype[0]!='T' and mtype[0]!='V'):
        raise Exception("mtype '"+mtype+"' not allowed")
    disc1 = makeSingleModel(M=M,mtype=mtype,s=s,q=q,a=a,b=b)
    s1    = disc1.scale_radius()
    b1    = disc1.scale_height()
    if rh > s1:
        raise Exception("rh = "+str(rh)+" > s = "+str(s1))
    if rh < b1:
        raise Exception("rh = "+str(rh)+" < b = "+str(b1))
    mtyp2 = mtype[0] + str(int(mtype[1]) + dk)
    disc2 = makeSingleModel(M=M,mtype=mtyp2,s=rh,q=disc1.scale_height()/rh)
    disc2.rescale_mass(disc1.density(0,0,0)/disc2.density(0,0,0))
    debugInfo(2, " ρ1(0)={:} ρ1(s,0)={:}".\
              format(disc1.density(0,0,0),disc1.density(s1,0,0)))
    debugInfo(2, " ρ2(0)={:} ρ2(s,0)={:}".\
              format(disc2.density(0,0,0),disc2.density(s1,0,0)))
    return disc1 - disc2

# function makeBulgeBar()
def makeBulgeBar():
    """
    create model 'bulge-bar' by Dehnen & Aly (2022)
    """
    peanut      = makeSingleModel(M=0.08,a=0.05,b=0.25,L=0.33,gamma=-0.95,
                                  mtype='V4')
    bar         = makeSingleModel(M=0.15,a=0.4,b=0.1,L=1.,gamma=0.1,phi=0.1047,
                                  mtype='V4')
    nuclearDisc = makeSingleModel(M=0.01,a=0.1,b=0.1,L=0.,gamma=0.,
                                  mtype='V4')
    outerDisc   = makeHoledDisc  (M=1.,s=1.6,q=0.05,rh=1.2,
                                  mtype='V2')
    return peanut + bar + nuclearDisc + outerDisc

# function makeThinBar()
def makeThinBar():
    """
    create model 'thin-bar' by Dehnen & Aly (2022)
    """
    bar         = makeSingleModel(M=0.15,a=0.3,b=0.1,L=1.,gamma=0.7,mtype='T3')
    nuclearDisc = makeSingleModel(M=0.02,a=0.3,b=0.1,L=0.,gamma=0.,mtype='V4')
    outerDisc   = makeHoledDisc  (M=1.,s=1.6,q=0.05,rh=1.2,mtype='V2')
    return bar + nuclearDisc + outerDisc
