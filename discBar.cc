// -*- C++ -*-
////////////////////////////////////////////////////////////////////////////////
///
/// \file    discBar.cc
/// \brief   implement discBar.h
///
/// \author  Walter Dehnen
///
/// \date    2022
///
/// copyright Walter Dehnen (2022)
///
/// \license GNU GENERAL PUBLIC LICENSE version 3.0
///          see file LICENSE
///
////////////////////////////////////////////////////////////////////////////////
///
/// \version jul-2022  WD  implemented models T1-4, V1-4
/// \version aug-2022  WD  minor refactoring, debugged models V2,V4
/// \version sep-2022  WD  'rod' --> 'needle' in nameing & comments
///
////////////////////////////////////////////////////////////////////////////////
#include "discBar.h"
#include <cmath>
#include <vector>
#include <cassert>
#include <iostream>
#include <exception>
#include <type_traits>

#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wpadded"
#  pragma clang diagnostic ignored "-Wunused-member-function"
#  pragma clang diagnostic ignored "-Wreturn-std-move-in-c++11"
#endif

namespace {

using namespace discBar;

using std::abs;
using std::log;
using std::sin;
using std::cos;
using std::sqrt;
using std::atan;
using std::atan2;
using std::hypot;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::make_unique;
using std::runtime_error;

using std::is_same_v;
using std::enable_if_t;

static constexpr double Pi = 3.14159265358979323846264338328;

inline constexpr double square(double x) noexcept
{ return x*x; }

inline constexpr double sign(double x) noexcept
{ return x<0.0? -1.0 : x>0.0? 1.0 : 0.0; }

inline constexpr char family(int T)
{ return "TV"[(T>>3) & 3]; }

inline constexpr int family(char F)
{ return (F=='V'? 1 : 0) << 3; }

inline constexpr int order(int T)
{ return T & 7; }

inline constexpr int order(char K)
{ return K=='0'? 0 : K=='2'? 2: K=='3'? 3 : K=='4'? 4 : 1; }

static constexpr int T1 =  1;
static constexpr int T2 =  2;
static constexpr int T3 =  3;
static constexpr int T4 =  4;
static constexpr int V1 =  9;
static constexpr int V2 = 10;
static constexpr int V3 = 11;
static constexpr int V4 = 12;

inline int thinType(int T)
{
    return T&7;                 // t --> T
}

inline int sphericalType(int T)
{
    return (T&24) | 1;          // k -> 1
}

inline int getType(string const&T)
{
    return family(T[0]) | order(T[1]);
}

inline string getName(int T)
{
    return family(T) + to_string(order(T)); 
}

/// internal representation of discBar parameters
struct internalPars
  : numericalParameters
{
    double F;      // actual mass normalisation in rFunc
    double C1,C2;  // actual mass normalisation in aFunc,bFunc
    int    T;

    bool isRazorThin() const noexcept
    { return B <= 0; }

    bool isSpherical() const noexcept
    { return A <= 0 && L <= 0; }

    bool isBarred() const noexcept
    { return L > 0; }

    string modelType() const
    { return getName(T); }

    internalPars(parameters const&pars) noexcept
      : numericalParameters(pars)
      , F ( -M * ((P>0 || P<0)? 0.5 : 1.0) )
      , C1( L <= 0? 0 : 0.5*F*(1-G)/L )
      , C2( L <= 0? 0 : F*G/(L*L) )
      , T ( getType(pars.T) )
    {
	if(A <= 0) T = sphericalType(T);  else 
	if(B <= 0) T = thinType(T);
    }
    
    parameters params() const
    {
	return {static_cast<const numericalParameters&>(*this),modelType()};
    }

    void rescaleMass(double factor) noexcept
    {
	M  *= factor;
	F  *= factor;
	C1 *= factor;
	C2 *= factor;
    }

    void rescaleHeight(double factor) noexcept
    {
	B *= factor;
    }

    void rescaleLength(double factor) noexcept
    {
	A *= factor;
	L *= factor;
	factor = 1/factor;
	C1 *= factor;
	C2 *= factor*factor;
    }

    void flipSign()
    {
	M = -M;
	F = -F;
    }

    void dump(std::ostream&out) const
    {
	out<<" A="<<A<<" B=" <<B <<" L=" << L<<" G="<<G<<" P="<<P<<" M="<<M
	   <<" F="<<F<<" C1="<<C1<<" C2="<<C2<<" T="<<T<<"='"<<getName(T)
	   <<"'\n";
    }
    
};  // struct internalPars

inline int type(internalPars const&pars) noexcept
{
    return pars.T;
}

inline int type(parameters const&pars) noexcept
{
    return getType(pars.T);
}

// Fn = ??/r^n  (unconvolved, used for axisymmetric models)
class rFunc
{
    int N;
    const double X,iR2;
    double F;

  public:

    static string name()
    { return "rFunc"; }

    rFunc(int n, double x, double y, double z, const internalPars*p) noexcept
      : N   ( (n&1)? 1 : 2 )
      , X   ( x )
      , iR2 ( 1/(x*x+y*y+z*z) )
      , F   ( N==1?  p->F*sqrt(iR2) : p->F*iR2 )
    {
	while(N < n)
	    incOrder();
    }
        
    static void assertPars(const internalPars*p) noexcept
    { assert(p->L <= 0); }

    int order() const noexcept
    { return N; }

    void incOrder() noexcept
    {
	F *= iR2;
	N += 2;
    }

    double operator()() const noexcept
    { return F; }

    double operator()(double &d1X) const noexcept
    {
	d1X = -N * X * iR2 * F;
	return F;
    } 
    
    double operator()(double &d1X, double&d2X) const noexcept
    {
	d1X  =-N * iR2 * F;
	d2X  = d1X * (1 - (N+2)*X*X*iR2);
	d1X *= X;
	return F;
    }
};  // class rFunc

///  An(x,u) := ?? ??? 1/r^n dx
///  Bn(x,u) := ?? ??? x/r^n dx
///  Cn(x,u) := x An - Bn
struct funcAC
{
    const double X,iU2,iR2;  // x, 1/u??, 1/r??
    double       FA,FB,AX;   // An, Bn, ???An/???x
    int          N;          // n

    funcAC(int n, double x, double u2, double r2, double f) noexcept
      : X(x), iU2(1/u2), iR2(1/r2)
    {
	if (n&1) {
	    double r = sqrt(r2);
	    AX = f / r;
	    if( n==1) {
		N  = 1;
		FA = f * log(X+r);
		FB = f * r;
	    } else {
		N  = 3;
		FB = -AX;
		FA = X * AX * iU2;
		AX*= iR2;
	    }
	} else {
	    double iU = sqrt(iU2);
	    N  = 2;
	    AX = f * iR2;
	    FA = f * iU * atan(iU*X);
	    FB = n==2? f * 0.5*log(r2) : 0.0;
	}
	while(N < n)
	    incOrder();
    }

  public:

    funcAC(int n, double x, double u2, double f=1.0) noexcept
      : funcAC(n, x, u2, x*x+u2, f) {}

    void incOrder() noexcept
    {
        //  B[n+2]    =-???An/???x / n
        //  A[n+2]    = ((n-1) An + x ???An/???x) / nu??
        // ???A[n+2]/???x = ???An/???x / (x??+u??)
	double iN = 1.0/N;
	FB  = -AX * iN;
	FA  = ((N-1)*FA + X*AX) * iU2 * iN;
	AX *= iR2;
	N  += 2;
    }

    double A() const noexcept
    { return FA; }

    double d1Ax() const noexcept
    { return AX; }

    double d2Ax() const noexcept
    { return -X*N*AX*iR2; }

    double C() const noexcept
    { return X*FA-FB; }

    double d1Cx() const noexcept
    { return FA; }

    double d2Cx() const noexcept
    { return AX; }
};  // class funcAC

// Fn = F / r^n convolved with constant needle density
class aFunc
{
    funcAC Am,Ap;

    aFunc(int n, double x, double u2, double l, double c1) noexcept
      : Am(n,x-l,u2,c1), Ap(n,x+l,u2,c1) {}

  public:

    static string name()
    { return "aFunc"; }

    aFunc(int n, double x, double y, double z, const internalPars*p) noexcept
      : aFunc(n, x, y*y+z*z, p->L, p->C1) {}

    static void assertPars(const internalPars*p) noexcept
    {
	assert(p->L > 0);
	assert(p->G <= 0 && p->G >= 0);
    }

    int order() const noexcept
    { return Am.N; }

    void incOrder() noexcept
    { Am.incOrder(); Ap.incOrder(); }
    
    double operator()() const noexcept
    { return Ap.A() - Am.A(); }

    double operator()(double&d1X) const noexcept
    {
	d1X =  Ap.d1Ax() - Am.d1Ax();
	return Ap.A()    - Am.A();
    }	    

    double operator()(double&d1X, double&d2X) const noexcept
    {
	d1X =  Ap.d1Ax() - Am.d1Ax();
	d2X =  Ap.d2Ax() - Am.d2Ax();
	return Ap.A()    - Am.A();
    }
};  // class aFunc


// F/r^n convolved with a needle at |x|<l of linear density
class bFunc
{
    const double C1,C2;
    funcAC    Am,A0,Ap;

    bFunc(int n, double x, double u2, double l, double c1, double c2) noexcept
      : C1(c1), C2(c2)
      , Am(n,x-l,u2), A0(n,x,u2), Ap(n,x+l,u2) {}

  public:

    static string name()
    { return "bFunc"; }

    bFunc(int n, double x, double y, double z, const internalPars*p) noexcept
      : bFunc(n, x, y*y+z*z, p->L, p->C1, p->C2) {}

    static void assertPars(const internalPars*p) noexcept
    {
	assert(p->L > 0);
	assert(p->G < 0 || p->G > 0);
    }

    int order() const noexcept
    { return Am.N; }

    void incOrder() noexcept
    { Am.incOrder(); A0.incOrder(); Ap.incOrder(); }

    double operator()() const noexcept
    { return  C1 * (Ap.A()    - Am.A())
	    + C2 * (Ap.C()    + Am.C()    - 2*A0.C()); }

    double operator()(double&d1X) const noexcept
    {
	d1X  = C1 * (Ap.d1Ax() - Am.d1Ax())
	    +  C2 * (Ap.d1Cx() + Am.d1Cx()  - 2*A0.d1Cx());
	return C1 * (Ap.A()    - Am.A())
	    +  C2 * (Ap.C()    + Am.C()     - 2*A0.C());
    }	    

    double operator()(double&d1X, double&d2X) const noexcept
    {
	d1X  = C1 * (Ap.d1Ax() - Am.d1Ax())
	    +  C2 * (Ap.d1Cx() + Am.d1Cx()  - 2*A0.d1Cx());
	d2X  = C1 * (Ap.d2Ax() - Am.d2Ax())
	    +  C2 * (Ap.d2Cx() + Am.d2Cx()  - 2*A0.d2Cx());
	return C1 * (Ap.A()    - Am.A())
	    +  C2 * (Ap.C()    + Am.C()     - 2*A0.C());
    }	    
};  // class bFunc


// convert {???x,Du,dZ}  to   ???{x,y,Z} 
//    with Du = 2???u?? = (1/u)???u
//    dZ = ???Z at fixed u??=y??+Z??
inline void convert2D(double*d1P, double y) noexcept
{
    d1P[1] *= y;
}

inline void convert2D(double*d1P, double*d2P, double y) noexcept
{
    d2P[1] *= y;
    d2P[2]  = d1P[1] + y * y * d2P[2];
    d1P[1] *= y;
}

template<bool hasZ1Deriv>
inline void convert3D(double*d1P, double y, double z) noexcept
{
    if constexpr (hasZ1Deriv)
	d1P[2] += z * d1P[1];
    else
	d1P[2]  = z * d1P[1];
    d1P[1] *= y;
}

template<bool hasZ1Deriv,bool hasZ2Deriv>
inline void convert3D(double*d1P, double*d2P, double y, double Z) noexcept
{
    /// ???x???Z = ???xdZ + Z ???xDu 
    if constexpr (hasZ1Deriv)
	d2P[2] += Z * d2P[1];
    else
	d2P[2]  = Z * d2P[1];
    /// ???x???y = y ???xDu
    d2P[1]     *= y;
    /// ???Z??  = dZ?? + Du + Z (2 dZDu + Z Du??)
    if      constexpr (hasZ2Deriv)
	d2P[5] += d1P[1] + Z * ( 2 * d2P[4] + Z * d2P[3]);
    else if constexpr (hasZ1Deriv)
	d2P[5]  = d1P[1] + Z * ( 2 * d2P[4] + Z * d2P[3]);
    else
	d2P[5]  = d1P[1] + Z *                Z * d2P[3];
    /// ???y???Z = y (dZDu + Z Du??)
    if constexpr (hasZ1Deriv)
	d2P[4] += Z * d2P[3];
    else
	d2P[4]  = Z * d2P[3];
    d2P[4]     *= y; 
    /// ???y??  = Du + y?? Du??
    d2P[3]      = d1P[1] + y *                y * d2P[3];
    /// ???Z   = dZ + Z Du
    if constexpr (hasZ1Deriv)
	d1P[2] += Z * d1P[1];
    else
	d1P[2]  = Z * d1P[1];
    /// ???y   = y Du
    d1P[1]     *= y;
}
    
template<typename, int> struct inputModel;

// type T1: Kuzmin/Plummer/Miyamoto-Nagai models
// ??(x,y,Z) = -M w1 C1
// w1 = 1
template<typename psiFunc> struct inputModel<psiFunc,T1>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;

    static constexpr auto Type = T1;
    
    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }

    // ??(r)
    double phi(double x, double y, double Z) const noexcept
    {
	psiFunc F{1,x,y,Z,this};
	return F();
    }
    // ??(r) and ????? = (???x,Du)??
    double phi2D_impl(double x, double y, double Z, double*d1P) const noexcept
    {
	psiFunc F{1,x,y,Z,this};
	double P = F(d1P[0]);
	F.incOrder();
	d1P[1]   =-F();
	return P;
    }
    // ??(r) and ????? = (???x,???y)??
    double phi2D(double x, double y, double Z, double*d1P) const noexcept
    {
	double P = phi2D_impl(x,y,Z,d1P);
	convert2D(d1P,y);
	return P;
    }
    // ??(r) and ????? = (???x,???y,???Z)??
    double phi3D(double x, double y, double Z, double*d1P) const noexcept
    {
	double P = phi2D_impl(x,y,Z,d1P);
	convert3D<false>(d1P,y,Z);
	return P;
    }
    // ??(r), ????? = (???x,???y)??, and ???????? = (???x,???y)????
    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	psiFunc F{1,x,y,Z,this};
	double P = F(d1P[0],d2P[0]);
	F.incOrder();
	d1P[1]   =-F(d2P[1]);
	d2P[1]   =  -d2P[1];
	F.incOrder();
	d2P[2]   = 3*F();
	convert2D(d1P,d2P,y);
	return P;
    }
    // ??(r), ????? = (???x,???y,???Z)??, and ???????? = (???x,???y,???Z)????
    double phi3D(double x, double y, double Z,
	double *d1P, double*d2P) const noexcept
    {
	psiFunc F{1,x,y,Z,this};
	double P = F(d1P[0],d2P[0]);
	F.incOrder();
	d1P[1]   =-F(d2P[1]);
	d2P[1]   =  -d2P[1];
	F.incOrder();
	d2P[3]   = 3*F();
	convert3D<false,false>(d1P,d2P,y,Z);
	return P;
    }
};  // struct inputModel<T1>

// type T2:
// ??(x,y,Z) = -M (w1 C1 + w3 C3)
// w1=1  w3=aZ
template<typename psiFunc> struct inputModel<psiFunc,T2>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;

    static constexpr auto Type = T2;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }

    double phi(double x, double y, double Z) const noexcept
    {
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double P = F();                 // f1 = C1
	F.incOrder();
	P += A * Z * F();               // f3 = w3 C3
	return P;
    }

    double phi2D(double x, double y, double Z, double*d1P) const noexcept
    {
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double d1C;
	double P = F(d1C);              //    f1 = C1
	d1P[0]   = d1C;                 // ???x f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                   // Du f1 =-C3
	double w3= A * Z;
	P       += w3 * C;              //    f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x f3 = w3 ???x C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F();
	d1P[1]  -= 3 * w3 * C;          // Du f3 =-3 w1 C5
	convert2D(d1P,y);
	return P;
    }

    double phi3D(double x, double y, double Z, double*d1P) const noexcept
    {
	psiFunc F{1,x,y,Z,this}; // F = C1 := -M/r
	double d1C;
	double P = F(d1C);              //    f1 = C1
	d1P[0]   = d1C;                 // ???x f1 = ???xC1
	F.incOrder();            // F = C3 := -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                   // Du f1 =-C3
	double w3= A * Z;
	P       += w3 * C;              //    f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x f3 = w3 ???x C3
	d1P[2]   = A * C;               // dZ f3 = w3' C3
	F.incOrder();            // F = C5 := -M/r^5
	C        = F();
	d1P[1]  -= 3 * w3 * C;          // Du f3 =-3 w3 C5
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double P = F(d1C,d2C);          //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	d2P[0]   = d2C;                 // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	d2P[1]   =-d1C;                 // ???xDu f1 = -???xC3
	double w3= A * Z;
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;            // ???x??  f3 = w3 ???x??C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d2P[2]   = 3 * C;               // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;        // ???xDu f3 =-3 w3 ???xC5
	F.incOrder();            // F = C7 = -M/r^7
	d2P[2]  += 15 * w3 * F();       // Du??  f3 = 15 * w3 * C7
	convert2D(d1P,d2P,y);
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double P = F(d1C,d2C);          //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	d2P[0]   = d2C;                 // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	d2P[1]   =-d1C;                 // ???xDu f1 = -???xC3
	double w3= A * Z;
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d1P[2]   = A * C;               // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;            // ???x??  f3 = w3 ???x??C3
	d2P[2]   = A * d1C;             // ???xdZ f3 = w3' ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d2P[3]   = 3 * C;               // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;        // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * A * C;           // dZDu f3 =-3 w3' C5
	F.incOrder();            // F = C7 = -M/r^7
	d2P[3]  += 15 * w3 * F();       // Du??  f3 = 15 * w3 * C7
	convert3D<true,false>(d1P,d2P,y,Z);
	return P;
    }
};  // struct inputModel<T2>
    
// type T3:
// ??(x,y,Z) = -M (w1 C1 + w3 C3 + w5 C5)
// w1=1  w3=a(Z-a/3)  w5=a??Z??
template<typename psiFunc> struct inputModel<psiFunc,T3>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;

    static constexpr auto Type = T3;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }
    
    double phi(double x, double y, double Z) const noexcept
    {
	psiFunc F {1,x,y,Z,this};
	double P = F();
	F.incOrder();
	P += A * (Z - A/3) * F();
	F.incOrder();
	P += square(A * Z) * F();
	return P;
    }

    double phi3D(double x, double y, double Z, double*d1P) const noexcept
    {
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C;
	double C = F(d1C);
	double P = C;                    //      f1 = C1
	d1P[0]   = d1C;                  // ???x   f1 = ???xC1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C);
	d1P[1]   =-C;                    // Du   f1 = -C3
	double w3= A * (Z - A/3);
	P       += w3 * C;               //      f3 = w3 C3
	d1P[0]  += w3 * d1C;             // ???x   f3 = w3 ???xC3
	d1P[2]   = A * C;                // dZ   f3 = w3' C3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;           // Du   f3 =-3 w3 C5
	double w5= A * Z;
	double d5= 2 * A * w5;
	w5      *= w5;
	P       += w5 * C;               //      f5 = w5 C5
	d1P[0]  += w5 * d1C;             // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;               // dZ   f5 = w5' C5
	F.incOrder();             // F = C7 = 0M/r^7
	C        = F();
	d1P[1]  -= 5 * w5 * C;           // Du   f5 = -5 w5 C7
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z, double*d1P) const noexcept
    {
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C;
	double C = F(d1C);
	double P = C;                    //      f1 = C1
	d1P[0]   = d1C;                  // ???x   f1 = ???xC1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C);
	d1P[1]   =-C;                    // Du   f1 = -C3
	double w3= A * (Z - A/3);
	P       += w3 * C;               //      f3 = w3 C3
	d1P[0]  += w3 * d1C;             // ???x   f3 = w3 ???xC3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;           // Du   f3 =-3 w3 C5
	double w5= square(A * Z);
	P       += w5 * C;               //      f5 = w5 C5
	d1P[0]  += w5 * d1C;             // ???x   f5 = w5 ???xC5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F();
	d1P[1]  -= 5 * w5 * C;           // Du   f5 = -5 w5 C7
	convert2D(d1P,y);
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double C = F(d1C,d2C);
	double P = C;                   //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	d2P[0]   = d2C;                 // ???x??  f1 = ???x??C1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C,d2C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	d2P[1]   =-d1C;                 // ???xDu f1 = -???xC3
	double w3= A * (Z - A/3);
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d1P[2]   = A * C;               // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;            // ???x??  f3 = w3 ???x??C3
	d2P[2]   = A * d1C;             // ???xdZ f3 = w3' ???xC3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[3]   = 3 * C;               // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;        // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * A * C;           // DudZ f3 =-3 w3' C5
	double w5= A * Z;
	double d5= 2 * A * w5;
	w5      *= w5;
	P       += w5 * C;              //      f5 = w5 C5
	d1P[0]  += w5 * d1C;            // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;              // dZ   f5 = w5' C5
	d2P[0]  += w5 * d2C;            // ???x??  f5 = w5 ???x??C5
	d2P[2]  += d5 * d1C;            // ???xdZ f5 = w5' ???xC5
	d2P[5]   = 2 * A * A * C;       // dZ??  f5 = w5" C5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F(d1C);
	d2P[3]  += 15 * w3 * C;         // Du??  f3 = 15 w3 C7
	d1P[1]  -=  5 * w5 * C;         // Du   f5 = -5 w5 C3
	d2P[1]  -=  5 * w5 * d1C;       // ???xDu f5 = -5 w5 ???xC3
	d2P[4]  -=  5 * d5 * C;         // DudZ f5 = -5 w5' C3
	F.incOrder();             // F = C9 = -M/r^9
	d2P[3]  += 35 * w5 * F();       // Du??  f5 = 35 w5 C9
	convert3D<true,true>(d1P,d2P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double C = F(d1C,d2C);
	double P = C;                   //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	d2P[0]   = d2C;                 // ???x??  f1 = ???x??C1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C,d2C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	d2P[1]   =-d1C;                 // ???xDu f1 = -???xC3
	double w3= A * (Z - A/3);
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;            // ???x??  f3 = w3 ???x??C3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[2]   = 3 * C;               // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;        // ???xDu f3 =-3 w3 ???xC5
	double w5= square(A * Z);
	P       += w5 * C;              //      f5 = w5 C5
	d1P[0]  += w5 * d1C;            // ???x   f5 = w5 ???xC5
	d2P[0]  += w5 * d2C;            // ???x??  f5 = w5 ???x??C5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F(d1C);
	d2P[2]  += 15 * w3 * C;         // Du??  f3 = 15 w3 C7
	d1P[1]  -=  5 * w5 * C;         // Du   f5 = -5 w5 C3
	d2P[1]  -=  5 * w5 * d1C;       // ???xDu f5 = -5 w5 ???xC3
	F.incOrder();             // F = C9 = -M/r^9
	d2P[2]  += 35 * w5 * F();       // Du??  f5 = 35 w5 C9
	convert2D(d1P,d2P,y);
	return P;
    }
};  // struct inputModel<T3>

// type T4:
// ??(x,y,Z) = -M (w1 C1 + w3 C3 + w5 C5 + w7 C7)
// w1=1  w3=a(Z-2a/5)  w5= (3/5)a??Z(2Z-a)  w7=a??Z??
template<typename psiFunc> struct inputModel<psiFunc,T4>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;

    static constexpr auto Type = T4;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }
    
    double phi(double x, double y, double Z) const noexcept
    {
	double aZ = A * Z;
	psiFunc F{1,x,y,Z,this};
	double P = F();
	F.incOrder();
	P += A * (Z - 0.4*A) * F();
	F.incOrder();
	P += 0.6 * A * aZ * (Z+Z-A) * F();
	F.incOrder();
	P += aZ * aZ * aZ * F();
	return P;
    }

    double phi3D(double x, double y, double Z, double*d1P) const noexcept
    {
	double aZ = A * Z;
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C;
	double C = F(d1C);
	double P = C;                   //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	double w3= A * (Z-0.4*A);
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d1P[2]   = A * C;               // dZ   f3 = w3' C3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	double d5= 0.6*A*A;
	double w5= d5*Z*(Z+Z-A);        // w5 = (3/5)a??Z(2Z-a)
	d5      *=      (4*Z-A);        // w5'= (3/5)a?? (4Z-a)
	P       += w5 * C;              //      f5 = w5 C5
	d1P[0]  += w5 * d1C;            // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;              // dZ   f5 = w5' C5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F(d1C);
	d1P[1]  -=  5 * w5 * C;         // Du   f5 = -5 w5 C7
	double d7= aZ * aZ;
	double w7= aZ * d7;             // w7      =  a??Z??
	d7      *= 3 * A;               // w7'     =  3 a??Z??
	P       += w7 * C;              //      f7 = w7 C7
	d1P[0]  += w7 * d1C;            // ???x   f7 = w7 ???xC7
	d1P[2]  += d7 * C;              // dZ   f7 = w7' C7
	F.incOrder();             // F = C9 = -M/r^9
	C        = F();
	d1P[1]  -=  7 * w7 * C;         // Du   f7 = -7 w7 C9
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z, double*d1P) const noexcept
    {
	const double aZ = A * Z;
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C;
	double C = F(d1C);
	double P = C;                   //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	double w3= A * (Z-0.4*A);
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	double w5= 0.6*A*A*Z*(Z+Z-A);   // w5 = (3/5)a??Z(2Z-a)
	P       += w5 * C;              //      f5 = w5 C5
	d1P[0]  += w5 * d1C;            // ???x   f5 = w5 ???xC5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F(d1C);
	d1P[1]  -=  5 * w5 * C;         // Du   f5 = -5 w5 C7
	double w7= aZ * aZ * aZ;        // w7      =  a??Z??
	P       += w7 * C;              //      f7 = w7 C7
	d1P[0]  += w7 * d1C;            // ???x   f7 = w7 ???xC7
	F.incOrder();             // F = C9 = -M/r^9
	C        = F();
	d1P[1]  -=  7 * w7 * C;         // Du   f7 = -7 w7 C9
	convert2D(d1P,y);
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double C = F(d1C,d2C);
	double P = C;                   //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	d2P[0]   = d2C;                 // ???x??  f1 = ???x??C1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C,d2C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	d2P[1]   =-d1C;                 // ???xDu f1 = -???xC3
	double w3= A * (Z-0.4*A);
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d1P[2]   = A * C;               // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;            // ???x??  f3 = w3 ???x??C3
	d2P[2]   = A * d1C;             // ???xdZ f3 = w3' ???xC3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[3]   = 3 * C;               // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;        // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * A * C;           // DudZ f3 =-3 w3' C5
	double t5= 2.4 * A * A;
	double d5= t5 * (Z - 0.25*A);
	double w5= 0.5 * t5 * Z * (Z-0.5*A);
	P       += w5 * C;              //      f5 = w5 C5
	d1P[0]  += w5 * d1C;            // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;              // dZ   f5 = w5' C5
	d2P[0]  += w5 * d2C;            // ???x??  f5 = w5 ???x??C5
	d2P[2]  += d5 * d1C;            // ???xdZ f5 = w5' ???xC5
	d2P[5]   = t5 * C;              // dZ??  f5 = w5" C5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F(d1C,d2C);
	d2P[3]  += 15 * w3 * C;         // Du??  f3 = 15 w3 C7
	d1P[1]  -=  5 * w5 * C;         // Du   f5 = -5 w5 C7
	d2P[1]  -=  5 * w5 * d1C;       // ???xDu f5 = -5 w5 ???xC7
	d2P[4]  -=  5 * d5 * C;         // DudZ f5 = -5 w5' C7
	double t7= A*A*A*Z;
	double d7= t7*Z;
	double w7= d7*Z;
	t7      *= 6;
	d7      *= 3;
	P       += w7 * C;              //      f7 = w7 C7
	d1P[0]  += w7 * d1C;            // ???x   f7 = w7 ???xC7
	d1P[2]  += d7 * C;              // dZ   f7 = w7' C7
	d2P[0]  += w7 * d2C;            // ???x??  f7 = w7 ???x??C3
	d2P[2]  += d7 * d1C;            // ???xdZ f7 = w7' ???xC3
	d2P[5]  += t7 * C;              // dZ??  f7 = w7" C3
	F.incOrder();             // F = C9 = -M/r^9
	C        = F(d1C);
	d2P[3]  += 35 * w5 * C;         // Du??  f5 = 35 w5 C9
	d1P[1]  -=  7 * w7 * C;         // Du   f7 = -7 w7 C9
	d2P[1]  -=  7 * w7 * d1C;       // ???xDu f7 = -7 w7 ???xC9
	d2P[4]  -=  7 * d7 * C;         // DudZ f7 = -7 w7' C9
	F.incOrder();             // F = C11 = -M/r^11
	C        = F();
	d2P[3]  += 63 * w7 * C;         // Du??  f7 = 63 w7 C11
	convert3D<true,true>(d1P,d2P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	double aZ = A * Z;
	psiFunc F {1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double C = F(d1C,d2C);
	double P = C;                   //      f1 = C1
	d1P[0]   = d1C;                 // ???x   f1 = ???xC1
	d2P[0]   = d2C;                 // ???x??  f1 = ???x??C1
	F.incOrder();             // F = C3 = -M/r^3
	C        = F(d1C,d2C);
	d1P[1]   =-C;                   // Du   f1 = -C3
	d2P[1]   =-d1C;                 // ???xDu f1 = -???xC3
	double w3= A * (Z-0.4*A);       // w3 = a(Z-2a/5)
	P       += w3 * C;              //      f3 = w3 C3
	d1P[0]  += w3 * d1C;            // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;            // ???x??  f3 = w3 ???x??C3
	F.incOrder();             // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[2]   = 3 * C;               // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;          // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;        // ???xDu f3 =-3 w3 ???xC5
	double w5= 0.6*A*A*Z*(Z+Z-A);   // w5 = (3/5)a??Z(2Z-a)
	P       += w5 * C;              //      f5 = w5 C5
	d1P[0]  += w5 * d1C;            // ???x   f5 = w5 ???xC5
	d2P[0]  += w5 * d2C;            // ???x??  f5 = w5 ???x??C5
	F.incOrder();             // F = C7 = -M/r^7
	C        = F(d1C,d2C);
	d2P[2]  += 15 * w3 * C;         // Du??  f3 = 15 w3 C7
	d1P[1]  -=  5 * w5 * C;         // Du   f5 = -5 w5 C7
	d2P[1]  -=  5 * w5 * d1C;       // ???xDu f5 = -5 w5 ???xC7
	double w7= aZ * aZ * aZ;        // w7      =  a??Z??
	P       += w7 * C;              //      f7 = w7 C7
	d1P[0]  += w7 * d1C;            // ???x   f7 = w7 ???xC7
	d2P[0]  += w7 * d2C;            // ???x??  f7 = w7 ???x??C3
	F.incOrder();             // F = C9 = -M/r^9
	C        = F(d1C);
	d2P[2]  += 35 * w5 * C;         // Du??  f5 = 35 w5 C9
	d1P[1]  -=  7 * w7 * C;         // Du   f7 = -7 w7 C9
	d2P[1]  -=  7 * w7 * d1C;       // ???xDu f7 = -7 w7 ???xC9
	F.incOrder();             // F = C11 = -M/r^11
	C        = F();
	d2P[2]  += 63 * w7 * C;         // Du??  f7 = 63 w7 C11
	convert2D(d1P,d2P,y);
	return P;
    }
};  // struct inputModel<T4>


// type V1:
// ??(x,y,Z) = -M (w1 C1 + w3 C3)
// w1=1  w3=b??Z/2??
template<typename psiFunc> struct inputModel<psiFunc,V1>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;
    const double hB2;

    static constexpr auto Type = V1;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p), hB2(0.5*B*B)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }

    double phi(double x, double y, double Z) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double P = F();                 // f1 = w1 C1
	F.incOrder();            // F = C3 = M/r^3
	double w3 = hB2*Z*iz;
	P += w3 * F();                  // f3 = w3 C3
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= hB2*Z*iz;           // w3  = B Z/??   with   B = b??/2
	double d3= (hB2-w3)*iz;        // w3' =(B-w3)/?? = -aB/????
	double t3= -2*d3*iz;           // w3" = -2w3'/?? = 2aB/????
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	d2P[2]   = d3 * d1C;           // ???xdZ f3 = w3' ???xC3
	d2P[5]   = t3 * C;             // dZ??  f3 = w3" C3
	F.incOrder();            // F = C5 = M/r^5
	C        = F(d1C);
	d2P[3]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * d3 * C;         // DudZ f3 =-3 w3' C5
	F.incOrder();            // F = C7 = -M/r^7
	d2P[3]  += 15 * w3 * F();      // Du??  f3 = 15 * w3 * C7
	convert3D<true,true>(d1P,d2P,y,Z);
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= hB2*Z*iz;           // w3  = B Z/??   with   B = b??/2
	double d3= (hB2-w3)*iz;        // w3' =(B-w3)/?? = -aB/????
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	F.incOrder();            // F = C5 = M/r^5
	C        = F();
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= hB2*Z*iz;           // w3  = B Z/??   with   B = b??/2
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	F.incOrder();            // F = C5 = M/r^5
	C        = F(d1C);
	d2P[2]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	F.incOrder();            // F = C7 = -M/r^7
	d2P[2]  += 15 * w3 * F();      // Du??  f3 = 15 * w3 * C7
	convert2D(d1P,d2P,y);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= hB2*Z*iz;           // w3 = B Z/??   with   B = b??/2
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	F.incOrder();            // F = C5 = M/r^5
	C        = F();
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	convert2D(d1P,y);
	return P;
    }
};  // struct inputModel<V1>

// type V2:
// ??(x,y,Z) = -M (w1 C1 + w3 C3 + w5 C5)
// w1 =1  w3 =aZ+B   w5 =3aB Z??/??          B=b??/2
// w1'=0  w3'=a      w5'=3aB (1-a??/????)
// w1"=0  w3"=0      w5"=6aB a??/????
template<typename psiFunc> struct inputModel<psiFunc,V2>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;
    const double hB2;

    static constexpr auto Type = V2;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p), hB2(0.5*B*B)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }

    double phi(double x, double y, double Z) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = -M/r
	double P = F();                 // f1 = w1 C1
	F.incOrder();            // F = C3 = -M/r^3
	double w3= A*Z + hB2;
	P += w3 * F();                  // f3 = w3 C3
	F.incOrder();            // F = C5 = -M/r^5
	double w5= 3*A*hB2*Z*Z*iz;
	P += w5 * F();                  // f5 = w5 C5
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= A*Z+hB2;
	double d3= A;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	d2P[2]   = d3 * d1C;           // ???xdZ f3 = w3' ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[3]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * d3 * C;         // DudZ f3 =-3 w3' C5
	double w5= 3 * A * hB2 * Z*Z*iz;  // w5 = 3aB Z??/??
	double d5= square(A*iz);
	double t5= 6 * A * hB2 * d5 * iz;  // w5"= 6aB a??/????
	d5       = 3 * A * hB2 * (1-d5);   // w5'= 3aB (1-a??/????)
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;             // dZ   f5 = w5' C5
	d2P[0]  += w5 * d2C;           // ???x??  f5 = w5 ???x??C5
	d2P[2]  += d5 * d1C;           // ???xdZ f5 = w5' ???xC5
	d2P[5]   = t5 * C;             // dZ??  f5 = w5" C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C);
	d2P[3]  += 15 * w3 * C;        // Du??  f3 = 15 * w3 * C7
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	d2P[1]  -= 5 * w5 * d1C;       // ???xDu f5 =-5 w5 ???xC7
	d2P[4]  -= 5 * d5 * C;         // DudZ f5 =-5 w5' C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F();
	d2P[3]  += 35 * w5 * C;        // Du??  f5 = 35 * w5 * C9
	convert3D<true,true>(d1P,d2P,y,Z);
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= A*Z+hB2;
	double d3= A;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	double w5= 3 * A * hB2 * Z*Z*iz;  // w5 = 3aB Z??/??
	double d5= 3 * A * hB2 * (1-square(A*iz));   // w5'= 3aB (1-a??/????)
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;             // dZ   f5 = w5' C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F();
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= A*Z+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[2]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	double w5= 3 * A * hB2 * Z*Z*iz;  // w5 = 3aB Z??/??
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d2P[0]  += w5 * d2C;           // ???x??  f5 = w5 ???x??C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C);
	d2P[2]  += 15 * w3 * C;        // Du??  f3 = 15 * w3 * C7
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C5
	d2P[1]  -= 5 * w5 * d1C;       // ???xDu f5 =-5 w5 ???xC5
	F.incOrder();            // F = C9 = -M/r^9
	C        = F();
	d2P[2]  += 35 * w5 * C;        // Du??  f5 = 35 * w5 * C9
	convert2D(d1P,d2P,y);
	return P;
    }

    double phi2D(double x, double y, double Z, double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double aZ= A*Z;
	double w3= aZ+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	double w5= 3 * A * hB2 * Z*Z*iz;  // w5 = 3aB Z??/??
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F();
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C5
	convert2D(d1P,y);
	return P;
    }
};  // struct inputModel<V2>

// type V3:
// ??(x,y,Z) = -M (w1 C1 + w3 C3 + w5 C5 + w7 C7)
// w1 =1  w3 =a(Z-a/3)+B   w5 =aZ(aZ+3B)    w7 =5a??BZ??/??
// w1'=0  w3'=a            w5'=2a??Z+3aB     w7'=5a??B(2Z-a-a??/????)
// w1"=0  w3"=0            w5"=2a??          w7"=10a??B(1-a??/????)
template<typename psiFunc> struct inputModel<psiFunc,V3>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;
    const double hB2;

    static constexpr auto Type = V3;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p), hB2(0.5*B*B)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double ai=A*iz;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= aZ-0.333333333333333*A2+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = A  * C;             // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	d2P[2]   = A  * d1C;           // ???xdZ f3 = w3' ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[3]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * A  * C;         // DudZ f3 =-3 w3' C5
	double tm= aZ + 3*hB2;
	double w5= aZ * tm;
	double d5= A * (aZ + tm);
	double t5= A2 + A2;
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;             // dZ   f5 = w5' C5
	d2P[0]  += w5 * d2C;           // ???x??  f5 = w5 ???x??C5
	d2P[2]  += d5 * d1C;           // ???xdZ f5 = w5' ???xC5
	d2P[5]   = t5 * C;             // dZ??  f5 = w5" C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C,d2C);
	d2P[3]  += 15 * w3 * C;        // Du??  f3 = 15 * w3 * C7
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	d2P[1]  -= 5 * w5 * d1C;       // ???xDu f5 =-5 w5 ???xC7
	d2P[4]  -= 5 * d5 * C;         // DudZ f5 =-5 w5' C7
	tm       = 5 * A2 * hB2;
	double w7= tm*Z*Z*Z*iz;          // w7 =5a??BZ??/??
	double d7= tm*(Z+Z+A-A2*ai*iz);  // w7'=5a??B(2Z+a-a??/????)
	double t7= (tm+tm)*(1+ai*ai*ai); // w7"=10a??B(1+a??/????)
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	d1P[2]  += d7 * C;             // dZ   f7 = w7' C7
	d2P[0]  += w7 * d2C;           // ???x??  f7 = w7 ???x??C7
	d2P[2]  += d7 * d1C;           // ???xdZ f7 = w7' ???xC7
	d2P[5]  += t7 * C;             // dZ??  f7 = w7" C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C);
	d2P[3]  += 35 * w5 * C;        // Du??  f5 = 35 * w5 * C9
	d1P[1]  -= 7 * w7 * C;         // Du   f5 =-7 w7 C9
	d2P[1]  -= 7 * w7 * d1C;       // ???xDu f5 =-7 w7 ???xC9
	d2P[4]  -= 7 * d7 * C;         // DudZ f5 =-7 w7' C9
	F.incOrder();            // F = C11 = -M/r^11
	C        = F();
	d2P[3]  += 63 * w7 * C;        // Du??  f7 = 63 * w7 * C11
	convert3D<true,true>(d1P,d2P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= aZ-0.333333333333333*A2+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[2]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	double w5= aZ * (aZ + 3*hB2);
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d2P[0]  += w5 * d2C;           // ???x??  f5 = w5 ???x??C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C,d2C);
	d2P[2]  += 15 * w3 * C;        // Du??  f3 = 15 * w3 * C7
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	d2P[1]  -= 5 * w5 * d1C;       // ???xDu f5 =-5 w5 ???xC7
	double w7= 5 * A2 * hB2*Z*Z*Z*iz;           // w7 =5a??BZ??/??
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	d2P[0]  += w7 * d2C;           // ???x??  f7 = w7 ???x??C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C);
	d2P[2]  += 35 * w5 * C;        // Du??  f5 = 35 * w5 * C9
	d1P[1]  -= 7 * w7 * C;         // Du   f5 =-7 w7 C9
	d2P[1]  -= 7 * w7 * d1C;       // ???xDu f5 =-7 w7 ???xC9
	F.incOrder();            // F = C11 = -M/r^11
	C        = F();
	d2P[2]  += 63 * w7 * C;        // Du??  f7 = 63 * w7 * C11
	convert2D(d1P,d2P,y);
	return P;
    }

    double phi3D(double x, double y, double Z,
	double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= aZ-0.333333333333333*A2+hB2;
	double d3= A;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	double w5= aZ * (aZ + 3*hB2);
	double d5= A * (aZ + aZ + 3*hB2);
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;             // dZ   f5 = w5' C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C);
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	double tm= 5 * A2 * hB2;
	double w7= tm*Z*Z*Z*iz;          // w7 =5a??BZ??/??
	double d7= tm*(Z+Z+A-A2*A*iz*iz);  // w7'=5a??B(2Z-a-a??/????)
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	d1P[2]  += d7 * C;             // dZ   f7 = w7' C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C);
	d1P[1]  -= 7 * w7 * C;         // Du   f5 =-7 w7 C9
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= aZ-0.333333333333333*A2+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	double w5= aZ * (aZ + 3*hB2);
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C);
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	double w7= 5 * A2 * hB2 *Z*Z*Z*iz;           // w7 =5a??BZ??/??
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C);
	d1P[1]  -= 7 * w7 * C;         // Du   f5 =-7 w7 C9
	convert2D(d1P,y);
	return P;
    }

    double phi(double x, double y, double Z) const noexcept
    {
	const double aZ=A*Z;
	psiFunc F{1,x,y,Z,this}; // F = C1
	double P = F();
	F.incOrder();            // F = C3 = -M/r^3
	P       += (A*(Z-0.333333333333333*A)+hB2) * F();
	F.incOrder();            // F = C5 = -M/r^5
	P       += aZ * (aZ + 3*hB2) * F();
	F.incOrder();            // F = C7 = -M/r^7
	P       += 5 * A * A * hB2 *Z*Z*Z * F()/(Z-A);
	return P;
    }
};  // struct inputModel<V3>

// type V4:
// ??(x,y,Z) = -M (w1 C1 + w3 C3 + w5 C5 + w7 C7 + w9 C9)
// w1 =1  w3 =a(Z-2a/5)+B  w5 =0.6a??Z(2Z-a)+3aB(Z-0.2a)
// w1'=0  w3'=a            w5'=0.6a?? (4Z-a)+3aB
// w1"=0  w3"=0            w5"=2.4a??
// w7 =Z?? a??(aZ+6B)        w9 =7a??B K9   K9 =    Z???     /??
// w7'=3Z a??(aZ+4B)        w9'=7a??B K9'  K9'=  (4Z??-K9 )/??
// w7"=6  a??(aZ+2B)        w9"=7a??B K9"  K9"= 2(6Z??-K9')/??
template<typename psiFunc> struct inputModel<psiFunc,V4>
  : internalPars
{
    using internalPars::A;
    using internalPars::B;
    const double hB2;

    static constexpr auto Type = V4;

    template<typename pars>
    explicit inputModel(const pars&p)
      : internalPars(p), hB2(0.5*B*B)
    {
	assert(type(p) == Type);
	psiFunc::assertPars(this);
    }

    double phi3D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= aZ-0.4*A2+hB2;
	double d3= A;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	d2P[2]   = d3 * d1C;           // ???xdZ f3 = w3' ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[3]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	d2P[4]   =-3 * d3 * C;         // DudZ f3 =-3 w3' C5
	double w5= 0.6*A*aZ*(Z+Z-A)+3*A*hB2*(Z-0.2*A);
	double d5= 0.6*A2  *(4*Z-A)+3*A*hB2;
	double t5= 2.4*A2;
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;             // dZ   f5 = w5' C5
	d2P[0]  += w5 * d2C;           // ???x??  f5 = w5 ???x??C5
	d2P[2]  += d5 * d1C;           // ???xdZ f5 = w5' ???xC5
	d2P[5]   = t5 * C;             // dZ??  f5 = w5" C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C,d2C);
	d2P[3]  += 15 * w3 * C;        // Du??  f3 = 15 * w3 * C7
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	d2P[1]  -= 5 * w5 * d1C;       // ???xDu f5 =-5 w5 ???xC7
	d2P[4]  -= 5 * d5 * C;         // DudZ f5 =-5 w5' C7
	double w7= aZ *aZ*(aZ+6*hB2);
	double d7= 3*A*aZ*(aZ+4*hB2);
	double t7= 6*A2  *(aZ+2*hB2);
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	d1P[2]  += d7 * C;             // dZ   f7 = w7' C7
	d2P[0]  += w7 * d2C;           // ???x??  f7 = w7 ???x??C7
	d2P[2]  += d7 * d1C;           // ???xdZ f7 = w7' ???xC7
	d2P[5]  += t7 * C;             // dZ??  f7 = w7" C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C,d2C);
	d2P[3]  += 35 * w5 * C;        // Du??  f5 = 35 * w5 * C9
	d1P[1]  -= 7 * w7 * C;         // Du   f7 =-7 w7 C9
	d2P[1]  -= 7 * w7 * d1C;       // ???xDu f7 =-7 w7 ???xC9
	d2P[4]  -= 7 * d7 * C;         // DudZ f7 =-7 w7' C9
	double tm= 7 * A2 * A * hB2;
	double Z2= Z*Z;
	double w9= Z2*Z2*iz;
	double d9= (4*Z2*Z-w9)*iz;
	double t9= 2*tm*(6*Z2-d9)*iz;
	w9      *= tm;
	d9      *= tm;
	P       += w9 * C;             //      f9 = w9 C9
	d1P[0]  += w9 * d1C;           // ???x   f9 = w9 ???xC9
	d1P[2]  += d9 * C;             // dZ   f9 = w9' C9
	d2P[0]  += w9 * d2C;           // ???x??  f9 = w9 ???x??C9
	d2P[2]  += d9 * d1C;           // ???xdZ f9 = w9' ???xC9
	d2P[5]  += t9 * C;             // dZ??  f9 = w9" C9
	F.incOrder();            // F = C11 = -M/r^11
	C        = F(d1C);
	d2P[3]  += 63 * w7 * C;        // Du??  f7 = 63 * w7 * C11
	d1P[1]  -= 9 * w9 * C;         // Du   f9 =-9 w9 C9
	d2P[1]  -= 9 * w9 * d1C;       // ???xDu f9 =-9 w9 ???xC9
	d2P[4]  -= 9 * d9 * C;         // DudZ f9 =-9 w9' C9
	F.incOrder();            // F = C13 = -M/r^13
	C        = F();
	d2P[3]  += 99 * w9 * C;        // Du??  f9 = 99 * w9 * C13
	convert3D<true,true>(d1P,d2P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z,
	double*d1P, double*d2P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C,d2C;
	double P = F(d1C,d2C);         //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	d2P[0]   = d2C;                // ???x??  f1 = ???x??C1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C,d2C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	d2P[1]   =-d1C;                // ???xDu f1 = -???xC3
	double w3= aZ-0.4*A2+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d2P[0]  += w3 * d2C;           // ???x??  f3 = w3 ???x??C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C,d2C);
	d2P[2]   = 3 * C;              // Du??  f1 = 3 * C5
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	d2P[1]  -= 3 * w3 * d1C;       // ???xDu f3 =-3 w3 ???xC5
	double w5= 0.6*A*aZ*(Z+Z-A)+3*A*hB2*(Z-0.2*A);
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d2P[0]  += w5 * d2C;           // ???x??  f5 = w5 ???x??C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C,d2C);
	d2P[2]  += 15 * w3 * C;        // Du??  f3 = 15 * w3 * C7
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	d2P[1]  -= 5 * w5 * d1C;       // ???xDu f5 =-5 w5 ???xC7
	double w7= aZ *aZ*(aZ+6*hB2);
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	d2P[0]  += w7 * d2C;           // ???x??  f7 = w7 ???x??C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C,d2C);
	d2P[2]  += 35 * w5 * C;        // Du??  f5 = 35 * w5 * C9
	d1P[1]  -= 7 * w7 * C;         // Du   f7 =-7 w7 C9
	d2P[1]  -= 7 * w7 * d1C;       // ???xDu f7 =-7 w7 ???xC9
	double w9= 7 * A2 * A * hB2*square(square(Z))*iz;
	P       += w9 * C;             //      f9 = w9 C9
	d1P[0]  += w9 * d1C;           // ???x   f9 = w9 ???xC9
	d2P[0]  += w9 * d2C;           // ???x??  f9 = w9 ???x??C9
	F.incOrder();            // F = C11 = -M/r^11
	C        = F(d1C);
	d2P[2]  += 63 * w7 * C;        // Du??  f7 = 63 * w7 * C11
	d1P[1]  -= 9 * w9 * C;         // Du   f9 =-9 w9 C9
	d2P[1]  -= 9 * w9 * d1C;       // ???xDu f9 =-9 w9 ???xC9
	F.incOrder();            // F = C13 = -M/r^13
	C        = F();
	d2P[2]  += 99 * w9 * C;        // Du??  f9 = 99 * w9 * C13
	convert2D(d1P,d2P,y);
	return P;
    }

    double phi3D(double x, double y, double Z, double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= aZ-0.4*A2+hB2;
	double d3= A;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	d1P[2]   = d3 * C;             // dZ   f3 = w3' C3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	double w5= 0.6*A*aZ*(Z+Z-A)+3*A*hB2*(Z-0.2*A);
	double d5= 0.6*A2  *(4*Z-A)+3*A*hB2;
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	d1P[2]  += d5 * C;             // dZ   f5 = w5' C5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C);
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	double w7= aZ *aZ*(aZ+6*hB2);
	double d7= 3*A*aZ*(aZ+4*hB2);
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	d1P[2]  += d7 * C;             // dZ   f7 = w7' C7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C);
	d1P[1]  -= 7 * w7 * C;         // Du   f7 =-7 w7 C9
	double tm= 7 * A2 * A * hB2;
	double Z2= Z*Z;
	double w9= Z2*Z2*iz;
	double d9= tm*(4*Z2*Z-w9)*iz;
	w9      *= tm;
	P       += w9 * C;             //      f9 = w9 C9
	d1P[0]  += w9 * d1C;           // ???x   f9 = w9 ???xC9
	d1P[2]  += d9 * C;             // dZ   f9 = w9' C9
	F.incOrder();            // F = C11 = -M/r^11
	C        = F(d1C);
	d1P[1]  -= 9 * w9 * C;         // Du   f9 =-9 w9 C9
	F.incOrder();            // F = C13 = -M/r^13
	C        = F();
	convert3D<true>(d1P,y,Z);
	return P;
    }

    double phi2D(double x, double y, double Z, double*d1P) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double d1C;
	double P = F(d1C);             //      f1 = C1
	d1P[0]   = d1C;                // ???x   f1 = ???xC1
	F.incOrder();            // F = C3 = -M/r^3
	double C = F(d1C);
	d1P[1]   =-C;                  // Du   f1 = -C3
	double w3= aZ-0.4*A2+hB2;
	P       += w3 * C;             //      f3 = w3 C3
	d1P[0]  += w3 * d1C;           // ???x   f3 = w3 ???xC3
	F.incOrder();            // F = C5 = -M/r^5
	C        = F(d1C);
	d1P[1]  -= 3 * w3 * C;         // Du   f3 =-3 w3 C5
	double w5= 0.6*A*aZ*(Z+Z-A)+3*A*hB2*(Z-0.2*A);
	P       += w5 * C;             //      f5 = w5 C5
	d1P[0]  += w5 * d1C;           // ???x   f5 = w5 ???xC5
	F.incOrder();            // F = C7 = -M/r^7
	C        = F(d1C);
	d1P[1]  -= 5 * w5 * C;         // Du   f5 =-5 w5 C7
	double w7= aZ *aZ*(aZ+6*hB2);
	P       += w7 * C;             //      f7 = w7 C7
	d1P[0]  += w7 * d1C;           // ???x   f7 = w7 ???xC7
	F.incOrder();            // F = C9 = -M/r^9
	C        = F(d1C);
	d1P[1]  -= 7 * w7 * C;         // Du   f7 =-7 w7 C9
	double w9= 7*A2*A*hB2*square(square(Z))*iz;
	P       += w9 * C;             //      f9 = w9 C9
	d1P[0]  += w9 * d1C;           // ???x   f9 = w9 ???xC9
	F.incOrder();            // F = C11 = -M/r^11
	C        = F(d1C);
	d1P[1]  -= 9 * w9 * C;         // Du   f9 =-9 w9 C9
	F.incOrder();            // F = C13 = -M/r^13
	C        = F();
	convert2D(d1P,y);
	return P;
    }

    double phi(double x, double y, double Z) const noexcept
    {
	const double iz=1/(Z-A);
	const double aZ=A*Z;
	const double A2=A*A;
	psiFunc F{1,x,y,Z,this}; // F = C1 = M/r
	double P = F();
	F.incOrder();            // F = C3 = -M/r^3
	P       += (aZ-0.4*A2+hB2) * F();
	F.incOrder();            // F = C5 = -M/r^5
	P       += (0.6*A*aZ*(Z+Z-A)+3*A*hB2*(Z-0.2*A)) * F();
	F.incOrder();            // F = C7 = -M/r^7
	P       += aZ *aZ*(aZ+6*hB2) * F();
	F.incOrder();            // F = C9 = -M/r^9
	P       += 7*A2*A*hB2*square(square(Z))*iz * F();
	return P;
    }

};  // struct inputModel<V4>

template<typename psiFunc, int Type, bool Xalgined>
struct singleModel;

/// a single model with ???? = 0
template<typename psiFunc, int Type>
struct singleModel<psiFunc, Type, true>
  : discBar::model
  , inputModel<psiFunc,Type>
{
    using InputModel = inputModel<psiFunc,Type>;
    using InputModel::A;
    using InputModel::B;

    template<typename Pars>
    explicit singleModel(Pars const&pars, bool checkAlgined=true) noexcept
      : InputModel(pars)
    { if(checkAlgined) assert(pars.P <= 0 && pars.P >= 0); }

    size_t numRazorThin() const noexcept override
    { return internalPars::isRazorThin()? 1 : 0; }

    size_t numBarred() const noexcept override
    { return internalPars::isBarred()? 1 : 0; }

    size_t numSpherical() const noexcept override
    { return internalPars::isSpherical()? 1 : 0; }
       
    string modelType() const override
    { return getName(internalPars::T); }

    double totalMass() const noexcept override
    { return internalPars::M; }

    void rescaleM(double factor) noexcept override
    { internalPars::rescaleMass(factor); }

    void rescaleH(double factor) noexcept override
    { internalPars::rescaleHeight(factor); }

    void rescaleL(double factor) noexcept override
    { internalPars::rescaleLength(factor); }

    parameters params() const override
    { return internalPars::params(); }

    internalPars const&iparams() const
    { return *this; }

    void scanParameters(
	std::function<void(parameters const&)> const&func) const override
    { func(params()); }

    modelPtr clone() const override
    { return make_unique<singleModel>(iparams()); }

    void flipSign() noexcept override
    { internalPars::flipSign(); }

    double pot3D(const double*x) const noexcept override
    { return InputModel::phi(x[0],x[1],hypot(x[2],B)+A); }

    double pot3D(const double*x, double*d1P) const noexcept override
    {
	if(B <= 0) {
	    double Z = abs(x[2]);
	    double P = InputModel::phi3D(x[0],x[1],Z+A,d1P);
	    d1P[2]  *= sign(x[2]);
	    return P;
	} else {
	    double Z = hypot(x[2],B);
	    double P = InputModel::phi3D(x[0],x[1],Z+A,d1P);
	    d1P[2]  *= x[2]/Z;
	    return P;
	}
    }

    double pot3D(const double*x, double*d1P, double*d2P) const noexcept override
    {
	if(B <= 0) {
	    double Z = abs(x[2]);
	    double P = InputModel::phi3D(x[0],x[1],Z+A,d1P,d2P);
	    double d1Z = sign(x[2]);
	    d1P[2] *= d1Z;               // ?????/???z    = ?????/???Z Z'
	    d2P[2] *= d1Z;               // ???????/???x???z = ???????/???x???Z Z'
	    d2P[4] *= d1Z;               // ???????/???y???z = ???????/???y???Z Z'
	    d2P[5] *= d1Z * d1Z;         // ???????/???z??  = ???????/???Z?? Z'??
	    return P;
	} else {
	    double Z = hypot(x[2],B);
	    double P = InputModel::phi3D(x[0],x[1],Z+A,d1P,d2P);
	    double d1Z = 1/Z;
	    double d2Z = B*B * d1Z*d1Z*d1Z;
	    d1Z    *= x[2];
	    d2P[2] *= d1Z;               // ???????/???x???z = ???????/???x???Z Z'
	    d2P[4] *= d1Z;               // ???????/???y???z = ???????/???y???Z Z'
	    d2P[5] *= d1Z * d1Z;
	    d2P[5] += d1P[2] * d2Z;      // ???????/???z??  = ???????/???Z?? Z'?? + ?????/???Z Z"
	    d1P[2] *= d1Z;               // ?????/???z    = ?????/???Z Z'
	    return P;
	}
    }

    double pot2D(const double*x) const noexcept override
    { return InputModel::phi(x[0],x[1],A+B); }

    double pot2D(const double*x, double*d1P) const noexcept override
    { return InputModel::phi2D(x[0],x[1],A+B,d1P); }

    double pot2D(const double*x, double*d1P, double*d2P) const noexcept override
    { return InputModel::phi2D(x[0],x[1],A+B,d1P,d2P); }

    void dump(std::ostream&out) const override
    {
	out << " discBar::singleModel<"
	    << psiFunc::name() << ", " << getName(Type) << ", 1>:";
	internalPars::dump(out);
    }

};  // struct singleModel<Xaligned = true>

/// a single model with ???? ??? 0
template<typename psiFunc, int Type>
struct singleModel<psiFunc, Type, false>
  : discBar::model
{
    using alignedModel = singleModel<psiFunc,Type,true>;
    
    alignedModel F;
    const double C,S;
    
    template<typename Pars>
    explicit singleModel(Pars const&pars) noexcept
      : F(pars,false), C(cos(pars.P)), S(sin(pars.P))
    {
	assert(pars.L > 0);
	assert(pars.P > 0 || pars.P < 0);
    }

    size_t numRazorThin() const noexcept override
    { return F.numRazorThin(); }

    size_t numBarred() const noexcept override
    { return F.numBarred(); }

    size_t numSpherical() const noexcept override
    { return F.numSpherical(); }
       
    size_t numComponents() const noexcept override
    { return 1; }

    string modelType() const override
    { return F.modelType(); }

    double totalMass() const noexcept override
    { return 2 * F.totalMass(); }

    void rescaleM(double factor) noexcept override
    { F.rescaleM(factor); }

    void rescaleH(double factor) noexcept override
    { F.rescaleH(factor); }

    void rescaleL(double factor) noexcept override
    { F.rescaleL(factor); }

    parameters params() const override
    {
	auto pars = F.params();
	pars.M *= 2.0;
	return pars;
    }

    void scanParameters(
	std::function<void(parameters const&)> const&func) const override
    { func(params()); }

    modelPtr clone() const override
    { return make_unique<singleModel>(F.iparams()); }

    void flipSign() noexcept override
    { F.flipSign(); }

    void rotateForward(const double*x, double*X) const noexcept
    {
	X[0] = C * x[0] + S * x[1];
	X[1] = C * x[1] - S * x[0];
    }

    void rotateBackward(const double*x, double*X) const noexcept
    {
	X[0] = C * x[0] - S * x[1];
	X[1] = C * x[1] + S * x[0];
    }

    double pot2D(const double*x) const noexcept override
    {
	double X[2];
	rotateForward(x,X);
	double P = F.pot2D(X);
	rotateBackward(x,X);
	return P + F.pot2D(X);
    }

    double pot2D(const double*x, double*d1P) const noexcept override
    {
	double X[2];
	rotateForward(x,X);
	double d1Q[2], P = F.pot2D(X,d1Q);
	d1P[0]  = C * d1Q[0] - S * d1Q[1];
	d1P[1]  = C * d1Q[1] + S * d1Q[0];
	rotateBackward(x,X);
	P      += F.pot2D(X,d1Q);
	d1P[0] += C * d1Q[0] + S * d1Q[1];
	d1P[1] += C * d1Q[1] - S * d1Q[0];
	return P;
    }

    double pot2D(const double*x, double*d1P, double*d2P) const noexcept override
    {
	const double C2=C*C, S2=S*S, CS=C*S;
	double X[2];
	rotateForward(x,X);
	double d1Q[2],d2Q[3], P = F.pot2D(X,d1Q,d2Q);
	d1P[0]  = C * d1Q[0] - S * d1Q[1];
	d1P[1]  = C * d1Q[1] + S * d1Q[0];
	d2P[0]  = C2    *d2Q[0] - 2*CS* d2Q[1] + S2*d2Q[2];
	d2P[1]  =(C2-S2)*d2Q[1] +   CS*(d2Q[0] -    d2Q[2]);
	d2P[2]  = S2    *d2Q[0] + 2*CS* d2Q[1] + C2*d2Q[2];
	rotateBackward(x,X);
	P      += F.pot2D(X,d1Q,d2Q);
	d1P[0] += C * d1Q[0] + S * d1Q[1];
	d1P[1] += C * d1Q[1] - S * d1Q[0];
	d2P[0] += C2    *d2Q[0] + 2*CS* d2Q[1] + S2*d2Q[2];
	d2P[1] +=(C2-S2)*d2Q[1] -   CS*(d2Q[0] -    d2Q[2]);
	d2P[2] += S2    *d2Q[0] - 2*CS* d2Q[1] + C2*d2Q[2];
	return P;
    }

    double pot3D(const double*x) const noexcept override
    {
	double X[3]; X[2]=x[2];
	rotateForward(x,X);
	double P = F.pot3D(X);
	rotateBackward(x,X);
	return P + F.pot3D(X);
    }

    double pot3D(const double*x, double*d1P) const noexcept override
    {
	double X[3]; X[2]=x[2];
	rotateForward(x,X);
	double d1Q[3], P = F.pot3D(X,d1Q);
	d1P[0]  = C * d1Q[0] - S * d1Q[1];
	d1P[1]  = C * d1Q[1] + S * d1Q[0];
	d1P[2]  = d1Q[2];
	rotateBackward(x,X);
	P      += F.pot3D(X,d1Q);
	d1P[0] += C * d1Q[0] + S * d1Q[1];
	d1P[1] += C * d1Q[1] - S * d1Q[0];
	d1P[2] += d1Q[2];
	return P;
    }

    double pot3D(const double*x, double*d1P, double*d2P) const noexcept override
    {
	const double C2=C*C, S2=S*S, CS=C*S;
	double X[3]; X[2]=x[2];
	rotateForward(x,X);
	double d1Q[3],d2Q[6], P = F.pot3D(X,d1Q,d2Q);
	d1P[0]  = C * d1Q[0] - S * d1Q[1];
	d1P[1]  = C * d1Q[1] + S * d1Q[0];
	d1P[2]  = d1Q[2];
	d2P[0]  = C2    *d2Q[0] - 2*CS* d2Q[1] + S2*d2Q[3];
	d2P[1]  =(C2-S2)*d2Q[1] +   CS*(d2Q[0] -    d2Q[3]);
	d2P[2]  = C * d2Q[2] - S * d2Q[4];
	d2P[3]  = S2    *d2Q[0] + 2*CS* d2Q[1] + C2*d2Q[3];
	d2P[4]  = C * d2Q[4] + S * d2Q[2];
	d2P[5]  = d2Q[5];
	rotateBackward(x,X);
	P      += F.pot3D(X,d1Q,d2Q);
	d1P[0] += C * d1Q[0] + S * d1Q[1];
	d1P[1] += C * d1Q[1] - S * d1Q[0];
	d1P[2] += d1Q[2];
	d2P[0] += C2    *d2Q[0] + 2*CS* d2Q[1] + S2*d2Q[3];
	d2P[1] +=(C2-S2)*d2Q[1] -   CS*(d2Q[0] -    d2Q[3]);
	d2P[2] += C * d2Q[2] + S * d2Q[4];
	d2P[3] += S2    *d2Q[0] - 2*CS* d2Q[1] + C2*d2Q[3];
	d2P[4] += C * d2Q[4] - S * d2Q[2];
	d2P[5] += d2Q[5];
	return P;
    }

    void dump(std::ostream&out) const override
    {
	out << " discBar::singleModel<"
	    << psiFunc::name() << ", " << getName(Type) << ", 0>:";
	static_cast<internalPars const&>(F).dump(out);
    }
};  // struct singleModel<Xaligned = false>

template<int Type, typename Pars>
modelPtr makeSingleT(Pars const&pars)
{
    if (pars.L <= 0.0)
	// axisymmetric
	return        make_unique<singleModel<rFunc,Type,1>>(pars);
    const auto P0 = pars.P <=0 && pars.P >= 0;
    if (pars.G <= 0.0 && pars.G >= 0.0) {
	// barred with constant needle-density
	if(P0) return make_unique<singleModel<aFunc,Type,1>>(pars);
	else   return make_unique<singleModel<aFunc,Type,0>>(pars);
    } else {
 	// barred with linear needle-density
	if(P0) return make_unique<singleModel<bFunc,Type,1>>(pars);
	else   return make_unique<singleModel<bFunc,Type,0>>(pars);
    }
}

template<typename Pars>
modelPtr makeSingle(Pars const&pars, int T)
{
    switch(T) {
    case T1: return makeSingleT<T1>(pars);
    case T2: return makeSingleT<T2>(pars);
    case T3: return makeSingleT<T3>(pars);
    case T4: return makeSingleT<T4>(pars);
    case V1: return makeSingleT<V1>(pars);
    case V2: return makeSingleT<V2>(pars);
    case V3: return makeSingleT<V3>(pars);
    case V4: return makeSingleT<V4>(pars);
    default: throw runtime_error("discBar: model '" + pars.modelType()
	+ "' not supported");
    }
}

inline modelPtr makeSingle(internalPars const&pars)
{
    return makeSingle(pars,pars.T);
}

inline modelPtr makeSingle(parameters const&pars)
{
    return makeSingle(pars,getType(pars.T));
}
    
/// a collection of individual models
struct collectionModel
  : model
{
    using collPtr = unique_ptr<collectionModel>;

    size_t numComponents() const noexcept override
    { return models.size(); }

    size_t numRazorThin() const noexcept override
    { return numThin; }

    size_t numBarred() const noexcept override
    { return numBars; }

    size_t numSpherical() const noexcept override
    { return numSphr; }
       
    double totalMass() const noexcept override
    { return Mtotal; }

    void rescaleM(double factor) noexcept override
    {
	for(auto&mod : models)
	    mod->rescaleM(factor);
	Mtotal *= factor;
    }

    void rescaleH(double factor) noexcept override
    {
	for(auto&mod : models)
	    mod->rescaleH(factor);
    }

    void rescaleL(double factor) noexcept override
    {
	for(auto&mod : models)
	    mod->rescaleL(factor);
    }

    void scanParameters(
	std::function<void(parameters const&)> const&func) const override
    {
	for(const auto&mod : models)
	    mod->scanParameters(func);
    }

    double pot3D(const double*x) const noexcept override
    {
	auto it = models.begin();
	double P = (*it)->pot3D(x);
	for(++it; it != models.end(); ++it)
	    P += (*it)->pot3D(x);
	return P;
    }

    double pot3D(const double*x, double*d1P) const noexcept override
    {
	auto it = models.begin();
	double P = (*it)->pot3D(x,d1P);
	for(++it; it!=models.end(); ++it) {
	    double d1[3];
	    P += (*it)->pot3D(x,d1);
	    d1P[0] += d1[0];
	    d1P[1] += d1[1];
	    d1P[2] += d1[2];
	}
	return P;
    }

    double pot3D(const double*x, double*d1P, double*d2P) const noexcept override
    {
	auto it = models.begin();
	double P = (*it)->pot3D(x,d1P,d2P);
	for(++it; it!=models.end(); ++it) {
	    double d1[3], d2[6];
	    P += (*it)->pot3D(x,d1,d2);
	    d1P[0] += d1[0];
	    d1P[1] += d1[1];
	    d1P[2] += d1[2];
	    d2P[0] += d2[0];
	    d2P[1] += d2[1];
	    d2P[2] += d2[2];
	    d2P[3] += d2[3];
	    d2P[4] += d2[4];
	    d2P[5] += d2[5];
	}
	return P;
    }

    double pot2D(const double*x) const noexcept override
    {
	auto it = models.begin();
	double P = (*it)->pot2D(x);
	for(++it; it!=models.end(); ++it)
	    P += (*it)->pot2D(x);
	return P;
    }

    double pot2D(const double*x, double*d1P) const noexcept override
    {
	auto it = models.begin();
	double P = (*it)->pot2D(x,d1P);
	for(++it; it!=models.end(); ++it) {
	    double d1[2];
	    P += (*it)->pot2D(x,d1);
	    d1P[0] += d1[0];
	    d1P[1] += d1[1];
	}
	return P;
    }

    double pot2D(const double*x, double*d1P, double*d2P) const noexcept override
    {
	auto it = models.begin();
	double P = (*it)->pot2D(x,d1P,d2P);
	for(++it; it!=models.end(); ++it) {
	    double d1[2], d2[3];
	    P += (*it)->pot2D(x,d1,d2);
	    d1P[0] += d1[0];
	    d1P[1] += d1[1];
	    d2P[0] += d2[0];
	    d2P[1] += d2[1];
	    d2P[2] += d2[2];
	}
	return P;
    }

    void flipSign() noexcept override
    {
	for(auto const&mod : models)
	    mod->flipSign();
    }

    bool isCollection() const noexcept override
    { return true; }

    collPtr cloneImpl(size_t reserveExtra, bool flipSign = false) const noexcept
    {
	auto coll = make_unique<collectionModel>
	    (reserveExtra + numComponents());
	for(const auto&mod : models)
	    coll->appendSingle(mod->clone(),flipSign);
	return coll;
    }

    modelPtr clone() const noexcept override
    { return cloneImpl(0); }

    void appendSingle(modelPtr const&mod, bool flipSign = false)
    {
	models.push_back(mod->clone());
	auto&appended = models.back();
	if( flipSign )
	    appended->flipSign();
	Mtotal  += appended->totalMass();
	numThin += appended->numRazorThin();
	numSphr += appended->numSpherical();
	numBars += appended->numBarred();
    }

    void appendSingle(modelPtr &&mod, bool flipSign = false)
    {
	models.push_back(std::move(mod));
	auto&appended = models.back();
	if( flipSign )
	    appended->flipSign();
	Mtotal  += appended->totalMass();
	numThin += appended->numRazorThin();
	numSphr += appended->numSpherical();
	numBars += appended->numBarred();
    }

    void append(modelPtr const&mod, bool flipSign = false)
    {
	if(mod->isCollection()) {
	    auto coll = static_cast<const collectionModel*>(mod.get());
	    for(const auto&submod : coll->models)
		appendSingle(submod,flipSign);
	} else
	    appendSingle(mod,flipSign);
    }

    void append(modelPtr &&mod, bool flipSign = false)
    {
	if(mod->isCollection()) {
	    auto coll = static_cast<collectionModel*>(mod.get());
	    for(auto&&submod : coll->models)
		appendSingle(std::move(submod),flipSign);
	    coll->models.clear();
	    coll->Mtotal = 0.0;
	} else
	    appendSingle(std::move(mod),flipSign);
    }

    collectionModel(size_t reserve = 2)
    { models.reserve(reserve); }
    
  private:
    std::vector<modelPtr> models;
    double Mtotal  = 0.0;
    size_t numThin = 0;
    size_t numSphr = 0;
    size_t numBars = 0;
};  // struct collectionModel

using collPtr = unique_ptr<collectionModel>;

collPtr makeHollowDisc(parameters const&p, double holeRadius)
{
    p.sanityCheck();
    if(p.T[1]=='0' || (p.T[0]!='T' && p.T[0]!='V'))
	throw runtime_error("discBar::makeHoledDisc(): model type '" + p.T
	    + "' not supported");
    internalPars pars(p);
    double b1=pars.B, s1=pars.A + b1;
    if(holeRadius >= s1)
	throw runtime_error("discBar::makeHoledDisc(): holeRadius="
	    + to_string(holeRadius) + " ??? scale-radius s = " + to_string(s1));
    if(holeRadius <= b1)
	throw runtime_error("discBar::makeHoledDisc(): holeRadius="
	    + to_string(holeRadius) + " ??? scale-height b = " + to_string(b1));
    auto disc1 = makeSingle(pars);
    pars.A = holeRadius - b1;
    auto disc2 = makeSingle(pars);
    const double x[3] = {0.0,0.0,0.0};
    disc2->rescaleMass(-disc1->density(x)/disc2->density(x));

    collPtr coll = make_unique<collectionModel>();
    coll->append(disc1);
    coll->append(disc2);
    return coll;
}

}   // namespace {

namespace discBar {
    
#define checkFinite(value,name,caller)		       		\
    if(isinf(value) || isnan(value))	       			\
	throw runtime_error(string(caller) + ": " + name + '='	\
	    + to_string(value) + " is not finite");

parameters const&
parameters::sanityCheck(const char* caller) const
{
    checkFinite(M, 'M', caller ? caller : "discBar::parameters");
    checkFinite(A, 'a', caller ? caller : "discBar::parameters");
    checkFinite(B, 'b', caller ? caller : "discBar::parameters");
    checkFinite(L, 'L', caller ? caller : "discBar::parameters");
    checkFinite(G, "??", caller ? caller : "discBar::parameters");
    checkFinite(P, "??", caller ? caller : "discBar::parameters");

#undef checkFinite

    if(G < -1)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": ??=" + to_string(G) + " < -1"); 
    if(G > 1)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": ??=" + to_string(G) + " > 1");
    if(A < 0)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": a=" + to_string(A) + " < 0");
    if(B < 0)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": b=" + to_string(B) + " < 0");
    if(A+B <= 0)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": a+b=" + to_string(A+B) + " ??? 0");
    if(L < 0)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": L=" + to_string(L) + " < 0");
    if(M <= 0 && M >= 0)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": M=" + to_string(M) + " = 0");

    if( T.length() != 2 ||
	string("TV").find(T[0]) == string::npos ||
	string("1234").find(T[1]) == string::npos)
	throw runtime_error(string(caller?caller:"orbit::discParameters")
	    + ": model type '" + T + "' unknown");
    return*this;
}

//  this is non-inline to force the generation of typyinfo and vtable
model::~model() {}

void model::rescaleMass(double factor)
{
    if(factor <= 0 && factor >= 0)
	throw runtime_error("mass re-scale factor = 0");
    rescaleM(factor);
}

void model::rescaleHeight(double factor)
{
    if(factor <= 0 && factor >= 0)
	throw runtime_error("height re-scale factor = 0");
    rescaleH(factor);
}

void model::rescaleLength(double factor)
{
    if(factor <= 0 && factor >= 0)
	throw runtime_error("length re-scale factor = 0");
    rescaleL(factor);
}

void model::rescaleSize(double factor)
{
    if(factor <= 0 && factor >= 0)
	throw runtime_error("size re-scale factor = 0");
    rescaleH(factor);
    rescaleL(factor);
}

double model::density(const double*x) const
{
    if(numRazorThin() == 0) {
	double d1P[3],d2P[6];
	potential(x,d1P,d2P);
	return 0.25 * (d2P[0] + d2P[3] + d2P[5]) / Pi;
    } else if(numRazorThin() == numComponents()) {
	double X[3]={x[0],x[1],0.0}, d1P[3];
	potential(X,d1P);
	return 0.5 * d1P[2] / Pi;
    } else
	throw runtime_error("discBar::model::density() "
	    "called for partially razor-thin model");
}

modelPtr makeSingleModel(parameters const&pars, bool check)
{
    if(check)
	pars.sanityCheck();
    return makeSingle(pars);
}

modelPtr sum(modelPtr const&a, modelPtr const&b)
{
    collPtr coll;
    if(a->isCollection()) {
	coll = static_cast<const collectionModel*>(a.get())->
	    cloneImpl(b->numComponents());
	coll->append(b);
    } else if(b->isCollection()) {
	coll = static_cast<const collectionModel*>(b.get())->
	    cloneImpl(a->numComponents());
	coll->append(a);
    } else {
	coll = make_unique<collectionModel>();
	coll->append(a);
	coll->append(b);
    }	
    return coll;
}

modelPtr diff(modelPtr const&a, modelPtr const&b)
{
    collPtr coll;
    if(a->isCollection()) {
	coll = static_cast<const collectionModel*>(a.get())->
	    cloneImpl(b->numComponents());
	coll->append(b,true);
    } else if(b->isCollection()) {
	coll = static_cast<const collectionModel*>(b.get())->
	    cloneImpl(a->numComponents(), true);
	coll->append(a);
    } else {
	coll = make_unique<collectionModel>();
	coll->append(a);
	coll->append(b,true);
    }	
    return coll;
}

modelPtr makeHoledDisc(parameters const&pars, double holeRadius)
{
    return makeHollowDisc(pars,holeRadius);
}

modelPtr makeBulgeBar()
{
    parameters pars;
    // hollow disc
    pars.T = "V2";
    pars.setSQ(1.6, 0.05);
    auto disc = makeHollowDisc(pars, 1.2);
    // nuclear disc
    pars.T = "V4";
    pars.M = 0.01;
    pars.A = 0.1;
    pars.B = 0.1;
    disc->append(makeSingleModel(pars));
    // bulge peanut
    pars.M = 0.08;
    pars.A = 0.05;
    pars.B = 0.25;
    pars.L = 0.33;
    pars.G =-0.95;
    disc->append(makeSingleModel(pars));
    // bar
    pars.M = 0.15;
    pars.A = 0.4;
    pars.B = 0.1;
    pars.L = 1.0;
    pars.G = 0.1;
    pars.P = 0.1047;
    disc->append(makeSingleModel(pars));
    return disc;
}

modelPtr makeThinBar()
{
    parameters pars;
    // hollow disc
    pars.T = "V2";
    pars.setSQ(1.6, 0.05);
    auto disc = makeHollowDisc(pars, 1.2);
    // nuclear disc
    pars.T = "V4";
    pars.M = 0.02;
    pars.A = 0.3;
    pars.B = 0.1;
    disc->append(makeSingleModel(pars));
    // bar
    pars.T = "T3";
    pars.M = 0.15;
    pars.A = 0.3;
    pars.B = 0.1;
    pars.L = 1.0;
    pars.G = 0.7;
    disc->append(makeSingleModel(pars));
    return disc;
}

}   // namespace discBar
