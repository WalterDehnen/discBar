// -*- C++ -*-
////////////////////////////////////////////////////////////////////////////////
///
/// \file    discBar.h
/// \brief   barred disc models (Dehnen & Aly 2022)
///
/// \author  Walter Dehnen
///
/// \date    2022
///
/// copyright Walter Dehnen (2022)
///
////////////////////////////////////////////////////////////////////////////////
///
/// \version jul-2022  WD  implemented
/// \version aug-2022  WD  minor refactoring
///
////////////////////////////////////////////////////////////////////////////////
#ifndef included_discBar_h
#define included_discBar_h

#include <string>        // for std::string
#include <memory>        // for std::unique_ptr
#include <exception>     // for std::runtime_error
#include <functional>    // for std::function


#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wc++98-compat"
#endif

namespace discBar {

using std::string;

struct potImpl;

/// parameters for a single disc-bar model
struct parameters
{
    double mass        = 1.0;  ///< total mass
    double scaleRadius = 1.0;  ///< scale radius s=a+b
    double axisRatio   = 0.0;  ///< axis ratio b/s
    double barRadius   = 0.0;  ///< bar half-length L
    double gamma       = 0.0;  ///< bar-slope parameter γ
    double phi         = 0.0;  ///< if > 0  opening angle ±φ with x-axis
    string modelType   = "T1"; ///< type of model: 'Tk' or 'Vk' with 1 ≤ k ≤ 4
    /// set Miyamoto-Nagai parameters a and b rather than s=a+b and q=b/(a+b)
    /// \throws if a < 0, b < 0, or a+b ≤ 0
    void setAB(double a, double b = 0.0);
    /// a = s(1-q)
    double A() const noexcept { return (1-axisRatio) * scaleRadius; }
    double scaleLength() const noexcept { return A(); }
    /// b = sq
    double B() const noexcept { return axisRatio * scaleRadius; }
    double scaleHeight() const noexcept { return B(); }
    /// perform sanity check of parameters
    /// \throws  if any value is inf or nan
    /// \throws  if gamma < -1  or  gamma > 1
    /// \throws  if mass == 0
    /// \throws  if scaleRadius <= 0
    /// \throws  if axisRatio < 0 or axisRatio > 1
    /// \throws  if modelType is unknown
    const parameters &sanityCheck(const char*caller=nullptr) const;
};  // struct discBar::parameters

/// abstract base class for a disc-bar model, single or compound
struct model
{
    /// # individual components
    virtual size_t numComponents() const noexcept
    { return 1; }

    /// # razor-thin components
    virtual size_t numRazorThin() const noexcept = 0;

    /// # barred components
    virtual size_t numBarred() const noexcept = 0;

    /// # spherical components
    virtual size_t numSpherical() const noexcept = 0;

    /// is this model razor thin (in all components)?
    bool isRazorThin() const noexcept
    { return numRazorThin() == numComponents(); }

    /// is this model axisymmetric?
    bool isAxisymmetric() const noexcept
    { return numBarred() == 0; }

    /// does this model contain a bar (in at least one component)?
    bool isBarred() const noexcept
    { return numBarred() > 0; }

    /// is this a spherical model (of possibly several components)?
    bool isSpherical() const noexcept
    { return numSpherical() == numComponents(); }

    /// is this a compound model?
    bool isCompound() const noexcept
    { return numComponents() > 1; }

    /// is this a single disc model?
    bool isSingle() const noexcept
    { return numComponents() == 1; }

    /// in case of a single-component model: parameters used to make it
    /// \throws if !isSingle()
    virtual parameters params() const
    {
	throw std::runtime_error("request for single-model parameters "
	    "of multi-component model");
    }

    /// call func(parameters) for the parameters of all single components
    /// \note useful for generating parameter output from any model
    virtual void scanParameters(
	std::function<void(parameters const&)> const&func) const = 0;

    /// type of single-component model or "compound"
    virtual string modelType() const
    { return "compound"; }

    /// total mass
    virtual double totalMass() const noexcept = 0;

    /// re-scale the mass
    /// \throws if factor == 0
    virtual void rescaleMass(double factor) = 0;

    /// Φ(x,y,z) and (optionally) its 1st and 2nd derivatives w.r.t. {x,y,z}
    /// \param[in]  x   [x,y,z]
    /// \param[out] d1P [∂Φ/∂x,∂Φ/∂y,∂Φ/∂z]
    /// \param[out] d2P [∂²Φ/∂x²,∂²Φ/∂x∂y,∂²Φ/∂x∂z,∂²Φ/∂y²,∂²Φ/∂y∂z,∂²Φ/∂z²]
    double potential(const double*x,
	double*d1P=nullptr, double*d2P=nullptr) const noexcept
    {
	return
	    d1P == nullptr? pot3D(x) :
	    d2P == nullptr? pot3D(x,d1P) : pot3D(x,d1P,d2P);
    }
    /// Φ(x,y,z=0) and (optionally) its 1st and 2nd derivatives w.r.t. {x,y}
    /// \param[in]  x   [x,y]
    /// \param[out] d1P [∂Φ/∂x,∂Φ/∂y]
    /// \param[out] d2P [∂²Φ/∂x²,∂²Φ/∂x∂y,∂²Φ/∂y²]
    double planarPotential(const double*x,
	double*d1P=nullptr, double*d2P=nullptr) const noexcept
    {
	return
	    d1P == nullptr? pot2D(x) :
	    d2P == nullptr? pot2D(x,d1P) : pot2D(x,d1P,d2P);
    }

    /// ρ(x,y,z) assuming G=1
    /// \param[in] x  [x,y,z]
    /// \throws  for compound model with some but not all razor-thin components
    /// \note    for razor-thin models Σ(x,y) is returned (ignoring z)
    /// \note    not optimised for efficiency
    double density(const double*x) const;

    virtual ~model();

    virtual void dump(std::ostream&) const {}

    virtual std::unique_ptr<model> clone() const = 0;

    virtual void flipSign() noexcept = 0;

    virtual bool isCollection() const noexcept
    { return false; }

    virtual double pot3D(const double*) const noexcept = 0;
    virtual double pot3D(const double*, double*) const noexcept = 0;
    virtual double pot3D(const double*, double*, double*) const noexcept = 0;

    virtual double pot2D(const double*) const noexcept = 0;
    virtual double pot2D(const double*, double*) const noexcept = 0;
    virtual double pot2D(const double*, double*, double*) const noexcept = 0;
    
};  // struct discBar::model

using modelPtr = std::unique_ptr<model>;

/// create a single model with given parameters
/// \throws if checkParameters==true and parameters::sanityCheck() fails
modelPtr makeSingleModel(parameters const&pars, bool checkParameters = true);

/// create a new compound model: the sum of two models
modelPtr sum(modelPtr const&, modelPtr const&);
inline
modelPtr operator+ (modelPtr const& a, modelPtr const& b) { return sum(a,b); }

/// create a new compound model: the difference between two models
modelPtr diff(modelPtr const&, modelPtr const&);
inline
modelPtr difference(modelPtr const& a, modelPtr const& b) { return diff(a,b); }
inline
modelPtr operator- (modelPtr const& a, modelPtr const& b) { return diff(a,b); }

/// create an axisymmetric model with a central hole
/// \param[in]  pars         parameters for model w/o hole, barRadius=0 set
/// \param[in]  holeRadius   < pars.scaleRadius: radius of hole
modelPtr makeHoledDisc(parameters const&pars, double holeRadius);

}   // namespace discBar

#endif
