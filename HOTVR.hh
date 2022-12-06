//----------------------------------------------------------------------
//
//  The Heavy Object Tagger with Variable R (HOTVR)
//
//  This package combines variable R jet clustering with a
//  mass jump veto. The resulting HOTVR jets have subjets,
//  accessible through a helper class (HOTVRinfo).
//  Rejected clusters and jets without a mass jump
//  can be accessed as well.
//
//  The code is based on the implementation of the ClusteringVetoPlugin
//  version 1.0.0 (by Seng-Pei Liew and Martin Stoll)
//  and the VariableR plugin version 1.2.0 (by David Krohn,
//  Gregory Soyez, Jesse Thaler and Lian-Tao Wang) in FastJet Contribs.
//  Please see the README file for more information.
//
//  For questions and comments, please contact:
//     Tobias Lapsien  <tobias.lapsien@desy.de>
//     Roman Kogler    <roman.kogler@uni-hamburg.de>
//     Johannes Haller <johannes.haller@uni-hamburg.de>
//
//----------------------------------------------------------------------
// This file is part of FastJet contrib.
//
// It is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
//
// It is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
// License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this code. If not, see <http://www.gnu.org/licenses/>.
//----------------------------------------------------------------------

#ifndef __FASTJET_CONTRIB_HOTVR_HH__
#define __FASTJET_CONTRIB_HOTVR_HH__

#include <fastjet/internal/base.hh>
#include "HOTVRinfo.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"
#include <fastjet/LimitedWarning.hh>
#include "math.h"

#include <queue>

FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

namespace contrib {

  //------------------------------------------------------------------------
  /// \class HOTVR
  ///
  class HOTVR : public JetDefinition::Plugin {

  public:

    /// The strategy to be used with the clustering
    enum Strategy{
      Best,      ///< currently N2Tiled or N2Plain for FJ>3.2.0, NNH for FastJet<3.2.0
      N2Tiled,   ///< the default (faster in most cases) [requires FastJet>=3.2.0]
      N2Plain,   ///< [requires FastJet>=3.2.0]
      NNH,       ///< slower but already available for FastJet<3.2.0
    };

    // Result of veto function, from ClusteringVetoPlugin 1.0.0
    enum VetoResult {
      CLUSTER,
      VETO,
      NOVETO
    };

    /// The clustering type is chosen by the "p" parameter
    /// generalised-kt algorithm. The definitions below are shorthand
    /// for the anti-kt, C/A and kt algorithm which also allow for
    /// backwards compatibility.
    static const double CALIKE;
    static const double KTLIKE;
    static const double AKTLIKE;

    /// Constructor that sets HOTVR algorithm parameters
    ///  - mu            mass jump threshold
    /// -  theta         mass jump strength
    ///  - min_r         minimum jet radius
    ///  - max_r         maximum jet radius
    ///  - rho           mass scale for effective radius (i.e. R ~ rho/pT)
    ///  - pt_sub        minimum pt of subjets
    ///  - clust_type    whether to use CA-like, kT-like, or anti-kT-like distance measure
    ///                  note that subjet finding has only been tested with CA-like clustering
    ///  - strategy      one of Best (the default), N2Tiled , N2Plain or NNH
    ///                  for FastJet>=3.2.0, the N2Tiled option is the default strategy,
    ///                  for earlier FastJet versions NNH is used
    ///
    /// Example usage:
    /// \code
    /// HOTVR hotvr_plugin(mu, theta, min_r, max_r, rho, pt_sub, HOTVR::CALIKE);
    /// fastjet::JetDefinition jet_def(&hotvr_plugin);
    /// fastjet::ClusterSequence clust_seq(event, jet_def);
    /// vector<fastjet::PseudoJet> inclusive_jets = clust_seq.inclusive_jets(ptmin);
    /// std::vector<fastjet::PseudoJet> hotvr_jets;
    /// hotvr_jets = hotvr_plugin.get_jets();
    /// \endcode
    //HOTVR(double mu, double theta, double min_r, double max_r,double rho, double pt_sub, double clust_type,
    //        Strategy requested_strategy = Best);

    /// ANNA: Constructor that sets HOTVR algorithm parameters for HOTVR with Softdrop
    ///  - beta          soft drop angle dependence
    ///  - z_cut         soft drop strength
    ///  - pt_threshold  minimum pt for the Soft Drop Check
    ///  - min_r         minimum jet radius
    ///  - max_r         maximum jet radius
    ///  - rho           mass scale for effective radius (i.e. R ~ rho/pT)
    ///  - alpha         exponent in the effective radius (i.e. R ~ rho/pT ^ alpha)
    ///  - pt_sub        minimum pt of subjets (default = 0)
    ///  - clust_type    whether to use CA-like, kT-like, or anti-kT-like distance measure
    ///                  note that subjet finding has only been tested with CA-like clustering
    ///  - strategy      one of Best (the default), N2Tiled , N2Plain or NNH
    ///                  for FastJet>=3.2.0, the N2Tiled option is the default strategy,
    ///                  for earlier FastJet versions NNH is used
    HOTVR(double beta, double z_cut, double pt_threshold, double min_r, double max_r, double rho, double pt_sub, double mu, double clust_type, double alpha, double a, double b, double c,
          Strategy requested_strategy = Best); //ANNA new constructor for HOTVR with Softdrop, possibility to change alpha

    void set_jetptmin(double ptmin){_jetptmin = ptmin;}

    // Virtual function from JetDefinition::Plugin that implements the algorithm
    void run_clustering(fastjet::ClusterSequence & cs) const;

    // Information string
    virtual std::string description() const;

    // NOTE: Required by JetDefinition::Plugin
    double R() const { return sqrt(_max_r2); }
    std::vector<PseudoJet>  get_jets(){return sorted_by_pt(_hotvr_jets);} //return the candidate jets
    std::vector<fastjet::PseudoJet> get_soft_cluster(){return _soft_cluster;} //return the soft clusters rejected by the soft drop condition
    std::vector<fastjet::PseudoJet> get_rejected_cluster(){return _rejected_cluster;} //return jets without subjets
    std::vector<fastjet::PseudoJet> get_rejected_subjets(){return _rejected_subjets;} //return subjets rejected by the pT criterion
    void Reset(){ _hotvr_jets.clear(); _soft_cluster.clear();  _rejected_cluster.clear(); _subjets.clear(); _jets.clear(); _rejected_subjets.clear();}

  private:

    // bool for banner printout
    static bool _already_printed;

    double _beta, _z_cut, _pt_threshold;
    double _min_r2, _max_r2, _max_r;
    double _rho2, _pt_sub, _mu;
    // ANNA added Parameters for soft drop condition

    // Parameters of the HOTVR

    double _clust_type;
    double _alpha;     // ANNA add _alpha
    double _a, _b, _c;
    Strategy _requested_strategy;
    double _jetptmin;
    double _theta;

    // some debugging output
    bool _debug;

    // the jets and rejected clusters
    mutable std::vector<fastjet::PseudoJet> _jets;
    mutable std::vector<fastjet::PseudoJet> _hotvr_jets;
    mutable std::vector<fastjet::PseudoJet> _soft_cluster;
    mutable std::vector<fastjet::PseudoJet> _rejected_cluster;
    mutable std::vector<fastjet::PseudoJet> _rejected_subjets;
    mutable std::vector<std::vector<fastjet::PseudoJet> >  _subjets;

    // helper function to decide what strategy is best
    // the optimal strategy will depend on the multiplicity and _max_r
    Strategy best_strategy(unsigned int N) const;

    // implementation of the clustering using FastJet NN*** classes
    template<typename NN>
    void NN_clustering(ClusterSequence &cs, NN &nn) const;

    // veto condition
    VetoResult CheckVeto(const PseudoJet& j1, const PseudoJet& j2) const;

    // ANNA: veto condition for soft drop
    VetoResult CheckVeto_SoftDrop(const PseudoJet& j1, const PseudoJet& j2) const;

    // print welcome message with some information
    void print_banner();

  };

  //----------------------------------------------------------------------
  // classes to help run the HOTVR algorithm using NN-type classes
  // the code below is based on the implementation of the Variable R clustering
  // and has been taken from VariableR/VariableRPlugin.cc, version 1.2.1

  // class carrying particle-independent information
  class HOTVRNNInfo {
  public:
    HOTVRNNInfo(double rho2_in, double min_r2_in, double max_r2_in,
                    double clust_type_in, double alpha_in, double a_in, double b_in, double c_in) // ANNA added alpha // add parameters for exp function
      : _rho2(rho2_in), _min_r2(min_r2_in), _max_r2(max_r2_in),
        _clust_type(clust_type_in),
        _alpha(alpha_in), _a(a_in), _b(b_in), _c(c_in) {}

    double rho2()  const  {return _rho2; }
    double min_r2() const {return _min_r2;}
    double max_r2() const {return _max_r2;}
    double alpha() const {return _alpha;}
    double a() const {return _a;}
    double b() const {return _b;}
    double c() const {return _c;}
    double momentum_scale_of_pt2(double pt2) const {
      return pow(pt2,_clust_type);
    }

  private:
    double _rho2;   ///< constant (squared) that controls the overall R magnitude
    double _min_r2; ///< minimal allowed radius squared
    double _max_r2; ///< maximal allowed radius squared
    double _clust_type; ///< cluster type (power "p" in distance measure)
    double _alpha; ///< exponent in the effective radius
    double _a; // parameter for exp function in mass term
    double _b; // parameter for exp function in mass term
    double _c; // parameter for exp function in mass term

  };

  // class carrying the minimal info required for the clustering
  class HOTVRBriefJet {

  public:

    void init(const PseudoJet & jet, HOTVRNNInfo *info) {
      _rap = jet.rap();
      _phi = jet.phi();
      _fourmom = jet;

      _max_r2 = info->max_r2();
      _min_r2 = info->min_r2();

      _a = info->a();
      _b = info->b();
      _c = info->c();

      double pt2 = jet.pt2();
      //double pt = sqrt(pt2);
      double m2 = jet.m2();
      double m = sqrt(m2);

      _debug = false;

      // get the effective "radius" Reff
      //_beam_R2 = info->rho2()/pt2;

      //  _beam_R2 = info->rho2()/pow(pt2,info->alpha()); // calculate effective radius with tunable exponent
      //  _beam_R2 = info->rho2()*m2/pow(pt2,info->alpha()); // calculate effective radius with tunable exponent and mass dependent

        //_beam_R2 = info->rho2()*m2/pow(pt2,info->alpha());
        //m = 170;
      //double beam_R = 0.15+2.7*m/pt + (1 + signbit(m-150))/2 * (0.15+0.1*m/pt);

      // try this one: R = 1/ET * (a + m^2/b) with a = 140 GeV and b = 50 GeV
      // add a counter-term such that very large masses (>>mt) are not preferred
      double ET2 = pt2 + m2;
    //  double mterm = 150. + m2/50 - 200*exp((m-200)/40);
      double mterm = 150. + m2/50 - info->a()*exp((m-info->b()/info->c()));

      if (mterm<0) mterm = 0;
      _beam_R2 = 1/ET2 * mterm*mterm;

      if      (_beam_R2 > info->max_r2()){ _beam_R2 = info->max_r2();}
      else if (_beam_R2 < info->min_r2()){ _beam_R2 = info->min_r2();}

      if (_debug){
         std::cout << "HOTVRBriefJet: ---------Alpha: " << info->alpha() << '\n';
         std::cout << "Rho2: " << info->rho2() << '\n';
         std::cout << "Rho: " << std::sqrt(info->rho2()) << '\n';
         std::cout << "pt2: " << pt2 << '\n';
         std::cout << "ET2: " << ET2 << ", ET = " << sqrt(ET2) << '\n';
         std::cout << "pt: " << std::sqrt(pt2) << '\n';
         std::cout << "m: " << m << '\n';
         std::cout << "beam_R2 "<< _beam_R2 << '\n';
         std::cout << "beam_R "<< std::sqrt(_beam_R2) << '\n';
      }

      // get the appropriate momentum scale
      _mom_factor2 = info->momentum_scale_of_pt2(pt2);
    }

    double geometrical_distance(const HOTVRBriefJet * jet) const {
      double dphi = std::abs(_phi - jet->_phi);
      double deta = (_rap - jet->_rap);
      if (dphi > pi) {dphi = twopi - dphi;}
      double dR2 = dphi*dphi + deta*deta;

      PseudoJet combj = _fourmom + jet->four_momentum();
      double m2 = combj.m2();
      double m = sqrt(m2);
      double pt2 = combj.pt2();
      //double jetR = 0.15+2.7*m/pt + (1 + signbit(m-150))/2 * (0.15+0.1*m/pt);
      //double jetR2 = jetR*jetR;

      double ET2 = pt2 + m2;
      //double mterm = 140. + m2/50;
      //double mterm = 150. + m2/50 - 200*exp((m-200)/40);
      double mterm = 150. + m2/50 - _a*exp((m-_b)/_c);

      if (mterm<0) mterm = 0;
      double jetR2 = 1/ET2 * mterm*mterm;

      if (jetR2 > _max_r2) {jetR2 = _max_r2;}
      if (jetR2 < _min_r2) {jetR2 = _min_r2;}

      return dR2/jetR2;
      //return dR2/_beam_R2;
    }

    double variable_distance() const { return _beam_R2; }

    //double geometrical_beam_distance() const { return _beam_R2; }
    double geometrical_beam_distance() const { return 1; }

    double momentum_factor() const{ return _mom_factor2; }

    PseudoJet four_momentum() const { return _fourmom; }

    /// make this BJ class compatible with the use of NNH
    double distance(const HOTVRBriefJet * other_bj_jet){
      double mom1 = momentum_factor();
      double mom2 = other_bj_jet->momentum_factor();
      return (mom1<mom2 ? mom1 : mom2) * geometrical_distance(other_bj_jet);
    }
    double beam_distance(){
      return momentum_factor() * geometrical_beam_distance();
    }

    // the following are required by N2Tiled
    inline double rap() const{ return _rap;}
    inline double phi() const{ return _phi;}

  private:
    PseudoJet _fourmom;
    double _max_r2, _min_r2;
    double _a, _b, _c;
    double _rap, _phi, _mom_factor2, _beam_R2;
    bool _debug;
  };

} // namespace contrib

FASTJET_END_NAMESPACE

#endif  // __FASTJET_CONTRIB_HOTVR_HH__
