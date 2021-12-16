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

#include "HOTVR.hh"

#include <cstdio>
#include "math.h"
#include <iomanip>
#include <cmath>
#include <sstream>
#include <memory>

#include <fastjet/NNH.hh>
#if FASTJET_VERSION_NUMBER >= 30200
#include <fastjet/NNFJN2Plain.hh>
#include <fastjet/NNFJN2Tiled.hh>
#endif


FASTJET_BEGIN_NAMESPACE      // defined in fastjet/internal/base.hh

namespace contrib {

  // make sure that welcome banner is printed only once
  bool HOTVR::_already_printed = false;

  /// The clustering type is chosen by the "p" parameter
  /// generalised-kt algorithm. The definitions below are shorthand
  /// for the anti-kt, C/A and kt algorithm which also allow for
  /// backwards compatibility.
  const double HOTVR::CALIKE  =  0.0;
  const double HOTVR::KTLIKE  =  1.0;
  const double HOTVR::AKTLIKE = -1.0;

  // Constructor that sets HOTVR algorithm parameters
  //  - mu            mass jump threshold
  // -  theta         mass jump strength
  //  - min_r         minimum jet radius
  //  - max_r         maximum jet radius
  //  - rho           mass scale for effective radius (i.e. R ~ rho/pT)
  //  - pt_sub        minimum pt of subjets
  //  - clust_type    whether to use CA-like, kT-like, or anti-kT-like distance measure
  //                  note that subjet finding has only been tested with CA-like clustering
  //  - strategy      one of Best (the default), N2Tiled , N2Plain or NNH
  HOTVR::HOTVR(double mu, double theta, double min_r, double max_r, double rho, double pt_sub, double clust_type, Strategy requested_strategy)
    :_mu(mu), _theta(theta), _min_r2(min_r*min_r), _max_r2(max_r*max_r), _max_r(max_r),
     _rho2(rho*rho), _pt_sub(pt_sub),
     _clust_type(clust_type), _requested_strategy(requested_strategy)
  {
    if (!_already_printed){
      print_banner();
      _already_printed = true;
    }

    // some sanity checks
    if (mu < 0.0) throw Error("HOTVR: mu must be positive.");
    if (theta > 1.0 || theta < 0.0) throw Error("HOTVR: theta must be in [0.0,1.0].");
    if (max_r < 0.0) throw Error("HOTVR: Maximum radius must be positive.");
    if (min_r < 0.0) throw Error("HOTVR: Minimum radius must be positive.");
    if (min_r>max_r)  throw Error("HOTVR: Minimum radius must be smaller than maximum radius.");
    if (rho<0)  throw Error("HOTVR: Rho must be positive.");
    if (pt_sub<0)  throw Error("HOTVR: pT threshold must be positive.");

    // decide the strategy
#if FASTJET_VERSION_NUMBER < 30200
    // this is only supported for the Best and Native strategies
    if ((requested_strategy!=Best) && (requested_strategy!=NNH))
      throw Error("HOTVR: with FastJet<3.2.0, Best and NNH are the only supported strategies.");
#endif

  }
  /// ANNA: Constructor that sets HOTVR algorithm parameters for HOTVR with Softdrop
  ///  - beta          soft drop angle dependence
  ///  - z_cut         soft drop strength
  ///  - pt_threshold  minimum pt for the Soft Drop Check
  ///  - min_r         minimum jet radius
  ///  - max_r         maximum jet radius
  ///  - rho           mass scale for effective radius (i.e. R ~ rho/pT)
  ///  - pt_sub        minimum pt of subjets (default = 0)
  ///  - clust_type    whether to use CA-like, kT-like, or anti-kT-like distance measure
  ///                  note that subjet finding has only been tested with CA-like clustering
  ///  - strategy      one of Best (the default), N2Tiled , N2Plain or NNH
  ///                  for FastJet>=3.2.0, the N2Tiled option is the default strategy,
  ///                  for earlier FastJet versions NNH is used
  ///  - alpha         change the slope of effective radius (default alpha=1)
  HOTVR::HOTVR(double beta, double z_cut, double pt_threshold, double min_r, double max_r,double rho, double pt_sub, double mu, double clust_type, double alpha,
    Strategy requested_strategy):
      _beta(beta), _z_cut(z_cut), _pt_threshold(pt_threshold), _min_r2(min_r*min_r), _max_r2(max_r*max_r), _max_r(max_r),
     _rho2(rho*rho), _pt_sub(pt_sub), _mu(mu),
     _clust_type(clust_type),  _alpha(alpha),
     _requested_strategy(requested_strategy)
  {
    if (!_already_printed){
      print_banner();
      _already_printed = true;
    }

    // some sanity checks
// ANNA: Do we set bounds on beta and z_cut?
  //  if (beta < 0.0) throw Error("HOTVR: beta must be positive.");
  //  if (z_cut > 1.0 || z_cut < 0.0) throw Error("HOTVR: z_cut must be in [0.0,1.0].");
    if (pt_threshold < 0.0) throw Error("HOTVR: pT threshold for Softdrop condition must be positive.");
    if (max_r < 0.0) throw Error("HOTVR: Maximum radius must be positive.");
    if (min_r < 0.0) throw Error("HOTVR: Minimum radius must be positive.");
    if (min_r>max_r)  throw Error("HOTVR: Minimum radius must be smaller than maximum radius.");
    if (rho<0)  throw Error("HOTVR: Rho must be positive.");
    if (pt_sub<0)  throw Error("HOTVR: pT threshold must be positive.");

    // decide the strategy
#if FASTJET_VERSION_NUMBER < 30200
    // this is only supported for the Best and Native strategies
    if ((requested_strategy!=Best) && (requested_strategy!=NNH))
      throw Error("HOTVR: with FastJet<3.2.0, Best and NNH are the only supported strategies.");
#endif
  }

  void HOTVR::run_clustering(ClusterSequence & cs) const {

    // set up clustering strategy
    Strategy strategy = _requested_strategy;

    // decide the best option upon request
    if (_requested_strategy==Best){
      strategy = best_strategy(cs.jets().size());
    }

    // set up NNH
    HOTVRNNInfo nninfo(_rho2, _min_r2, _max_r2, _clust_type, _alpha); // ANNA added alpha here

    // the following code has been written by G. Soyez and is taken from
    // VariableR/VariableRPlugin.cc, version 1.2.1
    // -> make use of the NN-type clustering in FastJet 3.2 and higher
#if FASTJET_VERSION_NUMBER >= 30200
    if (strategy==N2Tiled){
      NNFJN2Tiled<HOTVRBriefJet,HOTVRNNInfo> nnt(cs.jets(), _max_r, &nninfo);
      NN_clustering(cs, nnt);
    } else if (strategy==N2Plain){
      NNFJN2Plain<HOTVRBriefJet,HOTVRNNInfo> nnp(cs.jets(), &nninfo);
      NN_clustering(cs, nnp);
    } else { // NNH is the only option left
#endif
      fastjet::NNH<HOTVRBriefJet,HOTVRNNInfo> nnh(cs.jets(), &nninfo);
      NN_clustering(cs, nnh);
#if FASTJET_VERSION_NUMBER >= 30200
    }
#endif
  }



  //---------------------------------------------------------------------
  // the actual implementation of the clustering using the NN helpers
  template<typename NN>
  void HOTVR::NN_clustering(ClusterSequence &cs, NN &nn) const{
    // ANNA
    bool _debug=true; //bool for debug option, that gives couts and warnings
    bool _found_subjets=false; // counts if the algorithm finds subjets -> for debugging
    // loop over pseudojets
    int njets = cs.jets().size();
    while (njets > 0) { // combine jets untill the list is empty
      bool existing_i=false;
      bool existing_j=false;
      int i(-1), j(-1);
      double dij = nn.dij_min(i, j); //ANNA returns the dij_min and indices i and j for the corresponding jets.
// If j<0 then i recombines with the beam
      if(_debug){std::cout << "dij is "<< dij << '\n';}
      if(j < 0) {//diB is smallest
        if(_debug){std::cout << ".................d_iB is smallest............." << '\n';}
	      cs.plugin_record_iB_recombination(i,dij);
	      _jets.push_back(cs.jets()[i]); //ANNA the last pseudojet (final HOTVR jet) i is stored as jet
	      _jets[_jets.size()-1].set_user_index(-2); //set the user index of the jet to -2

	      bool set=false;
	      std::vector<fastjet::PseudoJet> subjets;

        //------------handle single particles, that are not in the jet --------------
        // if (last particle that has been clustered into the jet is no subjet jet) -> merge into nearest subjet
        // TODO make sure that the jet has other subjets
        if (_jets[_jets.size()-2].user_index()!=i) { // last particle that has been clustered has not the same user index (is no subjet)
          // find closest subjet (with index i)
          if (_debug) { std::cout << "BEGIN: Handle single particle, that was not part of any subjet at the end of the clustering. " << '\n';}
          double dist;
          double min_dist = 100;
          PseudoJet closest_subjet;
          int position;
            if (_debug) {std::cout << "Before: Go through all stored jets to find closest subjet " << '\n';}
          for(uint o=0;o<_jets.size();o++){ // go through all stored jets
              if (_debug) {std::cout << "Go through all stored jets to find closest subjet " << '\n';}
            if(_jets[o].user_index()==i) { // find the corresponding subjets (same user index)
              if (_debug) {std::cout << "jet with same user index found " << '\n';}
              dist = _jets[o].delta_R(_jets[_jets.size()-2]);
              if (dist < min_dist) {
                  if (_debug) {std::cout << "dist < min_dist " << '\n';}
                min_dist = dist;
                closest_subjet = _jets[o];
                position = o;
              }
            }
          }
          // merge the particle into the clostest subjet
            if (_debug) {std::cout << "Merge particle into subjet " << '\n';}
          JetDefinition jet_def;
          jet_def = cs.jet_def();
            if (_debug) {std::cout << "Jet def done " << '\n';}
          const JetDefinition::Recombiner* recombiner;
          recombiner = jet_def.recombiner();
            if (_debug) {std::cout << "recombiner done " << '\n';}
          recombiner->recombine(_jets[position], _jets[_jets.size()-2], _jets[position]); // combine jet 1 and 2 and save into 3
            if (_debug) {std::cout << "recombined jets " << '\n';}
          // remove the particle from the list
          _jets.erase(_jets.end()-2);
            if (_debug) { std::cout << "END: Handle single particle, that was not part of any subjet at the end of the clustering. " << '\n';
              std::cout << "min_dist = " << min_dist << '\n';
              std::cout << "position = " << position << '\n';
            }
        }
        //------------END handle single particles --------------------

        // now go through all stored jets
	      for(uint o=0;o<_jets.size();o++){ //ANNA go through all stored jets
	        if(_jets[o].user_index()==i) { // find the corresponding subjets (same user index)
            if(_debug){std::cout << "User index ==i " << _jets[o].user_index() << '\n';}
	          _jets[_jets.size()-1].set_user_index(i*100); // combined jet is the last one in the list of jets -> index i*100
	          if(!set) { // save the jet candidates only once
	            _hotvr_jets.push_back(cs.jets()[i]);//save jet candidates (jets with massjumps / softdrop)
	            _hotvr_jets[_hotvr_jets.size()-1].set_user_index(_hotvr_jets.size()-1);
              if(_debug){std::cout << "Hotvr jets size " << _hotvr_jets.size() << '\n';}
              //ANNA user index = position of HOTVR jet
	            set=true;
	          }
	          subjets.push_back(_jets[o]);//save subjets if the user index is i
	        } //end if user index = i
        } // end jet loop
        //ANNA if the previous condition was fulfilled, means the jet has subjets
        if(_jets[_jets.size()-1].user_index()!=-2) { // if the HOTVR jet has subjets

          if(_debug){std::cout << "user index != 2 is "<< _jets[_jets.size()-1].user_index() << '\n';}
	        _subjets.push_back(sorted_by_pt(subjets));
	        _hotvr_jets.at(_hotvr_jets.size()-1).set_user_info(
              new HOTVRinfo(_hotvr_jets.at(_hotvr_jets.size()-1),
              _subjets[_hotvr_jets.at(_hotvr_jets.size()-1).user_index()]) );
          if(_debug){std::cout << "Set the HOTVR info." << '\n';}

          if(_debug){//begin only debug
          for (size_t l = 0; l < _subjets.size(); l++) {
           std::cout << "Size of the subjet list " <<   _subjets[l].size() << '\n';
           if (_subjets[l].size()==1 && _subjets[l].at(0).m() > _mu) { // check mass for jets with only 1 subjet
             std::cout << "WARNING: found jet with 1 subjet, but has m="<< _subjets[l].at(0).m() << '\n';
           }
            for (size_t m = 0; m < _subjets[l].size(); m++) {
                std::cout << " pt "<< _subjets[l].at(m).pt() << ", mass "<<_subjets[l].at(m).m() << '\n';
                if(_subjets[l].at(m).pt() < _pt_threshold){
                  std::cout << "WARNING subjet pt smaller pt threshold!" << '\n';
                }
            }
          }
          } //debug end
        } // end "if the HOTVR jet has subjets"
        else { //all jets that have no subjets
          if(_debug){std::cout << "fill rejected cluster, size =  "<< _rejected_cluster.size() << '\n';
          std::cout << "User index "<< cs.jets()[i].user_index()<< " pt "<<cs.jets()[i].pt() << '\n';}
          _rejected_cluster.push_back(cs.jets()[i]); //fill vector with jets that have no massjumps
// Anna: this part is new!
          subjets.push_back(cs.jets()[i]); // save the single HOTVR jet as subjet
          _subjets.push_back(subjets); // add subjets to _subjets
          _hotvr_jets.push_back(cs.jets()[i]); //add jets that have no softjumps to hotvrjets list
          _hotvr_jets[_hotvr_jets.size()-1].set_user_index(_hotvr_jets.size()-1);
          _hotvr_jets.at(_hotvr_jets.size()-1).set_user_info( // save the hotvr jet as subjet
              new HOTVRinfo(_hotvr_jets.at(_hotvr_jets.size()-1),
              _subjets[_hotvr_jets.at(_hotvr_jets.size()-1).user_index()]) );
        } // end "if the HOTVR jet has no subjets"

        nn.remove_jet(i);//remove jet from list
	      njets--;
        // for debugging: WARNING if the algorithm found subjets but stored no HOTVR jets
        if(_debug && njets==0 && _found_subjets && _hotvr_jets.size()==0){std::cout << "WARNING: found subjets but stored no HOTVR jet!" << '\n';}
	     continue;
      }


      int k=-1;//dij is smallest
    // ANNA: replaced the massjump switch by a softdrop switch
    //  switch ( CheckVeto ( cs.jets()[i],cs.jets()[j] ) ) {//below the massjump threshold mu: cluster
      switch ( CheckVeto_SoftDrop ( cs.jets()[i],cs.jets()[j] ) ) {

      case CLUSTER: {
      //  std::cout << "entering CLUSTER" << '\n';
	      k=-1;
	      cs.plugin_record_ij_recombination(i, j, dij, k);
//cluster i and j //ANNA k is the index for the combined jet, that is set by the function
	      nn.merge_jets(i, j, cs.jets()[k], k);
	      njets--;
	      break;
      }

    //   case VETO: { //above massjump threshold mu: massjump found
    //   if(_debug){std::cout << "entering VETO" << '\n';}
    //
	  //     k=-1;
	  // 	  if(cs.jets()[i].pt()<_pt_sub) {
    //   if(_debug){std::cout << "VETO: remove first subjet with pt  "<< cs.jets()[i].pt() << '\n';}
    //
	  //       cs.plugin_record_iB_recombination(i,dij);     //subjet i below pT threshold?
		//       _rejected_subjets.push_back(cs.jets().at(i)); //save rejected subjets here
	  //       nn.remove_jet(i);
	  //       njets--;
	  //     }
	  //     if(cs.jets()[j].pt()<_pt_sub) {
    // if(_debug){  std::cout << "VETO: remove second subjet with pt  "<< cs.jets()[j].pt() << '\n';}
    //
	  //       cs.plugin_record_iB_recombination(j,dij);     //subjet j below pT threshold?
		//       _rejected_subjets.push_back(cs.jets().at(j)); //save rejected subjets here
	  //       nn.remove_jet(j);
	  //       njets--;
	  //     }
	  //     if(cs.jets()[i].pt()>=_pt_sub && cs.jets()[j].pt()>=_pt_sub) {//check if the subjet pT is higher than the threshold
    //   if(_debug){std::cout << "store subjets with pt "<< cs.jets()[i].pt() << " and "<< cs.jets()[j].pt() << '\n';}
    //   _found_subjets=true;
    //     cs.plugin_record_ij_recombination(i, j, dij, k);
    //     // set all user indices to k, e.g i and j consist of subjets
  	//       for(uint o=0;o<_jets.size();o++){
	  //         if(_jets[o].user_index()==j) {
	  //           _jets[o].set_user_index(k);
	  //           existing_j=true;
	  //         }
	  //         if( _jets[o].user_index()==i){
	  //           _jets[o].set_user_index(k);
	  //           existing_i=true;
	  //         }
	  //       }
    //       // the jets are not yet in our list
	  //       if(!existing_j){ // store the subjet j into the list of jets
	  //         _jets.push_back(cs.jets()[j]);
	  //         _jets[_jets.size()-1].set_user_index(k);
	  //       }
	  //       if(!existing_i){
	  //         _jets.push_back(cs.jets()[i]);
	  //         _jets[_jets.size()-1].set_user_index(k);
	  //       }
	  //       nn.merge_jets(i, j, cs.jets()[k], k);
	  //       njets--;
	  //       }
	  //     break;
    //   }

      case VETO: { // ANNA SoftDrop case: soft drop found, added a mass criterion here
      if(_debug){std::cout << "entering VETO" << '\n';}

        k=-1;
        if(cs.jets()[i].pt()<_pt_sub) { // check pt of first potential subjet
      if(_debug){std::cout << "VETO: remove first subjet with pt  "<< cs.jets()[i].pt() << '\n';}

          cs.plugin_record_iB_recombination(i,dij);     //subjet i below pT sub?
          _rejected_subjets.push_back(cs.jets().at(i)); //save rejected subjets here
          nn.remove_jet(i);
          njets--;
        }
        if(cs.jets()[j].pt()<_pt_sub) { // check pt of second potential subjet
    if(_debug){  std::cout << "VETO: remove second subjet with pt  "<< cs.jets()[j].pt() << '\n';}

          cs.plugin_record_iB_recombination(j,dij);     //subjet j below pT sub?
          _rejected_subjets.push_back(cs.jets().at(j)); //save rejected subjets here
          nn.remove_jet(j);
          njets--;
        }
        if(cs.jets()[i].pt()>=_pt_sub && cs.jets()[j].pt()>=_pt_sub) {//both subjets have pT higher than the threshold ptsub
      _found_subjets=true;
        cs.plugin_record_ij_recombination(i, j, dij, k);
        bool _masscondition = false;
        // CASE 1: no subjets stored previously
        if (_jets.size()==0) {
          if (_debug) {std::cout << "CASE 1: no jets stored yet, check mass condition for pseudojets that should be clustered. " << '\n';}
          PseudoJet combj = cs.jets()[i] + cs.jets()[j];
          double mcombj = abs(combj.m());
          if(mcombj > _mu){_masscondition=true;}
        }
        // CASE 2: subjets stored previously
        // Check mass threshold for all combinations of subjets
        else{
          std::vector<PseudoJet> subjets_j;
          std::vector<PseudoJet> subjets_i;
          // find all previously stored subjets
          for(uint o=0;o<_jets.size();o++){ // for list of jets
            if (_debug) {
              std::cout << "CASE 2: Loop over jets to store subjets " << '\n';
            }
            if(_jets[o].user_index()==j) { // jet j is in list
              subjets_j.push_back(_jets[o]); // save all jets with index j in list of subjets of jet j
              if (_debug) {std::cout << "Jet j has a subjet " << '\n';}
            }

            if( _jets[o].user_index()==i){ // jet i is in list
              if (_debug) {std::cout << "Jet i has a subjet " << '\n';}
              subjets_i.push_back(_jets[o]);
            }
          }
          if(subjets_j.size() == 0) {
            subjets_j.push_back(cs.jets()[j]);
            if (_debug) {std::cout << "Jet j has no subjet, store jet in list of subjets" << '\n';}
          }
          if(subjets_i.size() == 0){
            subjets_i.push_back(cs.jets()[i]);
            if (_debug) {std::cout << "Jet i has no subjet, store jet in list of subjets" << '\n';}
          }

          for (size_t n = 0; n < subjets_j.size(); n++) {
            for (size_t m = 0; m < subjets_i.size(); m++) {
              PseudoJet combj = subjets_j[n]+subjets_i[m];
              double mcombj = abs(combj.m());
              if (_debug) {std::cout << "check mass condition, mcombj = " << mcombj << '\n';}
              if(mcombj > _mu){
                _masscondition=true;
                break;
              }
            }
          }
      } // end at least one jet has subjets
        // if the mass condition was fulfilled for at least one pair of subjets
        if(_masscondition){// code body copied from old implementation -> store subjets
          if(_debug){std::cout << "masscut fullfilled, store subjets with pt "<< cs.jets()[i].pt() << " and "<< cs.jets()[j].pt() << '\n';}

          // set all user indices to k, e.g i and j consist of subjets
            for(uint o=0;o<_jets.size();o++){ // for list of jets
              if(_jets[o].user_index()==j) { // jet j is in list
                _jets[o].set_user_index(k);
                existing_j=true;
              }
              if( _jets[o].user_index()==i){ // jet i is in list
                _jets[o].set_user_index(k);
                existing_i=true;
              }
            }
            // the jets are not yet in our list
            if(!existing_j){ // store the subjet j into the list of jets
              _jets.push_back(cs.jets()[j]); // add the subjet to the list
              _jets[_jets.size()-1].set_user_index(k);
            }
            if(!existing_i){
              _jets.push_back(cs.jets()[i]);
              _jets[_jets.size()-1].set_user_index(k);
            }
            nn.merge_jets(i, j, cs.jets()[k], k); // combine jets
        } // end if mu

        // if no combination of subjets fullfills mcombj>mu
        else{
          if(_debug){std::cout << "masscut not fullfilled, subjets with pt "<< cs.jets()[i].pt() << " and "<< cs.jets()[j].pt() << '\n';}
          nn.merge_jets(i, j, cs.jets()[k], k); // combine jets
          for(uint o=0;o<_jets.size();o++){ // for list of jets, check if one pseudojet is already a stored subjet
            if(_jets[o].user_index()==j) { // jet j is in list of subjets
              _jets[o]=cs.jets()[k]; // overwrite the old subjet with the new jet
              _jets[o].set_user_index(k);
            }
            if( _jets[o].user_index()==i){ // jet i is in list
              _jets[o]=cs.jets()[k]; // overwrite the old subjet with the new jet
              _jets[o].set_user_index(k);
            }
          } // end list of jets
        } // end mu not fulfilled
        njets--;
      } // end ptsub check
        break;
      } // end VETO

      // case NOVETO: {  //above massjump threshold mu:  no massjump found
      //   if(cs.jets()[i].m()<cs.jets()[j].m()) {
      //     cs.plugin_record_iB_recombination(i,dij);
      //     _soft_cluster.push_back(cs.jets()[i]); //fill vector with jets that were rejected
      //     nn.remove_jet(i);
      //   } else {
      //     cs.plugin_record_iB_recombination(j,dij);
      //     _soft_cluster.push_back(cs.jets()[j]); //fill vector with jets that were rejected
      //     nn.remove_jet(j);
      //   }
      //   njets--;
      // }

      // ANNA: NOVETO in SoftDrop case
      case NOVETO: {  //soft drop condition not fulfilled -> discard softer jet
    if(_debug){std::cout << "entering NOVETO" << '\n';}
	      if(cs.jets()[i].pt()<cs.jets()[j].pt()) {
    if(_debug){std::cout << "NOVETO: removed 1. jet with pt "<< cs.jets()[i].pt() << '\n';
    std::cout << "the other had a pt of "<< cs.jets()[j].pt() << '\n';}
	        cs.plugin_record_iB_recombination(i,dij);
	        _soft_cluster.push_back(cs.jets()[i]); //fill vector with jets that were rejected
	        nn.remove_jet(i);
	      } else {
    if(_debug){std::cout << "NOVETO: removed 2. jet with pt "<<  cs.jets()[j].pt()<< '\n';
    std::cout << "the other had a pt of "<< cs.jets()[i].pt() << '\n';}
	        cs.plugin_record_iB_recombination(j,dij);
	        _soft_cluster.push_back(cs.jets()[j]); //fill vector with jets that were rejected
	        nn.remove_jet(j);
	      }
	      njets--;
      }

      } // end switch over veto condition

    } // end while loop over pseudojets

  }


  std::string HOTVR::description() const{
    std::stringstream sstream("");

    sstream << "HOTVR (1606.04961), ";

    if (_clust_type<0) {
      sstream << "AKT";
    } else if (_clust_type==0.) {
       sstream << "CA";
    } else {
      sstream << "KT";
    }
    sstream << "-like";
    sstream << std::fixed << std::setprecision(1) << ", theta=" << _theta;
    sstream << ", mu=" << _mu;
    sstream << ", max_r=" << sqrt(_max_r2);
    sstream << ", min_r=" << sqrt(_min_r2);
    sstream << ", rho=" << sqrt(_rho2);
    sstream << ", pt_sub=" << _pt_sub;

    switch (_requested_strategy){
    case Best:    sstream << ", strategy=Best"; break;
    case N2Tiled: sstream << ", strategy=N2Tiled"; break;
    case N2Plain: sstream << ", strategy=N2Plain"; break;
    case NNH:     sstream << ", strategy=NNH"; break;
    };

    return sstream.str();
  }


  void HOTVR::print_banner(){
    std::cout << "#------------------------------------------------------------------------------\n";
    std::cout << "#                                 HOTVR 1.0.0                                  \n";
    std::cout << "#                      T. Lapsien, R. Kogler, J. Haller                        \n";
    std::cout << "#                                arXiv:1606.04961                              \n";
    std::cout << "#                                                                              \n";
    std::cout << "# The Heavy Object Tagger with Variable R.                                     \n";
    std::cout << "#                                                                              \n";
    std::cout << "# If you use this package, please cite the following papers:                   \n";
    std::cout << "#                                                                              \n";
    std::cout << "# - Tobias Lapsien, Roman Kogler, Johannes Haller,                             \n";
    std::cout << "#   \"A new tagger for hadronically decaying heavy particles at the LHC\",     \n";
    std::cout << "#   arXiv:1606.04961                                                           \n";
    std::cout << "#                                                                              \n";
    std::cout << "# - David Krohn, Jesse Thaler, Lian-Tao Wang, \"Jets with Variable R\",        \n";
    std::cout << "#   JHEP 06, 059 (2009) [arXiv:0903.0392 [hep-ph]]                             \n";
    std::cout << "#                                                                              \n";
    std::cout << "# - Martin Stoll, \"Vetoed jet clustering: The mass-jump algorithm\",          \n";
    std::cout << "#   JHEP 04, 111 (2015) [arXiv:1410.4637 [hep-ph]]                             \n";
    std::cout << "#                                                                              \n";
    std::cout << "# The HOTVR algorithm is provided without warranty under the terms of the      \n";
    std::cout << "# GNU GPLv2. See COPYING file for details.                                     \n";
    std::cout << "#------------------------------------------------------------------------------\n";
}

  // check the mass jump terminating veto
  // this function is taken from ClusteringVetoPlugin 1.0.0
  HOTVR::VetoResult HOTVR::CheckVeto(const PseudoJet& j1, const PseudoJet& j2) const {

    PseudoJet combj = j1+j2;

    double mj1 = abs (j1.m());
    double mj2 = abs (j2.m());
    double mcombj = abs (combj.m());

    if (mcombj < _mu)  return CLUSTER; // recombine
    else if (_theta*mcombj > std::max(mj1,mj2)) return VETO;
    else return NOVETO; // mass jump

  }

  // check the SoftDrop terminating veto
  HOTVR::VetoResult HOTVR::CheckVeto_SoftDrop(const PseudoJet& j1, const PseudoJet& j2) const {
    bool _debug=false;
    PseudoJet combj = j1+j2;
    double pt = combj.pt();
    double pt2 = combj.pt2();
    //double beam_R2 = _rho2/pt2;
    double beam_R2 = _rho2/pow(pt2,_alpha); // calculate effective radius with tunable exponent

    if      (beam_R2 > _max_r2){ beam_R2 = _max_r2;}
    else if (beam_R2 < _min_r2){ beam_R2 = _min_r2;}

    if (_debug) {
      std::cout << "Alpha " << _alpha << '\n';
      std::cout << "Beam R2 " << beam_R2 << '\n';
        }

    double beam_R = std::sqrt(beam_R2);
    double DeltaR = j1.delta_R(j2);

    double ptj1 = abs (j1.pt()); // pt values are always >0
    double ptj2 = abs (j2.pt());
    double ptcomb = abs (ptj1+ptj2);

    if(_debug){std::cout << "-----------------Check Veto-----------------------" << '\n';
    std::cout << "pt of 1. Jet "<< ptj1 << "pt of 2. Jet "<< ptj2 << '\n';
    std::cout << "pt_threshold" <<_pt_threshold << '\n';
    std::cout << "pt "<< pt << '\n';
    std::cout << "ptcomb "<< ptcomb << '\n';
    std::cout << "z_cut "<< _z_cut << '\n';
    std::cout << "DeltaR "<< DeltaR << '\n';
    std::cout << "beta "<< _beta << '\n';
    double a = std::min(ptj1, ptj2) /ptcomb;
    double b = (_z_cut * std::pow((DeltaR / beam_R),_beta));
    std::cout <<  a << " > " << b << '\n';}

    //if (ptj1 < _pt_threshold || ptj2 < _pt_threshold){ //ANNA 27.08. if we use this option we find subjets but do not store them, because more particles can be clustered to these jets and so the information about the subjets gets lost :(
    //if (ptcomb < _pt_threshold){ // ANNA leads to ca 20 subjets
    if(std::max(ptj1, ptj2)< _pt_threshold ){
        if(_debug){std::cout << ".............CLUSTER............." << '\n';}
        return CLUSTER; // recombine
    }
    else {
        if ( std::min(ptj1, ptj2) /ptcomb  >
              _z_cut * std::pow((DeltaR / beam_R),_beta)) { //check SD condition
          if(_debug){std::cout << ".............VETO............." << '\n';}
           return VETO; // if check pt > _ptsub + mu threshold for subjets, store subjets
        }
        else {
          if(_debug){std::cout << ".............NOVETO............." << '\n';}
           return NOVETO; // remove softer jet
        }
     }

  }

  // decide the optimal strategy
  // taken from VariableR/VariableRPlugin.cc, version 1.2.1
  HOTVR::Strategy HOTVR::best_strategy(unsigned int N) const{
#if FASTJET_VERSION_NUMBER >= 30200
    // use the FastJet (v>/3.1) transition between N2Plain and N2Tiled
    if (N <= 30 || N <= 39.0/(std::max(_max_r, 0.1) + 0.6)) return N2Plain;
    else return N2Tiled;
#else
    return NNH;
#endif
  }


} // namespace contrib

FASTJET_END_NAMESPACE
