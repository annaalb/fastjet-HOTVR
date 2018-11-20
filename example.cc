//
//  The HOTVR FastJet contrib was written and is 
//  maintained and developed by:
//      Tobias Lapsien  <tobias.lapsien@desy.de>
//      Roman Kogler    <roman.kogler@uni-hamburg.de>
//      Johannes Haller <johannes.haller@uni-hamburg.de>
// 
//  Example showing the usage of the HOTVR algorithm
//
//  Compile with "make example" and run with
//    ./example < ../data/Pythia-Zp2jets-lhc-pileup-1ev.dat
//
//  The HOTVR code is based on the implementation of the ClusteringVetoPlugin 
//  version 1.0.0 (by Seng-Pei Liew and Martin Stoll)
//  and the VariableR plugin version 1.2.0 (by David Krohn, 
//  Gregory Soyez, Jesse Thaler and Lian-Tao Wang) in FastJet Contribs.
//  Please see the README file for more information.
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

#include <iostream>
#include <sstream>
#include <stdio.h>

#include "fastjet/PseudoJet.hh"
#include <sstream>
#include "HOTVRinfo.hh"
#include "HOTVR.hh" // In external code, this should be fastjet/contrib/HOTVR.hh

using namespace std;
using namespace fastjet;
using namespace contrib;

// forward declaration to make things clearer
void read_event(vector<PseudoJet> &event);
void print_jets(const vector<PseudoJet> &jets,
		const ClusterSequence &clust_seq);


//----------------------------------------------------------------------
int main(){

  // read in input particles
  vector<PseudoJet> event;
  read_event(event);
  cout << "# read an event with " << event.size() << " particles" << endl;

   //----------------------------------------------------------
  // Illustration of HOTVR with Cambridge-Aachen-like clustering
  //----------------------------------------------------------

  {
    // parameters for HOTVR (optimized for top-tagging)
    double mu(30.),     // massjump threshold
      theta(0.7),       // massjump parameter
      max_r(1.5),       // maximum allowed distance R
      min_r(0.1),       // minimum allowed distance R
      rho(600),         // cone shrinking parameter
      pt_threshold(30.), // minimum pT of subjets
      ptmin(5.);        // minimum pT of large jets
 
    // initialize plugin
    HOTVR hotvr_plugin(mu, theta,min_r, max_r,rho,pt_threshold, HOTVR::CALIKE); 
 
    // set up jet definition and cluster sequence
    JetDefinition jet_def(&hotvr_plugin);
    ClusterSequence clust_seq(event, jet_def);
 
    // print setup
    cout << endl << "Run " << jet_def.description() << endl;
    
    // get HOTVR inclusive jets (includes rejected pseudojets and jets without a mass jump)
    vector<fastjet::PseudoJet> jets = sorted_by_pt(clust_seq.inclusive_jets(ptmin));
  
    // print out inclusive jets
    cout << "\nInclusive jets with pT > " << ptmin << " GeV" << endl;
	  cout << "Number of jets: " << jets.size() << endl;
    print_jets(jets, clust_seq);
    
    // get HOTVR jets (the ones that saw a mass jump)
    std::vector<fastjet::PseudoJet> hotvr_jets;
    hotvr_jets = hotvr_plugin.get_jets();
    
    // print out  
    cout << "\nHOTVR jets " << endl;
    cout << "Number of HOTVR jets with mass jump: " << hotvr_jets.size() << endl;
    print_jets(hotvr_jets, clust_seq);
    
    // now access subjets with the info class and print them
    for(uint i=0;i<hotvr_jets.size();i++)
      {
        // vector with subjets of fatjet i
        // subjets saved in user_info class HOTVRinfo
        // each HOTVR jet has this info 

        // get the user info
        HOTVRinfo hi = hotvr_jets.at(i).user_info<HOTVRinfo>();

        // get the subjets
        std::vector<fastjet::PseudoJet> subjets;
        subjets = hi.subjets();

        // some print out
	      cout<<"\nSubjets for HOTVR jet " << i << endl;
	      cout << "Number of subjets: " << subjets.size() << endl;
       	print_jets(subjets,clust_seq);
      }
   
  }

  return 0;

}

// read in input particles
void read_event(vector<PseudoJet> &event){  
  string line;
  while (getline(cin, line)) {
    istringstream linestream(line);
    // take substrings to avoid problems when there are extra "pollution"
    // characters (e.g. line-feed).
    if (line.substr(0,4) == "#END") {return;}
    if (line.substr(0,1) == "#") {continue;}
    double px,py,pz,E;
    linestream >> px >> py >> pz >> E;
    PseudoJet particle(px,py,pz,E);

    // push event onto back of full_event vector
    event.push_back(particle);
  }
}

// prints a vector of jets
void print_jets(const vector<PseudoJet> &jets,
		const ClusterSequence &clust_seq){

  if (jets.size()==0) return;

  // columns labels
  if(jets.at(0).has_user_info<HOTVRinfo>()){
    printf("%5s %10s %10s %10s %10s %10s %10s\n",
					 "jet #", "pt", "rap", "phi", "m", "last d_ij", "max. distance");
  } else { 
    printf("%5s %10s %10s %10s %10s %10s\n",
	         "jet #", "pt", "rap", "phi", "m", "last d_ij");
  }

  // print out the jets
  for (unsigned i=0; i<jets.size(); ++i) {

    if(jets.at(0).has_user_info<HOTVRinfo>()){
      printf("%5u %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", 
             i, jets[i].pt(), jets[i].rap(), jets[i].phi(), jets[i].m(), 
             clust_seq.exclusive_subdmerge ( jets[i], 1 ),
						 jets[i].user_info<HOTVRinfo>().max_distance());
    } else  {
      printf("%5u %10.3f %10.3f %10.3f %10.3f %10.3f\n", 
             i, jets[i].pt(), jets[i].rap(), jets[i].phi(), jets[i].m(), 
             clust_seq.exclusive_subdmerge ( jets[i], 1 ));
    }
  }

}



