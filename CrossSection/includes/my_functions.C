#include "MYLIBRARY.h"

// ===================== //
//   INPUT INFORMATION   //
// ===================== //

// if you change any file, make sure to update their POT as well

string filename_data = "rootfiles/checkout_data_numi_run1_morestat.root";
string filename_ext = "rootfiles/checkout_data_extnumi_run1.root";
string filename_mc = "rootfiles/checkout_prodgenie_numi_overlay_run1_OFFSETFIXED2.root";
string filename_nue = "rootfiles/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root";

const double dataPOT = 2.064e+50;
const double EXTPOT = 1.1603984e+50;
double MCPOT; // calculated in the code
double NuePOT; // calculated in the code

float wData = 1;
float wEXT; // calculated in the code
float wNue; // calculated in the code
float wMC; // calculated in the code

// ========================== //
//   DECLARE VARIABLES HERE   //
// ========================== //

enum file_type {k_file_data=0, k_file_ext, k_file_mc, k_file_nue, k_FILE_TYPE_MAX};
enum topologies_list {k_ccnue=0, k_ccnuebar, k_ncpi0, k_nc, k_ccnumupi0, k_ccnumu, k_outFV, k_cosmic, k_dirt, k_ext, k_data, k_MAX_topologies};

// ========================== //
//   DECLARE FUNCTIONS HERE   //
// ========================== //

void fixWeights(float &weight_cv, float &weight_spline);                     // fix weight_cv, weight_spline to a finite number
int getTopology(int file, EvalInfo &eval, PFevalInfo &pfeval);                // returns the topology of the event
float getGenieWeight(EvalInfo &e);                                            // calculate the GENIE weight for the overlay and intrinsic nue samples

// ========================= //
//   DEFINE FUNCTIONS HERE   //
// ========================= //

void fixWeights(float &weight_cv, float &weight_spline){

  // --- weight_cv
  if     ( weight_cv <= 0 ) weight_cv = 1;
  else if( weight_cv > 30 ) weight_cv = 1;
  else if( isnan(weight_cv) ) weight_cv = 1;
  else if( isinf(weight_cv) ) weight_cv = 1;

  // --- weight_spline
  if     ( weight_spline <= 0 ) weight_spline = 1;
  else if( weight_spline > 30 ) weight_spline = 1;
  else if( isnan(weight_spline) ) weight_spline = 1;
  else if( isinf(weight_spline) ) weight_spline = 1;

}

int getTopology(int file, EvalInfo &eval, PFevalInfo &pfeval){

    int top;

    if( file == k_file_mc ){

        bool isCosmic = false;
        bool isNueCC = false;
        bool isNueBarCC = false;
        bool isNumuCC = false;
        bool isNC = false; 
        bool hasPi0 = false;
        bool hasChargedPi = false;
        bool hasProton = false; 
        bool isOutFV = false;

        // classify true information
        if (eval.truth_energyInside!=0 && (eval.match_completeness_energy/eval.truth_energyInside<0.1)) isCosmic = true;
        if (eval.match_completeness_energy/eval.truth_energyInside>=0.1 && eval.truth_vtxInside==0) isOutFV = true;
        if (pfeval.truth_NprimPio>0) hasPi0 = true;
        if (eval.truth_isCC){
            if (eval.truth_nuPdg==12) isNueCC = true;
            else if (eval.truth_nuPdg==-12) isNueBarCC = true;
            else if (abs(eval.truth_nuPdg)==14) isNumuCC = true;
        }
        else isNC = true;

        // topology
        top = -1;
        if (isCosmic) top = k_cosmic;
        else if (isOutFV) top = k_outFV;
        else if (isNueCC) top = -1; // use intrinsic nue sample
        else if (isNueBarCC) top = -1; // use intrinsic nue sample
        else if (isNumuCC){
            if (hasPi0) top = k_ccnumupi0;
            else top = k_ccnumu;
        }
        else if (isNC){
            if (hasPi0) top = k_ncpi0;
            else top = k_nc;
        }

    }

    else if ( file == k_file_nue ){

        bool isOutFV = false;
        bool isCosmic = false;
        bool isNueCC = false;
        bool isNueBarCC = false;

        // classify true information
        if (eval.truth_energyInside!=0 && (eval.match_completeness_energy/eval.truth_energyInside<0.1)) isCosmic = true;
        if (eval.match_completeness_energy/eval.truth_energyInside>=0.1 && eval.truth_vtxInside==0) isOutFV = true;
        if (eval.truth_isCC){
            if (eval.truth_nuPdg==12) isNueCC = true;
            else if (eval.truth_nuPdg==-12) isNueBarCC = true;
        }

        // topology
        if (isCosmic) top = k_cosmic;
        else if (isOutFV) top = k_outFV;
        else if (isNueCC) top = k_ccnue;
        else if (isNueBarCC) top = k_ccnuebar;
    }

    return top;

}

float getGenieWeight(int file, EvalInfo &eval){

    float genie_weight;

    if( file==k_file_mc || file==k_file_nue ){
        genie_weight = (float)(eval.weight_cv * eval.weight_spline);
    }

    return genie_weight;
}