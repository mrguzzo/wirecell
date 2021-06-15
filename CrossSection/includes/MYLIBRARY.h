// This file contains all the libraries that will be used throughout the script.

#ifndef MYLIBRARY
#define MYLIBRARY

// ROOT
#include "TChain.h"
#include "TString.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "THStack.h"
#include "TH1.h"
#include "TH2.h"
#include "TList.h"
#include "TRatioPlot.h"
#include "TLatex.h"
#include <iostream>
#include <cmath>
#include "TStyle.h"
#include <tuple>

// CHECKOUT
// --- This is where the variables from the checkout files are defined and
//     where the branches are called as well.
#include "pot.h"
#include "eval.h"
#include "kine.h"
#include "pfeval.h"
#include "tagger.h"

// MY FUNCTIONS
// --- This is where I define the functions I use throughout the script.
//     the idea is to have the simplest main code possible, and use functions
//     to do the rest of the job.
#include "my_functions.C"
#include "histogramHelper.C"

#endif