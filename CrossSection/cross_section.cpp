#include "includes/MYLIBRARY.h"

void cross_section(){

    // ======================================================================================= //
    //                                      OPENING FILES                                      //
    // ======================================================================================= //

    std::cout << "Opening files..." << std::endl;

    // MC
    TChain* T_pot = new TChain("wcpselection/T_pot"); T_pot->Add(filename_mc.c_str()); POTInfo pot_info; loadPOTInfo(T_pot, pot_info);
    TChain* T_kine = new TChain("wcpselection/T_KINEvars"); T_kine->Add(filename_mc.c_str()); KineInfo kine_info; loadKineInfo(T_kine, kine_info);
    TChain* T_eval = new TChain("wcpselection/T_eval"); T_eval->Add(filename_mc.c_str()); EvalInfo eval_info; loadEvalInfo(T_eval, eval_info);
    TChain* T_pfeval = new TChain("wcpselection/T_PFeval"); T_pfeval->Add(filename_mc.c_str()); PFevalInfo pfeval_info; loadPFevalInfo(T_pfeval, pfeval_info);
    TChain* T_tagger = new TChain("wcpselection/T_BDTvars"); T_tagger->Add(filename_mc.c_str()); TaggerInfo tagger_info; loadTaggerInfo(T_tagger, tagger_info);

    // intrinsic nue 
    TChain* T_pot_nue = new TChain("wcpselection/T_pot"); T_pot_nue->Add(filename_nue.c_str()); POTInfo pot_info_nue; loadPOTInfo(T_pot_nue, pot_info_nue);
    TChain* T_kine_nue = new TChain("wcpselection/T_KINEvars"); T_kine_nue->Add(filename_nue.c_str()); KineInfo kine_info_nue; loadKineInfo(T_kine_nue, kine_info_nue);
    TChain* T_eval_nue = new TChain("wcpselection/T_eval"); T_eval_nue->Add(filename_nue.c_str()); EvalInfo eval_info_nue; loadEvalInfo(T_eval_nue, eval_info_nue);
    TChain* T_pfeval_nue = new TChain("wcpselection/T_PFeval"); T_pfeval_nue->Add(filename_nue.c_str()); PFevalInfo pfeval_info_nue; loadPFevalInfo(T_pfeval_nue, pfeval_info_nue);
    TChain* T_tagger_nue = new TChain("wcpselection/T_BDTvars"); T_tagger_nue->Add(filename_nue.c_str()); TaggerInfo tagger_info_nue; loadTaggerInfo(T_tagger_nue, tagger_info_nue);

    // EXT
    TChain* T_kine_ext = new TChain("wcpselection/T_KINEvars"); T_kine_ext->Add(filename_ext.c_str()); KineInfo kine_info_ext; loadKineInfo(T_kine_ext, kine_info_ext);
    TChain* T_eval_ext = new TChain("wcpselection/T_eval"); T_eval_ext->Add(filename_ext.c_str()); EvalInfo eval_info_ext; loadEvalInfo(T_eval_ext, eval_info_ext, 0);
    TChain* T_tagger_ext = new TChain("wcpselection/T_BDTvars"); T_tagger_ext->Add(filename_ext.c_str()); TaggerInfo tagger_info_ext; loadTaggerInfo(T_tagger_ext, tagger_info_ext);
    TChain* T_pfeval_ext = new TChain("wcpselection/T_PFeval"); T_pfeval_ext->Add(filename_ext.c_str()); PFevalInfo pfeval_info_ext; loadPFevalInfo(T_pfeval_ext, pfeval_info_ext, 0);

    // DATA
    TChain* T_kine_data = new TChain("wcpselection/T_KINEvars"); T_kine_data->Add(filename_data.c_str()); KineInfo kine_info_data; loadKineInfo(T_kine_data, kine_info_data);
    TChain* T_eval_data = new TChain("wcpselection/T_eval"); T_eval_data->Add(filename_data.c_str()); EvalInfo eval_info_data; loadEvalInfo(T_eval_data, eval_info_data, 0);
    TChain* T_tagger_data = new TChain("wcpselection/T_BDTvars"); T_tagger_data->Add(filename_data.c_str()); TaggerInfo tagger_info_data; loadTaggerInfo(T_tagger_data, tagger_info_data);
    TChain* T_pfeval_data = new TChain("wcpselection/T_PFeval");  T_pfeval_data->Add(filename_data.c_str()); PFevalInfo pfeval_info_data;  loadPFevalInfo(T_pfeval_data, pfeval_info_data, 0);

    // ===================================================================================== //
    //                                      GETTING POT                                      //
    // ===================================================================================== //

    // get overlay POT
    for (int n = 0; n < T_pot->GetEntries(); ++n) {
        T_pot->GetEntry(n);
        MCPOT += (float)pot_info.pot_tor875;
    }

    // get intrinsic nue POT
    for (int n = 0; n < T_pot_nue->GetEntries(); ++n) {
        T_pot_nue->GetEntry(n);
        NuePOT += (float)pot_info_nue.pot_tor875;
    }

    // ========================================================================================= //
    //                                      GETTING WEIGHTS                                      //
    // ========================================================================================= //

    wMC = dataPOT/MCPOT; // POT weight   
    wNue = dataPOT/NuePOT; // nue POT weight 
    wEXT = 0.98*2393691.0/2993141.315; // EXT weight
    if (MCPOT == 0) wMC = 1;
    if (NuePOT == 0) wMC = 1;

    std::cout   << "-----------------------------" << "\n"
                << "data POT:     " << dataPOT << "\n"
                << "MC POT:       " << MCPOT << "\n"
                << "Nue POT:      " << NuePOT << "\n"
                << "-----------------------------" << "\n"
                << "MC weight:    " << wMC << "\n"
                << "Nue weight:   " << wNue << "\n"
                << "EXT weight:   " << wEXT << "\n"
                << "data weight:  " << wData << "\n"
                << "-----------------------------" << std::endl;

    // ============================================================================================= //
    //                                      PROCESS INPUT FILES                                      //
    // ============================================================================================= //

    // =============== //
    //    MC SAMPLE    //
    // =============== //

    int nevents = T_eval->GetEntries();

    for(unsigned int n=0; n<nevents; n++){

        if (n%10000 == 0) std::cout << "[MC] processing event " << n << "/" << nevents << std::endl;

        // get entries
        T_eval->GetEntry(n);
        T_pfeval->GetEntry(n);
        T_kine->GetEntry(n);
        T_tagger->GetEntry(n);

        // fix weights
        fixWeights(eval_info.weight_cv, eval_info.weight_spline);

        // get true topology
        int top = getTopology(k_file_mc, eval_info, pfeval_info);

        // get weight
        float genie_weight = getGenieWeight(k_file_mc, eval_info);

        // fill stacked histograms
        if (top>-1){

        }
    }

    // ======================== //
    //   INTRINSIC NUE SAMPLE   //
    // ======================== //

    nevents = T_eval_nue->GetEntries();

    for(unsigned int n=0; n<nevents; n++){

        if (n%10000 == 0) std::cout << "[Nue] processing event " << n << "/" << nevents << std::endl;

        // get entries
        T_eval_nue->GetEntry(n);
        T_pfeval_nue->GetEntry(n);
        T_kine_nue->GetEntry(n);
        T_tagger_nue->GetEntry(n);

        // fix weights
        fixWeights(eval_info_nue.weight_cv, eval_info_nue.weight_spline);

        // get true topology
        int top = getTopology(k_file_nue, eval_info_nue, pfeval_info_nue);

        // get weight
        float genie_weight = getGenieWeight(k_file_nue, eval_info_nue);

        // fill stacked histograms
        if (top>-1)
    }

    // ============== //
    //   EXT SAMPLE   //
    // ============== //

    nevents = T_eval_ext->GetEntries();

    for (int n = 0; n < nevents; ++n) {

        if (n%10000 == 0) std::cout << "[EXT] processing event " << n << "/" << nevents << std::endl;

        T_eval_ext->GetEntry(n);
        T_kine_ext->GetEntry(n);
        T_tagger_ext->GetEntry(n);

        // fill stacked histograms
        //fillStackedHists(eval_info_ext, tagger_info_ext, kine_info_ext, ext, wEXT);

    }

    // =============== //
    //   DATA SAMPLE   //
    // =============== //

    nevents = T_eval_data->GetEntries();

    for (int n = 0; n < nevents; ++n) {

        if (n%10000 == 0) std::cout << "[Data] processing event " << n << "/" << nevents << std::endl;

        T_eval_data->GetEntry(n);
        T_kine_data->GetEntry(n);
        T_tagger_data->GetEntry(n);

        // stacked histograms
        //fillStackedHists(eval_info_data, tagger_info_data, kine_info_data, data, wData);

    }

}