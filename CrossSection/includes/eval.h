struct EvalInfo{
  bool is_match_found_int;
  bool is_file_type;
    
  Int_t run;
  Int_t subrun;
  Int_t event;
  Bool_t flash_found;
  Int_t flash_found_asInt;
  Float_t flash_time;
    
  Float_t flash_measPe;
  Float_t flash_predPe;
  Bool_t match_found;
  Int_t match_found_asInt;
  UInt_t match_type;
  Bool_t match_isFC;
    
  Bool_t match_isTgm;
  Bool_t match_notFC_FV;
  Bool_t match_notFC_SP;
  Bool_t match_notFC_DC;
  Float_t match_chargeU;
    
  Float_t match_chargeV;
  Float_t match_chargeY;
  Float_t match_energyY;
  Bool_t light_mismatch;
  Float_t match_charge;
    
  Float_t match_energy;
  Int_t stm_eventtype;
  Int_t stm_lowenergy;
  Int_t stm_LM;
  Int_t stm_TGM;
    
  Int_t stm_STM;
  Int_t stm_FullDead;
  Float_t stm_clusterlength;
  Float_t truth_nuEnergy;
  Float_t truth_energyInside;
    
  Float_t truth_electronInside;
  Int_t truth_nuPdg;
  Bool_t truth_isCC;
  Bool_t truth_isEligible;
  Bool_t truth_NCisEligible;
    
  Bool_t truth_isFC;
  Bool_t truth_vtxInside;
  Float_t truth_vtxX;
  Float_t truth_vtxY;
  Float_t truth_vtxZ;
    
  Float_t truth_nuTime;
  Float_t match_completeness;
  Float_t match_completeness_energy;
  Float_t match_purity;
  Float_t match_purity_xz;
    
  Float_t match_purity_xy;
  Float_t weight_spline;
  Float_t weight_cv;
  Float_t weight_lee;
    
  Bool_t weight_change;
  
}; // EvalInfo

void loadEvalInfo(TChain* tree0, EvalInfo& eval_info, int flag=1) {
  eval_info.is_match_found_int = false;
  eval_info.is_file_type = false;
  
  tree0->SetBranchAddress("run", &eval_info.run);
  tree0->SetBranchAddress("subrun", &eval_info.subrun);
  tree0->SetBranchAddress("event", &eval_info.event);
  tree0->SetBranchAddress("flash_found", &eval_info.flash_found);
  if (tree0->GetBranch("flash_found_asInt"))   {
    tree0->SetBranchAddress("flash_found_asInt", &eval_info.flash_found_asInt);
    eval_info.is_match_found_int = true;
  }

  tree0->SetBranchAddress("flash_time", &eval_info.flash_time);

  tree0->SetBranchAddress("flash_measPe", &eval_info.flash_measPe);
  tree0->SetBranchAddress("flash_predPe", &eval_info.flash_predPe);
  tree0->SetBranchAddress("match_found", &eval_info.match_found);
  if (tree0->GetBranch("match_found_asInt"))   tree0->SetBranchAddress("match_found_asInt", &eval_info.match_found_asInt); 
  tree0->SetBranchAddress("match_type", &eval_info.match_type);
  tree0->SetBranchAddress("match_isFC", &eval_info.match_isFC);

  tree0->SetBranchAddress("match_isTgm", &eval_info.match_isTgm);
  tree0->SetBranchAddress("match_notFC_FV", &eval_info.match_notFC_FV);
  tree0->SetBranchAddress("match_notFC_SP", &eval_info.match_notFC_SP);
  tree0->SetBranchAddress("match_notFC_DC", &eval_info.match_notFC_DC);
  tree0->SetBranchAddress("match_chargeU", &eval_info.match_chargeU);
  
  tree0->SetBranchAddress("match_chargeV", &eval_info.match_chargeV);
  tree0->SetBranchAddress("match_chargeY", &eval_info.match_chargeY);
  tree0->SetBranchAddress("match_energyY", &eval_info.match_energyY);
  tree0->SetBranchAddress("light_mismatch", &eval_info.light_mismatch);
  tree0->SetBranchAddress("match_charge", &eval_info.match_charge);

  tree0->SetBranchAddress("match_energy", &eval_info.match_energy);
  tree0->SetBranchAddress("stm_eventtype", &eval_info.stm_eventtype);
  tree0->SetBranchAddress("stm_lowenergy", &eval_info.stm_lowenergy);
  tree0->SetBranchAddress("stm_LM", &eval_info.stm_LM);
  tree0->SetBranchAddress("stm_TGM", &eval_info.stm_TGM);

  tree0->SetBranchAddress("stm_STM", &eval_info.stm_STM);
  tree0->SetBranchAddress("stm_FullDead", &eval_info.stm_FullDead);
  tree0->SetBranchAddress("stm_clusterlength", &eval_info.stm_clusterlength);

  if (flag==1){
    tree0->SetBranchAddress("truth_nuEnergy", &eval_info.truth_nuEnergy);
    tree0->SetBranchAddress("truth_energyInside", &eval_info.truth_energyInside);
    
    tree0->SetBranchAddress("truth_electronInside", &eval_info.truth_electronInside);
    tree0->SetBranchAddress("truth_nuPdg", &eval_info.truth_nuPdg);
    tree0->SetBranchAddress("truth_isCC", &eval_info.truth_isCC);
    tree0->SetBranchAddress("truth_isEligible", &eval_info.truth_isEligible);
    tree0->SetBranchAddress("truth_NCisEligible", &eval_info.truth_NCisEligible);
    
    tree0->SetBranchAddress("truth_isFC", &eval_info.truth_isFC);
    tree0->SetBranchAddress("truth_vtxInside", &eval_info.truth_vtxInside);
    tree0->SetBranchAddress("truth_vtxX", &eval_info.truth_vtxX);
    tree0->SetBranchAddress("truth_vtxY", &eval_info.truth_vtxY);
    tree0->SetBranchAddress("truth_vtxZ", &eval_info.truth_vtxZ);
    
    tree0->SetBranchAddress("truth_nuTime", &eval_info.truth_nuTime);
    tree0->SetBranchAddress("match_completeness", &eval_info.match_completeness);
    tree0->SetBranchAddress("match_completeness_energy", &eval_info.match_completeness_energy);
    tree0->SetBranchAddress("match_purity", &eval_info.match_purity);
    tree0->SetBranchAddress("match_purity_xz", &eval_info.match_purity_xz);
    
    tree0->SetBranchAddress("match_purity_xy", &eval_info.match_purity_xy);
    tree0->SetBranchAddress("weight_spline", &eval_info.weight_spline);
    tree0->SetBranchAddress("weight_cv", &eval_info.weight_cv);
    tree0->SetBranchAddress("weight_lee", &eval_info.weight_lee);
    
    if (tree0->GetBranch("weight_change")) tree0->SetBranchAddress("weight_change",&eval_info.weight_change);
  }
} // loadEvalInfo
