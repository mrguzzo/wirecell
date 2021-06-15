struct TaggerInfo {
  
  //  float match_isFC;
  float numu_cc_flag;
  float numu_score;
  float nue_score;
  
}; // TaggerInfo

void loadTaggerInfo(TChain* tree0, TaggerInfo& tagger_info, int flag=1) {  
  //  tree0->SetBranchAddress("match_isFC",&tagger_info.match_isFC);
  tree0->SetBranchAddress("numu_cc_flag",&tagger_info.numu_cc_flag);
  tree0->SetBranchAddress("numu_score",&tagger_info.numu_score);
  tree0->SetBranchAddress("nue_score",&tagger_info.nue_score);
} // loadTaggerInfo
