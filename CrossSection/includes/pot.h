/* --------------------------------------- */
struct POTInfo {
  Int_t runNo;
  Int_t subRunNo;
  Double_t pot_tor875;
  Double_t pot_tor875good;
  Double_t spill_tor875;
  Double_t spill_tor875good;

}; // POTInfo                                                                                                       


/* --------------------------------------- */
void loadPOTInfo(TChain* tree0, POTInfo& pot_info) {
  tree0->SetBranchAddress("runNo",&pot_info.runNo);
  tree0->SetBranchAddress("subRunNo", &pot_info.subRunNo);
  tree0->SetBranchAddress("pot_tor875", &pot_info.pot_tor875);
  tree0->SetBranchAddress("pot_tor875good", &pot_info.pot_tor875good);
  tree0->SetBranchAddress("spill_tor875", &pot_info.spill_tor875);
  tree0->SetBranchAddress("spill_tor875good", &pot_info.spill_tor875good);
} // loadPOTInfo                                                                                                    

