
import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import gridspec

import uproot3 as uproot

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, accuracy_score, f1_score

from xgboost import XGBClassifier

legend_size = 12


#===========================================================================================================================================================================================
# SELECT VARIABLES TO IMPORT FROM TREES

KINE_vars = ['kine_reco_Enu', 'kine_reco_add_energy', 'kine_pio_mass', 'kine_pio_flag', 'kine_pio_vtx_dis', 'kine_pio_energy_1', 'kine_pio_theta_1', 'kine_pio_phi_1', 'kine_pio_dis_1', 'kine_pio_energy_2', 'kine_pio_theta_2', 'kine_pio_phi_2', 'kine_pio_dis_2', 'kine_pio_angle']

BDT_variab_all = ['cosmic_n_solid_tracks', 'cosmic_energy_main_showers', 'cosmic_energy_direct_showers', 'cosmic_energy_indirect_showers', 'cosmic_n_direct_showers', 'cosmic_n_indirect_showers', 'cosmic_n_main_showers', 
                  'gap_flag_prolong_u', 'gap_flag_prolong_v', 'gap_flag_prolong_w', 'gap_flag_parallel', 'gap_n_points', 'gap_n_bad', 'gap_energy', 'gap_num_valid_tracks', 'gap_flag_single_shower', 
                  'mip_quality_energy', 'mip_quality_overlap', 'mip_quality_n_showers', 'mip_quality_n_tracks', 'mip_quality_flag_inside_pi0', 'mip_quality_n_pi0_showers', 'mip_quality_shortest_length', 'mip_quality_acc_length', 'mip_quality_shortest_angle', 'mip_quality_flag_proton',
                  'mip_energy', 'mip_n_end_reduction', 'mip_n_first_mip', 'mip_n_first_non_mip', 'mip_n_first_non_mip_1', 'mip_n_first_non_mip_2', 'mip_vec_dQ_dx_0', 'mip_vec_dQ_dx_1', 'mip_vec_dQ_dx_2', 'mip_vec_dQ_dx_3', 'mip_vec_dQ_dx_4', 'mip_vec_dQ_dx_5', 'mip_vec_dQ_dx_6', 'mip_vec_dQ_dx_7', 'mip_vec_dQ_dx_8',
                  'mip_vec_dQ_dx_9', 'mip_vec_dQ_dx_10',
                  'mip_vec_dQ_dx_11', 'mip_vec_dQ_dx_12',
                  'mip_vec_dQ_dx_13', 'mip_vec_dQ_dx_14',
                  'mip_vec_dQ_dx_15', 'mip_vec_dQ_dx_16',
                  'mip_vec_dQ_dx_17', 'mip_vec_dQ_dx_18',
                  'mip_vec_dQ_dx_19', 'mip_max_dQ_dx_sample',
                  'mip_n_below_threshold', 'mip_n_below_zero',
                  'mip_n_lowest', 'mip_n_highest',
                  'mip_lowest_dQ_dx', 'mip_highest_dQ_dx',
                  'mip_medium_dQ_dx', 'mip_stem_length',
                  'mip_length_main', 'mip_length_total',
                  'mip_angle_beam', 'mip_iso_angle',
                  'mip_n_vertex', 'mip_n_good_tracks',
                  'mip_E_indirect_max_energy', 'mip_flag_all_above',
                  'mip_min_dQ_dx_5', 'mip_n_other_vertex',
                  'mip_n_stem_size', 'mip_flag_stem_trajectory',
                  'mip_min_dis', 
                  # -----------------------------------------------
                  'pio_mip_id', 'pio_flag_pio', 
                  # -----------------------------------------------
                  'pio_1_mass', 'pio_1_pio_type', 
                  'pio_1_energy_1', 'pio_1_energy_2', 
                  'pio_1_dis_1', 'pio_1_dis_2', 
                  # -----------------------------------------------
                  'mgo_energy',
                  'mgo_max_energy', 'mgo_total_energy',
                  'mgo_n_showers', 'mgo_max_energy_1',
                  'mgo_max_energy_2', 'mgo_total_other_energy',
                  'mgo_n_total_showers', 'mgo_total_other_energy_1',
                  # -----------------------------------------------
                  'mgt_flag_single_shower', 'mgt_max_energy',
                  'mgt_energy', 'mgt_total_other_energy',
                  'mgt_max_energy_1', 'mgt_e_indirect_max_energy',
                  'mgt_e_direct_max_energy', 'mgt_n_direct_showers',
                  'mgt_e_direct_total_energy', 'mgt_flag_indirect_max_pio',
                  'mgt_e_indirect_total_energy',
                  # -----------------------------------------------
                  'stw_1_energy', 'stw_1_dis',
                  'stw_1_dQ_dx', 'stw_1_flag_single_shower',
                  'stw_1_n_pi0', 'stw_1_num_valid_tracks',
                  # -----------------------------------------------
                  'spt_flag_single_shower', 'spt_energy',
                  'spt_shower_main_length', 'spt_shower_total_length',
                  'spt_angle_beam', 'spt_angle_vertical',
                  'spt_max_dQ_dx', 'spt_angle_beam_1',
                  'spt_angle_drift', 'spt_angle_drift_1',
                  'spt_num_valid_tracks', 'spt_n_vtx_segs',
                  'spt_max_length', 
                  # -----------------------------------------------
                  'stem_len_energy', 'stem_len_length',
                  'stem_len_flag_avoid_muon_check',
                  'stem_len_num_daughters', 'stem_len_daughter_length',
                  # -----------------------------------------------
                  'lem_shower_total_length',
                  'lem_shower_main_length', 'lem_n_3seg',
                  'lem_e_charge', 'lem_e_dQdx',
                  'lem_shower_num_segs', 'lem_shower_num_main_segs',
                  # -----------------------------------------------
                  'brm_n_mu_segs', 'brm_Ep',
                  'brm_energy', 'brm_acc_length',
                  'brm_shower_total_length', 'brm_connected_length',
                  'brm_n_size', 'brm_acc_direct_length',
                  'brm_n_shower_main_segs', 'brm_n_mu_main',
                  # -----------------------------------------------
                  'cme_mu_energy', 'cme_energy',
                  'cme_mu_length', 'cme_length',
                  'cme_angle_beam',
                  # -----------------------------------------------
                  'anc_energy', 'anc_angle',
                  'anc_max_angle', 'anc_max_length',
                  'anc_acc_forward_length', 'anc_acc_backward_length',
                  'anc_acc_forward_length1', 'anc_shower_main_length',
                  'anc_shower_total_length', 'anc_flag_main_outside',
                  # -----------------------------------------------
                  'stem_dir_flag_single_shower', 'stem_dir_angle',
                  'stem_dir_energy', 'stem_dir_angle1',
                  'stem_dir_angle2', 'stem_dir_angle3',
                  'stem_dir_ratio',
                  # -----------------------------------------------
                  'vis_1_n_vtx_segs', 'vis_1_energy',
                  'vis_1_num_good_tracks', 'vis_1_max_angle',
                  'vis_1_max_shower_angle', 'vis_1_tmp_length1',
                  'vis_1_tmp_length2', 'vis_1_particle_type',                                      
                  # -----------------------------------------------
                  'vis_2_n_vtx_segs', 'vis_2_min_angle',
                  'vis_2_min_weak_track', 'vis_2_angle_beam',
                  'vis_2_min_angle1', 'vis_2_iso_angle1',
                  'vis_2_min_medium_dQ_dx', 'vis_2_min_length',
                  'vis_2_sg_length', 'vis_2_max_angle',
                  'vis_2_max_weak_track',
                  # -----------------------------------------------
                  'br1_1_shower_type',
                  'br1_1_vtx_n_segs', 'br1_1_energy',
                  'br1_1_n_segs', 'br1_1_flag_sg_topology',
                  'br1_1_flag_sg_trajectory', 'br1_1_sg_length',
                  # -----------------------------------------------
                  'br1_2_energy', 'br1_2_n_connected',
                  'br1_2_max_length', 'br1_2_n_connected_1',
                  'br1_2_vtx_n_segs', 'br1_2_n_shower_segs',
                  'br1_2_max_length_ratio', 'br1_2_shower_length',
                  # -----------------------------------------------
                  'br1_3_energy', 'br1_3_n_connected_p',
                  'br1_3_max_length_p', 'br1_3_n_shower_segs',
                  'br1_3_flag_sg_topology', 'br1_3_flag_sg_trajectory',
                  'br1_3_n_shower_main_segs', 'br1_3_sg_length',
                  # -----------------------------------------------
                  'br2_flag_single_shower', 'br2_num_valid_tracks',
                  'br2_energy', 'br2_angle1',
                  'br2_angle2', 'br2_angle',
                  'br2_angle3', 'br2_n_shower_main_segs',
                  'br2_max_angle', 'br2_sg_length',
                  'br2_flag_sg_trajectory',                                     
                  # -----------------------------------------------
                  'br3_1_n_shower_segments', 'br3_1_sg_flag_trajectory',
                  'br3_1_sg_direct_length', 'br3_1_sg_length',
                  'br3_1_total_main_length', 'br3_1_total_length',
                  'br3_1_iso_angle', 'br3_1_sg_flag_topology',
                  # -----------------------------------------------
                  'br3_2_n_ele', 'br3_2_n_other',
                  'br3_2_energy', 'br3_2_total_main_length',
                  'br3_2_total_length', 'br3_2_other_fid',
                  # -----------------------------------------------
                  'br3_4_acc_length', 'br3_4_total_length',
                  'br3_4_energy', 
                  # -----------------------------------------------
                  'br3_7_energy', 'br3_7_min_angle', 
                  'br3_7_sg_length', 'br3_7_main_length', 
                  # -----------------------------------------------
                  'br3_8_max_dQ_dx', 'br3_8_energy', 'br3_8_n_main_segs',
                  'br3_8_shower_main_length', 'br3_8_shower_length',
                  # -----------------------------------------------
                  'br4_1_shower_main_length', 'br4_1_shower_total_length',
                  'br4_1_min_dis', 'br4_1_energy',
                  'br4_1_n_vtx_segs', 'br4_1_n_main_segs',
                  # -----------------------------------------------
                  'br4_2_ratio_45', 'br4_2_ratio_35',
                  'br4_2_ratio_25', 'br4_2_ratio_15',
                  'br4_2_energy', 'br4_2_ratio1_45',
                  'br4_2_ratio1_35', 'br4_2_ratio1_25',
                  'br4_2_ratio1_15', 'br4_2_iso_angle',
                  'br4_2_iso_angle1', 'br4_2_angle',                                     
                  # -----------------------------------------------
                  'tro_3_stem_length', 'tro_3_n_muon_segs',
                  'tro_3_energy',
                  # -----------------------------------------------
                  'hol_1_n_valid_tracks', 'hol_1_min_angle',
                  'hol_1_energy', 'hol_1_flag_all_shower',
                  'hol_1_min_length', 
                  # -----------------------------------------------
                  'hol_2_min_angle', 'hol_2_medium_dQ_dx',
                  'hol_2_ncount', 'hol_2_energy',
                  # -----------------------------------------------
                  'lol_3_angle_beam', 'lol_3_n_valid_tracks', 
                  'lol_3_min_angle', 'lol_3_vtx_n_segs', 
                  'lol_3_energy', 'lol_3_shower_main_length', 
                  'lol_3_n_out', 'lol_3_n_sum',
                  # -----------------------------------------------
                  'cosmict_2_particle_type', 'cosmict_2_n_muon_tracks',
                  'cosmict_2_flag_inside', 'cosmict_2_angle_beam',
                  'cosmict_2_flag_dir_weak', 'cosmict_2_dQ_dx_end',
                  'cosmict_2_dQ_dx_front', 'cosmict_2_theta',
                  'cosmict_2_phi', 'cosmict_2_valid_tracks',
                  # -----------------------------------------------
                  'cosmict_3_flag_inside',
                  'cosmict_3_angle_beam', 'cosmict_3_flag_dir_weak',
                  'cosmict_3_dQ_dx_end', 'cosmict_3_dQ_dx_front',
                  'cosmict_3_theta', 'cosmict_3_phi',
                  'cosmict_3_valid_tracks', 
                  # -----------------------------------------------
                  'cosmict_4_flag_inside', 'cosmict_4_angle_beam',
                  # -----------------------------------------------
                  'cosmict_5_flag_inside', 'cosmict_5_angle_beam', 
                  # -----------------------------------------------
                  'cosmict_6_flag_dir_weak', 'cosmict_6_flag_inside',
                  'cosmict_6_angle', 
                  # -----------------------------------------------
                  'cosmict_7_flag_sec', 'cosmict_7_n_muon_tracks',
                  'cosmict_7_flag_inside', 'cosmict_7_angle_beam',
                  'cosmict_7_flag_dir_weak', 'cosmict_7_dQ_dx_end',
                  'cosmict_7_dQ_dx_front', 'cosmict_7_theta',
                  'cosmict_7_phi', 
                  # -----------------------------------------------
                  'cosmict_8_flag_out', 'cosmict_8_muon_length',
                  'cosmict_8_acc_length',
                  # -----------------------------------------------
                  'numu_cc_3_particle_type',
                  'numu_cc_3_max_length', 'numu_cc_3_track_length',
                  'numu_cc_3_max_length_all', 'numu_cc_3_max_muon_length',
                  'numu_cc_3_n_daughter_all', 
                  # -----------------------------------------------                                      
                  'pio_2_score', 'sig_1_score',
                  'sig_2_score', 'stw_2_score',
                  'stw_3_score', 'stw_4_score',
                  'br3_3_score', 'br3_5_score',
                  'br3_6_score', 'lol_1_score',
                  'lol_2_score', 'tro_1_score',
                  'tro_2_score', 'tro_4_score',
                  'tro_5_score', 'cosmict_10_score',
                  'numu_1_score', 'numu_2_score',
                  'numu_score', 'nue_score',
                  'cosmict_flag', 'numu_cc_flag']

Non_unique = ['mip_quality_energy', 'mgo_energy', 'mgt_energy', 'stw_1_energy', 'spt_energy', 
              'stem_len_energy', 'brm_energy', 'cme_energy', 'anc_energy', 'stem_dir_energy', 
              'br1_1_energy', 'br1_2_energy', 'br1_3_energy', 'br2_energy', 'br3_2_energy', 
              'br3_4_energy', 'br3_7_energy', 'br3_8_energy', 'br4_1_energy', 'br4_2_energy', 
              'tro_3_energy', 'lol_3_energy', 'br2_num_valid_tracks', 'mgt_flag_single_shower', 
              'stw_1_flag_single_shower', 'spt_flag_single_shower', 'stem_dir_flag_single_shower', 
              'br2_flag_single_shower', 'spt_angle_drift', 'mgt_max_energy', 'br1_1_flag_sg_trajectory', 
              'br1_3_flag_sg_trajectory', 'br2_flag_sg_trajectory', 'br3_1_sg_flag_trajectory', 
              'spt_max_dQ_dx', 'lem_shower_main_length', 'anc_shower_main_length', 
              'br3_1_total_main_length', 'br3_2_total_main_length', 'br3_4_total_length', 
              'br3_7_main_length', 'br3_8_shower_main_length', 'br4_1_shower_main_length', 
              'lol_3_shower_main_length', 'lem_shower_total_length', 'brm_shower_total_length', 
              'cme_length', 'anc_shower_total_length', 'br1_2_shower_length', 'br3_2_total_length', 
              'br3_8_shower_length', 'br4_1_shower_total_length', 'anc_angle', 'br1_1_vtx_n_segs', 
              'br1_2_vtx_n_segs', 'br4_1_n_vtx_segs', 'lol_3_vtx_n_segs', 'br1_1_sg_length', 
              'br1_3_sg_length', 'br2_sg_length', 'br3_1_sg_length', 'br3_7_sg_length', 
              'brm_n_shower_main_segs', 'br1_1_n_segs', 'br1_2_n_shower_segs', 'br1_3_n_shower_segs', 
              'br3_1_n_shower_segments', 'br1_3_n_shower_main_segs', 'br2_n_shower_main_segs', 
              'br3_8_n_main_segs', 'br4_1_n_main_segs', 'br2_angle', 'br2_angle1', 'br2_angle2', 
              'br2_angle3', 'vis_1_tmp_length2', 'br1_3_flag_sg_topology', 'br3_1_sg_flag_topology', 
              'hol_2_energy', 'cosmict_2_theta', 'cosmict_4_angle_beam', 'cosmict_5_flag_inside', 
              'cosmict_3_theta', 'cosmict_5_angle_beam', 'cosmict_7_theta']

BDT_vars = [x for x in BDT_variab_all if x not in Non_unique]

pot_vars = ['pot_tor875']

pfeval_vars = ['truth_NprimPio', 'truth_NCDelta']

eval_vars = ['truth_isCC', 'truth_nuPdg', 'truth_nuEnergy', 'truth_vtxInside', 
                            'truth_vtxX', 'truth_vtxY', 'truth_vtxZ', 'weight_spline', 'weight_cv', 
                            'weight_lee', 'truth_energyInside', 'match_completeness_energy', 
                            'match_isFC', 'stm_clusterlength', 'match_found', 'stm_eventtype', 
                            'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead']

# --- merge variables together
variables_w = BDT_vars + KINE_vars + ['weight_cv']
variables = BDT_vars + KINE_vars

#========================================================================================#
#                                       OPEN FILE
#========================================================================================#

print('\n\033[1mOpening input file...\033[0m')

# --- file name
file = "~/Desktop/wirecell_marina/rootfiles/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root"

print('\n\033[1mImporting trees...\033[0m')

# --- import trees and variables
T_pot = uproot.open(file)['wcpselection/T_pot']
df_pot = T_pot.pandas.df(pot_vars, flatten=False)

T_KINE = uproot.open(file)['wcpselection/T_KINEvars']
df_KINE = T_KINE.pandas.df(KINE_vars, flatten=False)

T_BDT = uproot.open(file)['wcpselection/T_BDTvars']
df_BDT = T_BDT.pandas.df(BDT_vars, flatten=False)
         
T_PFeval = uproot.open(file)['wcpselection/T_PFeval']
df_PFeval = T_PFeval.pandas.df(pfeval_vars, flatten=False)

T_eval = uproot.open(file)['wcpselection/T_eval']
df_eval = T_eval.pandas.df(eval_vars, flatten=False)

# --- merge dataframes        
df = pd.concat([df_KINE, df_PFeval, df_BDT, df_eval], axis=1)

print('\n\033[1mGeneral dataframe created...\033[0m')

print('     Total entries = %s'%(len(df)))

# --- make sure weights are valid numbers        
df['weight_cv'] = np.where((df.weight_cv <= 0), 1, df.weight_cv)
df['weight_cv'] = np.where((df.weight_cv > 30), 1, df.weight_cv)
df['weight_cv'] = np.where((df.weight_cv == np.nan), 1, df.weight_cv)
df['weight_cv'] = np.where((df.weight_cv == np.inf), 1, df.weight_cv)
df['weight_cv'] = np.where((df['weight_cv'].isna()), 1, df.weight_cv)
df['weight_spline'] = np.where((df.weight_spline <= 0), 1, df.weight_spline)
df['weight_spline'] = np.where((df.weight_spline > 30), 1, df.weight_spline)
df['weight_spline'] = np.where((df.weight_spline == np.nan), 1, df.weight_spline)
df['weight_spline'] = np.where((df.weight_spline == np.inf), 1, df.weight_spline)
df['weight_spline'] = np.where((df['weight_spline'].isna()), 1, df.weight_spline)

# --- calculate POT
POT_file = sum(df_pot.pot_tor875)
print('     POT = %s'%(POT_file))

#===========================================================================================================================================================================================

# signal = nuebar
df_nuebar = df[df.truth_nuPdg==-12]
print('     Signal = %s'%(len(df_nuebar)))

# background = everything else
df_notnuebar = df[df.truth_nuPdg!=-12]
print('     Background = %s \n'%(len(df_notnuebar)))

#===========================================================================================================================================================================================

def split_train_val_test(df,tag):
    
    # test = 1/3 of the sample
    # validation = 1/6 of the sample
    # training = 1/2 of the sample
    
    df_test = df.iloc[(df.index % 3 == 0).astype(bool)].reset_index(drop=True)
    df_train = df.iloc[(df.index % 3 != 0).astype(bool)].reset_index(drop=True)
    
    df_val = df_train.iloc[(df_train.index % 4 == 0).astype(bool)].reset_index(drop=True)
    df_train = df_train.iloc[(df_train.index % 4 != 0).astype(bool)].reset_index(drop=True)
    
    return df_train, df_val, df_test

def Gen(df):
    
    # Generic Nu selection (reco)
    
    df_ = df[(df.match_found == 1) & 
             (df.stm_eventtype != 0) &
             (df.stm_lowenergy == 0) &
             (df.stm_LM == 0) &
             (df.stm_TGM == 0) &
             (df.stm_STM == 0) &
             (df.stm_FullDead == 0) &
             (df.stm_clusterlength > 15)]
    return df_

# --- create train, val, test samples from DF
# --- it is only done for events that pass the generic Nu selection
df_nuebar_train, df_nuebar_val, df_nuebar_test = split_train_val_test(Gen(df_nuebar), 'Signal')
df_notnuebar_train, df_notnuebar_val, df_notnuebar_test = split_train_val_test(Gen(df_notnuebar), 'Background')

# ------------------------------------------------------------------------------------------
# --- training sample

df_sig_train = shuffle(df_nuebar_train).reset_index(drop=True)[variables_w]
df_bkg_train = shuffle(df_notnuebar_train).reset_index(drop=True)[variables_w]

df_sig_train.head()

df_sig_train.loc[:,'Y'] = [1]*df_sig_train.shape[0] # add extra column, 1=signal and 0=background
df_bkg_train.loc[:,'Y'] = [0]*df_bkg_train.shape[0]

df_train = shuffle(pd.concat([df_sig_train, df_bkg_train]), random_state=1).reset_index(drop=True)

x_train = df_train[df_train.columns[:-2]] # Removes weight and Y for training
y_train = df_train['Y']
w_train = df_train['weight_cv']

# ------------------------------------------------------------------------------------------
# --- validation sample
df_sig_val = shuffle(df_nuebar_val).reset_index(drop=True)[variables_w]
df_bkg_val = shuffle(df_notnuebar_val).reset_index(drop=True)[variables_w]

df_sig_val.loc[:,'Y'] = [1]*df_sig_val.shape[0] # add extra column, 1=signal and 0=background
df_bkg_val.loc[:,'Y'] = [0]*df_bkg_val.shape[0]

df_val = shuffle(pd.concat([df_sig_val, df_bkg_val]), random_state=1).reset_index(drop=True)

x_val = df_val[df_val.columns[:-2]] # Removes weight and Y for training
y_val = df_val['Y']
w_val = df_val['weight_cv']

# ------------------------------------------------------------------------------------------
# --- test sample
df_sig_test = shuffle(df_nuebar_test).reset_index(drop=True)[variables_w]
df_bkg_test = shuffle(df_notnuebar_test).reset_index(drop=True)[variables_w]

df_sig_test.loc[:,'Y'] = [1]*df_sig_test.shape[0] # add extra column, 1=signal and 0=background
df_bkg_test.loc[:,'Y'] = [0]*df_bkg_test.shape[0]

df_test = shuffle(pd.concat([df_sig_test, df_bkg_test]), random_state=1).reset_index(drop=True)

x_test = df_test[df_test.columns[:-2]] # Removes weight and Y for training
y_test = df_test['Y']
w_test = df_test['weight_cv']

#===========================================================================================================================================================================================

use_label_encoder=False # removes warning message because XGBClassifier won't be used in future releases

model = XGBClassifier(n_estimators=550,                         # maximum number of rounds
                            max_depth=3,                        # number of cuts
                            scale_pos_weight = 5,               # sum(df_bkg_train.weight) / sum(df_sig_train.weight) (you should change it manually for your case)
                            learning_rate=0.1,                  # steps
                            objective='binary:logistic',        # bdt score 0-1
                            colsample_bytree=0.8)

eval_set = [(x_train,y_train), (x_val,y_val)]                 # Train: validation_0, Test: validation_1
eval_metric = ['auc', 'error']
sample_weight_eval_set = [w_train, w_val]
    
model.fit(x_train, y_train, sample_weight=w_train, 
          eval_set=eval_set, 
          eval_metric=eval_metric, 
          sample_weight_eval_set=sample_weight_eval_set,
          early_stopping_rounds=200,                             # <-- after this number, stop
          verbose=100)

#===========================================================================================================================================================================================

def plot_important_features(features, feature_importances_, number, name):
    
    zipped = zip(features, feature_importances_)
    zipped_sort = sorted(zipped, key = lambda x:x[1], reverse=True)
    zipped_sort_reduced = zipped_sort[:number]
    
    res = [[ i for i, j in zipped_sort_reduced], 
           [ j for i, j in zipped_sort_reduced]]
    red_features = res[0]
    red_importances = res[1]
    
    plt.barh(range(len(red_importances)), red_importances, align='center')
    plt.yticks(np.arange(len(red_features)), red_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Top %i features"%(number))
    #plt.xscale('log')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.tight_layout()
    plt.savefig('training_validation_important_features.pdf')
    
    return red_features

results = model.evals_result()
n_estimators = len(results['validation_0']['error']) # validation_0 = train, validation_1 = test
auc_train = results['validation_0']['auc']
auc_val = results['validation_1']['auc']
error_train = results['validation_0']['error']
error_val = results['validation_1']['error']

plt.figure(figsize=(15,5))

plt.subplot(121)
plt.plot(range(0,n_estimators), auc_train, c='blue', label='train')
plt.plot(range(0,n_estimators), auc_val, c='orange', label='validation')
ymin = min(min(auc_train),min(auc_val))
ymax = max(max(auc_train),max(auc_val))
plt.ylabel('AUC')
plt.ylim(ymin, ymax)
plt.vlines(model.best_iteration, ymin=ymin, ymax=ymax, ls='--', color='red', label='best iteration', alpha=0.5)
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.plot(range(0,n_estimators), error_train, c='blue', label='train')
plt.plot(range(0,n_estimators), error_val, c='orange', label='validation')
ymin = min(min(error_train),min(error_val))
ymax = max(max(error_train),max(error_val))
plt.ylabel('Classification Error')
plt.ylim(ymin, ymax)
plt.vlines(model.best_iteration, ymin=ymin, ymax=ymax, ls='--', color='red', label='best iteration', alpha=0.5)
plt.legend(loc='best', prop={'size': legend_size})
plt.savefig('training_validation.pdf')

plt.figure(figsize=(8,5))

list_feat = plot_important_features(variables_w[:-2], model.feature_importances_, 16, 'NC') # number not greater than the number of variables

#===========================================================================================================================================================================================

plt.figure(figsize=(13,13))
plt.title('Signal correlation')
labels = list_feat
plt.imshow(df_sig_train[list_feat].corr())
plt.xticks(range(len(labels)), labels, rotation ='vertical')
plt.yticks(range(len(labels)), labels, rotation ='horizontal')
plt.colorbar()
plt.tight_layout()
plt.savefig('signal_correlation.pdf')


plt.figure(figsize=(13,13))
plt.title('Background correlation')
labels = list_feat
plt.imshow(df_bkg_train[list_feat].corr())
plt.xticks(range(len(labels)), labels, rotation ='vertical')
plt.yticks(range(len(labels)), labels, rotation ='horizontal')
plt.colorbar()
plt.tight_layout()
plt.savefig('background_correlation.pdf')

#===========================================================================================================================================================================================

pred_sig_train = model.predict_proba(df_sig_train[variables])[:,1] # column 1=success, 0=fail
pred_sig_test = model.predict_proba(df_sig_test[variables])[:,1]
pred_bkg_train = model.predict_proba(df_bkg_train[variables])[:,1]
pred_bkg_test = model.predict_proba(df_bkg_test[variables])[:,1]

plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_sig_train['weight_cv'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight_cv'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight_cv'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight_cv'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_sig_train['weight_cv'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight_cv'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight_cv'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight_cv'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('bdt_1.pdf')


plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_sig_train['weight_cv'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight_cv'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight_cv'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight_cv'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_sig_train['weight_cv'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight_cv'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight_cv'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight_cv'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('bdt_2.pdf')
