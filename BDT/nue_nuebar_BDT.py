# ideas to implement
# (done) merge intrinsic nue and overlay sample (for better diversity in the background)
# (done) should I cut down the signal and background (make sure to shuffle it) to the same number of entries?
# - gridsearch (take a look at the end of the tutorial) -- optimise the cuts
# - cut down variables (using the correltion matrix for it)


# ================= #
#     TO-DO LIST    #
# ================= #

# [ok] merge intrinsic nue and overlay samples
# [ok] cut down the signal and background to the same number of entries
# [ok] implement grid search
# [ok] cut down variables
# [  ] scale_pos_weight = ratio between sum of the weights, instead of the number of entries
# [  ] I don't need n_estimators if I'm using early_stopping_rounds
# [ok] check the overlay POT, it should be ~e20
# [  ] use variables k_something ?
# [  ] include particle ID

# ================= #
#     INCLUDES      #
# ================= #

import math 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import uproot3 as uproot

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, accuracy_score, f1_score

import xgboost
from xgboost import XGBClassifier

#================================================ #
#      SELECT VARIABLES TO IMPORT FROM TREES      #
# =============================================== #

KINE_vars = ['kine_reco_Enu', 'kine_reco_add_energy', 'kine_pio_mass', 'kine_pio_flag', 'kine_pio_vtx_dis', 
             'kine_pio_energy_1', 'kine_pio_theta_1', 'kine_pio_phi_1', 'kine_pio_dis_1', 'kine_pio_energy_2', 
             'kine_pio_theta_2', 'kine_pio_phi_2', 'kine_pio_dis_2', 'kine_pio_angle']

BDT_variab_all = ['cosmic_n_solid_tracks', 'cosmic_energy_main_showers',
                  'cosmic_energy_direct_showers', 'cosmic_energy_indirect_showers',
                  'cosmic_n_direct_showers', 'cosmic_n_indirect_showers',
                  'cosmic_n_main_showers', 
                  # -----------------------------------------------
                  'gap_flag_prolong_u', 'gap_flag_prolong_v',
                  'gap_flag_prolong_w', 'gap_flag_parallel',
                  'gap_n_points', 'gap_n_bad',
                  'gap_energy', 'gap_num_valid_tracks',
                  'gap_flag_single_shower', 
                  # -----------------------------------------------
                  'mip_quality_energy', 'mip_quality_overlap',
                  'mip_quality_n_showers', 'mip_quality_n_tracks', 
                  'mip_quality_flag_inside_pi0', 'mip_quality_n_pi0_showers', 
                  'mip_quality_shortest_length', 'mip_quality_acc_length',
                  'mip_quality_shortest_angle', 'mip_quality_flag_proton',
                  # -----------------------------------------------
                  'mip_energy',
                  'mip_n_end_reduction', 'mip_n_first_mip',
                  'mip_n_first_non_mip', 'mip_n_first_non_mip_1',
                  'mip_n_first_non_mip_2', 'mip_vec_dQ_dx_0',
                  'mip_vec_dQ_dx_1', 'mip_vec_dQ_dx_2',
                  'mip_vec_dQ_dx_3', 'mip_vec_dQ_dx_4',
                  'mip_vec_dQ_dx_5', 'mip_vec_dQ_dx_6',
                  'mip_vec_dQ_dx_7', 'mip_vec_dQ_dx_8',
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
                  #'cosmict_2_particle_type', 'cosmict_2_n_muon_tracks',
                  #'cosmict_2_flag_inside', 'cosmict_2_angle_beam',
                  #'cosmict_2_flag_dir_weak', 'cosmict_2_dQ_dx_end',
                  #'cosmict_2_dQ_dx_front', 'cosmict_2_theta',
                  #'cosmict_2_phi', 'cosmict_2_valid_tracks',
                  # -----------------------------------------------
                  #'cosmict_3_flag_inside',
                  #'cosmict_3_angle_beam', 'cosmict_3_flag_dir_weak',
                  #'cosmict_3_dQ_dx_end', 'cosmict_3_dQ_dx_front',
                  #'cosmict_3_theta', 'cosmict_3_phi',
                  #'cosmict_3_valid_tracks', 
                  # -----------------------------------------------
                  #'cosmict_4_flag_inside', 'cosmict_4_angle_beam',
                  # -----------------------------------------------
                  #'cosmict_5_flag_inside', 'cosmict_5_angle_beam', 
                  # -----------------------------------------------
                  #'cosmict_6_flag_dir_weak', 'cosmict_6_flag_inside',
                  #'cosmict_6_angle', 
                  # -----------------------------------------------
                  #'cosmict_7_flag_sec', 'cosmict_7_n_muon_tracks',
                  #'cosmict_7_flag_inside', 'cosmict_7_angle_beam',
                  #'cosmict_7_flag_dir_weak', 'cosmict_7_dQ_dx_end',
                  #'cosmict_7_dQ_dx_front', 'cosmict_7_theta',
                  #'cosmict_7_phi', 
                  # -----------------------------------------------
                  #'cosmict_8_flag_out', 'cosmict_8_muon_length',
                  #'cosmict_8_acc_length',
                  # -----------------------------------------------
                  'numu_cc_3_particle_type',
                  'numu_cc_3_max_length', 'numu_cc_3_track_length',
                  'numu_cc_3_max_length_all', 'numu_cc_3_max_muon_length',
                  'numu_cc_3_n_daughter_all', 
                  # -----------------------------------------------                                      
                  #'pio_2_score', 'sig_1_score',
                  #'sig_2_score', 'stw_2_score',
                  #'stw_3_score', 'stw_4_score',
                  #'br3_3_score', 'br3_5_score',
                  #'br3_6_score', 'lol_1_score',
                  #'lol_2_score', 'tro_1_score',
                  #'tro_2_score', 'tro_4_score',
                  #'tro_5_score', 'cosmict_10_score',
                  'numu_1_score', 'numu_2_score',
                  'numu_score', 'nue_score',
                  'cosmict_flag', 'numu_cc_flag']  # Last 2 or 4 variables should not be included in training

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

pfeval_vars = ['truth_NprimPio', 'truth_NCDelta', 'reco_nuvtxX', 'reco_nuvtxY', 'reco_nuvtxZ']

eval_vars = ['truth_isCC', 'truth_nuPdg', 'truth_nuEnergy', 'truth_vtxInside', 
             'truth_vtxX', 'truth_vtxY', 'truth_vtxZ', 'weight_spline', 'weight_cv', 
             'weight_lee', 'truth_energyInside', 'match_completeness_energy', 
             'match_isFC', 'stm_clusterlength', 'match_found', 'stm_eventtype', 
             'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead']

# =================== #
#      FUNCTIONS      #
# =================== #

legend_size = 12

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
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    plt.tight_layout()
    plt.savefig('plots/bdt_important_features.pdf')
    
    return red_features

def create_dataframe(file, family):

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

    # -------------------------------------------------- #
    #     calculate cos_theta wrt the beam direction     #
    # -------------------------------------------------- #
    
    T_PFeval_cos_theta = uproot.open(file)['wcpselection/T_PFeval']
    df_PFeval_cos_theta = T_PFeval_cos_theta.pandas.df("reco_showerMomentum", flatten=False)

    # get vectors
    v_targ_uboone = [-31387.58422, -3316.402543, -60100.2414]
    v_shower_direction = [df_PFeval_cos_theta['reco_showerMomentum[0]'],df_PFeval_cos_theta['reco_showerMomentum[1]'],df_PFeval_cos_theta['reco_showerMomentum[2]']]

    # normalise vectors
    unit_v_targ_uboone = v_targ_uboone / np.linalg.norm(v_targ_uboone)
    unit_v_shower_direction = v_shower_direction / np.linalg.norm(v_shower_direction)

    # calculate cos theta
    cos_theta = np.dot(-unit_v_targ_uboone,unit_v_shower_direction)

    df.loc[:,'cos_theta'] = cos_theta

    # ------------------- #
    #    calculate POT    #
    # ------------------- #

    POT = sum(df_pot.pot_tor875)
    #print('POT = %.2e' % POT)

    # ----------------- #
    #    fix weights    #
    # ----------------- #

    # --- make sure weights are valid numbers  
  
    df.loc[ df['weight_cv']<=0, 'weight_cv' ] = 1
    df.loc[ df['weight_cv']>30, 'weight_cv' ] = 1
    df.loc[ df['weight_cv']==np.nan, 'weight_cv' ] = 1
    df.loc[ df['weight_cv']==np.inf, 'weight_cv' ] = 1
    df.loc[ df['weight_cv'].isna(), 'weight_cv' ] = 1
    df.loc[ df['weight_spline']<=0, 'weight_spline' ] = 1
    df.loc[ df['weight_spline']>30, 'weight_spline' ] = 1
    df.loc[ df['weight_spline']==np.nan, 'weight_spline' ] = 1
    df.loc[ df['weight_spline']==np.inf, 'weight_spline' ] = 1
    df.loc[ df['weight_spline'].isna(), 'weight_spline'] = 1

    # --- calculate weight

    if(family=='NUE'): W_ = 1
    elif(family=='MC'): W_ = 1#POT/POT_NUE

    #print('W_ = %.2e' % W_)

    df.loc[:,'weight_genie'] = df['weight_cv']*df['weight_spline']
    df.loc[:,'weight'] = [W_]*df.shape[0]*df['weight_genie']

    # -------------------------------------------------------------------------------------- intrinsuc nue/overlay tag

    # variable created to classify signal and background dataframes

    if(family=='NUE'): df.loc[:,'original_file'] = 0
    elif(family=='MC'): df.loc[:,'original_file'] = 1

    # -------------------------------------------------------------------------------------- delete dataframes

    del df_pot
    del df_KINE
    del df_BDT 
    del df_PFeval 
    del df_eval

    return df, POT

def split_train_val_test(df,tag):
    
    # test = 1/3 of the sample
    # validation = 1/6 of the sample
    # training = 1/2 of the sample
    
    df_test = df.iloc[(df.index % 3 == 0).astype(bool)].reset_index(drop=True)
    df_train = df.iloc[(df.index % 3 != 0).astype(bool)].reset_index(drop=True)
    
    df_val = df_train.iloc[(df_train.index % 4 == 0).astype(bool)].reset_index(drop=True)
    df_train = df_train.iloc[(df_train.index % 4 != 0).astype(bool)].reset_index(drop=True)
    
    return df_train, df_val, df_test

def printtitle(string):
    print('\n\033[1m%s \033[0m\n' % string)

def print_summary(df,label):

    nentries = len(df)
    
    nentries_intrinsic = len(df[df.original_file==0])
    percentage_intrinsic = nentries_intrinsic/nentries

    nentries_mc = len(df[df.original_file==1])
    percentage_mc = nentries_mc/nentries

    weight_intrinsic = sum(df[df.original_file==0].weight)
    weight_mc = sum(df[df.original_file==1].weight)
    weight = weight_intrinsic + weight_mc

    print('\n%s' % label)
    print('Total entries: %.2e           nue = %.2e (%.2f)            overlay = %.2e (%.2f)' % (nentries,nentries_intrinsic,percentage_intrinsic,nentries_mc,percentage_mc))
    print('Total Weight:  %.2e           nue = %.2e                   overlay = %.2e' % (weight, weight_intrinsic, weight_mc))

def print_particle_id(df,title):

    # original_file = 0    intrinsic nue
    # original_file = 1    overlay

    # create subdataframes

    # intrinsic nue
    df_nue_IN = df[(df.truth_nuPdg==12) & (df.original_file==0)]
    df_antinue_IN = df[(df.truth_nuPdg==-12) & (df.original_file==0)]
    df_numu_IN = df[(df.truth_nuPdg==14) & (df.original_file==0)]
    df_antinumu_IN = df[(df.truth_nuPdg==-14) & (df.original_file==0)]

    # overlay
    df_nue_MC = df[(df.truth_nuPdg==12) & (df.original_file==1)]
    df_antinue_MC = df[(df.truth_nuPdg==-12) & (df.original_file==1)]
    df_numu_MC = df[(df.truth_nuPdg==14) & (df.original_file==1)]
    df_antinumu_MC = df[(df.truth_nuPdg==-14) & (df.original_file==1)]

    d = {"Sample" : [len(df),"","|","",""],
         "Nue"    : [len(df_nue_IN)+len(df_nue_MC),(len(df_nue_IN)+len(df_nue_MC))/len(df),"|",len(df_nue_IN), len(df_nue_MC)],
         "AntiNue": [len(df_antinue_IN)+len(df_antinue_MC),(len(df_antinue_IN)+len(df_antinue_MC))/len(df),"|",len(df_antinue_IN), len(df_antinue_MC)],
         "Numu"   : [len(df_numu_IN)+len(df_numu_MC),(len(df_numu_IN)+len(df_numu_MC))/len(df),"|", len(df_numu_IN), len(df_numu_MC)],
         "AntiNumu" : [len(df_antinumu_IN)+len(df_antinumu_MC),(len(df_antinumu_IN)+len(df_antinumu_MC))/len(df),"|", len(df_antinumu_IN), len(df_antinumu_MC)],
         "Rest"   : [len(df)-len(df_nue_IN)-len(df_nue_MC)-len(df_antinue_IN)-len(df_antinue_MC)-len(df_numu_IN)-len(df_numu_MC)-len(df_antinumu_IN)-len(df_antinumu_MC),"","|","",""]}

    print("\n%s" % title)
    print("{:<15} {:<10} {:<9} {:<5} {:<15} {:<10}".format('Particle ID','Total','','|','Intrinsic Nue','Overlay'))
    for k, v in d.items():
        a1, a2, a3, a4, a5 = v
        print ("{:<15} {:<10} ({:<7.2}) {:<5} {:<15} {:<10}".format(k, a1, a2, a3, a4, a5))

# cuts on the dataframes

def apply_gen_nu_selection(df):
    
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

def apply_vtx_quality(df):

    distX = df.truth_vtxX - df.reco_nuvtxX
    distY = df.truth_vtxY - df.reco_nuvtxY
    distZ = df.truth_vtxZ - df.reco_nuvtxZ

    min_dist = 5 # unit = cm
    squared_min_dist = min_dist * min_dist

    dist_squared = distX*distX + distY*distY + distZ*distZ

    df.loc[:,'vtx_dist'] = dist_squared

    df_ = df[df.vtx_dist < squared_min_dist]

    return df_
    
















#==================== #
#      OPEN FILE      #
#==================== #

printtitle('Open files')
filename_nue = '../rootfiles/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root'
filename_overlay = '../rootfiles/checkout_prodgenie_numi_overlay_run1.root'

print('intrinsic nue: %s' % filename_nue)
print('overlay:       %s\n' % filename_overlay)

# create dataframes
df_intrinsic_nue, POT_NUE = create_dataframe(filename_nue,'NUE')
df_overlay, POT_MC = create_dataframe(filename_overlay,'MC')

# merge dataframes
df = pd.concat([df_intrinsic_nue,df_overlay], ignore_index=True)
POT_MERGED = POT_NUE + POT_MC

extra_variables = ['cos_theta'] # variables that were calculated

print('Intrinsic Nue       entries = %.2e      POT = %.2e      Tot Weight = %.2e' % (len(df_intrinsic_nue),POT_NUE,sum(df_intrinsic_nue.weight)))
print('Overlay             entries = %.2e      POT = %.2e      Tot Weight = %.2e' % (len(df_overlay),POT_MC,sum(df_overlay.weight)))
print('Merged sample       entries = %.2e      POT = %.2e      Tot Weight = %.2e' % (len(df), POT_MERGED, sum(df.weight)))

print_particle_id(df_intrinsic_nue,"Original Intrinsic Nue Sample")
print_particle_id(df_overlay,"Original Overlay")
print_particle_id(df,"Merged Sample")

df_nue_score = df_overlay[(df_overlay.numu_cc_flag>=0) & (df_overlay.nue_score>7)]
print_particle_id(df_nue_score,'After Nue Score')




















# ==================================================== #
#       DEFINE DATAFRAMS FOR SIGNAL AND BACKGROUD      #
# ==================================================== #

# --- signal
df_signal = df[ (df.truth_nuPdg==-12) | (df.truth_nuPdg==12) ]               # definition
df_signal = df_signal[(df_signal.truth_isCC==1)]                             # apply CC interaction condition           
df_signal = df_signal[(df_signal.truth_vtxInside==1)]                        # apply in active volume condition
df_signal = apply_gen_nu_selection(df_signal)                                # apply generic neutrino selection
df_signal = apply_vtx_quality(df_signal)                                     # check reco-true vertex distance

# --- background
#     nue+antinue, out AV
#     not nue/antinue
df_background = apply_gen_nu_selection(df)                                   # apply generic neutrino selection
df_background = df_background[ (((df_background.truth_nuPdg==12)|(df_background.truth_nuPdg==-12)) & (df_background.truth_vtxInside!=1)) |
                    (((df_background.truth_nuPdg!=12)&(df_background.truth_nuPdg!=-12)))]
#df_background = df[ (df.truth_nuPdg!=-12) & (df.truth_nuPdg!=12) ]           # definition
#df_background = df_background[(df_background.truth_isCC==1)]                 # apply CC interaction condition
#df_background = df_background[(df_background.truth_vtxInside==1)]            # apply in active volume condition

# --- resize signal and background
if(len(df_signal)>len(df_background)):
    df_signal = shuffle(df_signal).reset_index(drop=True)
    df_signal = df_signal.iloc[:len(df_background),:]
else:
    df_background = shuffle(df_background).reset_index(drop=True)
    df_background = df_background.iloc[:len(df_signal),:]

# --- delete label
df_signal.drop('original_file',axis='columns',inplace=True)
df_background.drop('original_file',axis='columns',inplace=True)


# ========================================================= #
#      CREATE VALIDATION, TESTING AND TRAINING SAMPLES      #
# ========================================================= #

# --- merge variables together
# --- # BDT_variables[:-2] means all variables in the list minus ['cosmict_flag', 'numu_cc_flag']
variables_w = extra_variables + KINE_vars + BDT_vars[:-2] + ['weight']
variables   = extra_variables + KINE_vars + BDT_vars[:-2]

# --- create training, validation and testing samples from DF
# --- it is only done for events that pass the generic Nu selection
df_signal_train, df_signal_val, df_signal_test = split_train_val_test(df_signal, 'Signal')
df_background_train, df_background_val, df_background_test = split_train_val_test(df_background, 'Background')

# ------------------------------------------------------------------------------------------
# --- training sample

# shuffle and reorganise for the variables in variables_w
# shuffles entries, but columns remain the same position
df_sig_train = shuffle(df_signal_train).reset_index(drop=True)[variables_w] # shuffle and keep only variables in the list variables_w
df_bkg_train = shuffle(df_background_train).reset_index(drop=True)[variables_w]

# add extra column, 1=signal and 0=background
df_sig_train.loc[:,'Y'] = [1]*df_sig_train.shape[0] 
df_bkg_train.loc[:,'Y'] = [0]*df_bkg_train.shape[0]

df_train = shuffle(pd.concat([df_sig_train, df_bkg_train]), random_state=1).reset_index(drop=True)

x_train = df_train[df_train.columns[:-2]] # Removes weight and Y for training
y_train = df_train['Y']
w_train = df_train['weight']

# ------------------------------------------------------------------------------------------
# --- validation sample
df_sig_val = shuffle(df_signal_val).reset_index(drop=True)[variables_w]
df_bkg_val = shuffle(df_background_val).reset_index(drop=True)[variables_w]

df_sig_val.loc[:,'Y'] = [1]*df_sig_val.shape[0] # add extra column, 1=signal and 0=background
df_bkg_val.loc[:,'Y'] = [0]*df_bkg_val.shape[0]

df_val = shuffle(pd.concat([df_sig_val, df_bkg_val]), random_state=1).reset_index(drop=True)

x_val = df_val[df_val.columns[:-2]] # Removes weight and Y for training
y_val = df_val['Y']
w_val = df_val['weight']

# ------------------------------------------------------------------------------------------
# --- test sample
df_sig_test = shuffle(df_signal_test).reset_index(drop=True)[variables_w]
df_bkg_test = shuffle(df_background_test).reset_index(drop=True)[variables_w]

df_sig_test.loc[:,'Y'] = [1]*df_sig_test.shape[0] # add extra column, 1=signal and 0=background
df_bkg_test.loc[:,'Y'] = [0]*df_bkg_test.shape[0]

df_test = shuffle(pd.concat([df_sig_test, df_bkg_test]), random_state=1).reset_index(drop=True)

x_test = df_test[df_test.columns[:-2]] # Removes weight and Y for training
y_test = df_test['Y']
w_test = df_test['weight']

# ====================== #
#      BDT TRAINING      #
# ====================== #

printtitle('Starting BDT training...')

print('                 Signal             Background')
print('Training:        %.2e           %.2e' % (len(df_signal_train),len(df_background_train)))
print('Validation:      %.2e           %.2e' % (len(df_signal_val),len(df_background_val)))
print('Testing:         %.2e           %.2e\n' % (len(df_signal_test),len(df_background_test)))

use_label_encoder=False # removes warning message because XGBClassifier won't be used in future releases

print('scale_pos_weight = %f' % (sum(df_bkg_train.weight) / sum(df_sig_train.weight)))

model = XGBClassifier(n_estimators=300,                   # maximum number of rounds    550
                      max_depth=3,                       # number of cuts              6
                      scale_pos_weight = 1,               # sum(df_bkg_train.weight) / sum(df_sig_train.weight) (you should change it manually for your case)
                      learning_rate=0.01,                  # steps   0.05
                      objective='binary:logistic',        # bdt score 0-1
                      colsample_bytree=0.8)

                                                                # understand the parameters: https://xgboost.readthedocs.io/en/latest/python/python_api.html
model.fit(x_train,                                              # feature matrix
          y_train,                                              # labels (Y=1 signal, Y=0 background)
          sample_weight=w_train,                                # instance weights
          eval_set = [(x_train,y_train), (x_val,y_val)],        # a list of (X,y) tuple pairs to use as validation sets ---> validation_0=train, validation_1=validation
          sample_weight_eval_set = [w_train, w_val],            # list of arrays storing instances weights for the i-th validation set
          eval_metric = ['auc', 'error'],                       # list of parameters under eval_metric: https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
          early_stopping_rounds=50,                             # validation metric needs to improve at least once in every early_stopping_rounds round(s)
          verbose=100)

results = model.evals_result()                            # takes the results from the BDT training above
n_estimators = len(results['validation_0']['error'])      # number of rounds used for the BDT training
auc_train = results['validation_0']['auc']                # subsample: auc for training
auc_val = results['validation_1']['auc']                  # subsample: auc for validation
error_train = results['validation_0']['error']            # subsample: error for training
error_val = results['validation_1']['error']              # subsample: error for validation

# save the bdt result to our dataframe
df.loc[:,'bdt_score'] = model.predict_proba(df[variables])[:,1]
df_overlay.loc[:,'bdt_score'] = model.predict_proba(df_overlay[variables])[:,1]

# ==================== #
#      MAKE PLOTS      #
# ==================== #

printtitle('Making plots...')

# --- plot auc and error for training and validation

plt.figure(figsize=(15,5))

plt.subplot(121)
plt.plot(range(0,n_estimators), auc_train, c='blue', label='train')
plt.plot(range(0,n_estimators), auc_val, c='orange', label='validation')
ymin = min(min(auc_train),min(auc_val))
ymax = max(max(auc_train),max(auc_val))
plt.ylabel('AUC')
plt.xlabel('Estimators')
plt.ylim(ymin, ymax)
plt.vlines(model.best_iteration, ymin=ymin, ymax=ymax, ls='--', color='red', label='best iteration', alpha=0.5)
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.plot(range(0,n_estimators), error_train, c='blue', label='train')
plt.plot(range(0,n_estimators), error_val, c='orange', label='validation')
ymin = min(min(error_train),min(error_val))
ymax = max(max(error_train),max(error_val))
plt.ylabel('Classification Error')
plt.xlabel('Estimators')
plt.ylim(ymin, ymax)
plt.vlines(model.best_iteration, ymin=ymin, ymax=ymax, ls='--', color='red', label='best iteration', alpha=0.5)
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('plots/bdt_auc_error.pdf')

# --- plot important features

plt.figure(figsize=(8,5))
list_feat = plot_important_features(variables_w[:-2], model.feature_importances_, 16, 'NC') # number not greater than the number of variables

# --- signal correlation

plt.figure(figsize=(13,13))

plt.title('Signal correlation')
labels = list_feat
plt.imshow(df_sig_train[list_feat].corr())
plt.xticks(range(len(labels)), labels, rotation ='vertical')
plt.yticks(range(len(labels)), labels, rotation ='horizontal')
plt.colorbar()
plt.tight_layout()
plt.savefig('plots/bdt_signal_correlation.pdf')

# --- background correlation

plt.figure(figsize=(13,13))
plt.title('Background correlation')
labels = list_feat
plt.imshow(df_bkg_train[list_feat].corr())
plt.xticks(range(len(labels)), labels, rotation ='vertical')
plt.yticks(range(len(labels)), labels, rotation ='horizontal')
plt.colorbar()
plt.tight_layout()
plt.savefig('plots/bdt_background_correlation.pdf')

#===========================================================================================

pred_sig_train = model.predict_proba(df_sig_train[variables])[:,1] # column 1=success, 0=fail
pred_sig_test = model.predict_proba(df_sig_test[variables])[:,1]
pred_bkg_train = model.predict_proba(df_bkg_train[variables])[:,1]
pred_bkg_test = model.predict_proba(df_bkg_test[variables])[:,1]

plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_sig_train['weight'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_sig_train['weight'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('plots/bdt_score_log_scale.pdf')


plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_sig_train['weight'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_sig_train['weight'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_bkg_train['weight'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_sig_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_bkg_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('plots/bdt_score.pdf')

# ==============================================================================================================================================================

# =================================== #
#   CALCULATE EFFICIENTY AND PURITY   #
# =================================== #

# plot bdt score

df_signal_bdt_score = df_overlay[(df_overlay.truth_nuPdg==12) | (df_overlay.truth_nuPdg==-12)]
df_background_bdt_score = df_overlay[(df_overlay.truth_nuPdg!=12) & (df_overlay.truth_nuPdg!=-12)]

# plot bdt score for signal and background
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.hist(df_signal_bdt_score.bdt_score, range=(0,1), label='Signal')
plt.subplot(122)
plt.hist(df_background_bdt_score.bdt_score, range=(0,1), label='Background')
plt.savefig('plots/testing.pdf')

# calculate efficiency and purity per cut
var_bdt_score = 0.0
step = 0.1
h_pur = []
h_eff = []
h_x_pur = []
h_x_eff = []

while(var_bdt_score<1):

    print('BDT score = %f' & var_bdt_score)

    df_signal_cut = df_signal_bdt_score[df_signal_bdt_score.bdt_score>var_bdt_score]
    df_background_cut = df_background_bdt_score[df_background_bdt_score.bdt_score>var_bdt_score]

    # calculate purity
    if((len(df_signal_cut)+len(df_background_cut))!=0):
        pur = len(df_signal_cut)/(len(df_signal_cut)+len(df_background_cut))
        h_pur.append(pur)
        h_x_pur.append(var_bdt_score)
        print('Purity = %f' % pur)
    
    # calculate efficiency
    if((len(df_signal_bdt_score)!=0)):
        eff = len(df_signal_cut)/len(df_signal_bdt_score)
        h_eff.append(eff)
        h_x_eff.append(var_bdt_score)
        print('Efficiency = %f' % eff)

    var_bdt_score = var_bdt_score + step

    print('')

# make plot
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

#plt.plot(hist_x, hist_y_eff, c='blue', label='Efficiency')
plt.plot(h_x_pur, h_pur, c='orange', label='Purity')
plt.plot(h_x_eff, h_eff, c='blue', label='Efficiency')
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})
plt.savefig('plots/eff.pdf')
