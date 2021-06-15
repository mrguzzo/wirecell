import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns 
import uproot3 as uproot
import pandas as pd

# --- open files

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
pfeval_vars = ['truth_NprimPio', 'truth_NCDelta']
eval_vars = ['truth_isCC', 'truth_nuPdg', 'truth_nuEnergy', 'truth_vtxInside', 
             'truth_vtxX', 'truth_vtxY', 'truth_vtxZ', 'weight_spline', 'weight_cv', 
             'weight_lee', 'truth_energyInside', 'match_completeness_energy', 
             'match_isFC', 'stm_clusterlength', 'match_found', 'stm_eventtype', 
             'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead']

def create_dataframe(file,family,label):

    print('\n\033[1mCreating dataframe for %s...\033[0m\n' % label)

    # --- open trees
    T_pot = uproot.open(file)['wcpselection/T_pot']
    T_KINE = uproot.open(file)['wcpselection/T_KINEvars']
    T_BDT = uproot.open(file)['wcpselection/T_BDTvars']
    T_PFeval = uproot.open(file)['wcpselection/T_PFeval']
    T_eval = uproot.open(file)['wcpselection/T_eval']

    # --- import variables and create dataframe
    df_pot = T_pot.pandas.df(pot_vars, flatten=False)
    df_KINE = T_KINE.pandas.df(KINE_vars, flatten=False)
    df_BDT = T_BDT.pandas.df(BDT_vars, flatten=False)
    df_PFeval = T_PFeval.pandas.df(pfeval_vars, flatten=False)
    df_eval = T_eval.pandas.df(eval_vars, flatten=False)

    # --- merge dataframes
    df = pd.concat([df_KINE, df_PFeval, df_BDT, df_eval], axis=1)

    # --- fix weights
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
    elif(family=='MC'): W_ = 1 #POT/POT_NUE
    df.loc[:,'weight_genie'] = df['weight_cv']*df['weight_spline']
    df.loc[:,'weight'] = [W_]*df.shape[0]*df['weight_genie']

    # --- create intrinsic nue/overlay tag
    if(family=='NUE'): df.loc[:,'original_file'] = 0
    elif(family=='MC'): df.loc[:,'original_file'] = 1

    # --- calculate cos_theta wrt the beam direction
    T_PFeval_cos_theta = uproot.open(file)['wcpselection/T_PFeval']
    df_PFeval_cos_theta = T_PFeval_cos_theta.pandas.df("reco_showerMomentum", flatten=False)
    # get vectors
    v_targ_uboone = [-31387.58422, -3316.402543, -60100.2414]
    v_shower_direction = [df_PFeval_cos_theta['reco_showerMomentum[0]'],
                          df_PFeval_cos_theta['reco_showerMomentum[1]'],
                          df_PFeval_cos_theta['reco_showerMomentum[2]']]
    # normalise vectors
    unit_v_targ_uboone = v_targ_uboone / np.linalg.norm(v_targ_uboone)
    unit_v_shower_direction = v_shower_direction / np.linalg.norm(v_shower_direction)
    # calculate cos theta
    cos_theta = np.dot(-unit_v_targ_uboone,unit_v_shower_direction)
    # add variable to the dataframe
    df.loc[:,'cos_theta'] = cos_theta

    # --- calculate POT
    POT = sum(df_pot.pot_tor875)
    print('POT = %.2e' % POT)
    
    # --- delete dataframes to save memory
    del df_pot
    del df_KINE
    del df_BDT 
    del df_PFeval 
    del df_eval

    return df

df_intrinsic_nue = create_dataframe("~/Desktop/organised_phd/wirecell/BDT/files/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root",'NUE','intrinsic nue')
df_overlay = create_dataframe("~/Desktop/organised_phd/wirecell/BDT/files/nu_overlay_run2.root",'MC','overlay')
