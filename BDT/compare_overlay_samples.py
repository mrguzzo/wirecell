# This file is going to plot the two overlay files to compare
# the input variables level, and try to see if they are different.
# I'm only going to plot the variables I'm using for the BDT, as
# they are the only ones that I care about at this stage.

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

# ------------------------------------------------------------------------

file1 = '../rootfiles/checkout_prodgenie_numi_overlay_run1.root'
file2 = '../rootfiles/output.root'
file3 = '../rootfiles/checkout_prodgenie_numi_overlay_run1_OFFSETFIXED2.root'

# ------------------------------------------------------------------------

kine_vars   = ['kine_reco_Enu','kine_pio_vtx_dis','kine_pio_energy_1']
bdt_vars    = ['numu_cc_flag','nue_score']
pot_vars    = ['pot_tor875']
pfeval_vars = ['truth_corr_nuvtxX','truth_corr_nuvtxY','truth_corr_nuvtxZ','reco_nuvtxX', 'reco_nuvtxY', 'reco_nuvtxZ']
eval_vars   = ['truth_isCC','truth_nuPdg','truth_vtxInside','weight_spline', 'weight_cv',
               'match_found', 'stm_eventtype', 'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead','stm_clusterlength',
               'truth_energyInside', 'match_completeness_energy']

# --- variables calculated by me
extra_vars  = ['cos_theta'] 


def create_dataframe(file, family):

    # --- import trees and variables
    T_pot = uproot.open(file)['wcpselection/T_pot']
    df_pot = T_pot.pandas.df(pot_vars, flatten=False)

    T_KINE = uproot.open(file)['wcpselection/T_KINEvars']
    df_KINE = T_KINE.pandas.df(kine_vars, flatten=False)

    T_BDT = uproot.open(file)['wcpselection/T_BDTvars']
    df_BDT = T_BDT.pandas.df(bdt_vars, flatten=False)
            
    T_PFeval = uproot.open(file)['wcpselection/T_PFeval']
    df_PFeval = T_PFeval.pandas.df(pfeval_vars, flatten=False)

    T_eval = uproot.open(file)['wcpselection/T_eval']
    df_eval = T_eval.pandas.df(eval_vars, flatten=False)

    # --- merge dataframes
    df = pd.concat([df_KINE, df_PFeval, df_BDT, df_eval, df_pot], axis=1)

    # --- calculate POT
    POT = sum(df_pot.pot_tor875)

    # --- fix weight variables, make sure they are valid numbers
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
    elif(family=='MC'): W_ = 1

    df.loc[:,'weight_genie'] = df['weight_cv']*df['weight_spline']  # calculate GENIE weight
    df.loc[:,'weight'] = [W_]*df.shape[0]*df['weight_genie']        # should I POT normalise it?

    # --- create variable to track file of origin
    if(family=='NUE'): df.loc[:,'original_file'] = 0
    elif(family=='MC'): df.loc[:,'original_file'] = 1

    # --- delete dataframes to save memory space
    del df_pot
    del df_KINE
    del df_BDT 
    del df_PFeval 
    del df_eval

    return df, POT

df_file1, POT1 = create_dataframe(file1,'MC')
df_file2, POT2 = create_dataframe(file2,'MC')
df_file3, POT3 = create_dataframe(file3,'MC')

print('File1  | %.2e POT | %i entries' % (POT1,len(df_file1)))
print('File2  | %.2e POT | %i entries' % (POT2,len(df_file2)))
print('File3  | %.2e POT | %i entries' % (POT3,len(df_file3)))

# ------------------------------------------------------------------------

# plot the variables to see if they are the same

legend_size = 12

plt.figure(figsize=(5,5))
plt.plot(df_file1.kine_reco_Enu, c='orange', marker='o', label='file1')
plt.plot(df_file2.kine_reco_Enu, c='blue', marker='o', label='new file')
plt.plot(df_file3.kine_reco_Enu, c='red', marker='o', label='file3 offsetfixed')
plt.legend(loc='best', prop={'size': legend_size})
plt.tight_layout()
plt.savefig('plots/comparison_kine_reco_Enu.pdf')