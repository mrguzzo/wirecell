# ---------------------------------------------------------------------
# Author: Marina Reggiani-Guzzo
# July 15, 2021
#
# What does this code do?
# 
# - Import intrinsic nue and overlay files and variables
# - Merge them together to create an unified dataframe with both files.
#   The reason for that is to increase the statistics for the nue+antinue
#   analysis that I want to do.
# - Split the merged sample into signal and background. This signal and
#   background definition is going to be used for the training. This is where
#   I tell BDT what signal and background should look like.
# - Creates validation (1/6), testing (1/3) and training (1/2) samples for
#   my BDT selection. It is split in a way to keep the same signal/background
#   ratio between those subsamples.
#   ------> I can try to optimise it using the k-fold method
#   ------> Maybe also change the ratio chosen for each subsample
# - 



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

# ------------- #
#   VARIABLES   #
# ------------- #

# --- variables imported from the checkout file
kine_vars   = ['kine_reco_Enu','kine_pio_vtx_dis','kine_pio_energy_1']
bdt_vars    = ['numu_cc_flag','nue_score']
pot_vars    = ['pot_tor875']
pfeval_vars = ['truth_corr_nuvtxX','truth_corr_nuvtxY','truth_corr_nuvtxZ','reco_nuvtxX', 'reco_nuvtxY', 'reco_nuvtxZ']
eval_vars   = ['truth_isCC','truth_nuPdg','truth_vtxInside','weight_spline', 'weight_cv',
               'match_found', 'stm_eventtype', 'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead','stm_clusterlength',
               'truth_energyInside', 'match_completeness_energy',
               'run','subrun','event']

# --- variables calculated by me
extra_vars  = ['cos_theta'] 

# ------------- #
#   FUNCTIONS   #
# ------------- #

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
    df = pd.concat([df_KINE, df_PFeval, df_BDT, df_eval], axis=1)

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

def calculate_extra_vars(df,file,label):

    # --- cos theta --- #
    
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
    
    if(label!='DATA'):

        # variables that depend on true information

        # --- true/reco vertex distance --- #

        distX = df.truth_corr_nuvtxX - df.reco_nuvtxX
        distY = df.truth_corr_nuvtxY - df.reco_nuvtxY
        distZ = df.truth_corr_nuvtxZ - df.reco_nuvtxZ

        min_dist = 1 # unit = cm
        squared_min_dist = min_dist * min_dist

        dist_squared = distX*distX + distY*distY + distZ*distZ

        df.loc[:,'vtx_dist'] = dist_squared

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

def print_nentries(df,text):

    print('   > %s %7.0i entries = %7.0i (%.2f%%) nue + %7.0i (%.2f%%) mc' % (text,len(df),
                                                          len(df[df.original_file==0]),100*(len(df[df.original_file==0])/len(df)),
                                                          len(df[df.original_file==1]),100*(len(df[df.original_file==1])/len(df))))

def calculate_ratios(df,label):

    df_sig = define_signal(df)
    df_bkg = define_background(df)

    df_inAV = df[df.truth_vtxInside==1]
    df_inAV_sig = define_signal(df_inAV)
    df_inAV_bkg = define_background(df_inAV)

    df_inAV_gen = apply_gen_nu_selection(df_inAV)
    df_inAV_gen_sig = define_signal(df_inAV_gen)
    df_inAV_gen_bkg = define_background(df_inAV_gen)

    df_gen = apply_gen_nu_selection(df)
    df_gen_sig = define_signal(df_gen)
    df_gen_bkg = define_background(df_gen)


    print('\n--------------------------')
    print('Ratios for %s\n' % label)
    print('  Raw entries                %i (1.00) = %i signal + %i background' % (len(df),len(df_sig),len(df_bkg)))
    print('  inAV                       %i (%.2f) = %i signal + %i background' % (len(df_inAV),len(df_inAV)/len(df),len(df_inAV_sig),len(df_inAV_bkg)))
    print('  GenNuSel                   %i (%.2f) = %i signal + %i background' % (len(df_gen),len(df_gen)/len(df),len(df_gen_sig),len(df_gen_bkg)))
    print('  inAV + GenNuSel            %i (%.2f) = %i signal + %i background' % (len(df_inAV_gen),len(df_inAV_gen)/len(df),len(df_inAV_gen_sig),len(df_inAV_gen_bkg)))
    print('')



























# ----------------- #
#    IMPORT FILES   #
# ----------------- #

# Import the files that you are going to use for your analysis. Remember that this
# code aims to look for nue+antinue CC interactions, so unfortunately the overlay
# sample doesn't have enough nue+antinue CC statistics for the BDT training. For this
# reason both overlay and intrinsic nue samples are used for this analysis. Because
# the intrinsic nue one is responsible for providing most of the nue+antinue events

# Note that I am not applying the generic neutrino selection to the events for the
# BDT selection since I want to keep most of my background interactions for my BDT
# training. Otherwise it would cut most of the background events, and we don't want that 

print('\nInput files:\n')

filename_nue = '../rootfiles/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root'
#filename_nue = '../rootfiles/checkout_intrinsic_nue_numi_run1_particle_flow.root'

#filename_overlay = '../rootfiles/checkout_prodgenie_numi_overlay_run1_OFFSETFIXED2.root'
filename_overlay = '../rootfiles/checkout_overlay_numi_run1_particle_flow.root'

df_intrinsic_nue, POT_NUE = create_dataframe(filename_nue,'NUE')          # create dataframe and calculate POT
calculate_extra_vars(df_intrinsic_nue,filename_nue,'MC')                  # calculate extra variables not in the checkout file

df_overlay, POT_MC = create_dataframe(filename_overlay,'MC')              # create dataframe and calculate POT
calculate_extra_vars(df_overlay,filename_overlay,'MC')                    # calculate extra variables not in the checkout file

# --- merge dataframes
df = pd.concat([df_intrinsic_nue,df_overlay], ignore_index=True)          # merge intrinsic/overlay dataframes and ignore index

# --- print summary
print('   > Intrinsic Nue     %.2e POT    %6.0f entries' % (POT_NUE,len(df_intrinsic_nue)))
print('   > Overlay           %.2e POT    %6.0f entries' % (POT_MC,len(df_overlay)))
print('   > Merged            %.2e POT    %6.0f entries' % (POT_MC+POT_NUE,len(df)))














# -------------------------------- #
#   DEFINE SIGNAL AND BACKGROUND   #
# -------------------------------- #

# Define here what signal and background should look like for your BDT training
# remember that background is defined as everything that is not signal, meaning
# nueCC+antinueCC that are outAV are classified as background. It means that
# my original sample should be perfectly split into signal and background, so
# nentries_signal + nentries_background = nentries_merged_sample

squared_min_dist = 1   # squared distance for the vertex reconstruction quality check

def define_signal(df):

    df_ = df[ (df.truth_nuPdg==-12) | (df.truth_nuPdg==12) ]                # PDG definition
    df_ = df_[df_.truth_isCC==1]                                            # apply CC interaction condition 
    df_ = df_[df_.truth_vtxInside==1]                                       # apply in active volume condition
    df_ = df_[df_.vtx_dist <= squared_min_dist]                             # check reco-true vertex distance

    return df_

def define_background(df):

    # --- background is defined as everything that is not signal
    df_ = df[ ((df.truth_nuPdg!=-12) & (df.truth_nuPdg!=12)) |          # not nue nor antinue
            (df.truth_isCC!=1) |                                        # not CC
            (df.truth_vtxInside!=1) |                                   # event out of active volume
            (df.vtx_dist > squared_min_dist)]                           # reco-true vertex too far

    return df_

print('\nDefine signal and background:\n')

df_signal = define_signal(df)
df_background = define_background(df)

# check the splitting process and prints the number of entries for signal and background
if((len(df_signal)+len(df_background))!=len(df)):
    print('\n   > !!! ATTENTION !!! Check your signal and background definition, they are not covering all entries.\n')

print_nentries(df_signal,    'Signal                   ')
print_nentries(df_background,'Background               ')
print('')
print('   > Signal                     %6.0f entries = %6.2f%% of the merged sample' % (len(df_signal),100*(len(df_signal)/len(df))))
print('   > Background                 %6.0f entries = %6.2f%% of the merged sample' % (len(df_background),100*(len(df_background)/len(df))))















# -------------------------------------------------- #
#   CREATE VALIDATION, TESTIN AND TRAINING SAMPLES   #
# -------------------------------------------------- #

# Create here a validation, testing and training sample for your signal and
# background. To make sure those samples contain the same proportion of signal
# and background, I first split signal and background into validation, testing
# and training, using the same ratio, and then I merge the sig_val with bkg_val,
# for example. It guarantees the same signal/background from the merged sample.
#
# Make sure you only use reconstructed variables for your training, otherwise
# you won't be able to calculate the BDT score for data, as it doesn't have
# true-level information about your events.
# How is GENIE weight being used? Is it being used for training?

print('\nCreate validation, testing and training samples:')

# --- variables used for my BDT training
variables_w = extra_vars + kine_vars + bdt_vars + ['weight']
variables   = extra_vars + kine_vars + bdt_vars

def split_train_val_test(df,tag):
    
    # test = 1/3 of the sample
    # validation = 1/6 of the sample
    # training = 1/2 of the sample
    
    # --- first split the dataframe into 1/3=test and 2/3=train
    df_test = df.iloc[(df.index % 2 == 0).astype(bool)].reset_index(drop=True)
    df_train = df.iloc[(df.index % 2 != 0).astype(bool)].reset_index(drop=True)
    
    # --- split train into 
    df_val = df_train.iloc[(df_train.index % 4 == 0).astype(bool)].reset_index(drop=True)
    df_train = df_train.iloc[(df_train.index % 4 != 0).astype(bool)].reset_index(drop=True)
    
    return df_train, df_val, df_test

# --- split signal and background into training, validation and testing
df_signal_train, df_signal_val, df_signal_test = split_train_val_test(df_signal, 'Signal')
df_background_train, df_background_val, df_background_test = split_train_val_test(df_background, 'Background')

print('\n   > Signal')
print_nentries(df_signal_train,   'Training   ')
print_nentries(df_signal_val,     'Validation ')
print_nentries(df_signal_test,    'Testing    ')
print('\n   > Background')
print_nentries(df_background_train,'Training   ')
print_nentries(df_background_val,  'Validation ')
print_nentries(df_background_test, 'Testing    ')

# --- associate variables to the dataframes

def associate_variables(df_sig,df_bkg):

    df_sig = shuffle(df_sig).reset_index(drop=True)[variables_w]
    df_bkg = shuffle(df_bkg).reset_index(drop=True)[variables_w]

    # true label, 1=signal, 0=background
    df_sig.loc[:,'Y'] = 1
    df_bkg.loc[:,'Y'] = 0

    df = shuffle(pd.concat([df_sig,df_bkg]),random_state=1).reset_index(drop=True)

    x = df[df.columns[:-2]] # removes weight and Y for training
    y = df['Y']
    w = df['weight']

    return df, x, y, w

print('\nMerge signal/background training, validation and testing:\n')

df_train, x_train, y_train, w_train = associate_variables(df_signal_train,df_background_train)
df_val, x_val, y_val, w_val = associate_variables(df_signal_val,df_background_val)
df_test, x_test, y_test, w_test = associate_variables(df_signal_test,df_background_test)

print('   > Training   : %6.i = %5.0i (%.2f%%) signal + %5.0i (%.2f%%) background' % (len(df_train),len(df_signal_train),100*(len(df_signal_train)/len(df_train)),len(df_background_train),100*(len(df_background_train)/len(df_train))))
print('   > Validation : %6.i = %5.0i (%.2f%%) signal + %5.0i (%.2f%%) background' % (len(df_val),len(df_signal_val),100*(len(df_signal_val)/len(df_val)),len(df_background_val),100*(len(df_background_val)/len(df_val))))
print('   > Testing    : %6.i = %5.0i (%.2f%%) signal + %5.0i (%.2f%%) background' % (len(df_test),len(df_signal_test),100*(len(df_signal_test)/len(df_test)),len(df_background_test),100*(len(df_background_test)/len(df_test))))



# --------------- #
#   BDT TRAINIG   #
# --------------- #

print('\nStarting BDT training...\n')

use_label_encoder=False # removes warning message because XGBClassifier won't be used in future releases

# --- hyperparameters used for the BDT
param = {'n_estimators':           300, 
         'max_depth':              4,
         'scale_pos_weight':       1,
         'learning_rate':          0.01,
         'objective':              'binary:logistic',
         'colsample_bytree':       0.8,
         'lambda' :                1}

model = xgboost.XGBClassifier( **param,)

# --- fit my data with the hyperparameters above
model.fit(x_train,                                              # feature matrix
          y_train,                                              # labels (Y=1 signal, Y=0 background)
          sample_weight=w_train,                                # instance weights
          eval_set = [(x_train,y_train), (x_val,y_val)],        # a list of (X,y) tuple pairs to use as validation sets ---> validation_0=train, validation_1=validation
          sample_weight_eval_set = [w_train, w_val],            # list of arrays storing instances weights for the i-th validation set
          eval_metric = ['auc', 'error'],                       # list of parameters under eval_metric: https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
          early_stopping_rounds=50,                             # validation metric needs to improve at least once in every early_stopping_rounds round(s)
          verbose=100)

# --- take results
results = model.evals_result()                            # takes the results from the BDT training above
n_estimators = len(results['validation_0']['error'])      # number of rounds used for the BDT training
auc_train = results['validation_0']['auc']                # subsample: auc for training
auc_val = results['validation_1']['auc']                  # subsample: auc for validation
error_train = results['validation_0']['error']            # subsample: error for training
error_val = results['validation_1']['error']              # subsample: error for validation

print('\nBDT training done...')

# --- calculate the bdt score back to the dataframe
df_overlay.loc[:,'bdt_score'] = model.predict_proba(df_overlay[variables])[:,1]
df_intrinsic_nue.loc[:,'bdt_score'] = model.predict_proba(df_intrinsic_nue[variables])[:,1]





































# --------------- #
#   MAKE PLOTS    #
# --------------- #

print('\nMaking plots...')
legend_size=12

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
list_feat = plot_important_features(variables_w[:-2], model.feature_importances_, 5, 'NC') # number not greater than the number of variables

# --- bdt score

pred_sig_train = model.predict_proba(df_signal_train[variables])[:,1] # column 1=success, 0=fail
pred_sig_test = model.predict_proba(df_signal_test[variables])[:,1]
pred_bkg_train = model.predict_proba(df_background_train[variables])[:,1]
pred_bkg_test = model.predict_proba(df_background_test[variables])[:,1]

plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, alpha=0)
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
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('plots/bdt_score.pdf')

# =================================== #
#   CALCULATE EFFICIENTY AND PURITY   #
# =================================== #

# We are now going to calculate efficiency, for this we have to make sure
# the definition for effiency/purity is clear:
# - purity: how much of the remaining sample, given a bdt cut, is made of
#           signal. Remember that here we use the signal definition declared
#           at the beginning of the code
# - efficiency: numerator = nue+antinue CC + GenNuSel + BDT
#               denominator = nue+antinue CC + inAV

bdt_var_min = 0.0      # initial value for your bdt score cut
bdt_var_max = 1.0      # final value for your bdt score cut
step = 0.01            # bdt score step

# efficiency
h_eff = []
h_eff_x = []

# purity
h_pur = []
h_pur_x = []

# number of entries per bdt cut
h_nentries_all = []     # overlay
h_nentries_nue_antinue = [] # nue+antinue
h_nentries_nue = []     # nue
h_nentries_antinue = [] # antinue
h_nentries_x = []

# percentage of nue and antinue
h_perc_nue = []
h_perc_antinue = []
h_perc_x = []

var_bdt_score = bdt_var_min
while(var_bdt_score < bdt_var_max):

    # ----------------------------------------------------------------------------------------
    # --- calculate efficiency

    # numerator
    df_eff_num = df_overlay
    df_eff_num = df_eff_num[((df_eff_num.truth_nuPdg==12) | (df_eff_num.truth_nuPdg==-12)) &   # true nue+antinue
                           (df_eff_num.truth_isCC==1) &                                        # true CC interactions
                           (df_eff_num.truth_vtxInside==1) &                                   # vertex in active volume
                           (df_eff_num.bdt_score>var_bdt_score)]                               # bdt score above cut
    df_eff_num = apply_gen_nu_selection(df_eff_num)                                            # pass generic nu selection

    # denominator
    df_eff_den = df_overlay
    df_eff_den = df_eff_den[((df_eff_den.truth_nuPdg==12) | (df_eff_den.truth_nuPdg==-12)) &  # true nue+antinue
                            (df_eff_den.truth_isCC==1) &                                      # true CC interactions
                            (df_eff_den.truth_vtxInside==1)]                                  # vertex in active volume

    eff_numerator = len(df_eff_num)
    eff_denominator = len(df_eff_den)

    if(eff_denominator!=0):
        h_eff.append(eff_numerator/eff_denominator)
        h_eff_x.append(var_bdt_score)

    # ----------------------------------------------------------------------------------------
    # --- calculate purity
    
    # numerator
    df_pur_num = df_overlay 
    df_pur_num = df_pur_num[((df_pur_num.truth_nuPdg==12) | (df_pur_num.truth_nuPdg==-12)) &   # true nue+antinue
                            (df_pur_num.truth_isCC==1) &                                       # true CC interactions
                            (df_pur_num.truth_vtxInside==1) &                                  # vertex in active volume
                            (df_pur_num.bdt_score>var_bdt_score)]                              # bdt score above cut
    
    # denominator
    df_pur_den = df_overlay 
    df_pur_den = df_pur_den[df_pur_den.bdt_score>var_bdt_score]                                # bdt score above cut

    pur_numerator = len(df_pur_num)
    pur_denominator = len(df_pur_den)
    if(pur_denominator!=0):
        h_pur.append(pur_numerator/pur_denominator)
        h_pur_x.append(var_bdt_score)

    # ----------------------------------------------------------------------------------------
    # --- calculate the number of entries

    df_n = df_overlay
    df_nentries_all = df_n[(df_n.bdt_score>var_bdt_score)]
    df_nentries_nue_antinue = df_n[((df_n.truth_nuPdg==-12) | (df_n.truth_nuPdg==12))  & (df_n.truth_isCC==1) & (df_n.truth_vtxInside==1) & (df_n.bdt_score>var_bdt_score)]
    df_nentries_nue = df_n[(df_n.truth_nuPdg==12)                                      & (df_n.truth_isCC==1) & (df_n.truth_vtxInside==1) & (df_n.bdt_score>var_bdt_score)]
    df_nentries_antinue = df_n[(df_n.truth_nuPdg==-12)                                 & (df_n.truth_isCC==1) & (df_n.truth_vtxInside==1) & (df_n.bdt_score>var_bdt_score)]

    h_nentries_all.append(len(df_nentries_all))
    h_nentries_nue_antinue.append(len(df_nentries_nue_antinue))
    h_nentries_nue.append(len(df_nentries_nue))
    h_nentries_antinue.append(len(df_nentries_antinue))
    h_nentries_x.append(var_bdt_score)

    # ----------------------------------------------------------------------------------------
    # --- calculate percentage of nue and antinue in your final selected sample

    if(len(df_nentries_all)!=0):
        percentage_nue = 100*(len(df_nentries_nue)/len(df_nentries_nue_antinue))
        percentage_antinue = 100*(len(df_nentries_antinue)/len(df_nentries_nue_antinue))

        h_perc_nue.append(percentage_nue)
        h_perc_antinue.append(percentage_antinue)
        h_perc_x.append(var_bdt_score)


    # ----------------------------------------------------------------------------------------
    # --- update variable
    var_bdt_score = var_bdt_score + step

# plot efficiency
plt.figure(figsize=(5,5))
plt.plot(h_eff_x, h_eff, c='blue', label='Efficiency')
plt.plot(h_pur_x, h_pur, c='orange', label='Purity')
plt.grid()
plt.ylim([0,1])
plt.xlim([0,1])
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})
plt.tight_layout()
plt.savefig('plots/efficiency_purity.pdf')

# number of entries
plt.figure(figsize=(5,5))
plt.plot(h_nentries_x,h_nentries_nue,c='blue',label='Nue')
plt.plot(h_nentries_x,h_nentries_antinue,c='orange',label='Antinue')
plt.plot(h_nentries_x,h_nentries_nue_antinue,c='red',label='Nue+Antinue')
plt.plot(h_nentries_x,h_nentries_all,c='black',label='Entire overlay')
plt.legend(loc='best', prop={'size': legend_size})
plt.title('%.2e POT' % POT_MC, loc='left')
plt.ylim([0,2000])
plt.grid()
plt.xlabel('BDT score')
plt.ylabel('Number of selected events')
plt.tight_layout()
plt.savefig('plots/nevents.pdf')

# nue/antinue percentage
plt.figure(figsize=(5,5))
plt.plot(h_perc_x,h_perc_nue,c='blue',label='Nue')
plt.plot(h_perc_x,h_perc_antinue,c='orange',label='Antinue')
plt.legend(loc='best', prop={'size': legend_size})
plt.grid()
plt.xlabel('BDT score')
plt.ylabel('Percentage')
plt.tight_layout()
plt.savefig('plots/percentage_nue_antinue.pdf')




































# -------------------------- #
#   START SECOND-STAGE BDT   #
# -------------------------- #

# Let's try to run a second-stage BDT selection to try to distinguish between
# nue and antinue events. So the idea is to try to split the nue+antinue sample
# with a BDT score above a certain cut between nue and antinue. Remember that
# we are still using overlay only for this selection. We will later on use data
# for comparison, and to see what is the efficiency and purity when it comes#
# to data, but let's focus on the second-stage BDT selection for now.

print('\n\n\nStarting 2-stage BDT selection:\n')

# ----------------------------------- #
#   CALCULATE PARTICLE MULTIPLICITY   #
# ----------------------------------- #

# First calculate the particle multiplicity for the interactions and add this
# column to your overlay dataframe, so it also contains this information.
# Remember that not all files have the entire particle flow saved right now,
# so make sure to use the correct overlay file here.

print('   - Calculate particle multiplicity...')

# First we need to import a few extra variables to calculate the particle multiplicity
pfeval_particle = ['reco_mother','reco_pdg']
T_PFeval = uproot.open(filename_overlay)['wcpselection/T_PFeval']
df_PFeval = T_PFeval.pandas.df(pfeval_particle, flatten=False)
df_overlay = pd.concat([df_overlay,df_PFeval], axis=1)

# pick here which dataframe you want to use for your BDT selection
df = df_overlay 

def calculate_particle_multiplicity(DF):

    DF.loc[:,'Num_muons'] = [0]*DF.shape[0]
    DF.loc[:,'Num_kaons'] = [0]*DF.shape[0]
    DF.loc[:,'Num_pions'] = [0]*DF.shape[0]
    DF.loc[:,'Num_protons'] = [0]*DF.shape[0]
    DF.loc[:,'Num_photons'] = [0]*DF.shape[0]
    DF.loc[:,'Num_electrons'] = [0]*DF.shape[0]
    DF.loc[:,'Num_neutrons'] = [0]*DF.shape[0]

    pdg_N = 2112
    pdg_mu = 13
    pdg_k = 321
    pdg_pi = 211
    pdg_P = 2212
    pdg_gamma = 22
    pdg_e = 11

    for x,y,w in zip(DF.reco_mother, DF.reco_pdg, range(len(DF.reco_mother))):
        if (len(x) > 0) & (pdg_N in y): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_N and x[i] ==0)]
            DF.loc[w,'Num_neutrons'] = len(index_)
        if (len(x) > 0) & ((pdg_mu in y) | (-pdg_mu in y)): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_mu and x[i] ==0)]
            DF.loc[w,'Num_muons'] = len(index_)
        if (len(x) > 0) & ((pdg_k in y) | (-pdg_k in y)): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_k and x[i] ==0)]
            DF.loc[w,'Num_kaons'] = len(index_)
        if (len(x) > 0) & ((pdg_pi in y) | (-pdg_pi in y)): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_pi and x[i] ==0)]
            DF.loc[w,'Num_pions'] = len(index_)
        if (len(x) > 0) & ((pdg_P in y) | (-pdg_P in y)): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_P and x[i] ==0)]
            DF.loc[w,'Num_protons'] = len(index_)
        if (len(x) > 0) & (pdg_gamma in y): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_gamma and x[i] ==0)]
            DF.loc[w,'Num_photons'] = len(index_)
        if (len(x) > 0) & ((pdg_e in y) | (-pdg_e in y)): 
            index_ = [i for i,z in enumerate(y) if (abs(z) == pdg_e and x[i] ==0)]
            DF.loc[w,'Num_electrons'] = len(index_)

    return DF

df_overlay = calculate_particle_multiplicity(df)

particle_multiplicity = ['Num_neutrons','Num_muons','Num_kaons','Num_pions','Num_protons','Num_photons','Num_electrons']

# -------------------------------- #
#   DEFINE SIGNAL AND BACKGROUND   #
# -------------------------------- #

# We want to choose events with a BDT score higher than a cut, and then split them
# into antinue=signal, and nue=background. The overall conditions for this selection
# remains the same as the one used earlier to define signal and background, just
# adding the BDT cut to it.

print('   - Define signal and background...')

# First, select the BDT score that you want to use for your selection
bdt_cut = 0.6

# Define signal=antinue and background=nue
def define_2bdt_signal(df):

    df_ = df[df.truth_nuPdg==-12]
    df_ = df_[df_.truth_isCC==1]
    df_ = df_[df_.truth_vtxInside==1]
    df_ = df_[df_.vtx_dist <= squared_min_dist]
    df_ = df_[df_.bdt_score > bdt_cut]

    return df_

def define_2bdt_background(df):

    df_ = df[(df.truth_nuPdg==12)]
    df_ = df_[df_.truth_isCC==1]
    df_ = df_[df_.truth_vtxInside==1]
    df_ = df_[df_.vtx_dist <= squared_min_dist]
    df_ = df_[df_.bdt_score > bdt_cut]

    return df_

df_2bdt_signal = define_2bdt_signal(df)
df_2bdt_background = define_2bdt_background(df)

print('Overlay (BDT score above cut)      %i' % len(df_overlay[df_overlay.bdt_score > bdt_cut]))
print('Signal (antinue)                   %i' % len(df_2bdt_signal))
print('Background (nue)                   %i' % len(df_2bdt_background))

variables_w = extra_vars + particle_multiplicity + kine_vars + bdt_vars + ['weight']
variables   = extra_vars + particle_multiplicity + kine_vars + bdt_vars

# --------------------------------------------------- #
#   CREATE VALIDATION, TESTING AND TRAINING SAMPLES   #
# --------------------------------------------------- #

print('   - Create validation, testing and training samples')

df_signal_train, df_signal_val, df_signal_test = split_train_val_test(df_2bdt_signal, 'Signal')
df_background_train, df_background_val, df_background_test = split_train_val_test(df_2bdt_background, 'Background')

df_train, x_train, y_train, w_train = associate_variables(df_signal_train,df_background_train)
df_val, x_val, y_val, w_val = associate_variables(df_signal_val,df_background_val)
df_test, x_test, y_test, w_test = associate_variables(df_signal_test,df_background_test)

# -------------------- #
#   2ND BDT TRAINING   #
# -------------------- #

use_label_encoder=False

param = {'n_estimators':           300, 
         'max_depth':              4,
         #'scale_pos_weight':       1,
         'learning_rate':          0.1,
         'objective':              'binary:logistic',
         'colsample_bytree':       0.8,
         'lambda' :                1}

model.fit(x_train,                                              # feature matrix
          y_train,                                              # labels (Y=1 signal, Y=0 background)
          sample_weight=w_train,                                # instance weights
          eval_set = [(x_train,y_train), (x_val,y_val)],        # a list of (X,y) tuple pairs to use as validation sets ---> validation_0=train, validation_1=validation
          sample_weight_eval_set = [w_train, w_val],            # list of arrays storing instances weights for the i-th validation set
          eval_metric = ['auc', 'error'],                       # list of parameters under eval_metric: https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
          early_stopping_rounds=100,                             # validation metric needs to improve at least once in every early_stopping_rounds round(s)
          verbose=100)

# --- take results
results = model.evals_result()                            # takes the results from the BDT training above
n_estimators = len(results['validation_0']['error'])      # number of rounds used for the BDT training
auc_train = results['validation_0']['auc']                # subsample: auc for training
auc_val = results['validation_1']['auc']                  # subsample: auc for validation
error_train = results['validation_0']['error']            # subsample: error for training
error_val = results['validation_1']['error']              # subsample: error for validation

df_overlay.loc[:,'bdt_score_2nd'] = model.predict_proba(df_overlay[variables])[:,1]

# --------------- #
#   MAKE PLOTS    #
# --------------- #

print('\nMaking plots...')
legend_size=12

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

plt.savefig('plots/2nd_bdt_auc_error.pdf')

# --- plot important features

plt.figure(figsize=(8,5))
list_feat = plot_important_features(variables_w[:-2], model.feature_importances_, 5, 'NC') # number not greater than the number of variables

# --- bdt score

pred_sig_train = model.predict_proba(df_signal_train[variables])[:,1] # column 1=success, 0=fail
pred_sig_test = model.predict_proba(df_signal_test[variables])[:,1]
pred_bkg_train = model.predict_proba(df_background_train[variables])[:,1]
pred_bkg_test = model.predict_proba(df_background_test[variables])[:,1]

plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.yscale('log')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('plots/2nd_bdt_score_log_scale.pdf')


plt.figure(figsize=(14,4))
nbins=50
xrange=(0,1)

plt.subplot(121)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, density=True, color='red', alpha=0.5, label='Sig train (pdf)')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, density=True, color='blue', alpha=0.5, label='Bkg train (pdf)')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test (pdf)', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, density=True, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test (pdf)', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.subplot(122)
plt.hist(pred_sig_train, weights=df_signal_train['weight'], bins=nbins, range=xrange, color='red', alpha=0.5, label='Sig train')
plt.hist(pred_bkg_train, weights=df_background_train['weight'], bins=nbins, range=xrange, color='blue', alpha=0.5, label='Bkg train')

hist_sig_test, bins1, _1 = plt.hist(pred_sig_test, weights=df_signal_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_sig_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='firebrick', label='Sig test', fmt='o')

hist_bkg_test, bins1, _1 = plt.hist(pred_bkg_test, weights=df_background_test['weight'], bins=nbins, range=xrange, alpha=0)
mid=0.5*(bins1[1:] + bins1[:-1])
plt.errorbar(x=mid, y=hist_bkg_test, xerr=0.5*xrange[1]/nbins, yerr=[0]*nbins, c='navy', label='Bkg test', fmt='o')

plt.xlim(xrange)
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})

plt.savefig('plots/2nd_bdt_score.pdf')

# =================================== #
#   CALCULATE EFFICIENTY AND PURITY   #
# =================================== #

# --- calculate the number of selected events after the second BDT

bdt_var_min = 0.0
bdt_var_max = 1.0
step = 0.1

var_bdt_score = bdt_var_min
while(var_bdt_score < bdt_var_max):

    # count the number of selected antinue with a new bdt score above a cut
    df = df_overlay
    df = df[(df.truth_nuPdg==-12) & (df.truth_isCC==1) & (df.truth_vtxInside==1) & (df.bdt_score_2nd>var_bdt_score)]
    print('BDT score = %0.2f     Selected antinue = %i' % (var_bdt_score, len(df)))

    # update the loop variable
    var_bdt_score = var_bdt_score + step





































'''



# --- Split overlay into signal and background to calculate efficiency and purity
# --- make sure the events pass the generic neutrino selection first

# --- select here what is the your denominator
df_overlay = df_overlay 
calculate_ratios(df_overlay,"Overlay")
#df_overlay = apply_gen_nu_selection(df_overlay)          # pass GenNuSelection
#calculate_ratios(df_overlay,"Overlay + GenNuSel")
df_overlay = df_overlay[df_overlay.truth_vtxInside==1]   # inAV
calculate_ratios(df_overlay,"Overlay + GenNuSel + inAV")

df_signal_bdt_score = define_signal(df_overlay)
df_background_bdt_score = define_background(df_overlay)

print('\nCalculate Efficiency and Purity')
print_nentries(df_signal_bdt_score,'Signal BDT')
print_nentries(df_background_bdt_score,'Background BDT')

# --- plot bdt score for signal and background
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.hist(df_signal_bdt_score.bdt_score, range=(0,1), label='Signal')
plt.title('Signal')
plt.xlabel('BDT score')
plt.subplot(122)
plt.hist(df_background_bdt_score.bdt_score, range=(0,1), label='Background')
plt.title('Background')
plt.xlabel('BDT score')
plt.savefig('plots/overlay_bdt_score.pdf')

# --- calculate efficiency and purity per cut
var_bdt_score = 0.0
step = 0.01

h_pur = []
h_eff = []
h_nevents_nue = []
h_nevents_antinue = []
h_nevents = []
h_nue = []
h_antinue = []

h_x_pur = []
h_x_eff = []
h_x_nevents = []
h_x = []

# fix eff definition
# eff = 
#       everything in AV

# --- calculate the initial percentage of the signal made of nue and antinue
nue = len(df_signal_bdt_score[df_signal_bdt_score.truth_nuPdg==12])/len(df_signal_bdt_score)
antinue = len(df_signal_bdt_score[df_signal_bdt_score.truth_nuPdg==-12])/len(df_signal_bdt_score)
print('nue=%.2f   antinue=%.2f' % (nue,antinue))

while(var_bdt_score<1):

    df_signal_cut = df_signal_bdt_score[df_signal_bdt_score.bdt_score>var_bdt_score]
    df_background_cut = df_background_bdt_score[df_background_bdt_score.bdt_score>var_bdt_score]

    if(len(df_signal_cut)!=0):
        h_nue.append(len(df_signal_cut[df_signal_cut.truth_nuPdg==12])/len(df_signal_cut))
        h_antinue.append(len(df_signal_cut[df_signal_cut.truth_nuPdg==-12])/len(df_signal_cut))
        h_x.append(var_bdt_score)

    # calculate purity
    if((len(df_signal_cut)+len(df_background_cut))!=0):
        pur = len(df_signal_cut)/(len(df_signal_cut)+len(df_background_cut))
        #print('BDT score = %.2f           Purity = %i/%i = %.4f' % (var_bdt_score,len(df_signal_cut),(len(df_signal_cut)+len(df_background_cut)),pur))
        h_pur.append(pur)
        h_x_pur.append(var_bdt_score)
    
    # calculate efficiency
    if((len(df_signal_bdt_score)!=0)):
        eff = len(df_signal_cut)/len(df_signal_bdt_score)
        #print('BDT score = %.2f           Efficiency = %i/%i = %.4f' % (var_bdt_score,len(df_signal_cut),len(df_signal_bdt_score),eff))
        h_eff.append(eff)
        h_x_eff.append(var_bdt_score)

    # calculate number of selected events
    h_nevents_nue.append(len(df_signal_cut[df_signal_cut.truth_nuPdg==12]))
    h_nevents_antinue.append(len(df_signal_cut[df_signal_cut.truth_nuPdg==-12]))
    h_nevents.append(len(df_signal_cut))
    h_x_nevents.append(var_bdt_score)

    var_bdt_score = var_bdt_score + step

# make plot
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

# efficiency and purity
plt.plot(h_x_pur, h_pur, c='orange', label='Purity')
plt.plot(h_x_eff, h_eff, c='blue', label='Efficiency')
plt.grid()
plt.xlim([0,1])
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})
plt.savefig('plots/efficiency_purity.pdf')

# number of events
plt.figure(figsize=(5,5))
plt.plot(h_x_nevents,h_nevents_nue,c='blue',label='Nue')
plt.plot(h_x_nevents,h_nevents_antinue,c='orange',label='Antinue')
plt.plot(h_x_nevents,h_nevents,c='red',label='Nue+Antinue')
plt.legend(loc='best', prop={'size': legend_size})
plt.grid()
plt.xlabel('BDT score')
plt.ylabel('Number of selected events')
plt.savefig('plots/nevents.pdf')

# percentage of nue and antinue in the selected sample
plt.figure(figsize=(5,5))
plt.grid()
plt.plot(h_x,h_nue,c='blue',label='Nue')
plt.plot(h_x,h_antinue,c='orange',label='Antinue')
plt.legend(loc='best', prop={'size': legend_size})
plt.xlim([0,1])
plt.xlabel('BDT score')
plt.ylabel('Percentage')
plt.savefig('plots/nue_antinue.pdf')

# efficiency as a function of energy
bdt_cut = 0.8                      # bdt score cut for the plot
energy_initial = 0                 # eV
energy_final = 5000                # eV
step = 200                         # eV
h_nevents_original = []            # number of events for the energy range
h_nevents_bdt = []                 # number of events for the energy range, with bdt_score > bdt_cut
h_nevents_x = []
h_energy_eff = []                  # efficiency as a function of energy
h_energy_eff_x = []                # energy
df_eff_calc = df_overlay           # choose the original sample
while ((energy_initial+step)<=energy_final):

    # number of events in the original overlay sample for the given energy range
    df_initial = define_signal(df_eff_calc)
    df_initial = df_initial[(df_initial.kine_reco_Enu>energy_initial) & (df_initial.kine_reco_Enu<(energy_initial+step))]
    nevents_initial = len(df_initial)
    h_nevents_original.append(nevents_initial)
    h_nevents_x.append(energy_initial)

    # number of events in the selected sample for the given energy range
    df_selected = define_signal(df_eff_calc)
    df_selected = df_selected[(df_selected.bdt_score>bdt_cut) & (df_selected.kine_reco_Enu>energy_initial) & (df_selected.kine_reco_Enu<(energy_initial+step))]
    nevents_selected = len(df_selected)
    h_nevents_bdt.append(nevents_selected)

    # calculate efficiency
    if(len(df_initial)!=0):
        eff = len(df_selected)/len(df_initial)
        h_energy_eff.append(eff)
        h_energy_eff_x.append(energy_initial)
        
    # update energy
    energy_initial = energy_initial + step
plt.figure(figsize=(5,5))
plt.grid()
plt.plot(h_energy_eff_x, h_energy_eff, c='blue')
plt.title('BDT score cut = %.2f' % bdt_cut)
plt.ylabel('Efficiency')
plt.xlabel('Neutrino Reco Energy [MeV]')
plt.tight_layout()
plt.savefig('plots/eff_energy.pdf')

plt.figure(figsize=(5,5))
plt.grid()
plt.plot(h_nevents_x, h_nevents_original, c='blue', label='W/o BDT cut')
plt.plot(h_nevents_x, h_nevents_bdt, c='red', label='With BDT cut')
plt.ylabel('Number of Events')
plt.xlabel('Neutrino Reco Energy [MeV]')
plt.legend(loc='best', prop={'size': legend_size})
plt.tight_layout()
plt.savefig('plots/eff_energy_nevents.pdf')







'''











'''


# ---------------------- #
#   DATA/MC COMPARISON   #
# ---------------------- #

def create_dataframe_reco(file):
    T_kine = uproot.open(file)['wcpselection/T_KINEvars']
    df_kine = T_kine.pandas.df(kine_vars, flatten=False)
    T_bdt = uproot.open(file)['wcpselection/T_BDTvars']
    df_bdt = T_bdt.pandas.df(bdt_vars, flatten=False)
    df_ = pd.concat([df_kine,df_bdt],axis=1)
    #calculate_extra_vars(df_,file,'DATA')                              # calculate extra variables
    #df_.loc[:,'bdt_score'] = model.predict_proba(df_[variables])[:,1]  # add bdt score
    return df_

def calculate_POT(file):
    T_pot = uproot.open(file)['wcpselection/T_pot']
    df_pot = T_pot.pandas.df("pot_tor875", flatten=False)
    POT = sum(df_pot.pot_tor875)
    return POT

# --- import files and variables

datafile = '../rootfiles/checkout_data_numi_run1_morestat.root'
df_data  = create_dataframe_reco(datafile)
DATA_POT = 2.064e+50

mcfile   = '../rootfiles/checkout_prodgenie_numi_overlay_run1.root'
df_mc    = create_dataframe_reco(mcfile)
MC_POT   = calculate_POT(mcfile)

extfile  = '../rootfiles/checkout_data_extnumi_run1.root'
df_ext   = create_dataframe_reco(extfile)
EXT_POT  = 1.1603984e+50

print('Data      %.2e POT' % DATA_POT)
print('Overlay   %.2e POT' % MC_POT)
print('EXT       %.2e POT' % EXT_POT)

# --- plot variables

def plot_variable(datavar,mcvar,filename):

    plt.figure(figsize=(5,5))

    # data
    plt.hist(datavar, bins=100, histtype='step')
    
    # mc
    plt.hist(mcvar, bins=100, weights=MC_POT/DATA_POT)

    # visual
    plt.ylim(0,10000)
    plt.xlim(0,2000)
    plt.grid()
    plt.xlabel('Reco Neutrino Energy [eV]')
    plt.ylabel('Entries')

    plt.tight_layout()
    plt.savefig('plots/%s.pdf' % filename)


plot_variable(df_data.kine_reco_Enu,df_mc.kine_reco_Enu,'data_mc_energy.pdf')

'''

'''

# --- create signal
# --- what is classified as signal according to my bdt score?
bdt_cut=0.8
df_data_sig = df_data[df_data.bdt_score>bdt_cut]
df_mc_sig = df_overlay[df_overlay.bdt_score>bdt_cut]
#

# --- check true info
# --- among the reco signal, what is real signal and what is real background?
df_mc_true_sig = define_signal(df_mc_sig)
df_mc_true_bkg = define_signal(df_mc_sig)

# --- POT normalise overlay
POT_ratio = POT_MC/DATA_POT

# --- plot data/MC comparison
def compare_mc_data(variable):

    # create array for data
    arr_data = df_data_sig[variable]

    # create array for mc sig and bkg
    arr_sig = df_mc_true_sig[variable]
    arr_bkg = df_mc_true_bkg[variable]

    # plot stacked hist
    plt.figure(figsize=(5,5))

    arr_hist = [arr_sig, arr_bkg]

    plt.hist(arr_hist,
             bins=40,
             stacked=True,
             histtype='bar',
             color=["blue","red"],
             label=["Signal","Background"])

    plt.hist(arr_data,
             bins=40,
             label=["Data"])

    plt.savefig('plots/%s.pdf' % variable)
    

compare_mc_data("bdt_score")
'''