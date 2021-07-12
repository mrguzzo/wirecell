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
               'match_found', 'stm_eventtype', 'stm_lowenergy', 'stm_LM', 'stm_TGM', 'stm_STM', 'stm_FullDead','stm_clusterlength']

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

def calculate_extra_vars(df,file):

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































# ----------------- #
#    IMPORT FILES   #
# ----------------- #

def print_nentries(df,text):

    print('---> %s: %i = %i (%.2f) nue + %i (%.2f) mc' % (text,len(df),
                                                          len(df[df.original_file==0]),len(df[df.original_file==0])/len(df),
                                                          len(df[df.original_file==1]),len(df[df.original_file==1])/len(df)))

print('\nImport files')

filename_nue = '../rootfiles/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root'
filename_overlay = '../rootfiles/checkout_prodgenie_numi_overlay_run1.root'

df_intrinsic_nue, POT_NUE = create_dataframe(filename_nue,'NUE')     # create dataframe and calculate POT
calculate_extra_vars(df_intrinsic_nue,filename_nue)                  # calculate extra variables not in the checkout file
print('---> Intrinsic Nue sample: %i' % len(df_intrinsic_nue))

df_overlay, POT_MC = create_dataframe(filename_overlay,'MC')         # create dataframe and calculate POT
calculate_extra_vars(df_overlay,filename_overlay)                    # calculate extra variables not in the checkout file
print('---> Overlay sample      : %i' % len(df_overlay))

# --- merge dataframes
print('\nMerge samples')
df = pd.concat([df_intrinsic_nue,df_overlay], ignore_index=True)     # merge intrinsic/overlay dataframes and ignore index
print_nentries(df,'Merged sample')

# --- apply the generic nu selection to make sure events are reconstructed
df = apply_gen_nu_selection(df)  
print_nentries(df,'Passed Gen   ')



# -------------------------------- #
#   DEFINE SIGNAL AND BACKGROUND   #
# -------------------------------- #

print('\nDefine signal/background dataframes')

squared_min_dist = 5*5

def define_signal(df):

    df_ = df[ (df.truth_nuPdg==-12) | (df.truth_nuPdg==12) ]                # PDG definition
    df_ = df_[df_.truth_isCC==1]                                            # apply CC interaction condition 
    df_ = df_[df_.truth_vtxInside==1]                                       # apply in active volume condition
    df_ = df_[df_.vtx_dist <= squared_min_dist]                             # check reco-true vertex distance

    return df_

def define_background(df):

    # --- background is defined as everything that is not signal

    df_ = df[ ((df.truth_nuPdg!=-12) & (df.truth_nuPdg!=12)) |            # not nue nor antinue
              (df.truth_isCC!=1) |                                        # not CC
              (df.truth_vtxInside!=1) |                                   # event out of active volume
              (df.vtx_dist > squared_min_dist)]                           # reco-true vertex too far

    return df_

df_signal = define_signal(df)
print_nentries(df_signal,    'Signal    ')

df_background = define_background(df)
print_nentries(df_background,'Background')

# -------------------------------------------------- #
#   CREATE VALIDATION, TESTIN AND TRAINING SAMPLES   #
# -------------------------------------------------- #

print('\nSplit dataframe')

# --- variables used for my BDT training
variables_w = extra_vars + kine_vars + bdt_vars + ['weight']
variables   = extra_vars + kine_vars + bdt_vars

def split_train_val_test(df,tag):
    
    # test = 1/3 of the sample
    # validation = 1/6 of the sample
    # training = 1/2 of the sample
    
    # --- first split the dataframe into 1/3=test and 2/3=train
    df_test = df.iloc[(df.index % 3 == 0).astype(bool)].reset_index(drop=True)
    df_train = df.iloc[(df.index % 3 != 0).astype(bool)].reset_index(drop=True)
    
    # --- split train into 
    df_val = df_train.iloc[(df_train.index % 4 == 0).astype(bool)].reset_index(drop=True)
    df_train = df_train.iloc[(df_train.index % 4 != 0).astype(bool)].reset_index(drop=True)
    
    return df_train, df_val, df_test

# --- split signal and background into training, validation and testing
df_signal_train, df_signal_val, df_signal_test = split_train_val_test(df_signal, 'Signal')
print('\nSignal')
print_nentries(df_signal_train,   'Training   ')
print_nentries(df_signal_val,     'Validation ')
print_nentries(df_signal_test,    'Testing    ')

df_background_train, df_background_val, df_background_test = split_train_val_test(df_background, 'Background')
print('\nBackground')
print_nentries(df_background_train,'Training   ')
print_nentries(df_background_val,  'Validation ')
print_nentries(df_background_test, 'Testing    ')

# --- associate variables to the dataframes

print('\nFinal')
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

df_train, x_train, y_train, w_train = associate_variables(df_signal_train,df_background_train)
df_val, x_val, y_val, w_val = associate_variables(df_signal_val,df_background_val)
df_test, x_test, y_test, w_test = associate_variables(df_signal_test,df_background_test)

print('---> Training   : %i' % len(df_train))
print('---> Validation : %i' % len(df_val))
print('---> Testing    : %i' % len(df_test))

# --------------- #
#   BDT TRAINIG   #
# --------------- #

print('\nStarting BDT training...')

use_label_encoder=False # removes warning message because XGBClassifier won't be used in future releases


# --- hyperparameters used for the BDT
param = {'n_estimators':           300,
         'max_depth':              3,
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

# --- save bdt scora back to the dataframe
df_overlay.loc[:,'bdt_score'] = model.predict_proba(df_overlay[variables])[:,1]





































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

# --- Split overlay into signal and background to calculate efficiency and purity
# --- make sure the events pass the generic neutrino selection first

df_overlay = apply_gen_nu_selection(df_overlay) # apply gen nu selection

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
h_x_pur = []
h_x_eff = []

while(var_bdt_score<1):

    df_signal_cut = df_signal_bdt_score[df_signal_bdt_score.bdt_score>var_bdt_score]
    df_background_cut = df_background_bdt_score[df_background_bdt_score.bdt_score>var_bdt_score]

    # calculate purity
    if((len(df_signal_cut)+len(df_background_cut))!=0):
        pur = len(df_signal_cut)/(len(df_signal_cut)+len(df_background_cut))
        h_pur.append(pur)
        h_x_pur.append(var_bdt_score)
    
    # calculate efficiency
    if((len(df_signal_bdt_score)!=0)):
        eff = len(df_signal_cut)/len(df_signal_bdt_score)
        h_eff.append(eff)
        h_x_eff.append(var_bdt_score)

    var_bdt_score = var_bdt_score + step

# make plot
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

#plt.plot(hist_x, hist_y_eff, c='blue', label='Efficiency')
plt.plot(h_x_pur, h_pur, c='orange', label='Purity')
plt.plot(h_x_eff, h_eff, c='blue', label='Efficiency')
plt.grid()
plt.xlabel('BDT score')
plt.legend(loc='best', prop={'size': legend_size})
plt.savefig('plots/eff.pdf')

