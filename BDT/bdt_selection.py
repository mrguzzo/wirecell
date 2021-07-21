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

# =================== #
#      FUNCTIONS      #
# =================== #

bdt_vars = [] # used for BDT
kine_vars = [] # used for BDT
pot_vars = ['pot_tor875']
eval_vars = ['truth_isCC','truth_nuPdg','truth_vtxInside','weight_spline','weight_cv']
pfeval_vars = ['reco_nuvtxX','reco_nuvtxY','reco_nuvtxZ',
               'truth_corr_nuvtxX','truth_corr_nuvtxY','truth_corr_nuvtxZ']
extra_vars = ['cos_theta']

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

    # --- fix weights
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
    elif(family=='MC'): W_ = 1     #POT/POT_NUE

    df.loc[:,'weight_genie'] = df['weight_cv']*df['weight_spline']
    df.loc[:,'weight'] = [W_]*df.shape[0]*df['weight_genie']

    # variable created to classify signal and background dataframes
    if(family=='NUE'): df.loc[:,'original_file'] = 0
    elif(family=='MC'): df.loc[:,'original_file'] = 1

    # --- delete dataframes to free memory space (not sure if it's necessary)
    del df_pot
    del df_KINE
    del df_BDT 
    del df_PFeval 
    del df_eval

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

    return df, POT

def apply_vtx_quality(df):

    distX = df.truth_corr_nuvtxX - df.reco_nuvtxX
    distY = df.truth_corr_nuvtxY - df.reco_nuvtxY
    distZ = df.truth_corr_nuvtxZ - df.reco_nuvtxZ

    min_dist = 1 # unit = cm
    squared_min_dist = min_dist * min_dist

    dist_squared = distX*distX + distY*distY + distZ*distZ

    df.loc[:,'vtx_dist'] = dist_squared

    df_ = df[df.vtx_dist < squared_min_dist]

    return df_

def define_signal(df):

    df_ = df[ (df.truth_nuPdg==-12) | (df.truth_nuPdg==12) ]                # definition
    df_ = df_[df_.truth_isCC==1]                                            # apply CC interaction condition 
    df_ = df_[df_.truth_vtxInside==1]                                       # apply in active volume condition
    df_ = apply_vtx_quality(df_)                                            # check reco-true vertex distance

    return df_

def define_background(df):

    df_ = df[ (df.truth_nuPdg!=-12) & (df.truth_nuPdg!=12) ]
    df_ = df_[df_.truth_isCC==1]
    df_ = df_[df_.truth_vtxInside==1]

    return df_

def resize_samples(df1,df2):

    # this function takes both input dataframe and
    # resizes them in a way that they end up with the
    # same number of entries

    df1_size = len(df1)
    df2_size = len(df2)

    if(df1_size > df2_size):
        df1 = shuffle(df1).reset_index(drop=True)
        df1 = df1.iloc[:df2_size,:]
    else:
        df2 = shuffle(df2).reset_index(drop=True)
        df2 = df2.iloc[:df1_size,:]

    return df1,df2

#==================== #
#      OPEN FILE      #
#==================== #

filename_intrinsic_nue = '../rootfiles/checkout_prodgenie_numi_intrinsic_nue_overlay_run1_OFFSETFIXED2.root'
filename_overlay = '../rootfiles/checkout_prodgenie_numi_overlay_run1.root'

# create dataframes
df_intrinsic_nue, POT_NUE = create_dataframe(filename_intrinsic_nue,'NUE')
df_overlay, POT_overlay = create_dataframe(filename_overlay,'MC')

# merge dataframes
# this is going to be considered my dataset of origin
df_merged = pd.concat([df_intrinsic_nue,df_overlay], ignore_index=True)

# ============================================= # 
#      DEFINE SIGNAL/BACKGROUND DATAFRAMES      #
# ============================================= #

# create signal/background datasets
df_signal = define_signal(df_merged)
df_background = define_background(df_merged)

# create "class" label for them
df_signal['class'] = 1
df_background['class'] = 0

df_signal, df_background = resize_samples(df_signal,df_background) # comment out if necessary

# combine signal and background datasets
df = df_signal.append(df_background)
df = df.sample(frac=1).reset_index(drop=True) # shuffle

# ========================================================= #
#      CREATE VALIDATION, TESTING AND TRAINING SAMPLES      #
# ========================================================= #

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

variables_w = extra_vars + kine_vars + bdt_vars + ['weight']
variables = extra_vars + kine_vars + bdt_vars

y_df = df.pop('class') # takes only 'class' from the dataset

X_unscaled = np.array(df[variables])
Y = np.array(y_df)

x_train_unscaled, x_test_unscaled, y_train, y_test = train_test_split(X_unscaled, Y, test_size=0.3, random_state=0, shuffle=True, stratify=Y)

# perform initial transformations
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_unscaled)
x_test = scaler.fit_transform(x_test_unscaled)
X = scaler.transform(X_unscaled)

# start BDT
dtrain = xgboost.DMatrix(x_train_unscaled, label = y_train)
dtest = xgboost.DMatrix(x_test_unscaled, label = y_test)

param = {#'n_estimators':           300,
         'max_depth':              3,
         'scale_pos_weight':       1,
         'learning_rate':          0.01,
         'objective':              'binary:logistic',
         'colsample_bytree':       0.8,
         'lambda' :                1}

num_round = 100 # maximum number of rounds
progress = dict()
watchlist =  [ (dtrain, 'train'),(dtest, 'validation')]

bst = xgboost.train(param, dtrain, num_round,
                    evals = watchlist,
                    verbose_eval = 5 ,
                   # early_stopping_rounds = 50, #You can set this to stop training when the validation set starts to turn over (overtraining is occurring)
                    evals_result=progress)

plt.plot(progress['train']['error'], label='Training')
plt.plot(progress['validation']['error'], label ='Validation')
plt.axvline(np.argmin(progress['validation']['error']), ls = '--')
plt.xlabel('Boosting Stage', fontsize = 16)
plt.ylabel('Error', fontsize = 16)
plt.title('BDT Training', fontsize = 18)
plt.legend()