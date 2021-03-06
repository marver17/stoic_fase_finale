#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:14:18 2022

@author: admin_Mario
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc,roc_auc_score,confusion_matrix
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.combine import SMOTEENN
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,QuantileTransformer,MinMaxScaler
from numpy import interp
import shutil
ARTIFACT_DIR = "/output/"

def do_train():
    
    
    data =pd.read_csv("./algorithm/value.csv",sep = ";",decimal=",")
    L = ["prob_covid","prob_severe","ID","age","vol_lesion","mean","n_con"]
    # L = ["probCOVID","probSevere","ID","age","Volume_lesione_normalizzataata","Media","# comp"]

    x = data[L].to_numpy()
    x = np.nan_to_num(x)
    Y_covid = data["prob_covid"].to_numpy()
    Y_severe = data["prob_severe"].to_numpy()
    positivi = np.where(Y_covid == 1)
    x_positivi = x[positivi]
    train,test = train_test_split(x, test_size=0.2, stratify=x[:,1])
    l = np.where(train[:,4]<0.003) 
    train = np.delete(train,(l), axis = 0)
    
    
    
    trainX = train[:,2:]
    testX = test[:,2:]
    
    
    trainy_covid = train[:,0]
    trainy_severe = train[:,1]
    testy_covid = test[:,0]
    testy_severe = test[:,1]
    
    
    trainX_Covid = trainX[:,1:]
    trainX_Severe = trainX[:,1:]
    
    
    ID_test = testX[:,0]
    testX = testX[:,1:]
    i = np.where(testX[:,1]<0.003) 
    tt = testX[i] 
    testX = np.delete(testX,(i), axis = 0)
    testX_Severe = testX
    
    testX_Covid = testX
    scartati_covid = testy_covid[i]
    scartati_severe = testy_severe[i]
    ID_test1 = ID_test[i]
    testy_covid= np.delete(testy_covid,(i), axis = 0)
    testy_severe= np.delete(testy_severe,(i), axis = 0)
    ID_test = np.delete(ID_test,(i), axis = 0)
    ID_test= np.concatenate((ID_test,ID_test1),axis=0)  
    fea = np.copy(testX)
    no_covid = np.where(testy_covid == 0)
    testy_severe = np.delete(testy_severe,(no_covid), axis = 0)
    testX_Severe = np.delete(testX_Severe,(no_covid), axis = 0)
    
    
    sample = SMOTEENN(sampling_strategy=0.75)
    X_over, y_over = sample.fit_resample(trainX_Covid,trainy_covid)
    trainX_C, valX, trainy_covid, valy_severe = train_test_split(X_over, y_over, test_size=0.2, random_state=2, stratify=y_over)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter= 1000,penalty="l2",class_weight="balanced"))   
    pipe.fit(trainX_C, trainy_covid)
    with open('/output/model_c.sav', 'wb') as files:
          pickle.dump(pipe, files)
    
    with open('./algorithm/model_c.sav', 'wb') as files:
          pickle.dump(pipe, files)
    
    #shutil.copy('./algorithm/model_c.sav', ARTIFACT_DIR)
    
    A = pipe.predict_proba(testX_Covid)
    pred_test = [ A[i][1] for i in range(len(A))]
    testy_covid = np.concatenate((testy_covid,scartati_covid),axis=0)  
    pred_test_covid = np.concatenate((np.array(pred_test),np.zeros(len(scartati_covid))),axis = 0) 
    AUC_covid = roc_auc_score(testy_covid,pred_test_covid)
    
 
    
    pipe1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter= 1000,penalty="l2",class_weight="balanced"))   
    pipe1.fit(trainX_Severe, trainy_severe)
    with open('/output/model_s.sav', 'wb') as files:
          pickle.dump(pipe1, files)
    
    with open('./algorithm/model_s.sav', 'wb') as files:
          pickle.dump(pipe1, files)
    
    #shutil.copy('./algorithm/model_s.sav', ARTIFACT_DIR)

    
    A = pipe1.predict_proba(testX_Severe)
    pred_test_severe = [ A[i][1] for i in range(len(A))]
    
    testy_severe = np.concatenate((testy_severe,scartati_severe),axis=0)  
    pred_test_severe =   np.concatenate((np.array(pred_test_severe),np.zeros(len(scartati_severe))),axis = 0) 
    AUC_severe = roc_auc_score(testy_severe,pred_test_severe)
 
    return AUC_covid,AUC_severe

