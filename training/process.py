#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:19:48 2022
script di test per testare l' interazione dei vari attori in gioco
@author: admin
"""

from ctdataset import CTDataset
from tqdm import tqdm 
import os 
import pandas  as pd
from sklearn.model_selection import train_test_split
from config import get_config
import  torch.multiprocessing as mp
from algorithm.lesion_segmentation import P_lesion_segmentation,blockPrint
from algorithm.features_ex import features_exctractor
from multiprocessing import Pool



def get_datasets(config, data_dir):
    image_dir = os.path.join(data_dir, "data/mha/")
    reference_path = os.path.join(data_dir, "metadata/reference.csv")
    df = pd.read_csv(reference_path)


    df["x"] = df.apply(lambda row: os.path.join(image_dir, str(row["PatientID"]) + ".mha"), axis=1)
    df["y"] = df.apply(lambda row: [row["probCOVID"], row["probSevere"]], axis=1)


    data_train = df[["x", "y"]].to_dict("records")



    train_ds = CTDataset(data_train,"/scratch/")
    return train_ds

def operazione(i):
    CONFIGFILE = "./config/baseline.json"
    data_dir = "/input/"
    config = get_config(CONFIGFILE)
    train_ds = get_datasets(config, data_dir)

    extractor = features_exctractor(train_ds)
    return extractor._all_(i)



def do_process(data_dir):
    CONFIGFILE = "./config/baseline.json"



    config = get_config(CONFIGFILE)
    train_ds =  get_datasets(config, data_dir)
    ID = []
    com_connesse = []
    age = []
    sex = []
    vol_lesion = []
    mean_intensity = []
    std_intensity = []
    check_lesion = []
    prob_covid = []
    prob_severe = []    
    n_con = []
    imm = "all"
    #imm = 200
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    train_ds.get_all_x(2,imm,True)
    path_image = train_ds.get_path_list(imm,True)[1]
    n_imm = train_ds.get_path_list(imm,True)[3]
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    

    P_lesion_segmentation(4,path_image,data_dir + "/preprocess/")
    
    with Pool(processes=4) as pool:
        results =  tqdm(pool.imap(operazione, range(n_imm)), total=n_imm)
        for a,b,c,d,e,f,g,h,i,l  in results :
             ID.append(a)
             age.append(b)
             sex.append(c)
             vol_lesion.append(d)
             mean_intensity.append(e)
             std_intensity.append(f)
             check_lesion.append(g)
             prob_covid.append(h)
             prob_severe.append(i)
             n_con.append(l)
             
             
             
    dic = {"ID" : ID,
            "age" :age,
            "sex" : sex,
            "prob_covid" :prob_covid,
            "prob_severe" : prob_severe,
            "vol_lesion" : vol_lesion,
            "mean" : mean_intensity,
            "std" : std_intensity,
            "n_con" : n_con,
           "check" : check_lesion
              }
    
    
    df = pd.DataFrame.from_dict(dic)
    df.to_csv("./algorithm/value.csv",sep = ";",index=False,decimal=",")




