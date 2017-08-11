# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 21:58:55 2017

script for moving and sorting images to this format:
data/
    train/
        invasive/
            1.jpg
            ...
        non/
            3.jpg
            ...
    validation/
        invasive/
            3.jpg
            ...
        non/
            6.jpg
            ...

@author: Lorand
"""
import os
import shutil
import pandas as pd

#data: 2295
#train: 2000
#valid: 295

src = 'data/train'
train_inv = 'data/train/invasive'
train_non = 'data/train/non'

valid_inv = 'data/validate/invasive'
valid_non = 'data/validate/non'

df = pd.read_csv('data/train_labels.csv')

#list files in src folder
src_files = os.listdir(src)
i = 0
#copy all listed files
for file_name in src_files:
    #chek if it is a file, not a directory
    full_file_name = os.path.join(src, file_name)
    if (os.path.isfile(full_file_name)):
        
        #check if it invasive or non..
        number = int(file_name[:-4])# "234.jpg" ->  234
        
        res = df.loc[df['name'] == number]
        
        if i < 295:
            #copy to validation
            if res.loc[res.index[0], 'invasive'] == 1:
                shutil.move(full_file_name, valid_inv)
            else:
                shutil.move(full_file_name, valid_non)
        else:
            #copy to train
            if res.loc[res.index[0], 'invasive'] == 1:
                shutil.move(full_file_name, train_inv)
            else:
                shutil.move(full_file_name, train_non)
        
    i = i + 1
    
############################################################
# move images back to rain base folder, for re-distribution
#folder_list = [train_inv,train_non,valid_inv,valid_non ]
#for folder in folder_list:# iterate the folders
#    src_files = os.listdir(src)
#    
#    for file_name in src_files:
#        full_file_name = os.path.join(src, file_name)
#        if (os.path.isfile(full_file_name)):#is it a file?
#            shutil.move(full_file_name, src)
############################################################