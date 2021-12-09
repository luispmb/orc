#!/usr/bin/env python
# coding: utf-8

# In[1]:


#inputs
import os
parent_directory = os.path.abspath('')
root = os.path.abspath(os.path.join(parent_directory, os.pardir))
data_folder = '01-Data'
classifiers_folder = '02-Classifiers'
gan_char_models_folder = 'models_char_gan'
model_digits_letters_name = 'class_char_model_{}.h5'
mixed_models_folder = 'models_mixed'
model_symbols_letters_name = 'model_0symbol_1letter.h5'
model_0_oO_name = 'model_0_oO.h5'

characters_folder = 'characters'
test_images_folder = os.path.join(parent_directory, characters_folder)
character_to_test = '{}.png'

data_input_folder = os.path.join(root, 'test', 'data', 'input')


# In[2]:


from functions_score import *
from functions_char_preparation import *
from char_classification import classification, image_char_prepr
import sys
import shutil
import cv2
import numpy as np
import tensorflow as tf
import string
import pandas as pd
import keras.backend as K #clear RAM
import docx
from os import listdir
from os.path import isfile, join


# In[3]:


cwd = os.getcwd()
cwd


# In[4]:


def buidDictionary():

    characters_all = list(string.printable)[:-6] #+['ç']# <
    j=-1
    dict_target=[]
    for char in characters_all:
        j=j+1
        dict_target.append([char,ord(char),j])
    dictionar=pd.DataFrame(dict_target).rename(columns={0:'Actual_char',1:'Actual_num',2:'Actual_id'})

    dictionar_symbols=dictionar[62:94]
    dictionar_letters=dictionar[0:62]

    #threshold for _,-
    from pandas import DataFrame
    pd_ = DataFrame(np.arange(0,32,1))
    Q1_ = pd_.quantile(0.25)
    Q3_ = pd_.quantile(0.75)
    IQR_ = Q3_ - Q1_

    return characters_all, dictionar, dictionar_letters, dictionar_symbols, Q1_, Q3_


# In[5]:


class ML_Models():
    def addModel(self, modelsPath, modelOgirinalName, modelName):
        #char_models_filename = modelsPath.format(ord(char))
        char_models_filename = os.path.join(modelsPath, modelOgirinalName)
        print('char_models_filename', char_models_filename)

        if not hasattr(self, modelName):
            ml_model=tf.keras.models.load_model(char_models_filename)

            if modelName:
                setattr(self, modelName, ml_model)
            else:
                setattr(self, modelOgirinalName, ml_model)
        else:
            print('model ', modelName, ' in memory')


# In[6]:


def loadModels(characters_all):
    
    if not 'charModels' in globals():
        print('charModels not in locals neither in globals')
        global charModels
        charModels = ML_Models()
        print('charModels', charModels)

        gan_char_models_path = os.path.join(root, classifiers_folder, gan_char_models_folder)
        for char in characters_all[0:62]:
            print(char)
            charModels.addModel(gan_char_models_path, model_digits_letters_name.format(ord(char)), f'model_letters{ord(char)}')
            #K.clear_session()


        mixed_models_path = os.path.join(root, classifiers_folder, mixed_models_folder)
        for char in characters_all[62:94]:
            print(char)
            charModels.addModel(mixed_models_path, model_symbols_letters_name, f'model_symbols{ord(char)}')
            #K.clear_session()
            
        charModels.addModel(mixed_models_path, model_symbols_letters_name, 'model_symbols_letters')      
        charModels.addModel(mixed_models_path, model_0_oO_name, 'model_0_oO')

    else:
        print('charModels instantiated')       

    return charModels


# In[7]:


def char_img_preprocessing(img):
    #print('char_img_preprocessing', img)

    #img = cv2.imread(char_image)
    #print('img', img)
    #img = cv2.imread('C:\\Users\\Administrator\\OCR\\Final\\04-Recognition\\characters\\2.png')

    sorted_ctrs = char_preprocessing_step_1(img)
    #print('sorted_ctrs', sorted_ctrs)
    new_sorted_ctrs = char_preprocessing_step_2(sorted_ctrs,img)
    #print('new_sorted_ctrs', new_sorted_ctrs)
    temp_max_yh,temp_min_y = char_preprocessing_step_3(new_sorted_ctrs) #normalize height
    #print('temp_max_yh', temp_max_yh)
    #print('temp_min_y', temp_min_y)
    
    char_image,char_image_nh = char_preprocessing_step_4(img,new_sorted_ctrs,temp_max_yh,temp_min_y)
    
    return char_image, char_image_nh


# In[8]:


def char_recognition(charModels, new_img, new_img_normheight, dictionar, dictionar_letters, dictionar_symbols, Q1_,Q3_): #recognize char     
    import cv2 
    import  numpy as np
    
    resized=image_char_prepr(new_img,2,20,0)
    resized_normheight=image_char_prepr(new_img_normheight,2,20,0)

    new_image_density=resized.sum()

    x_test_right = np.expand_dims(resized, axis=-1)
    x_test = np.expand_dims(x_test_right, axis=0)
    forecast,dictionartemp = classification(charModels, resized, resized_normheight, x_test, Q1_, Q3_, dictionar, dictionar_letters, dictionar_symbols)
    text_char =str(forecast)
    return text_char


# In[9]:


'''
caracteres com problemas: 132, 140, 147, 186, 194, 233, 241, 280, 288
'''
unhandledChars = [140, 147, 186, 194, 233, 241, 280, 288]




# In[12]:


def main(char_image):
    print('CHAR_RECOGNITION')
    if len(char_image) == 0:
        data_input_folder = os.path.join(root, 'test', 'data', 'input')
        data_input_char_folder = os.path.join(data_input_folder, 'char_recognition')
        char_images_names = [f for f in listdir(data_input_char_folder) if isfile(join(data_input_char_folder, f))]
        print('char_images_names', char_images_names)
        char_image_name = char_images_names[0]
        char_image=cv2.imread(os.path.join(data_input_char_folder, char_image_name))

    characters_all, dictionar, dictionar_letters, dictionar_symbols, Q1_, Q3_ = buidDictionary()
    
    charModels = loadModels(characters_all)

    _char_image, char_image_nh = char_img_preprocessing(char_image)

    text_char = char_recognition(charModels, _char_image[0], char_image_nh[0], dictionar, dictionar_letters, dictionar_symbols, Q1_, Q3_)
    print('text_char', text_char)

    return text_char


if __name__ == '__main__':
    main(sys.argv[1]) 

