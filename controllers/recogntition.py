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


# In[2]:


root


# In[ ]:





# In[3]:


from functions_score import *
from functions_char_preparation import *
import shutil
import cv2
import numpy as np
import tensorflow as tf
import string
import pandas as pd
import keras.backend as K #clear RAM
import docx


# In[4]:


cwd = os.getcwd()
cwd


# In[5]:


characters_all = list(string.printable)[:-6] #+['ç']# <
j=-1
dict_target=[]
for char in characters_all:
    j=j+1
    dict_target.append([char,ord(char),j])
dictionar=pd.DataFrame(dict_target).rename(columns={0:'Actual_char',1:'Actual_num',2:'Actual_id'})

dictionar_symbols=dictionar[62:94]
dictionar_letters=dictionar[0:62]
#print('dictionar_symbols', dictionar_symbols)
# print('dictionar_letters', dictionar_letters)


gan_char_models_path = os.path.join(root, classifiers_folder, gan_char_models_folder, model_digits_letters_name)
#gan_char_models_path.replace('%s', ord(char))
for char in characters_all[0:62]:
    print(char)

    #model_temp=tf.keras.models.load_model('C:\\Users\\Administrator\\OCR\\Final\\02-Classifiers\\models_char_gan\\32_\\class_char_model_%s.h5'%(ord(char)))
    gan_char_models_filename = gan_char_models_path.format(ord(char))
    print('gan_char_models_filename', gan_char_models_filename)
    model_temp=tf.keras.models.load_model(gan_char_models_filename)
    #creates variable name dynamically
    exec(f'model_letters{ord(char)} = model_temp')
    K.clear_session()

mixed_models_path = os.path.join(root, classifiers_folder, mixed_models_folder)
for char in characters_all[62:94]:
    print(char)
    
    # model_temp=tf.keras.models.load_model('C:\\Users\\Administrator\\OCR\\Final\\02-Classifiers\\\models_char_gan\\32_\\class_char_model_%s.h5'%(ord(char)))
    gan_char_models_filename = gan_char_models_path.format(ord(char))
    print('gan_char_models_filename', gan_char_models_filename)
    model_temp=tf.keras.models.load_model(gan_char_models_filename)
    #creates variable name dynamically
    exec(f'model_symbols{ord(char)} = model_temp')
    K.clear_session()
    
    #model_symbols_letters=tf.keras.models.load_model('C:\\Users\\Administrator\\OCR\\Final\\02-Classifiers\\\models_mixed\\model_0symbol_1letter.h5')
    model_symbols_letters_path = os.path.join(mixed_models_path, model_symbols_letters_name)  
    model_symbols_letters=tf.keras.models.load_model(model_symbols_letters_path)         

#0 takes 0 ,oO takes 1
#model_0_oO =tf.keras.models.load_model('C:\\Users\\Administrator\\OCR\\Final\\02-Classifiers\\\models_mixed\\model_0_oO.h5') 
model_0_oO_path = os.path.join(mixed_models_path, model_0_oO_name)
model_0_oO =tf.keras.models.load_model(model_0_oO_path)             

#threshold for _,-
from pandas import DataFrame
pd_=DataFrame(np.arange(0,32,1))
Q1_ = pd_.quantile(0.25)
Q3_ = pd_.quantile(0.75)
IQR_ = Q3_ - Q1_

############################################################################################################################################################

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 120, 300)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


# In[6]:


#eval(model_letters_48)


# In[7]:


def output(resized, resized_normheight, dictionar, x_test, Q1_, Q3_):

    characters_all =  list(string.printable)[:-6] #+['ç']# <

    dictionar_symbols['1']=None
    dictionar_symbols['0']=None
    dictionar_letters['1']=None
    dictionar_letters['0']=None
    
    # predict se é simbolo ou letra
    forecast_symbol_letter=model_symbols_letters.predict(x_test.astype(np.float32)).argmax()
    print('forecast_symbol_letter', forecast_symbol_letter)
    
    if forecast_symbol_letter==1:
       
        for char in characters_all[0:62]:
            #O QUE SAO MODEL_LETTERS??
            #print('ord(char)', ord(char))
            test = eval('model_letters' + str(ord(char)))
            #print('test', test)
            #print('x_test.astype(np.float32)', x_test.astype(np.float32))
            modeltemp=eval('model_letters' + str(ord(char))).predict(x_test.astype(np.float32))
            dictionar_letters['0'][dictionar_letters['Actual_num']==ord(char)]=modeltemp[0][0]
            dictionar_letters['1'][dictionar_letters['Actual_num']==ord(char)]=modeltemp[0][1]

        forecast = dictionar_letters['Actual_char'][dictionar_letters['0']==dictionar_letters['0'].min()].values[0]
        print('forecast', forecast)
        dictionartemp=dictionar_letters
        if forecast in ['0','o','O']:
            
            forecast_0_oO=model_0_oO.predict(x_test.astype(np.float32)).argmax()
            if forecast_0_oO=='0':
                forecast='0'
            else:
                forecast,dictionartemp=mixed_o_O(dictionar_letters,x_test)
        if forecast in ['r','k','K']:
            forecast,dictionartemp=mixed_4(dictionar_letters,x_test)
            if forecast == 'k': 
                forecast,dictionartemp=mixed_k_K(dictionar_letters,x_test)

        if forecast in ['m','n']:
            forecast,dictionartemp=mixed_m_n(dictionar_letters,x_test)

        if forecast in ['y','Y']:
            forecast,dictionartemp=mixed_y_Y(dictionar_letters,x_test)

        if forecast in ['x','X']:
            forecast,dictionartemp=mixed_x_X(dictionar_letters,x_test)

        if forecast in ['s','S']:
            forecast,dictionartemp=mixed_s_S(dictionar_letters,x_test)

        if forecast in ['c','C']:
            forecast,dictionartemp=mixed_c_C(dictionar_letters,x_test)

        if forecast in ['v','V']:
            forecast,dictionartemp=mixed_v_V(dictionar_letters,x_test)

        if forecast in ['w','W']:
            forecast,dictionartemp=mixed_w_W(dictionar_letters,x_test)

        if forecast in ['z','Z']:
            forecast,dictionartemp=mixed_z_Z(dictionar_letters,x_test)

        print('Forecast_letters: ',forecast)
        # print('Dictionar_letters',dictionartemp.sort_values('0'))
        print('...........................................................')
        K.clear_session()
    
    else:
        
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(resized,kernel,iterations = 1)
        
        x_test_right2 = np.expand_dims(erosion, axis=-1)
        x_test2 = np.expand_dims(x_test_right2, axis=0)
        
        for char in characters_all[62:94]:
            modeltemp=eval('model_symbols' + str(ord(char))).predict(x_test2.astype(np.float32))
            dictionar_symbols['0'][dictionar_symbols['Actual_num']==ord(char)]=modeltemp[0][0]
            dictionar_symbols['1'][dictionar_symbols['Actual_num']==ord(char)]=modeltemp[0][1]

        forecast=dictionar_symbols['Actual_char'][dictionar_symbols['0']==dictionar_symbols['0'].min()].values[0] 
        dictionartemp=dictionar_symbols
        
        if forecast in [':','%']:
            forecast,dictionartemp=mixed_1(dictionar_symbols,x_test2)
            
        if forecast in ['-','_','=']:
            forecast,dictionartemp=mixed_2(dictionar_symbols,x_test2) 
            
            if forecast in ['-','_']:
                
                contours, hierarchy = cv2.findContours(resized_normheight.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                for c in sorted_ctrs: 
                    x,y,w,h = cv2.boundingRect(c) 
                    outlier=(y < (Q1_)) |(y > (Q3_))
                    if outlier.values[0]==False: 
                        forecast='-'
                    else: 
                        forecast='_'
                print('outlier detection')
                
        if forecast in [".", ',',"'",'`']:
            #forecast,dictionartemp=mixed_3(dictionar_symbols,x_test2) 
            
            contours, hierarchy = cv2.findContours(resized_normheight.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
            for c in sorted_ctrs: 
                x,y,w,h = cv2.boundingRect(c) 
                outlier=(y < round((Q3_+Q1_)/2))
                if outlier.values[0]==True: 
                    forecast=forecast,dictionartemp=mixed_5(dictionar_symbols,x_test2) 
                else: 
                    forecast=forecast,dictionartemp=mixed_6(dictionar_symbols,x_test2) 
                        
                print('outlier detection')
            
        print('Forecast_symbols: ',forecast)
        #print('Dictionar_symbols',dictionartemp.sort_values('0'))
        print('...........................................................')
        K.clear_session()
        
    return forecast,dictionartemp


# In[8]:


def char_detection(new_img,new_img_normheight,dictionar,Q1_,Q3_): #detect char 
    
    def image_char_prepr(img,dis_bound,blur,gray_thres):
    
        import cv2
        import numpy as np
        import warnings
        warnings.simplefilter('ignore')

        gray_char = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_thres>0:
        #https://stackoverflow.com/questions/49907382/how-to-remove-whitespace-from-an-image-in-opencv
            gray = 255*(gray_char < gray_thres).astype(np.uint8) 
            coords = cv2.findNonZero(gray) # Find all non-zero points (text)
            x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
            gray_char = gray_char[y:y+h, x:x+w] # Crop the image - note we do this on the original image

        blur_char = cv2.bilateralFilter(gray_char, blur,blur, blur) #90, 90, 90
        thresh_char = cv2.adaptiveThreshold(blur_char, 255, cv2.ADAPTIVE_THRESH_MEAN_C , 
                                           cv2.THRESH_BINARY_INV, 23, 19)
        constant= cv2.copyMakeBorder(thresh_char.copy(),dis_bound,dis_bound,dis_bound,dis_bound,cv2.BORDER_CONSTANT)

        resized=cv2.resize(constant,(32,32), interpolation = cv2.INTER_NEAREST)
        return resized

    
    import cv2 
    import  numpy as np
    resized=image_char_prepr(new_img,2,20,0)
    resized_normheight=image_char_prepr(new_img_normheight,2,20,0)

    new_image_density=resized.sum()

    x_test_right = np.expand_dims(resized, axis=-1)
    x_test = np.expand_dims(x_test_right, axis=0)
    forecast,dictionartemp = output(resized,resized_normheight,dictionar,x_test,Q1_,Q3_)
    text_char =str(forecast)
    return text_char   


# In[9]:


from functions_score import *
from functions_char_preparation import *


# In[10]:


'''
caracteres com problemas
132
140
147
186
194
233
241
280
288
'''
unhandledChars = [140, 147, 186, 194, 233, 241, 280, 288]


# In[11]:


dictionar


# In[12]:


test_char_list = range(1, 560)
for test_char in test_char_list:
    if not test_char  in unhandledChars:
        test_char = str(test_char)
        print('test_char', test_char)
        test_char_image = os.path.join(test_images_folder, character_to_test.format(test_char))
        print('test_char_image', test_char_image)
        img = cv2.imread(test_char_image)
        #print('img', img)
        #img = cv2.imread('C:\\Users\\Administrator\\OCR\\Final\\04-Recognition\\characters\\2.png')

        sorted_ctrs = char_preprocessing_step_1(img)
        #print('sorted_ctrs', sorted_ctrs)
        new_sorted_ctrs = char_preprocessing_step_2(sorted_ctrs,img)
        print('new_sorted_ctrs', new_sorted_ctrs)
        temp_max_yh,temp_min_y = char_preprocessing_step_3(new_sorted_ctrs) #normalize height
        print('temp_max_yh', temp_max_yh)
        print('temp_min_y', temp_min_y)
        char_image,char_image_nh = char_preprocessing_step_4(img,new_sorted_ctrs,temp_max_yh,temp_min_y)

        #print('Q1_', Q1_)
        #print('Q3_', Q3_)
        #print('char_image[0]', char_image)
        #print('char_image_nh[0]', char_image_nh)
        text_char = char_detection(char_image[0],char_image_nh[0],dictionar,Q1_,Q3_)
        #print('text_char', text_char)


# In[ ]:




