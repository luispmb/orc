#!/usr/bin/env python
# coding: utf-8

# In[1]:

#inputs
import os
parent_directory = os.path.abspath('')
root = os.path.abspath(os.path.join(parent_directory, os.pardir))
classifiers_folder = '02-Classifiers'
gan_char_models_folder = 'models_char_gan'
model_digits_letters_name = 'class_char_model_{}.h5'
mixed_models_folder = 'models_mixed'
model_symbols_letters_name = 'model_0symbol_1letter.h5'
model_0_oO_name = 'model_0_oO.h5'
mixed_models_path = os.path.join(root, classifiers_folder, mixed_models_folder)




import matplotlib.pyplot as plt
import cv2
import numpy as np
import warnings
warnings.simplefilter('ignore')


     
def mixed_o_O(dictionar,x_test): #['o','O']
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_o_O=dictionar[(dictionar['Actual_num']==111)|(dictionar['Actual_num']==79)]
    #model_o_O=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_o_O.h5')
    model_filename = os.path.join(mixed_models_path, 'model_o_O.h5')
    print('model_filename', model_filename)
    model_o_O=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_o_O['o']=None
    dictionar_o_O['O']=None
    modeltemp=model_o_O.predict(x_test.astype(np.float32))
    
    dictionar_o_O['o'].iloc[0]=modeltemp[0][0]
    dictionar_o_O['O'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_o_O['Actual_char'].iloc[pd.to_numeric(dictionar_o_O[["o", 'O']].max()).argmax()]
        
    return forecast,dictionar_o_O

def mixed_m_n(dictionar,x_test):
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_m_n=dictionar[(dictionar['Actual_num']==109)|(dictionar['Actual_num']==110)]
    #model_m_n=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_m_n.h5')
    model_filename = os.path.join(mixed_models_path, 'model_m_n.h5')
    print('model_filename', model_filename)
    model_m_n=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_m_n['m']=None
    dictionar_m_n['n']=None
    modeltemp=model_m_n.predict(x_test.astype(np.float32))
    
    dictionar_m_n['m'].iloc[0]=modeltemp[0][0]
    dictionar_m_n['n'].iloc[1]=modeltemp[0][1]

    print(dictionar_m_n)

    #print(dictionar_m_n['Actual_char'])

    forecast=dictionar_m_n['Actual_char'].iloc[pd.to_numeric(dictionar_m_n[["m", 'n']].max()).argmax()]
      
    return forecast,dictionar_m_n

    
def mixed_y_Y(dictionar,x_test):
    
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_y_Y=dictionar[(dictionar['Actual_num']==121)|(dictionar['Actual_num']==89)]
    #model_y_Y=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_y_Y.h5')
    model_filename = os.path.join(mixed_models_path, 'model_y_Y.h5')
    print('model_filename', model_filename)
    model_y_Y=tf.keras.models.load_model(model_filename)

    K.clear_session() 
    
    dictionar_y_Y['y']=None
    dictionar_y_Y['Y']=None
    modeltemp=model_y_Y.predict(x_test.astype(np.float32))
    
    dictionar_y_Y['y'].iloc[0]=modeltemp[0][0]
    dictionar_y_Y['Y'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_y_Y['Actual_char'].iloc[pd.to_numeric(dictionar_y_Y[["y", 'Y']].max()).argmax()]
        
    return forecast,dictionar_y_Y



def mixed_x_X(dictionar,x_test):
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_x_X=dictionar[(dictionar['Actual_num']==120)|(dictionar['Actual_num']==88)]
    #model_x_X=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_x_X.h5')
    model_filename = os.path.join(mixed_models_path, 'model_x_X.h5')
    print('model_filename', model_filename)
    model_x_X=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_x_X['x']=None
    dictionar_x_X['X']=None
    modeltemp=model_x_X.predict(x_test.astype(np.float32))
    
    dictionar_x_X['x'].iloc[0]=modeltemp[0][0]
    dictionar_x_X['X'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_x_X['Actual_char'].iloc[pd.to_numeric(dictionar_x_X[["x", 'X']].max()).argmax()]
        
    return forecast,dictionar_x_X

def mixed_s_S(dictionar,x_test):
    
    
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_s_S=dictionar[(dictionar['Actual_num']==115)|(dictionar['Actual_num']==83)]
    #model_s_S=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_s_S.h5')
    model_filename = os.path.join(mixed_models_path, 'model_s_S.h5')
    print('model_filename', model_filename)
    model_s_S=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_s_S['s']=None
    dictionar_s_S['S']=None
    modeltemp=model_s_S.predict(x_test.astype(np.float32))
    
    dictionar_s_S['s'].iloc[0]=modeltemp[0][0]
    dictionar_s_S['S'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_s_S['Actual_char'].iloc[pd.to_numeric(dictionar_s_S[["s", 'S']].max()).argmax()]
        
    return forecast,dictionar_s_S


def mixed_c_C(dictionar,x_test):
    
    
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_c_C=dictionar[(dictionar['Actual_num']==99)|(dictionar['Actual_num']==67)]
    #model_c_C=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_c_C.h5')
    model_filename = os.path.join(mixed_models_path, 'model_c_C.h5')
    print('model_filename', model_filename)
    model_c_C=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_c_C['c']=None
    dictionar_c_C['C']=None
    modeltemp=model_c_C.predict(x_test.astype(np.float32))
    
    dictionar_c_C['c'].iloc[0]=modeltemp[0][0]
    dictionar_c_C['C'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_c_C['Actual_char'].iloc[pd.to_numeric(dictionar_c_C[["c", 'C']].max()).argmax()]
        
    return forecast,dictionar_c_C


def mixed_k_K(dictionar,x_test):
    
    
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_k_K=dictionar[(dictionar['Actual_num']==107)|(dictionar['Actual_num']==75)]
    #model_k_K=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_k_K.h5')
    model_filename = os.path.join(mixed_models_path, 'model_k_K.h5')
    print('model_filename', model_filename)
    model_k_K=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_k_K['k']=None
    dictionar_k_K['K']=None
    modeltemp=model_k_K.predict(x_test.astype(np.float32))
    
    dictionar_k_K['k'].iloc[0]=modeltemp[0][0]
    dictionar_k_K['K'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_k_K['Actual_char'].iloc[pd.to_numeric(dictionar_k_K[["k", 'K']].max()).argmax()]
        
    return forecast,dictionar_k_K



def mixed_v_V(dictionar,x_test):
  
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    
    dictionar_v_V=dictionar[(dictionar['Actual_num']==118)|(dictionar['Actual_num']==86)]
    #model_v_V=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_v_V.h5')
    model_filename = os.path.join(mixed_models_path, 'model_v_V.h5')
    print('model_filename', model_filename)
    model_v_V=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_v_V['v']=None
    dictionar_v_V['V']=None
    modeltemp=model_v_V.predict(x_test.astype(np.float32))
    
    dictionar_v_V['v'].iloc[0]=modeltemp[0][0]
    dictionar_v_V['V'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_v_V['Actual_char'].iloc[pd.to_numeric(dictionar_v_V[["v", 'V']].max()).argmax()]
        
    return forecast,dictionar_v_V


def mixed_w_W(dictionar,x_test):
    
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_w_W=dictionar[(dictionar['Actual_num']==99)|(dictionar['Actual_num']==67)]
    #model_w_W=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Char-Classification\\models_mixed\\model_w_W.h5')
    model_filename = os.path.join(mixed_models_path, 'model_w_W.h5')
    print('model_filename', model_filename)
    model_w_W=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_w_W['w']=None
    dictionar_w_W['W']=None
    modeltemp=model_w_W.predict(x_test.astype(np.float32))
    
    dictionar_w_W['w'].iloc[0]=modeltemp[0][0]
    dictionar_w_W['W'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_w_W['Actual_char'].iloc[pd.to_numeric(dictionar_w_W[["w", 'W']].max()).argmax()]
        
    return forecast,dictionar_w_W



def mixed_z_Z(dictionar,x_test):
    
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_z_Z=dictionar[(dictionar['Actual_num']==122)|(dictionar['Actual_num']==90)]
    #model_z_Z=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\model_z_Z.h5')
    model_filename = os.path.join(mixed_models_path, 'model_z_Z.h5')
    print('model_filename', model_filename)
    model_z_Z=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_z_Z['z']=None
    dictionar_z_Z['Z']=None
    modeltemp=model_z_Z.predict(x_test.astype(np.float32))
    
    dictionar_z_Z['z'].iloc[0]=modeltemp[0][0]
    dictionar_z_Z['Z'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_z_Z['Actual_char'].iloc[pd.to_numeric(dictionar_z_Z[["z", 'Z']].max()).argmax()]
        
    return forecast,dictionar_z_Z

def mixed_1(dictionar,x_test): #   :,% model_
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_1=dictionar[(dictionar['Actual_num']==37)|(dictionar['Actual_num']==58)]
    #model_1=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\mixed_1.h5')
    model_filename = os.path.join(mixed_models_path, 'mixed_1.h5')
    print('model_filename', model_filename)
    model_1=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_1[':']=None
    dictionar_1['%']=None
    modeltemp=model_1.predict(x_test.astype(np.float32))
    
    dictionar_1[':'].iloc[0]=modeltemp[0][1]
    dictionar_1['%'].iloc[1]=modeltemp[0][0]

    forecast=dictionar_1['Actual_char'].iloc[pd.to_numeric(dictionar_1[[":", '%']].max()).argmax()]
        
    return forecast,dictionar_1


def mixed_2(dictionar,x_test): #=['-','_','=']
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_2=dictionar[(dictionar['Actual_num']==45)|(dictionar['Actual_num']==61)]
    #model_2=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\odels_mixed\\mixed_2.h5')
    model_filename = os.path.join(mixed_models_path, 'mixed_2.h5')
    print('model_filename', model_filename)
    model_2=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_2['-']=None
    dictionar_2['=']=None
    modeltemp=model_2.predict(x_test.astype(np.float32))
    
    dictionar_2['-'].iloc[0]=modeltemp[0][0]
    dictionar_2['='].iloc[1]=modeltemp[0][1]

    forecast=dictionar_2['Actual_char'].iloc[pd.to_numeric(dictionar_2[["-",'=']].max()).argmax()]
        
    return forecast,dictionar_2


def mixed_3(dictionar,x_test): #['.',',',"'",'`']
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_3=dictionar[(dictionar['Actual_num']==46)|(dictionar['Actual_num']==44)|(dictionar['Actual_num']==39)|(dictionar['Actual_num']==96)]
    #model_3=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\mixed_3.h5')
    model_filename = os.path.join(mixed_models_path, 'mixed_3.h5')
    print('model_filename', model_filename)
    model_3=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_3['.']=None
    dictionar_3[',']=None
    dictionar_3["'"]=None
    dictionar_3['`']=None
    modeltemp=model_3.predict(x_test.astype(np.float32))
    
    dictionar_3['.'].iloc[0]=modeltemp[0][0]
    dictionar_3[','].iloc[1]=modeltemp[0][1]
    dictionar_3["'"].iloc[2]=modeltemp[0][2]
    dictionar_3['`'].iloc[2]=modeltemp[0][3]

    forecast=dictionar_3['Actual_char'].iloc[pd.to_numeric(dictionar_3[[".", ',',"'",'`']].max()).argmax()]
        
    return forecast,dictionar_3

def mixed_4(dictionar,x_test): 
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_4=dictionar[(dictionar['Actual_num']==107)|(dictionar['Actual_num']==114)]
    #model_4=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\mixed_4.h5')
    model_filename = os.path.join(mixed_models_path, 'mixed_4.h5')
    print('model_filename', model_filename)
    model_4=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_4['r']=None
    dictionar_4['k']=None

    modeltemp=model_4.predict(x_test.astype(np.float32))
    
    dictionar_4['r'].iloc[0]=modeltemp[0][1]#deksia
    dictionar_4['k'].iloc[1]=modeltemp[0][0]#aristera


    forecast=dictionar_4['Actual_char'].iloc[pd.to_numeric(dictionar_4[["r", 'k']].max()).argmax()]
        
    return forecast,dictionar_4


def mixed_5(dictionar,x_test): #['.',',']
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_5=dictionar[(dictionar['Actual_num']==46)|(dictionar['Actual_num']==44)]
    #model_5=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\mixed_5.h5')
    model_filename = os.path.join(mixed_models_path, 'mixed_5.h5')
    print('model_filename', model_filename)
    model_5=tf.keras.models.load_model(model_filename)
    K.clear_session() 
    
    dictionar_5['.']=None
    dictionar_5[',']=None

    modeltemp=model_5.predict(x_test.astype(np.float32))
    
    dictionar_5['.'].iloc[0]=modeltemp[0][0]
    dictionar_5[','].iloc[1]=modeltemp[0][1]

    forecast=dictionar_5['Actual_char'].iloc[pd.to_numeric(dictionar_5[[".", ',']].max()).argmax()]
        
    return forecast,dictionar_5



def mixed_6(dictionar,x_test): #['.',',',"'",'`']
    import tensorflow as tf
    import string
    import pandas as pd
    import keras.backend as K
    
    dictionar_6=dictionar[(dictionar['Actual_num']==39)|(dictionar['Actual_num']==96)]
    #model_6=tf.keras.models.load_model('C:\\Users\Administrator\\OCR\\Final\\02-Classifiers\\models_mixed\\mixed_6.h5')
    model_filename = os.path.join(mixed_models_path, 'mixed_6.h5')
    print('model_filename', model_filename)
    model_6=tf.keras.models.load_model(model_filename)
    K.clear_session() 

    dictionar_6["'"]=None
    dictionar_6['`']=None
    modeltemp=model_6.predict(x_test.astype(np.float32))
    
    dictionar_6["'"].iloc[0]=modeltemp[0][0]
    dictionar_6['`'].iloc[1]=modeltemp[0][1]

    forecast=dictionar_6['Actual_char'].iloc[pd.to_numeric(dictionar_6[["'",'`']].max()).argmax()]
        
    return forecast,dictionar_6

        

    return forecast,dctionartemp



