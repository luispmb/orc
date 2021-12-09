import numpy as np 
import cv2
import string

def classification(charModels, resized, resized_normheight, x_test, Q1_, Q3_, dictionar, dictionar_letters, dictionar_symbols):
    characters_all =  list(string.printable)[:-6] #+['ç']# <

    dictionar_symbols['1']=None
    dictionar_symbols['0']=None
    dictionar_letters['1']=None
    dictionar_letters['0']=None
    
    # predict se é simbolo ou letra
    forecast_symbol_letter=charModels.model_symbols_letters.predict(x_test.astype(np.float32)).argmax()
    print('forecast_symbol_letter', forecast_symbol_letter)
    
    if forecast_symbol_letter==1:
       
        for char in characters_all[0:62]:
            #O QUE SAO MODEL_LETTERS??
            #print('ord(char)', ord(char))
            
            #test = eval('model_letters' + str(ord(char)))
            
            #print('test', test)
            #print('x_test.astype(np.float32)', x_test.astype(np.float32))
            modeltemp=eval('charModels.model_letters' + str(ord(char))).predict(x_test.astype(np.float32))
            dictionar_letters['0'][dictionar_letters['Actual_num']==ord(char)]=modeltemp[0][0]
            dictionar_letters['1'][dictionar_letters['Actual_num']==ord(char)]=modeltemp[0][1]

        forecast = dictionar_letters['Actual_char'][dictionar_letters['0']==dictionar_letters['0'].min()].values[0]
        print('forecast', forecast)
        dictionartemp=dictionar_letters
        if forecast in ['0','o','O']:
            
            forecast_0_oO=charModels.model_0_oO.predict(x_test.astype(np.float32)).argmax()
            if forecast_0_oO=='0':
                forecast='0'
            else:
                forecast,dictionartemp=charModels.mixed_o_O(dictionar_letters,x_test)
        if forecast in ['r','k','K']:
            forecast,dictionartemp=charModels.mixed_4(dictionar_letters,x_test)
            if forecast == 'k': 
                forecast,dictionartemp=charModels.mixed_k_K(dictionar_letters,x_test)

        if forecast in ['m','n']:
            forecast,dictionartemp=charModels.mixed_m_n(dictionar_letters,x_test)

        if forecast in ['y','Y']:
            forecast,dictionartemp=charModels.mixed_y_Y(dictionar_letters,x_test)

        if forecast in ['x','X']:
            forecast,dictionartemp=charModels.mixed_x_X(dictionar_letters,x_test)

        if forecast in ['s','S']:
            forecast,dictionartemp=charModels.mixed_s_S(dictionar_letters,x_test)

        if forecast in ['c','C']:
            forecast,dictionartemp=charModels.mixed_c_C(dictionar_letters,x_test)

        if forecast in ['v','V']:
            forecast,dictionartemp=charModels.mixed_v_V(dictionar_letters,x_test)

        if forecast in ['w','W']:
            forecast,dictionartemp=charModels.mixed_w_W(dictionar_letters,x_test)

        if forecast in ['z','Z']:
            forecast,dictionartemp=charModels.mixed_z_Z(dictionar_letters,x_test)

        print('Forecast_letters: ',forecast)
        # print('Dictionar_letters',dictionartemp.sort_values('0'))
        print('...........................................................')
        #K.clear_session()
    
    else:
        
        kernel = np.ones((2,2),np.uint8)
        erosion = cv2.erode(resized,kernel,iterations = 1)
        
        x_test_right2 = np.expand_dims(erosion, axis=-1)
        x_test2 = np.expand_dims(x_test_right2, axis=0)
        
        for char in characters_all[62:94]:
            modeltemp=eval('charModels.model_symbols' + str(ord(char))).predict(x_test2.astype(np.float32))
            dictionar_symbols['0'][dictionar_symbols['Actual_num']==ord(char)]=modeltemp[0][0]
            dictionar_symbols['1'][dictionar_symbols['Actual_num']==ord(char)]=modeltemp[0][1]

        forecast=dictionar_symbols['Actual_char'][dictionar_symbols['0']==dictionar_symbols['0'].min()].values[0] 
        dictionartemp=dictionar_symbols
        
        if forecast in [':','%']:
            forecast,dictionartemp=charModels.mixed_1(dictionar_symbols,x_test2)
            
        if forecast in ['-','_','=']:
            forecast,dictionartemp=charModels.mixed_2(dictionar_symbols,x_test2) 
            
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
                    forecast=forecast,dictionartemp=charModels.mixed_5(dictionar_symbols,x_test2) 
                else: 
                    forecast=forecast,dictionartemp=charModels.mixed_6(dictionar_symbols,x_test2) 
                        
                print('outlier detection')
            
        print('Forecast_symbols: ',forecast)
        #print('Dictionar_symbols',dictionartemp.sort_values('0'))
        print('...........................................................')
        #K.clear_session()
        
    return forecast,dictionartemp


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