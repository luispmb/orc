#!/usr/bin/env python
# coding: utf-8

# In[1]:

def origin_img_resize(img):
    from cv2 import cv2
    basewidth = int(img.shape[1]*3)
    wpercent = (basewidth / float(img.shape[1]))
    hsize = int((float(img.shape[0]) * float(wpercent)))
    img_res=cv2.resize(img,( basewidth,hsize), interpolation = cv2.INTER_BITS)
    return img_res
    

    
def line_preprocessing(img_res, AD1,AD2,AD3,AD4,b,BLURSTEPS,ADSTEPS,del1,del2,type_pr):
    from cv2 import cv2
    import  numpy as np
    #####to gray
    if type_pr=='one':
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        #######blur
        blur = cv2.bilateralFilter(gray, b, b, b) 
        if BLURSTEPS >0 :
            for i in range(0,BLURSTEPS):
                blur = cv2.bilateralFilter(blur, b, b, b) 

        ######thresholding
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, AD1, AD2)
        if ADSTEPS>0:
            for i in range(0,2):
                thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, AD3, AD4)
        #dilation
        if del1>0:
            kernel = np.ones((del1,del2), np.uint8) #can be from here depends the word seperation
            thresh = cv2.dilate(thresh, kernel, iterations=1)
    elif type_pr=='two':
        gray = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)

        ######thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, AD1, AD2)
        if ADSTEPS>0:
            for i in range(0,2):
                thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, AD3, AD4)
                
        #######blur
        blur = cv2.bilateralFilter(thresh, b, b, b) 
        if BLURSTEPS >0 :
            for i in range(0,BLURSTEPS):
                blur = cv2.bilateralFilter(blur, b, b, b) 

        #dilation
        if del1>0:
            kernel = np.ones((del1,del2), np.uint8) #can be from here depends the word seperation
            thresh = cv2.dilate(blur, kernel, iterations=1)

    return thresh,img_res
    

def line_preprocessing_advanced(img_res): #to delete vertical lines
    from cv2 import cv2
    gray_1 = cv2.cvtColor(img_res, cv2.COLOR_BGR2GRAY)
    
    #######blur
    b=30
    blur_1 = cv2.bilateralFilter(gray_1, b, b, b) #100,100,100
    for i in range(0,4):
         blur_1 = cv2.bilateralFilter(blur_1, b, b, b) 

    ######thresholding
    thresh_1 = cv2.adaptiveThreshold(blur_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, 11, 9)
    for i in range(0,2):
        thresh_1 = cv2.adaptiveThreshold(thresh_1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, 11, 9)

    #dilation
    import numpy as np
    kernel = np.ones((10,50), np.uint8)
    img_dilation = cv2.dilate(thresh_1, kernel, iterations=1)
    image_erode = cv2.erode(img_dilation, kernel, iterations=1)

    #vertical lines remove
    import math
    horizontal = np.copy(image_erode)
    cols = horizontal.shape[1] # Specify size on horizontal axis
    horizontal_size = math.ceil(cols / 10)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure )# Apply morphology operations
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    #find the text positionin the resized original image and crop it
    contours, hierarchy = cv2.findContours(horizontal.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #https://answers.opencv.org/question/179510/how-can-i-sort-the-contours-from-left-to-right-and-top-to-bottom/
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
    x_min=10000000
    wx_max=-1
    y_min=10000000
    yh_max=-1
    for c in sorted_ctrs: 
        x,y,w,h = cv2.boundingRect(c) 
        xw=x+w
        hy=y+h
        x_min=min(x,x_min)
        wx_max=max(xw,wx_max)
        y_min=min(y,y_min)
        yh_max=max(hy,yh_max)

    y_dif=abs(y_min-0)
    if y_dif >20:

        text_image=img_res[y_min-20:yh_max+20,x_min:wx_max] 
    else:
        text_image=img_res[y_min-y_dif:yh_max+20,x_min:wx_max] 
    #########################################################################
    #prepare new croped image for line detection
    #####to gray
    gray = cv2.cvtColor(text_image, cv2.COLOR_BGR2GRAY)

    #######blur
    b=30
    blur = cv2.bilateralFilter(gray, b, b, b) #100,100,100

    ######thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, 7, 5)
    for i in range(0,1):
        thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV,7, 5)
        
    return thresh,text_image




def line_detector(img_score,img_res,path_lines,idl):
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<line_detector>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<')
    import cv2
    pixels_sum=[]
    pixels=[]
    pixels_fin=[]
    line_images=[]
    for row in img_score:
        row_sum=row.sum() #sum every white pixel  of each row
        pixels_sum.append(row_sum)

    x_start=0
    x_end=img_score.shape[1]
    y_start=-1
    y_end=-1
    new_temp_start=False
    
    
    for row in range(0,len(pixels_sum)):
        if new_temp_start!= True:

            if pixels_sum[row]>0:
                new_temp_start=True
                y_start=row
        elif ((new_temp_start== True) and (pixels_sum[row]>255)) and ((img_score.shape[0]-1)>row):
            y_end=row

        else :  #this part help to connect _ with F and exclude redundant rows
            new_temp_start=False

            slices_y_start=int(abs(y_start-y_end)*0.40) #int(img_res.shape[0] *0.005)
            slices_y_end=int(abs(y_start-y_end)*0.35) #int(img_res.shape[0] *0.09)

            if (y_end-y_start) > 1: #slices if I want to filter

                if  0 > y_start-slices_y_start:
                    while 0 > y_start-slices_y_start:
                        slices_y_start=slices_y_start-1
                if y_end+slices_y_end > img_score.shape[0]:
                    while y_end+slices_y_end > img_score.shape[0]:
                        slices_y_end=slices_y_end-1
                    pixels.append([y_start-slices_y_start, y_end+slices_y_end])   
                else:
                    pixels.append([y_start-slices_y_start, y_end+slices_y_end])
    if y_start>-1:
        if len(pixels)<1: #image to short to be something
            pixels_fin.append([0, 0]) 
            line_images.append(img_res)
            cv2.imwrite(path_lines+str(idl) + '.png', img_res) 
            idl=idl+1
        else:

            to_drop=[]
            for i in range(1,len(pixels)) :
                try:
                    if 2> pixels[i][1]-pixels[i][0] : #point whene can fail if image is small
                        pixels[i-1][1]=pixels[i][1]
                        to_drop.append(i)
                except:
                    print('rows decreased')

            to_drop.sort(reverse = True)
            for i in to_drop:
                pixels.remove(pixels[i])

            for i in range(len(pixels)): 
                if pixels[i][0]>-1 and pixels[i][1]>pixels[i][0]:
                    new_img=img_res[pixels[i][0]:pixels[i][1],x_start:x_end] 
                    line_images.append(new_img)
                    # print('gonna add pixels to line_pixel ->', pixels[i])
                    pixels_fin.append(pixels[i])
                    cv2.imwrite(path_lines+str(idl) + '.png', new_img) 

                    idl=idl+1
    else:
        pixels_fin.append([0, 0])   #the image is black or white. No letters
        line_images.append(img_res)
        cv2.imwrite(path_lines+str(idl) + '.png', img_res) 
        idl=idl+1
        
    return line_images,pixels_fin,idl




def tesseract_recognition(line_image,path_to_my_tesseract,speller,blur):
    import pytesseract
    from cv2 import cv2
    import docx
    from autocorrect import Speller
    from langdetect import detect_langs
    from langdetect import detect
    import re
    import os
    
    #if "RUNNING_LOCALLY" in os.environ.keys() and os.environ["RUNNING_LOCALLY"] == "True":
    if "PLATFORM" in os.environ.keys() and os.environ["PLATFORM"] == "WINDOWS":
        from config.tesseract import path_to_my_tesseract
        pytesseract.pytesseract.tesseract_cmd = path_to_my_tesseract
    
    #custom_config = r'--oem 3 --psm 6'
    custom_config = r'-l por+eng --psm 6'
    
    line_res=origin_img_resize(line_image)
    gray = cv2.cvtColor(line_res, cv2.COLOR_BGR2GRAY)
    blur_char = cv2.bilateralFilter(gray, blur,blur, blur) #90, 90, 90
    
    text_tesseract = pytesseract.image_to_string(blur_char, config=custom_config) #, lang='por'
    combined_pat = r'|'.join(("\x0c","\n\x0c ","\n\x0c"))
    text_char_tesseract_clean=re.sub(combined_pat, '', str(text_tesseract))
    print(text_char_tesseract_clean)
    try:
        if speller == True and len(text_char_tesseract_clean)>0:
            lng=str(detect_langs(text_tesseract)[0]).split(":", 1)
            if lng[0]=='pt' and float(lng[1]) >0.98:
                spell = Speller(lang='pt')
                text_char_tesseract_clean=spell(text_char_tesseract_clean)
                print('spelled in pt:',text_char_tesseract_clean)
            elif  lng[0]=='en' and float(lng[1]) >0.98:
                spell = Speller(lang='en')
                text_char_tesseract_clean=spell(text_char_tesseract_clean)
                print('spelled in en:',text_char_tesseract_clean)
            else: None
    except: None
    
    return text_char_tesseract_clean


