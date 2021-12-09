#!/usr/bin/env python
# coding: utf-8

# In[1]:
def create_path(path):
    import os
    import shutil
    if not os.path.exists(path):
        print('PATH', path)
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)
    return None
        
def preprocess_orig_img(img1):
    from cv2 import cv2
    import numpy as np
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, 3, 1)


    return thresh,gray

def preprocess_tesseract_img(imgt,blur,thb,ths):
    from cv2 import cv2
    import numpy as np
    gray = cv2.cvtColor(imgt.copy(), cv2.COLOR_BGR2GRAY)
    blur_char = cv2.bilateralFilter(gray, blur,blur, blur) #90, 90, 90
    thresh = cv2.adaptiveThreshold(blur_char, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C  , cv2.THRESH_BINARY_INV, thb, ths)
    return thresh



def get_horizodal_lines(img2,size): #thresh, size of line
    import numpy as np
    from cv2 import cv2
    horizontal = img2.copy()
    rows,cols = horizontal.shape
    horizontalsize = int(cols / size)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize,1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    return horizontal


def get_vertical_lines(img3,size): #thresh, size of line
    import numpy as np
    from cv2 import cv2
    vertical = img3.copy()
    rows,cols = vertical.shape
    verticalsize = int(rows / 50)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    return vertical



def mask_morph(img4_src, img5_out):
    from cv2 import cv2
    import  numpy as np
    #inverse the image, so that lines are black for masking
    horizontal_inv = cv2.bitwise_not(img4_src)
    #perform bitwise_and to mask the lines with provided mask
    masked_img = cv2.bitwise_and(img5_out, img5_out, mask=horizontal_inv)
    #reverse the image back to normal
    #masked_img_inv = cv2.bitwise_not(masked_img)
    return masked_img

def expot(img8):
    multi= img8.shape[0]* img8.shape[1]
    
    if multi < 40000:
        par= 0.04
        
    elif 39999 <multi < 100000:
        par= 0.02
        
    elif 99999 < multi < 200000:
        par= 0.01  
        
    elif 199999 <multi < 800000: 
        par= 0.006
        
    else: 
        par= 0.002 #0.001
        
    return par


def wide_white_contours(img6,iss,js): #increase the space of the white pixels to boost lines
    iss=int(iss/2)
    for i in range(iss,(img6.shape[0]-iss)):
        for j in range (js,(img6.shape[1]-int(2.5*js))):
            if img6[i][j]>250:
                for kj in range (j-js,j+int(2.5*js)):
                    for  ki in range(i-1, i+1):
                        img6[ki][kj]=250   #ki   
    return img6



def remove_noise(img7):
    from cv2 import cv2
    contours, hierarchy = cv2.findContours(img7,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 10 and h < 10:
            z=0
            img7[y-z:y+h+z,x-z:x+w+z]= 0
        if  h < 6 or w < 14 : #12
            z=0
            img7[y-z:y+h+z,x-z:x+w+z]= 0
    return img7


def extract_contoured_image_1(img9, img10,zy,zx,ps, pb):# source, noised
    from cv2 import cv2
    import numpy as np
    import math
    img10v2=img10.copy()
    img10v3=img10.copy()
    contours, hierarchy = cv2.findContours(img10.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for c in sorted_ctrs: 
            x,y,w,h = cv2.boundingRect(c) 
            z=0
            img10[y-zy:y+h+zy,x-zx:x+w+zx]= 255 
            
    masked_img = cv2.bitwise_or(img9,img9,mask=img10)
    masked_img_weighted_big = cv2.addWeighted(img9,ps,masked_img,pb,0)
    #########################################
    contoursb, hierarchyb = cv2.findContours(img10.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrsb = sorted(contoursb, key=lambda ctr: cv2.boundingRect(ctr)[0])
   
    for c in sorted_ctrsb: 
            x,y,w,h = cv2.boundingRect(c) 
       
            if 50 < int((w-h)*100/h):#w-h apo 7 ews 9
                img10v2[y-z:y+h+z,x-z:x+w+z]= 255
                img10v3[y-z:y+h+z,x-z:x+w+z]= 0
            else:
                img10v3[y-z:y+h+z,x-z:x+w+z]= 255 
                img10v2[y-z:y+h+z,x-z:x+w+z]= 0
                
    masked_imgf = cv2.bitwise_or(img9,img9,mask=img10v2)
    masked_img_weighted_bigf = cv2.addWeighted(img9,ps,masked_imgf,pb,0)
    
    masked_imgfs= cv2.bitwise_or(img9,img9,mask=img10v3)
    masked_img_weighted_bigfs = cv2.addWeighted(img9,ps,masked_imgfs,pb,0)

   
    return  masked_img, masked_img_weighted_big,masked_imgf,masked_img_weighted_bigf,masked_imgfs,masked_img_weighted_bigfs,img10v2



def extract_contoured_image_2(img12,img11,zy,zx,ps, pb):# source, masked_imgf    
    
    from cv2 import cv2
    import numpy as np
    
    z=0
    contours, hierarchy = cv2.findContours(img11.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    thresh,gray=preprocess_orig_img(img12)
    for c in sorted_ctrs: 
            x,y,w,h = cv2.boundingRect(c)
            threshcut=thresh[y-z:y+h+z,x-z:x+w+z]
            list_white=[]
            list_black=[]
            for i in range(0,int(threshcut.shape[0])):
                numBlackPixel=None
                numWhitePixel=None
                numBlackPixel = len(np.extract(threshcut[i]==0, threshcut[i]))
                numWhitePixel = len(np.extract(threshcut[i]==255, threshcut[i]))
                
                list_white.append(numWhitePixel)
                list_black.append(numBlackPixel)
            numBlackPixel=sum(list_black)
            numWhitePixel=sum(list_white)
       
            if numBlackPixel==0 or numBlackPixel==255:
                 img11[y-z:y+h+z,x-z:x+w+z]= 0
           
           
            masked_imgf2 = cv2.bitwise_or(img12,img12,mask=img11)
            masked_img_weighted_bigf2 = cv2.addWeighted(img12,ps,masked_imgf2,pb,0)
    return  masked_imgf2,masked_img_weighted_bigf2,img11


def extract_object(img14,img13,p,path_object):
    from cv2 import cv2
    obj={}
    obj['idi'] = p
    lines=[]
    coordinates=[]
    coordinates_line=[]
    text=[]
    sub_coordinates=[]
    contours, hierarchy = cv2.findContours(img13.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    thresh=preprocess_tesseract_img(img14.copy(),0,11,9)
    idxc=0
    print(thresh.shape)
    for c in sorted_ctrs: 
        x,y,w,h = cv2.boundingRect(c)
        zy=3
        zx=3
        while 0>(y-zy) or (y+h+zy)>thresh.shape[0]:
            zy-=1
            
        while 0>(x-zx) or (x+w+zx)>thresh.shape[1]:
            zx-=1
            
        line_coordinates = [y-zy, y+h+zy, x-zx, x+w+zx]
        line = img14[line_coordinates[0]:line_coordinates[1],line_coordinates[2]:line_coordinates[3]]
        #line=img14[y-zy:y+h+zy,x-zx:x+w+zx]
        cv2.imwrite(path_object+str(idxc) + '.png', line)
        idxc+=1
        lines.append(line)
        coordinates.append([x,y,w,h])
        coordinates_line.append(line_coordinates)
        text.append('')
        sub_coordinates.append([0,0])

    obj['coordinates']=coordinates
    obj['coordinates_line']=coordinates_line
    obj['line_images']=lines
    obj['ocr_text']=text
    obj['sub_coordinates']=sub_coordinates
    return lines,obj




def write_images_pre(masked_imgf2,masked_img_weighted_bigf2,p):
    from cv2 import cv2
    
    cv2.imwrite('.\\app_output\\filtered\\output_all\\'+p+'.png', masked_imgf2)
    cv2.imwrite('.\\app_output\\filtered\\output_all_br\\'+p+'.png', masked_img_weighted_bigf2)

    #return print('written done')
    return




