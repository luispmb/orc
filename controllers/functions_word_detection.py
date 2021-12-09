#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

# In[1]:

def IQR(size,Q1,Q3):
    from pandas import DataFrame
    import numpy as np
    pd_=DataFrame(np.arange(0,size,1))
    Q1_ = pd_.quantile(Q1)
    Q3_ = pd_.quantile(Q3)
    IQR_ = Q3_ - Q1_
    return IQR_, Q3_, Q1_

def count_white_pixels(thresh,minq,maxq,X_buckets_s,X_buckets_e):
    import numpy as np
    list_white=[]
    temp_image=thresh[int(minq):int(maxq),X_buckets_s:X_buckets_e]
    
    for i in range(0,temp_image.shape[0]):
        numWhitePixel=None
        numWhitePixel = sum(np.extract(temp_image[i]==255, temp_image[i]))
        list_white.append(numWhitePixel)
    numWhitePixeltotal=sum(list_white)   
    return numWhitePixeltotal

#check_for_complicated_lines(img,0.15,0.85,13,9,0,0,30,2,0,0,0,'one',10)
def check_for_complicated_lines(Q1,Q3,buckets,pr_function):
    import numpy as np
    from numpy import matrix
    from cv2 import cv2
    
    whitepixelq3=[]
    whitepixelq1=[]
    complicado=False
    thresh,img_res=pr_function
    IQR_, Q3_, Q1_= IQR(thresh.shape[0],Q1,Q3) 
    
    X_buckets_end=int((thresh.shape[1])/buckets) 
    X_buckets_start=0

    while thresh.shape[1]>X_buckets_end:
        whitepixelq3.append(count_white_pixels(thresh,Q3_.values[0],thresh.shape[0],X_buckets_start,X_buckets_end))
        whitepixelq1.append(count_white_pixels(thresh,0,Q1_.values[0],X_buckets_start,X_buckets_end))
  
        X_buckets_start=X_buckets_start+int(thresh.shape[1]/buckets)
        X_buckets_end=X_buckets_end+int(thresh.shape[1]/buckets)
    
    q3 = matrix(whitepixelq3)
    q1 = matrix(whitepixelq1)
    ret = abs(q3 - q1)
    maax=np.max(ret)
    
    if maax/sum(whitepixelq3) > 0.25 and maax >10000: #15000:
        complicado=True
    return complicado


def word_detection(idw,path_words,pr_function):
    import numpy as np
    from cv2 import cv2
   
    words_images=[]
    words_pixels=[]
    
    #line_preprocessing_function(img,13,9,0,0,30,2,0,3,3,'one')
    thresh,img_res=pr_function  

    #find contours
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    words_sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for c in words_sorted_ctrs: 
        x,y,w,h = cv2.boundingRect(c)
        zy=2
        zx=2
        while 0>(y-zy) or (y+h+zy)>thresh.shape[0]:
            zy-=1
            
        while 0>(x-zx) or (x+w+zx)>thresh.shape[1]:
            zx-=1
            
        words=img_res[y-zy:y+h+zy,x-zx:x+w+zx]
        cv2.imwrite(path_words+str(idw) + '.png', words)
        idw+=1
        words_images.append(words)
        words_pixels.append([x,y,w,h])
    return words_images,words_pixels,idw

        
