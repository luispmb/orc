import cv2
import numpy as np

def green_blue_swap(image):
    # 3-channel image (no transparency)
    if image.shape[2] == 3:
        b,g,r = cv2.split(image)
        image[:,:,0] = g
        image[:,:,1] = b
    # 4-channel image (with transparency)
    elif image.shape[2] == 4:
        b,g,r,a = cv2.split(image)
        image[:,:,0] = g
        image[:,:,1] = b
    return image 

def char_preprocessing_step_1(wim): #return word primary contours
    temp_word_image=wim.copy()
    #to gray
    print('temp_word_image', temp_word_image)
    swapped = green_blue_swap(np.array(np.float32(temp_word_image)*255))
    gray = cv2.cvtColor(swapped, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(temp_word_image, cv2.COLOR_BGR2GRAY)
    res,gray_thresh = cv2.threshold(gray,195,255,cv2.THRESH_TRUNC)
    #blur
    b=20
    blur = cv2.bilateralFilter(gray_thresh, b, b, b) 
    #thresholding
    #print('BLUR', blur)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 3)
    #dilation
    kernel = np.ones((6,1), np.uint8) #can be 4,2
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #find contours
    contours, hierarchy = cv2.findContours(img_dilation.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print('contours', contours)
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return sorted_ctrs


def char_preprocessing_step_2(sorted_ctrs,wim):#Delete redundant contours, identify i j := ;...
    import cv2
    word_to_chars_cnts=[]
    word_to_chars_dif=[]
    new_sorted_ctrs=[]
    
    for c in sorted_ctrs: 
            x,y,w,h = cv2.boundingRect(c) 
            word_to_chars_dif.append(x+w) #calculate all the xw x_axis size
            word_to_chars_cnts.append([x,y,w,h]) #save primary contours as array


    j=len(word_to_chars_dif)
    i=0

    while i < 1*j:
        if (len(word_to_chars_dif)-i>1):
            diff = word_to_chars_dif[i+1]-word_to_chars_dif[i]

            if (diff/wim.shape[1])*100<=2: #if the distance is less than 2% of image size merge cnts
                x1,y1,w1,h1=word_to_chars_cnts[i]
                x2,y2,w2,h2=word_to_chars_cnts[i+1]
                x=min(x1,x2)
                y=min(y1,y2)
                w=max(w1,w2)
                h=max(h1,h2)
                yh=max(y1+h1,y2+h1,y1+h2,y2+h2)
                xw=max(x1+h1,x2+h1,x1+h2,x2+h2)
                new_sorted_ctrs.append([x,y,w,h,yh,xw])
                i=i+2
            else:

                x2,y2,w2,h2=word_to_chars_cnts[i]
                yh=y2+h2
                xw=x2+h2
                new_sorted_ctrs.append([x2,y2,w2,h2,yh,xw])
                i=i+1
        else:
                x2,y2,w2,h2=word_to_chars_cnts[i]
                yh=y2+h2
                xw=x2+h2
                new_sorted_ctrs.append([x2,y2,w2,h2,yh,xw])
                i=i+1
    return new_sorted_ctrs

def char_preprocessing_step_3(new_sorted_ctrs):#normalize  height 
    temp_min_y=100000
    temp_max_yh=0
    for c in new_sorted_ctrs:
        x,y,w,h,yh,xw = c
        if y<temp_min_y: 
            temp_min_y=y

        yh_temp=yh
        if temp_max_yh<yh_temp:
            temp_max_yh=yh_temp 

    return temp_max_yh,temp_min_y

def char_preprocessing_step_4(wim,new_sorted_ctrs,temp_max_yh,temp_min_y):

    char_images=[]
    char_images_nh=[]
    z=0
    for c in new_sorted_ctrs: 
        x,y,w,h,yh,xw = c
        #if w>7 and h>7 : ############################################################# exclude really smalle moise

        new_img=wim[y-z:yh+z,x-z:x+w+z] 
        new_img_normheight=wim[temp_min_y-z:temp_max_yh+z,x-z:x+w+z] 

        char_images.append(new_img)
        char_images_nh.append(new_img_normheight)
    return char_images,char_images_nh

def char_detection(new_img,new_img_normheight,dictionar,Q1_,Q3_,output): #detect char 
    
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
    forecast,dictionartemp=output(resized,resized_normheight,dictionar,x_test,Q1_,Q3_)
    text_char =str(forecast)
    return text_char   