#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DEPENDENCIES
#######################################################################################################################
import sys
sys.path.insert(0, '..')
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append("")

if "RUNNING_LOCALLY" in os.environ.keys() and os.environ["RUNNING_LOCALLY"] == "True":
    print('RUNNING LOCALLY')
    #sys.path.insert(0, '..')
    #sys.path.append("")
    from utils.util_s3 import instantiateS3
    s3 = instantiateS3()

myDir = os.getcwd()
print('myDir', myDir)

parentDir = os.path.dirname(os.getcwd())
print('parentDir', parentDir)

os.chdir(parentDir) #mudar wd pra root
#sys.path.append("")
print(' os.getcwd()',  os.getcwd())


if "RUNNING_LOCALLY" in os.environ.keys() and os.environ["RUNNING_LOCALLY"] == "True":
    print('A')
    if 'runApplication' in locals():
        platformDirOCR = os.path.join(os.getcwd(), 'controllers')
    else:
        platformDirOCR = os.path.join(os.getcwd(),'controllers')

else:
    print('B')
    s3DirOCR = parentDir
    platformDirOCR = os.path.join(os.getcwd(), 'ocr', 'controllers')

s3({"path": parentDir, "localPath": '/'}, "readFolder")

#platformDirOCR = os.path.join(os.getcwd(), 'ocr', 'controllers')
print('platformDirOCR', platformDirOCR)
os.chdir(platformDirOCR) #mudar wd pra root
print(' os.getcwd()',  os.getcwd())


from functions_object_detection import create_path, preprocess_orig_img, mask_morph, get_horizodal_lines, get_vertical_lines, wide_white_contours, remove_noise, extract_contoured_image_2, extract_contoured_image_1, expot, write_images_pre, extract_object
#from functions_line_detection import *
from config.tesseract import path_to_my_tesseract
#from functions_word_detection import check_for_complicated_lines, worqd_detection
#from functions_char_detection import *
#from cut_to_characters import *
from utils.util_miscellaneous import changeDirectory
from line_detection import line_detection

#dont require pip
import os
import shutil
import string
from os import listdir
from os.path import isfile, join

#require pip
import cv2
import numpy as np
import pandas as pd
import docx
import pytesseract
import re

######################################################################################################################################
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = path_to_my_tesseract
#pytesseract.pytesseract.tesseract_cmd = os.environ['path_to_my_tesseract']
#custom_config = r'--oem 3 --psm 6'
custom_config = r'-l por+eng --psm 6'


# In[2]:


#inputs
parent_directory = os.path.abspath('')
root = os.path.abspath(os.path.join(parent_directory, os.pardir))

data_input_folder = os.path.join(root, 'test', 'data', 'input')
data_output_folder = os.path.join(root, 'test', 'data', 'output')


#clean and create paths to save images after preprocessing
create_path(os.path.join(data_output_folder, 'filtered', 'output_all_br')) #source image preprocessing output with blured backround to compare
create_path(os.path.join(data_output_folder, 'filtered', 'output_all')) #source image preprocessing output ready to exctract objects and lines
create_path(os.path.join(data_output_folder, 'recognition')) # extracted objects, lines, and classification results


#read all the source images for recognition
#sourcepath='.\\data\\\\'
sourcepath=data_input_folder


# In[3]:


# estrutura de dados
def convertToFinalStructure(temp_object):
    result = {'result': []}

    
    for imgIdx in range(0, len(temp_object['base_images'])):
        #print('imgIdx', imgIdx)
        img = temp_object['base_images'][imgIdx]
        annot_img = img.copy()
        image_name = temp_object['image_name'][imgIdx]
        image_annotation_folder = os.path.join(data_output_folder, 'images_annotated', image_name)
        create_path(image_annotation_folder)

        imgResult = {'image': img.tolist(), 'objBboxes': []}

        #objDetected = {'objBbox': {}, 'text': '', 'chars': [{'charBbox': {}, 'char': ''}] }

        for objectBboxIdx in range (0, len(temp_object['coordinates_line'])):
            objectBbox = temp_object['coordinates_line'][objectBboxIdx]  #coordinates in the form of y0, y1, x0, x1
            #print('objectBbox', objectBbox)

            snippet = img[objectBbox[0]: objectBbox[1], objectBbox[2]: objectBbox[3]]
            cv2.imwrite(os.path.join(image_annotation_folder, "snippet_" + str(objectBboxIdx) + ".png"), snippet)
            
            objectDetected = {
                'text': '',
                'snippet': snippet.tolist() ,
                'chars': [],
                'objBbox': {
                    'y0': objectBbox[0],
                    'y1': objectBbox[1],
                    'x0': objectBbox[2],
                    'x1': objectBbox[3]
                }
            }

            imgResult['objBboxes'].append(objectDetected)
            

            for coordIdx in range (0,len(temp_object['sub_coordinates'][0][2:][0])): #coordinates in the form of y0, y1, x0, x1
                charBbox = temp_object['sub_coordinates'][objectBboxIdx][2:][0][coordIdx]
                #print('charBbox', charBbox)
                y0 = charBbox[0]
                y1 = charBbox[1]
                x0 = charBbox[2]
                x1 = charBbox[3]

                relBbox = [y0, y1, x0, x1]
                #print('relBbox',relBbox)

                snippet_char = snippet[y0: y1, x0: x1]

                snippet_char_path = "snippet_" + str(objectBboxIdx) + "char_" + str(coordIdx) + ".png"
                cv2.imwrite(os.path.join(image_annotation_folder, "snippet_" + str(objectBboxIdx) + "_char_" + str(coordIdx) + ".png"), snippet_char)
                
                y0 = objectBbox[0] + charBbox[0]
                y1 = objectBbox[0] + charBbox[1]
                x0 = objectBbox[2] + charBbox[2]
                x1 = objectBbox[2] + charBbox[3]
                absBbox = [y0, y1, x0, x1]
                #print('absBbox', absBbox)

                charDetected = {
                    'text': '',
                    'snippet': img[y0: y1, x0: x1].tolist() ,
                    'charBox': {
                        'y0': y0,
                        'y1': y1,
                        'x0': x0,
                        'x1': x1
                    }
                }
                
                imgResult['objBboxes'][objectBboxIdx]['chars'].append(charDetected)
                
                cv2.rectangle(annot_img, (x0, y0), (x1, y1), (255, 0, 0), 1)
                
        result['result'].append(imgResult)

        image_annotated_path = os.path.join(image_annotation_folder, 'annotation.png')
        #print('image_annotated_path', image_annotated_path)
        cv2.imwrite(image_annotated_path, annot_img)
        

    return result


# # Object detection

# In[4]:


def object_detection(imageName):

    dictionary=[]
    #create a path for each image to store the results of text objects, text lines and the word file with the classification output.
    path_objs=os.path.join(data_output_folder, 'recognition', 'objects\\'+imageName+'\\')
    create_path(path_objs)
    path_lines=os.path.join(data_output_folder, 'recognition', 'lines\\'+imageName+'\\') #path to objext text lines
    create_path(path_lines)
    path_words=os.path.join(data_output_folder, 'recognition', 'words\\'+imageName+'\\') 
    create_path(path_words)
    path_letters=os.path.join(data_output_folder, 'recognition', 'letters\\'+imageName+'\\') 
    create_path(path_letters)
    path_quant=os.path.join(data_output_folder, 'recognition', 'quantized\\'+imageName+'\\') 
    create_path(path_quant)
    
    imName = os.path.join(data_input_folder, imageName)
    print('imName', imName)
    img=cv2.imread(os.path.join(data_input_folder, imageName))
    #print('img', img)

    thresh,gray=preprocess_orig_img(img)
    horizontal_p1=mask_morph(get_horizodal_lines(thresh.copy(),25),thresh.copy())
    vertical=mask_morph(get_vertical_lines(thresh.copy(),50),horizontal_p1.copy()) #20
    image_wide=wide_white_contours(vertical.copy(),int(img.shape[0]*expot(vertical.copy())),int(img.shape[1] * expot(vertical.copy())))
    noise_removed=remove_noise(image_wide.copy())  
    masked_img_n, masked_img_weighted_big_n,masked_img_nf, masked_img_weighted_big_nf,masked_img_nfs, masked_img_weighted_big_nfs,filter1_nf=extract_contoured_image_1(img.copy(), noise_removed.copy(),0,0,0.2,0.8)
    masked_imgf2,masked_img_weighted_bigf2,filter1_f2=extract_contoured_image_2(img.copy(), filter1_nf.copy(),0,0,0.2,0.8)
    write_images_pre(masked_imgf2, masked_img_weighted_bigf2, imageName)

    lines, temp_object = extract_object(img.copy(), filter1_f2, imageName, path_objs)

    if not 'base_images' in temp_object:
        temp_object['base_images'] = []
    
    temp_object['base_images'].append(img)

    if not 'image_name' in temp_object:
        temp_object['image_name'] = []
    
    temp_object['image_name'].append(imageName)

            
    ##Line detection
    ###################################################################################################################
    idl=0 #id for line per image
    idw=0 #id for word per image
    idq=0 #quantized image
    idt=0 #text letter
    mydoc_tesseract = docx.Document()
    for key in range (0,len(temp_object['line_images'])):
        # print('line image', key)
        
        #get an object image
        img_line=temp_object['line_images'][key]

        pixels, text = line_detection(img_line, temp_object, path_lines, idl, key, mydoc_tesseract, path_quant, idw, idq, path_words, idt, path_letters)
        
        temp_object['ocr_text'][key]=text

        # print('pixels', pixels)

        temp_object['sub_coordinates'][key]=pixels

    #mydoc_tesseract.save(path_lines+"OCR_output_teserract.docx")
    mydoc_tesseract.save(os.path.join(path_lines, 'OCR_output_teserract.docx'))
    print('lines recogized for  image:', imageName)
            
    #save the object
    # np.save(os.path.join(data_output_folder, 'object_output.npy'), temp_object)
    
    return temp_object


# In[5]:
def main(imagesNames = []):
    if len(imagesNames) == 0:
        imagesNames = [f for f in listdir(sourcepath) if isfile(join(sourcepath, f))]

    for imageName in imagesNames:
        print('imageName', imageName)
        temp_object = object_detection(imageName)
        result = convertToFinalStructure(temp_object)
        print('<<<<<<<<<<<<<<<<<<<< DONE DETECTING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('result', result)
    
    return result

print('sys.argv', sys.argv)

if 'eai_payload' in locals() or 'eai_payload' in globals():
    print('eai_payload', eai_payload)
    output = main(eai_payload)
else:
    output = main(['ex_001.png'])

print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<output>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', type(output))


import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# output = json.dumps(output, cls=NumpyEncoder)

#output = {}

#print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<output>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', output)

if __name__ == '__main__':
    print('<<<<<<<<<<<<<<<<<<<<< MAIN >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', sys.argv)
    main(sys.argv) 

'''
if __name__ == 'detect':
    detect()[sys.argv[1]]
'''


