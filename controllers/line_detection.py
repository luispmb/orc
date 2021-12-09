#DEPENDENCIES
#######################################################################################################################
import sys
sys.path.insert(0, '..')
import os
import numpy as np
from functions_line_detection import line_preprocessing, line_preprocessing_advanced, line_detector
from line_breakdown import *


#inputs
parent_directory = os.path.abspath('')
root = os.path.abspath(os.path.join(parent_directory, os.pardir))

data_output_folder = os.path.join(root, 'test', 'data', 'output')



def line_detection(img_line, temp_object, path_lines, idl, key, mydoc_tesseract, path_quant, idw, idq, path_words, idt, path_letters):  
    # print('line image', key)
        
    #preprocess it
    try:
        img_score,img_res=line_preprocessing(img_line, 31,29,19,17,30,0,0,4,3,'two')  
        #print('Base Image')
    except:
        img_score,img_res=line_preprocessing_advanced(img_line) 
        print('Base Image v2 / can be deleted')
            
    #get all the lines per object and line pixels write them on a path
    line_images, pixels, idl = line_detector(img_score,img_res,path_lines,idl)
    # print('line detection pixels', pixels)

    #1 text output per maximum hierarchy
    text=[]
    
    #recognition
    for line in range (0,len(line_images)):
        # import line breakdown
        line_image = line_images[line]
        pixels_line = pixels[line]
        pixels_line = line_breakdown(line_image, mydoc_tesseract, path_quant, idw, path_words, idq, text, pixels_line, idt, path_letters)

    temp_object['ocr_text'][key]=text
    # print('key is', key)

    temp_object['sub_coordinates'][key]=pixels

    #mydoc_tesseract.save(path_lines+"OCR_output_teserract.docx")
    mydoc_tesseract.save(os.path.join(path_lines, 'OCR_output_teserract.docx'))
    # print('lines recognized for image')
                
    #save the object
    np.save(os.path.join(data_output_folder, 'object_output.npy'), temp_object)

    return pixels_line, text