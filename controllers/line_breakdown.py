#import cv2
from functions_word_detection import check_for_complicated_lines, word_detection
from functions_line_detection import line_preprocessing, line_preprocessing_advanced, tesseract_recognition
from cut_to_characters import cut_to_characters
from functions_char_detection import colors_decrease_minkowski_v2, quantizate_colors, get_color_ids, count_occur_color_by_row_and_column, check_for_underscore, horizodal_lines_remove_Hough
from config.tesseract import path_to_my_tesseract


def line_breakdown(line_image, mydoc_tesseract, path_quant, idw, path_words, idq, text, pixels_line, idt, path_letters): 
    #get all the words for complicated objects that lines cant be identified
    if check_for_complicated_lines(0.2,0.85,10,line_preprocessing(line_image,13,9,0,0,30,2,0,0,0,'one'))==True:
        words_images,words_pixels,idw=word_detection(idw,path_words,line_preprocessing(line_image,13,9,0,0,30,2,0,3,3,'one'))
                
        for w in range(0, len(words_images)):
            idq+=1
            text_tesseract=tesseract_recognition(words_images[w],path_to_my_tesseract,False,20)
            mydoc_tesseract.add_paragraph(text_tesseract)
            text.append(text_tesseract)
            
            #start letter recognition
            colors_count_sorted,color_arr,backround_id = get_color_ids(words_images[w])
            image_color_decr=colors_decrease_minkowski_v2(color_arr,backround_id,words_images[w],1.5)
            quant1=quantizate_colors(image_color_decr,14,2)
            quant2=quantizate_colors(quant1,2,30)
            colors_count_sorted,color_array,backround_id=get_color_ids(quant2)
            image_underscore=check_for_underscore(quant2,color_array)
            image_hough= horizodal_lines_remove_Hough(image_underscore,backround_id,0.15,0.05)
            #cv2.imwrite(path_quant+str(idq) + '.png', quant2)
            colordictioary, color_array,backround_id=get_color_ids(image_hough)
            colors_first_color_index,colors_last_color_index,colors_count_by_column_all_colors=count_occur_color_by_row_and_column(image_hough,color_array)
            try:
                y_dif=abs(colors_first_color_index[colors_first_color_index>0].min()-colors_last_color_index[colors_last_color_index>0].max())
                #returns letter pixels by line word
                idt,letter_pixels,letter_images=cut_to_characters(words_images[w],image_hough,colors_count_by_column_all_colors,colors_first_color_index,colors_last_color_index,idt,path_letters)
            except:
                #support filtering of symbols
                letter_pixels=[]
                #print('icon')
            #aggregate the pixels with lines     
            pixels_line=pixels_line+[words_pixels[w]+[letter_pixels]]
            #print('IF TRUE PIXELS LINE', pixels_line)
    else:
        text_tesseract=tesseract_recognition(line_image,path_to_my_tesseract,False,20)
        mydoc_tesseract.add_paragraph(text_tesseract)
        text.append(text_tesseract) 
        #start letter recognition
        
        colors_count_sorted,color_arr,backround_id =get_color_ids(line_image)
        image_color_decr=colors_decrease_minkowski_v2(color_arr,backround_id,line_image,1.5)
        quant1=quantizate_colors(image_color_decr,14,2)
        quant2=quantizate_colors(quant1,2,30)
        colors_count_sorted,color_array,backround_id=get_color_ids(quant2)
        image_underscore=check_for_underscore(quant2,color_array)
        image_hough= horizodal_lines_remove_Hough(image_underscore,backround_id,0.15,0.05)
        #cv2.imwrite(path_quant+str(idq) + '.png', quant2)
        colordictioary, color_array,backround_id=get_color_ids(image_hough)

        colors_first_color_index,colors_last_color_index,colors_count_by_column_all_colors=count_occur_color_by_row_and_column(image_hough,color_array)
        try:
            y_dif=abs(colors_first_color_index[colors_first_color_index>0].min()-colors_last_color_index[colors_last_color_index>0].max())
            #returns letter pixels by line
            idt,letter_pixels,letter_images=cut_to_characters(line_image,image_hough,colors_count_by_column_all_colors,colors_first_color_index,colors_last_color_index,idt,path_letters)
        except Exception as e:
            #helps to reduce symbols
            letter_pixels=[]
            print(str(e))
            
        #print('<<<<<<<<<<<<<<<<<<<<[letter_pixels] aqui >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><', [letter_pixels])
        pixels_line=pixels_line+[letter_pixels]
        #print('ELSE pixels line', pixels_line)
        
    return pixels_line

    