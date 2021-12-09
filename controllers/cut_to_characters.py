def cut_to_characters(img,quant,colors_count_by_column_all_colors,colors_first_color_index,colors_last_color_index,idt,path_letter):
    from functions_char_detection import adjust_x
    import numpy as  np
    import cv2
    pixels=[]
    letter_images=[]
    
    index_color_occur = np.where(colors_count_by_column_all_colors>0)
    img5=quant.copy()
    x_start=-1
    x_end=-1
    y_start=0
    y_end=img5.shape[0]
    new_temp_start=False
    
    y_dif=abs(colors_first_color_index[colors_first_color_index>0].min()-colors_last_color_index[colors_last_color_index>0].max())

    for col in range(0,img5.shape[1]):

        if new_temp_start!= True:

            if col in np.array(index_color_occur):#WHERE COLOR EXIST
                new_temp_start=True
                x_start=col

        elif (new_temp_start== True) and col in np.array(index_color_occur): 
            x_end=col


        else:  
            '''At this this level my images are cut in characters based on pure backround in between letters.
               Some of them are not perfectly cut it and they need further check. My x_start in this level is identical to 
               first color pixel and my x_end is plus 1 pixel. Because this adds white space at the end I will decrease x_end 
                by 1 pixels.'''
            x_end=col-1# to be in sample range else takes null letter part
            new_temp_start=False

            '''Next I am going to check which detected characters require further check. For those that are very hight
               or smaller than 5 pixels I will asume they are ok and append them. In the append I will add 1 pixel in each side
               because when I cut the images it is considered as open range() so I lose 1 pixel. '''

            if  img5.shape[0] > 25 or x_end-x_start< 5: #had 5
                #if initial image too big or current image width less than 3 pixels
                x1_start=x_start-1
                x1_end=x_end+1
                '''check if the limits are out of image range and adjust'''
                x_start1,x_end1=adjust_x(x1_start,x1_end,img5)
                pixels.append([y_start,y_end,x_start1,x_end1]) 

            else:
                '''Here we start the advanced filter. Our x_start is always 1 pixel before the actual starter. 
                Scope of that if the flexibility to detect M and N if they are in the beginning.'''

                col2=x_start-1 #the image start always -1
                col2max=x_end #the image end actual end
                endmax=colors_last_color_index.max() #end pixel max of the current image
                endmedian=np.median(colors_last_color_index)# end pixel median of the current image
                startmedian=np.median(colors_first_color_index)
                startmax=np.max(colors_first_color_index)
                print('object initial coordinates',col2,col2max)
                letter_round2=False #false when the letter is the first pixels that checked
                
                while col2 <= (col2max):
        
                    '''Use dynamic variable creation for first and last pixel position for current and next image columns
                       In case of out of range will return -1.  This can only happend in first or end col_0. Or if the rest
                       image after cutting is smaller than 5 pixels,  4,3.... '''
                    for x in range(0,11): #take 7 points
                        try:
                            globals()['col_%s_start' % x] = colors_first_color_index[col2+x]
                            globals()['col_%s_end' % x] = colors_last_color_index[col2+x]

                        except:
                            globals()['col_%s_start' % x] = -1
                            globals()['col_%s_end' % x] = -1

                    '''Use dynamic variable to calculate the diferencs.  In case it not exists will return 999999'''
                    for y in range (0,10):
                        if globals()['col_%s_start' %y] >=0 and globals()['col_%s_start' % (y+1)]  >=0:#goes down
                            globals()['dif_col_%s_%s_start' % (y,y+1)] =globals()['col_%s_start' % (y+1)]-globals()['col_%s_start' %y]
                        else:
                            globals()['dif_col_%s_%s_start' % (y,y+1)] =999999
                            
                    for y in range (0,10):
                        if globals()['col_%s_end' %y] >=0 and globals()['col_%s_end' % (y+1)]  >=0:#goes down
                            globals()['dif_col_%s_%s_end' % (y,y+1)] =globals()['col_%s_end' % (y+1)]-globals()['col_%s_end' %y]
                        else:
                            globals()['dif_col_%s_%s_end' % (y,y+1)] =999999
###########################START ADVANCED LETTER CUT####################################################################

                    if new_temp_start!= True: 
                        '''In the end  of the first level we always end with new_temp_start==False. 
                           So the first pixel will always be a starter'''
                        '''i dont care if this is negative I will fix at the end with while loop.'''
                        x_start2=col2 
                        new_temp_start= True 
                        '''check for the m'''
                        if col_1_start >0 and col_2_start >0 and col_3_start >0 and col_4_start >0 and col_5_start >0 and col_6_start >0 and col_7_start >0 and col_8_start >0:
                            print('we are inside m search',col2)
                            if  dif_col_1_2_start==dif_col_2_3_start==dif_col_3_4_start==dif_col_6_7_start==0:

                                if dif_col_5_6_start==-1 and dif_col_4_5_start==1:
                                    print('m  found')
                                    new_temp_start=False
                                    x_end2= col2+8 
                                    col2=col2+8
                                    x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                    pixels.append([y_start,y_end,x2_start,x2_end])
                                    letter_round2=True
                            elif dif_col_1_2_start==dif_col_2_3_start==dif_col_3_4_start==dif_col_4_5_start==dif_col_7_8_start==0:
                                if dif_col_6_7_start==-1 and dif_col_5_6_start==1:
                                    print('m  found')
                                    new_temp_start=False
                                    x_end2= col2+9 
                                    col2=col2+9
                                    x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                    pixels.append([y_start,y_end,x2_start,x2_end])
                                    letter_round2=True
                                    
###############################################################################################################
                        '''check for the k'''
                        if col_1_start >0 and col_2_start >0 and col_3_start >0 and col_4_start >0 and col_5_start >0 and col_6_start >0 and abs(col_1_end-col_1_start)>4:
                            if dif_col_1_2_start>0 and dif_col_2_3_start<0 and dif_col_3_4_start<0 and dif_col_4_5_start<1:
                                print('we are inside k search',col2)
                                if  dif_col_1_2_end<0 and dif_col_2_3_end>0 and dif_col_3_4_end>0:
                                    print('V10  found big k ')
                                    new_temp_start=False   
                                    x_end2=col2+5 #I stop in the plus 1 here. Thats the letter end.
                                    col2=col2+5
                                    x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                    pixels.append([y_start,y_end,x2_start,x2_end])
                                    letter_round2=True
######################################################################################################################
                        '''check for the v AND w'''
                        if col_1_start >0 and col_2_start >0 and col_3_start >0 and col_4_start >0 and col_5_start >0 and abs(col_1_end-col_1_start)<4:
                            if col_6_start >0 and col_7_start >0 and col_8_start >0 and col_9_start >0  :
                                w_search=False
                                if  dif_col_1_2_start>0 and dif_col_2_3_start>0 and dif_col_3_4_start<0 and dif_col_4_5_start<0 and dif_col_5_6_start>0 and dif_col_6_7_start>0 and dif_col_7_8_start<0:
                                    print('we are inside w search',col2)
                                    if dif_col_9_10_start<0:
                                        print('V10  found big w ')
                                        new_temp_start=False   
                                        x_end2=col2+10 #I stop in the plus 1 here. Thats the letter end.
                                        col2=col2+10
                                        x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                        pixels.append([y_start,y_end,x2_start,x2_end])
                                        letter_round2=True
                                        w_search=True
                                
                                if  w_search==False and dif_col_1_2_start>=0 and dif_col_2_3_start>0 and dif_col_3_4_start<0 and dif_col_4_5_start<0:
                                        print('we are inside v/V search',col2)
                                        print('V5  found small v ')
                                        new_temp_start=False   
                                        x_end2=col2+5 #I stop in the plus 1 here. Thats the letter end.
                                        col2=col2+5
                                        x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                        pixels.append([y_start,y_end,x2_start,x2_end])
                                        letter_round2=True
                                elif w_search==False and dif_col_1_2_start>=0 and dif_col_2_3_start>0 and dif_col_3_4_start>0 and dif_col_4_5_start<0 and dif_col_5_6_start<0 and dif_col_6_7_start<0:
                                        print('we are inside v/V search',col2)
                                        print('V5  found biv V ')
                                        new_temp_start=False   
                                        x_end2=col2+7 #I stop in the plus 1 here. Thats the letter end.
                                        col2=col2+7
                                        x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                        pixels.append([y_start,y_end,x2_start,x2_end])
                                        letter_round2=True
                        

######################################################################################################################

                        '''check for the M, N patter only if its in the beginning. This will only occur if the pixels exist'''
                        if col_4_start >0 and col_1_start >0 and col_2_start >0 and col_3_start >0 and col_5_start >0 and col_4_start<endmax and abs(col_1_end-col_1_start)>3:
                            print('we are inside M,N search /max5', col2)
                            if col_6_start >0 and col_5_start<endmax:
                                print('we are inside M,N search /max6', col2)
                                if col_7_start >0 and col_6_start<endmax:
                                    print('we are inside M,N search /max7', col2)
                                    if col_8_start >0 and col_7_start<endmax:
                                        print('we are inside M,N search /max8', col2)
                                        if dif_col_1_2_start>-1 and dif_col_2_3_start >-1 and dif_col_3_4_start>0 and dif_col_4_5_start >0 and dif_col_5_6_start >0 and dif_col_6_7_start >-1  and dif_col_7_8_start < 0 :
                                            if col2+8==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration
                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end])
                                                letter_round2=True
                                                print('M8 pattern', col2)
                                            else:
                                                x_end2=col2+7
                                                col2=col2+7
                                                print('DE NADA in M8', col2)
                                        elif dif_col_1_2_start>-1 and dif_col_2_3_start >-1 and dif_col_3_4_start>0 and dif_col_4_5_start >0 and dif_col_5_6_start >-1 and dif_col_6_7_start < 0 :
                                            if col2+7==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end]) 
                                                letter_round2=True
                                                print('M7 pattern', col2)
                                            else:
                                                x_end2=col2+6
                                                col2=col2+6
                                                print('DE NADA in M7', col2)
                                        elif  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start>-1  and dif_col_5_6_start < 0 :
                                            if col2+6==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end]) 
                                                letter_round2=True
                                                print('M6 pattern', col2)
                                            else:
                                                x_end2=col2+5 
                                                col2=col2+5
                                                print('DE NADA in M6', col2)

                                        elif  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start < 0 :
                                            if col2+5==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end]) 
                                                letter_round2=True
                                                print('M5 pattern', col2)
                                            else:
                                                x_end2=col2+4 
                                                col2=col2+4
                                                print('DE NADA in M5', col2)
                                    else:
                                        print('ok')

                                        if dif_col_1_2_start>-1 and dif_col_2_3_start >-1 and dif_col_3_4_start>0 and dif_col_4_5_start >0 and dif_col_5_6_start >-1 and dif_col_6_7_start < 0 :
                                            if col2+7==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end])
                                                letter_round2=True
                                                print('M7 pattern V2', col2)
                                            else:
                                                x_end2=col2+6
                                                col2=col2+6
                                                print('DE NADA in M7 V2', col2)
                                        elif  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start>-1  and dif_col_5_6_start < 0 :
                                            if col2+6==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end])
                                                letter_round2=True
                                                print('M6 pattern V2', col2)
                                            else:
                                                x_end2=col2+5
                                                col2=col2+5
                                                print('DE NADA in M6 V2', col2)

                                        elif  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start < 0 :
                                            if col2+5==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end])
                                                letter_round2=True
                                                print('M5 pattern V2', col2)
                                            else:
                                                x_end2=col2+4
                                                col2=col2+4
                                                print('DE NADA in M5 V2', col2)
                                else:
                                        if  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start>-1  and dif_col_5_6_start < 0 :
                                            if col2+6==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                    new_temp_start=False
                                                    col2=col2max #make it max to stop the iteration

                                                    x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                    x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                    while 0 >x2_start:
                                                        x2_start=x2_start+1
                                                    while x2_end>img5.shape[1]:
                                                        x2_end=x2_end-1
                                                    pixels.append([y_start,y_end,x2_start,x2_end]) 
                                                    letter_round2=True
                                                    print('M6 pattern V3', col2)
                                            else:
                                                x_end2=col2+5
                                                col2=col2+5
                                                print('DE NADA in M6 V3', col2)

                                        elif  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start < 0 :
                                            if col2+5==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                                new_temp_start=False
                                                col2=col2max #make it max to stop the iteration

                                                x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                                x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                                while 0 >x2_start:
                                                    x2_start=x2_start+1
                                                while x2_end>img5.shape[1]:
                                                    x2_end=x2_end-1
                                                pixels.append([y_start,y_end,x2_start,x2_end]) 
                                                letter_round2=True
                                                print('M5 pattern V3', col2)
                                            else:
                                                x_end2=col2+4
                                                col2=col2+4
                                                print('DE NADA in M5 V3', col2)

                            else:
                                                                                                                          #col_4_end                                   
                                if  dif_col_1_2_start>-1 and dif_col_2_3_start >0 and dif_col_3_4_start>0 and dif_col_4_5_start < 0 :
                                    if col2+5==col2max: #if I reached the end,mean only 1 letter M or N stop!
                                        new_temp_start=False
                                        col2=col2max #make it max to stop the iteration

                                        x2_start=x_start2 #I already add extra pixel here in the beginning dont need
                                        x2_end=col2max #dont need to define x_end2 here, its the end of the process.
                                        while 0 >x2_start:
                                            x2_start=x2_start+1
                                        while x2_end>img5.shape[1]:
                                            x2_end=x2_end-1
                                        pixels.append([y_start,y_end,x2_start,x2_end]) 
                                        letter_round2=True
                                        print('M5 pattern V4', col2)
                                else:
                                    x_end2=col2+4
                                    col2=col2+4
                                    print('DE NADA in M5 V4', col2)
 #####################################################################################################################

######################################################################################################################


                    elif new_temp_start== True and dif_col_0_1_start< 1 and col2<col2max and dif_col_0_1_start!=999999:
                        '''I dont need to create an else for  the 999999 case, because It will move to the next pixel until colmax.
                        This case can happen though, because the program is built to assign that value only in the starter'''
                        #if goes up or equal keep letter 


                        x_end2=col2
                        print('level 1 merge detection- goes  up',col2)

                    elif new_temp_start== True and dif_col_0_1_start >0 and col2<col2max and dif_col_0_1_start!=999999:
                        #if goes down then:
                        '''I dont need to create an else for  the 999999 case, because It will move to the next pixel until colmax.
                        This case can happen though, because the program is built to assign that value only in the starter'''
                        print('level 2 merge detection- goes  down v1',col2)
                        '''if it goes down and its in the lower level of  median then check for U.'''
                        if col_1_end ==col_1_start and col_1_end >= endmedian: #U vs z-a
                            '''if the next point end height is further that 1 pixel. Then its not a U.Its a cut. like \A'''
                            print('level 2 merge detection- gowes up down. U or \A ')

                            if col_0_end==col_0_start: #check for TA
                                new_temp_start=False
                                x_end2=col2+1 #I stop in the plus 1 here. Thats the letter end.
                                x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                pixels.append([y_start,y_end,x2_start,x2_end])
                                letter_round2=True
                                print('PP TA pattern')

                            else:

                                if abs(col_2_end - col_1_end) > 0 : #an to  end point exei apostasi apo to epomeno start then break
                                    new_temp_start=False
                                    x_end2=col2+1 #I stop in the plus 1 here. Thats the letter end.
                                    x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                    pixels.append([y_start,y_end,x2_start,x2_end])
                                    letter_round2=True
                                    print('PP \A  pattern ',col2)

                                else:
                                    '''else its a U'''
                                    x_end2=col2
                                    print('U pattern',col2)



                        else:
                            '''if it goes down but not lower level.'''
                            '''check for 2 specific patterns. If there are 3 extra points!!!!'''
                            if col_0_start >0 and col_1_start >0 and col_2_start >0 and col_3_start >0:
                                print('level 3 Down check v2',col2)
                                if dif_col_1_2_start>-1 and dif_col_2_3_start < 0 and col_1_start >= startmedian and col_2_start  >= startmedian:
                                    #if gowes down for 2 points like v (-,-=,+)
                                    new_temp_start=False
                                    x_end2=col2+2 #adjust  by 2 letter end
                                    col2=col2+2##adjust  by 1 the counting. Will be fixed at the end also
                                    x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                    pixels.append([y_start,y_end,x2_start,x2_end])
                                    letter_round2=True
                                    print('PP (-,-=,+) pattern',col2)

                                elif dif_col_1_2_start< 0 and col_1_start  >= startmedian and col_2_start >= startmedian: 
                                    #if gowes down for 2 points like v (-,+)
                                    new_temp_start=False
                                    x_end2=col2+1
                                    col2=col2+1 # Will be fixed at the end also
                                    x2_start,x2_end=adjust_x(x_start2,x_end2,img5)
                                    pixels.append([y_start,y_end,x2_start,x2_end])
                                    letter_round2=True
                                    print('PP (-,+) pattern',col2)


                                else:
                                    '''if no pattern found move on!!!!''' 
                                    print('No pattern found in level 3 down check v1',col2)
                                    x_end2=col2
                                    col2=col2  
                            else:
                                '''if not all col pixels exist move on,-1 error'''
                                print('No pattern found in level 3 doown check v2',col2)
                                x_end2=col2
                                col2=col2  

                    elif new_temp_start== True and col2==col2max:
                        '''if I reached max stop no matter!!!!''' 
                        print('PP level 4 - max columns ',col2)
                        new_temp_start=False

                        x2_start,x2_end=adjust_x(x_start2,col2max+1,img5)
                        pixels.append([y_start,y_end,x2_start,x2_end])
                        letter_round2=True

                    '''when I start search I am always -1 col. But if 2 letters are merged. Then the second search start
                       is from col 0. To avoid that I decrease start col by 1'''  


                    if letter_round2==True:
                        print('True letter round')
                        letter_round2=False
                        if abs(col2-col2max)<=1 :
                            print('2 pixels  left move on')
                            col2=col2max+1
                        else:
                            col2=col2
                    else:
                        col2=col2+1


                '''gurantee that after this process, we will deactivate'''   
                        
                new_temp_start=False
                print('...................')
    # print('cut to characters pixels ->', pixels)
    #cv2.imwrite('col' + str(col) + '.png', img)
    #print('Vou cortar a imagem em caracters')
    for i in range(len(pixels)): 
        idt+=1
        x_start,x_end=adjust_x(pixels[i][2],pixels[i][3],img5)
        new_img=img[y_start:y_end,x_start:x_end]
        #cv2.imwrite('test' + str(idt) + '.png', new_img)
        letter_images.append(new_img)
        #cv2.imshow('Contours', new_img) 
        cv2.imwrite(path_letter+str(idt)+ '.png', new_img)
    return idt,pixels,letter_images
    #return idt,[y_start, y_end,x_start, x_end],letter_images
                
   