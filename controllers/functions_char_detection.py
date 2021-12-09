def quantizate_colors(img1,num_colors,bl):
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans,MiniBatchKMeans
    #https://www.pyimagesearch.com/2014/07/07/color-quantization-opencv-using-k-means-clustering/
    blur = cv2.bilateralFilter(img1.copy(), bl, bl, bl)
    (h, w) = blur.shape[:2]
    if len(blur.shape)> 2:
        
        img_lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
        img_resh = img_lab.reshape((img_lab.shape[0] * img_lab.shape[1], 3))
        kmeans = MiniBatchKMeans(n_clusters = num_colors)
        labels = kmeans.fit_predict(img_resh)
        quant = kmeans.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        gray_quant = cv2.cvtColor(quant, cv2.COLOR_BGR2GRAY)
    
    else:
        img_lab=blur.copy()
        img_resh = img_lab.reshape((img_lab.shape[0] * img_lab.shape[1])).reshape(-1, 1)
        kmeans = MiniBatchKMeans(n_clusters = num_colors)
        labels = kmeans.fit_predict(img_resh)
        quant = kmeans.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w))
        gray_quant = quant
      
    return quant



def get_color_ids(img2):
    import  numpy as np
    import cv2
    import pandas as pd
    colors_count = {}
    color_arr=[]
    
    if len(img2.shape)> 2:
        
        (channel_b, channel_g, channel_r) = cv2.split(img2)
        channel_b = channel_b.flatten()
        channel_g = channel_g.flatten()  # ""
        channel_r = channel_r.flatten()  # ""

        for i in range(len(channel_r)):
            BGR = "(" + str(channel_b[i]) + "," + \
                str(channel_g[i]) + "," + str(channel_r[i]) + ")"

            if BGR in colors_count:
                colors_count[BGR] += 1
            else:
                colors_count[BGR] = 1

    else:
        
        channel_g = img2.flatten() 

        for i in range(len(channel_g)):
            BGR = "(" + str(channel_g[i])+ ")"

            if BGR in colors_count:
                colors_count[BGR] += 1
            else:
                colors_count[BGR] = 1
    
    colors_count_sorted=dict(sorted(colors_count.items(), key=lambda item: item[1],reverse=True))
    result = colors_count_sorted.items() 
    data = list(result) 
        
    backround=pd.DataFrame(data)[1].max()

    for i in range(0,len(colors_count_sorted)):
        color_id=np.array(data)[i][0]
        color_value=int(np.array(data)[i][1])
        
        if ((backround-color_value)/backround)>0.25:
            if len(img2.shape)> 2:
                color_id = [int(i) for i in (color_id[1:len(color_id)-1]).split(',')] 
            else:
                color_id = color_id[1:len(color_id)-1]
            color_arr.append(color_id)
            
    backroundid=color_id=np.array(data)[0][0]
    if len(img2.shape)> 2:
        backround_id = [int(i) for i in (backroundid[1:len(backroundid)-1]).split(',')] 
    else:
        backround_id = backroundid[1:len(backroundid)-1]
    #textcolor=color_arr[1]
    #backroundcolor=color_arr[0]
    #color_arr=color_arr[1:len(color_arr)]
    return colors_count_sorted,color_arr,backround_id #returns a color array either list either unique color of the letter colors
#colors_count_sorted returns all colors 
# color_arr remmoves backround


def check_for_underscore(img7,array_color):
    
    import numpy as np
    import cv2
    def column(matrix, i):
        return [row[i] for row in matrix]

    colors_occur_by_row_column=np.zeros(img7.shape[0:2])

    if len(img7.shape)> 2:
        for val in array_color:
            temp_col=np.where(img7==(val), 1, 0) #isolate 1 color by puting 1 where color occure else 0
            #index shows how many colors occur by row split by columns 2D
            colors_occur_by_row_column=colors_occur_by_row_column+cv2.split(temp_col)[0] #inverse to row by row view
    else:
        for val in array_color:
            temp_col=np.where(img7==int(val), 1, 0) #isolate 1 color
            colors_occur_by_row_column=colors_occur_by_row_column+temp_col 
            
    for i in range(len(colors_occur_by_row_column)-1,int(len(colors_occur_by_row_column)/2),-1):
        if np.sum(colors_occur_by_row_column [i])==0 and np.sum(colors_occur_by_row_column [i-1])>0 and np.sum(colors_occur_by_row_column [i-2])<3 :
            index=i-1 #keep index of middle
            img7 = np.delete(img7, (index), axis=0)
            break
    
    return img7


def colors_decrease_minkowski_v2(color_array,backround_id,img9,p): #p=1 manhatan, p=2 eucledian....
    from math import sqrt
    import numpy as np

    def minkowski_distance(a, b, p1):
        return sum(abs(e1-e2)**p1 for e1, e2 in zip(a,b))**(1/p1)
    dist_c_min=100000000000

    color_mean=np.mean(color_array[0:int(len(color_array))],axis=0).round()
    color_median=np.median(color_array[0:int(len(color_array))],axis=0).round()

    
    for val in range(0,len(color_array)-1):
        dist_b = minkowski_distance(backround_id ,np.array(color_array[val]),p)
     
        if dist_c_min>dist_b:
            dist_c_min=dist_b #v1
            color_c_min=np.array(color_array[val]) #closest backround  color

                                 
    for val in range(0,len(color_array)-1):
        dist_b = minkowski_distance(color_c_min ,np.array(color_array[val]),p)
        dist_c = minkowski_distance(color_median , np.array(color_array[val]),p)
        
        if dist_c>dist_b: #if color closer to backround
            img9[np.where((img9==color_array[val]).all(axis=2))] = backround_id

    return img9

def horizodal_lines_remove_Hough(img12,idbackround,perx,pery):
    from cv2 import cv2
    import numpy as np
    img22=img12.copy()
    def preprocessing_for_houghlines(img22,idbackround):
        per_x=int(img22.shape[0]*perx)
        per_y=int(img22.shape[1]*pery)
        img22[per_x:img22.shape[0]-per_x,per_y:img22.shape[1]-per_y]=idbackround
        gray=cv2.cvtColor(img22,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 200)
        return edges
    
    def houghLines_delete(img12,edges,idbackround):
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,threshold=50, minLineLength=int(edges.shape[1]*25/100), maxLineGap=int(edges.shape[1]/8))
        lines_to_delete=[]
        try:
            for line in lines:
                x1,y1,x2,y2=line[0][0],line[0][1],line[0][2],line[0][3]
                
                if x2-x1!= 0:
                    lines_to_delete.append(line)
                    #if abs(img13.shape[1]-x)/(img13.shape[1]) <0.25:
                     #   img12[y1-1:y2+1,x1:x2]=idbackround
                    #img12[y1:y2,x1:x2]=idbackround    
            x11,y11,x12,y12=np.min(lines_to_delete[0:int(len(lines_to_delete))],axis=0).round()[0]
            x21,y21,x22,y22=np.max(lines_to_delete[0:int(len(lines_to_delete))],axis=0).round()[0]
            x=x22-x11
            y=y22-y11

            img12[y11:y22,0:int(img12.shape[1])]=idbackround

            return img12
        except:
            return img12
  

    img13=preprocessing_for_houghlines(img22,idbackround)
    img12= houghLines_delete(img12,img13,idbackround)
    return  img12



def count_occur_color_by_row_and_column(img6,array_color): 

    import numpy as np
    import cv2
    def column(matrix, i):
        return [row[i] for row in matrix]

    colors_occur_by_row_column=np.zeros(img6.shape[0:2])
    colors_first_color_index=np.zeros(img6.shape[1])#
    colors_last_color_index=np.zeros(img6.shape[1])#

    colors_count_by_column = {}
    colors_count_by_column_all_colors=np.zeros(img6.shape[1])#


    if len(img6.shape)> 2:
        for val in array_color:
            temp_col=np.where(img6==(val), 1, 0) #isolate 1 color by puting 1 where color occure else 0
            #index shows how many colors occur by row split by columns 2D
            colors_occur_by_row_column=colors_occur_by_row_column+cv2.split(temp_col)[0] #inverse to row by row view

            #index  by color if it occured by each column dict 1D
            colors_count_by_column[str(val)] =column((temp_col * (temp_col != 0)).sum(axis=0), 0)
            #count how many colors exist in each column 1D
            colors_count_by_column_all_colors=colors_count_by_column[str(val)]+colors_count_by_column_all_colors #count how many color pixells  exist by column
    else:
        for val in array_color:
            temp_col=np.where(img6==int(val), 1, 0) #isolate 1 color
            colors_occur_by_row_column=colors_occur_by_row_column+temp_col 

            colors_count_by_column[val] =(temp_col * (temp_col != 0)).sum(axis=0) #count how many times (pixels) this color occur by column
            colors_count_by_column_all_colors=colors_count_by_column[val]+colors_count_by_column_all_colors #count how many color pixells  exist by column

            #find upper and down limit by column       
    for col in range(0,img6.shape[1]): 

        temp_cons_index=np.where(colors_occur_by_row_column[:, [col]]>0)[0] #col index where values are not zero


        if len(temp_cons_index)> 0:
            colors_first_color_index[col]=temp_cons_index[0]
            colors_last_color_index[col]=temp_cons_index[len(temp_cons_index)-1]

        else:
            colors_first_color_index[col]=0
            colors_last_color_index[col]=0
    return colors_first_color_index,colors_last_color_index,colors_count_by_column_all_colors


def check_M_N(*args,endmax,col2):
    
    '''Use dynamic variable creation for first and last pixel position for current and next image columns
       In case of out of range will return -1.  This can only happend in first or end col_0. Or if the rest
       image after cutting is smaller than 5 pixels,  4,3.... '''
    for x in range(0,9): #take 7 points
        try:
            globals()['col_%s_start' % x] = colors_first_color_index[col2+x]
            globals()['col_%s_end' % x] = colors_last_color_index[col2+x]

        except:
            globals()['col_%s_start' % x] = -1
            globals()['col_%s_end' % x] = -1

    '''Use dynamic variable to calculate the diferencs.  In case it not exists will return 999999'''
    for y in range (0,8):
        if globals()['col_%s_start' %y] >=0 and globals()['col_%s_start' % (y+1)]  >=0:
            globals()['dif_col_%s_%s_start' % (y,y+1)] =globals()['col_%s_start' % (y+1)]-globals()['col_%s_start' %y]
        else:
            globals()['dif_col_%s_%s_start' % (y,y+1)] =999999

    boolean_1=False
    boolean_2=False    
    boolean_1=eval('col_{}_start'.format(len(args)+1))<endmax #6
    boolean_2=eval('dif_col_{}_{}_start'.format(len(args),len(args)+1))<0 #6 and 5
    
    for i in args:
        boolean_1=boolean_1 and eval('col_{}_start'.format(i))>0
        
    if boolean_1==True:
        if len(args)>5:

            boolean_2=boolean_2 and eval('dif_col_{}_{}_start'.format(1,2))>-1  and eval('dif_col_{}_{}_start'.format(2,3))>-1 and eval('dif_col_{}_{}_start'.format(len(args)-1,len(args)))>-1 
            for i in range (3,len(args)-1):
                boolean_2=boolean_2 and eval('dif_col_{}_{}_start'.format(i,i+1))>0
        else:
            boolean_2=boolean_2 and eval('dif_col_{}_{}_start'.format(1,2))>-1  and  eval('dif_col_{}_{}_start'.format(len(args)-1,len(args)))>-1 
            for i in range (2,len(args)-1):
                boolean_2=boolean_2 and eval('dif_col_{}_{}_start'.format(i,i+1))>0

    return boolean_2

def adjust_x(xs,xe,img5):
    while 0 >xs:
        xs=xs+1
    while xe>img5.shape[1]:
        xe=xe-1
      
    return xs,xe



