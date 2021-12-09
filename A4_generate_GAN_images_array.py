#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def generateGanImagesArray():
    import tensorflow as tf
    import numpy as np
    import string
    import pandas as pd

    get_ipython().run_line_magic('store', '-r character')
    get_ipython().run_line_magic('store', '-r number')


    print('characters:',character)
    print('number:',number)


    latent_dim=100
    #train , test split
    step=round(number/(number*0.26))
    split=list(range(1,number ,step))

    #character dictionary
    characters_all = list(string.printable)[:-6] #+['ç']# <
    j=-1
    dict_target=[]
    for char1 in characters_all:
        j=j+1
        dict_target.append([char1,ord(char1),j])
    pd_dict_target=pd.DataFrame(dict_target).rename(columns={0:'Actual_char',1:'Actual_num',2:'Actual_id'})


    #def for deprocess
    def deprocess(x):
        return np.uint8((x+1)/2*255)

    gan_images_train = [] #image array
    gan_images_pr_train=[] #normalized image array
    gan_classes_train=[] #ord character
    gan_classes_tr_train=[] #numerical target instead of character

    gan_images_test = []
    gan_classes_test=[]
    gan_classes_tr_test=[]
    gan_images_pr_test=[]



    for char in character:
        generator = tf.keras.models.load_model('models\\generator%s.h5'%(ord(char)))
        j=pd_dict_target['Actual_id'][pd_dict_target['Actual_num']==ord(char)].values[0]
        for i in range(0,number):

            noise = np.random.randn(1 * 1, latent_dim)
            imgs = (generator.predict(noise))
            imgs_reshaped=imgs.reshape(32,32)
            imgs_depr=deprocess(imgs_reshaped)

            if i in split:
                gan_images_test.append(imgs_depr)
                gan_classes_test.append(int(ord(char)))
                gan_classes_tr_test.append(j)
                gan_images_pr_test.append(imgs_reshaped)

            else:
                gan_images_train.append(imgs_depr)
                gan_classes_train.append(int(ord(char)))
                gan_classes_tr_train.append(j)
                gan_images_pr_train.append(imgs_reshaped)


        print('done')
        
    import matplotlib.pyplot as plt
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, axs = plt.subplots() #create images plots and save them
    axs.imshow(gan_images_train[0], cmap='gray')


# In[ ]:




