# -*- coding: utf-8 -*-
"""
Created on Thu May 28 22:20:44 2020

@author: toshiba
"""
import csv,random,os,cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import img_as_ubyte

###Function to read the given csv######
def read_csv(path,lables):
    inactive_file_name=[]
    active_file_name=[]
    inactive_file_lables_dict={}
    with open(path,newline='') as csvfile:
        csvreader=csv.DictReader(csvfile, delimiter=";")
        for i in csvreader:
            if(i['inactive']=='1'):
                inactive_file_name.append(i['filename'])
                label_list=[]
                label_list.extend([i['poly_wafer'],i['crack'],i['inactive']])
                inactive_file_lables_dict.update({i['filename']:label_list})
                
            else:
                active_file_name.append(i['filename'])
    return inactive_file_lables_dict,active_file_name

######function to see the image####
def show_image(image):
    plt.imshow(image)
    plt.show()
    return 1

#####Augmenting functions######
def flipped(image):
    return tf.image.flip_left_right(image)

def saturation(image):
    return tf.image.adjust_saturation(image, 3)

def brightness(image):
    return tf.image.adjust_brightness(image, 0.2)

transformation_dict={"flipped":flipped,
                "saturation":saturation,
                "brightness":brightness
                }

        

path=r'C:\Users\toshiba\Desktop\DMML\DMML2\challenge\src_toimplement\data\train.csv'
labels=['filename','poly_wafer','crack','inactive']
inactive_file_lables_dict,active_file_name=read_csv(path,labels)
diff=len(active_file_name)-len(inactive_file_lables_dict)

i=0
while(i<diff):
    img_file_name_path=random.choice(list(inactive_file_lables_dict.keys()))
    img_file_name=img_file_name_path.replace('images/','')
    i=i+1
    
    path_data=r'C:\Users\toshiba\Desktop\DMML\DMML2\challenge\src_toimplement\data\images'
    image=cv2.imread(os.path.join(path_data,img_file_name))  ##read the file from local directory

    for j in range(len(transformation_dict)):
        transformed_image=transformation_dict[random.choice(list(transformation_dict.keys()))](image)
        image=transformed_image
        
        
    path_transformed_image=r'C:\Users\toshiba\Desktop\DMML\DMML2\challenge\src_toimplement\data\images'
    file_name="tr_img_"+str(i)+'_'+img_file_name
    complete_file_name=os.path.join(path_transformed_image,file_name)
    print(complete_file_name)
    #print(complete_file_name)
    image1=img_as_ubyte(image)
    cv2.imwrite(complete_file_name,image1)  ## writing the file into a specified directory
    
    file_name_to_upload=f"images/{file_name}"
    fields=[file_name_to_upload,
            inactive_file_lables_dict[img_file_name_path][0],
            inactive_file_lables_dict[img_file_name_path][1],
            inactive_file_lables_dict[img_file_name_path][2]]
        
    name=r"C:\Users\toshiba\Desktop\DMML\DMML2\challenge\src_toimplement\data\train.csv"
    with open(name, 'a',newline='') as f:
        writer = csv.writer(f, delimiter =';')
        writer.writerow(fields) ##updating the given csv file