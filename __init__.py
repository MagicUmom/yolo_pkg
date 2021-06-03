import train
import predict
import darknet2tf

import os
import shutil
import re
import numpy as np
from PIL import Image
from shutil import copyfile
import gdown


def initial():
    # 創建資料夾
    if os.path.exists(LOCAL_CFG_DIR_PATH):
        shutil.rmtree(LOCAL_CFG_DIR_PATH)
    os.makedirs(LOCAL_CFG_DIR_PATH, exist_ok=True)

    if os.path.exists(LOCAL_YOLOS_DIR_PATH):
        shutil.rmtree(LOCAL_YOLOS_DIR_PATH)
    os.makedirs(LOCAL_YOLOS_DIR_PATH, exist_ok=True)

    os.makedirs(IMAGES_DIR_PATH, exist_ok=True)
    os.makedirs(LABELS_DIR_PATH, exist_ok=True)
    os.makedirs(WEIGHTS_DIR_PATH, exist_ok=True)
    os.makedirs(CFG_DIR_PATH, exist_ok=True)


def train(pretrain_weight = None):
    matching()
    Convert_VOC2YOLO_format()
    parse_obj_files()
    create_train_and_test_files()
    # !cp {IMAGES_DIR_PATH}/*.jpg {LOCAL_YOLOS_DIR_PATH}
    
    if pretrain_weight == None:
        gdown.download('https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp', 'yolov4.conv.137')
        pretrain_weight = 'yolov4.conv.137'
        
    
        
    
def matching():
    def rename_cp_file_img(src_path, dst_path):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for f in os.listdir(src_path):
            try:
                split = re.split('\.|\s', f)
                if split[-1] == 'jpg' or split[-1] == 'jpeg' or split[-1] == 'JPG' or split[-1] == 'JPEG':
                    split[-1] = 'jpg'
                    new_name = '.'.join(['_'.join(split[:-1]), split[-1]])
                    copyfile( os.path.join(src_path, f), os.path.join(dst_path,new_name))

                elif split[-1] == 'png':
                    split[-1] = 'jpg'
                    new_name = '.'.join(['_'.join(split[:-1]), split[-1]])
                    im = Image.open(os.path.join(src_path, f))
                    im.convert('RGB').save( os.path.join(dst_path,new_name) ,"JPEG") #this converts png image as jpeg

                else:
                    print("未知/未定義副檔名: ", f)
            except:
                print("something error:" ,f)

    def rename_cp_file_xml(src_path, dst_path):
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        for f in os.listdir(src_path):
            try:
                split = re.split('\.|\s', f)
                new_name = '.'.join(['_'.join(split[:-1]), split[-1]])

                open( os.path.join(dst_path,new_name) , 'w').writelines([ line.rstrip('\r') for line in open(os.path.join(src_path, f))])

    #             copyfile( os.path.join(src_path, f), os.path.join(dst_path,new_name))

            except Exception as e: 
                print(e)
                print("something error:" ,f)
    
    def remove(image_path, label_path):
        image_name = []
        for i in os.listdir(image_path):
            image_name.append(i.split('.')[0])

        label_name = []
        for i in os.listdir(label_path):
            label_name.append(i.split('.')[0])

        image_name = np.array(image_name)
        label_name = np.array(label_name)

        image_match = []
        label_match = [] 
        for i, img in enumerate(image_name):
            for l, label in enumerate(label_name):
                if img == label:
                    image_match.append(i)
                    label_match.append(l)

        image_mismatch = np.setdiff1d(np.arange(len(image_name)), image_match)
        label_mismatch = np.setdiff1d(np.arange(len(label_name)), label_match)

        for file in np.array(os.listdir(image_path))[image_mismatch]:
            print('image file mismatch, remove name:', file)
            os.remove(os.path.join(image_path, file))


        for file in np.array(os.listdir(label_path))[label_mismatch]:
            print('label file mismatch, remove name:', file)
            os.remove(os.path.join(label_path, file))
    
    rename_cp_file_img(IMAGES_DIR_PATH, 'tmp_new_images/')
    rename_cp_file_xml(LABELS_DIR_PATH, 'tmp_new_labels/')
    remove('tmp_new_images/', 'tmp_new_labels/')