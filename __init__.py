from yolo_pkg.Train import VOC, YOLO_format

import os
import shutil
import re
import numpy as np
from PIL import Image
from shutil import copyfile
import gdown
import random
import glob

class YOLO():
    def __init__(self):
        self.IMAGES_DIR_PATH = None
        self.LABELS_DIR_PATH = None
        self.WEIGHTS_DIR_PATH = None
                
        self.LOCAL_YOLOS_DIR_PATH = None
        self.LOCAL_CFG_DIR_PATH = None
        
        self.darknet_path = None
        
        self.tmp_imgs = './tmp_images/'
        self.tmp_labels = './tmp_labels/'
        
        self.BASE_PATH = os.path.dirname(os.path.realpath(__file__))
        self.CFG_DIR_PATH = os.path.join( self.BASE_PATH , 'Train/cfg')
        
    def initial(self):

        # 創建資料夾
        if os.path.exists(self.LOCAL_CFG_DIR_PATH):
            shutil.rmtree(self.LOCAL_CFG_DIR_PATH)
        os.makedirs(self.LOCAL_CFG_DIR_PATH, exist_ok=True)

        if os.path.exists(self.tmp_imgs):
            shutil.rmtree(self.tmp_imgs)
        os.makedirs(self.tmp_imgs, exist_ok=True)

        if os.path.exists(self.tmp_labels):
            shutil.rmtree(self.tmp_labels)
        os.makedirs(self.tmp_labels, exist_ok=True)
        
        if os.path.exists(self.LOCAL_YOLOS_DIR_PATH):
            shutil.rmtree(self.LOCAL_YOLOS_DIR_PATH)
        os.makedirs(self.LOCAL_YOLOS_DIR_PATH, exist_ok=True)

        os.makedirs(self.IMAGES_DIR_PATH, exist_ok=True)
        os.makedirs(self.LABELS_DIR_PATH, exist_ok=True)
        os.makedirs(self.WEIGHTS_DIR_PATH, exist_ok=True)
        os.makedirs(self.CFG_DIR_PATH, exist_ok=True)


    def train(self, pretrain_weight = None):

        if pretrain_weight == None:
            gdown.download('https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp', os.path.join(self.BASE_PATH, 'yolov4.conv.137'))
            pretrain_weight = os.path.join(self.BASE_PATH, 'yolov4.conv.137')
        
        
        os.chdir(self.darknet_path)
        cmd = "./darknet detector train {}/obj.data {}/yolov4-custom.cfg {} -dont_show | grep 'avg loss'".format(self.LOCAL_CFG_DIR_PATH, self.LOCAL_CFG_DIR_PATH, pretrain_weight)
        print(cmd)
        return cmd
#         os.system(cmd)
            
    def mkcfg(self):
        self.matching()
        self.Convert_VOC2YOLO_format()
        self.parse_obj_files()
        self.create_train_and_test_files()
        
        files = glob.iglob(os.path.join(self.tmp_imgs, "*.jpg"))
        for file in files:
            if os.path.isfile(file):
                shutil.copy2(file, self.LOCAL_YOLOS_DIR_PATH)
    
    def matching(self):
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

        rename_cp_file_img(self.IMAGES_DIR_PATH, self.tmp_imgs)
        rename_cp_file_xml(self.LABELS_DIR_PATH, self.tmp_labels)
        remove(self.tmp_imgs, self.tmp_labels)
        
        
    # Convert VOC xmls into Yolo's format
    # 原本資料標注的檔案有 POSCAL VOC 及 YOLO兩種格式，我們將原本使用 VOC 格式標記的內容轉換成 YOLO格式
    def Convert_VOC2YOLO_format(self):
        labels = set()
        for path in glob.glob(os.path.join(self.tmp_labels, "*.xml")):
            with open(path, 'r') as f:
                content = f.read()

            # extract label names
            matches = re.findall(r'<name>([\w_]+)<\/name>', content, flags=0)
            labels.update(matches)

        # write label into file`
        with open(os.path.join(self.LOCAL_CFG_DIR_PATH, "obj.names"), 'w') as f:
            f.write("\n".join(labels))

        print('Read in %d labels: %s' % (len(labels), ", ".join(labels)))

    # 查詢 CFG 資料夾內，我們設定了多少的標記檔
    def parse_obj_files(self):
        voc = VOC()
        yolo_format = YOLO_format(os.path.join(self.LOCAL_CFG_DIR_PATH, "obj.names"))

        flag, data = voc.parse(self.tmp_labels)
        flag, data = yolo_format.generate(data)

        flag, data = yolo_format.save(data,
            save_path=self.LOCAL_YOLOS_DIR_PATH,
            img_path=self.tmp_imgs, img_type=".jpg", manipast_path="./")

    # 創建訓練要用到的檔案，其中包含設定yolov4的cfg，例如不同的從10個預測目標變成20個，那cfg檔案就要重新設定
    def create_train_and_test_files(self):
        # fetch label_names
        with open(os.path.join(self.LOCAL_CFG_DIR_PATH, "obj.names"), 'r') as f:
            f_content = f.read()
        label_names = f_content.strip().splitlines()

        # update the cfg file
        with open(os.path.join(self.CFG_DIR_PATH, "yolov4-custom.cfg"), 'r') as f:
            content = f.read()
        with open(os.path.join(self.LOCAL_CFG_DIR_PATH, "yolov4-custom.cfg"), 'w') as f:
            num_max_batches = len(label_names)*2000
            content = content.replace("%NUM_CLASSES%", str(len(label_names)))
            content = content.replace("%NUM_MAX_BATCHES%", str(num_max_batches))
            content = content.replace("%NUM_MAX_BATCHES_80%", str(int(num_max_batches*0.8)))
            content = content.replace("%NUM_MAX_BATCHES_90%", str(int(num_max_batches*0.9)))
            content = content.replace("%NUM_CONVOLUTIONAL_FILTERS%", str((len(label_names)+5)*3))

            f.write(content)

        txt_paths = glob.glob(os.path.join(self.LOCAL_YOLOS_DIR_PATH, "*.txt"))

        random.shuffle(txt_paths)
        num_train_images = int(len(txt_paths)*0.8)

        assert num_train_images>0, "There's no training images in folder %s" % (self.LOCAL_YOLOS_DIR_PATH)

        with open(os.path.join(self.LOCAL_CFG_DIR_PATH, "train.txt"), 'w') as f:
            for path in txt_paths[:num_train_images]:
                f.write("%s/%s\n" % (self.LOCAL_YOLOS_DIR_PATH, os.path.basename(path).replace(".txt", ".jpg")))
        with open(os.path.join(self.LOCAL_CFG_DIR_PATH, "test.txt"), 'w') as f:
            for path in txt_paths[num_train_images:]:
                f.write("%s/%s\n" % (self.LOCAL_YOLOS_DIR_PATH, os.path.basename(path).replace(".txt", ".jpg")))

        # create obj
        with open(os.path.join(self.LOCAL_CFG_DIR_PATH, "obj.data"), 'w') as f:
            f.write("classes=%d\n" % (len(label_names)))
            f.write("train=%s/train.txt\n" % (self.LOCAL_CFG_DIR_PATH))
            f.write("valid=%s/test.txt\n" % (self.LOCAL_CFG_DIR_PATH))
            f.write("names=%s/obj.names\n" % (self.LOCAL_CFG_DIR_PATH))
            f.write("backup=%s\n" % (self.WEIGHTS_DIR_PATH))
