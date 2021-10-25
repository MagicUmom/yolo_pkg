from posixpath import abspath
from yolo_pkg.Train import VOC, YOLO_format

import tensorflow as tf
from yolo_pkg.Darknet2tf.core.yolov4 import YOLO_BASE, decode, filter_boxes
from yolo_pkg.Darknet2tf.core import utils

from yolo_pkg.mAP import mAP
from yolo_pkg.mAP.scripts.extra.convert_gt_yolo import convert_yolo_coordinates_to_voc

import os
import shutil
import re
import numpy as np
from PIL import Image
from shutil import copyfile
import gdown
import random
import glob
from easydict import EasyDict
import xml.etree.cElementTree as ET
from xml.dom import minidom

class YOLO():
    def __init__(self):
        
        self.FLAGS = dict()

        self.IMAGES_DIR_PATH = None
        self.LABELS_DIR_PATH = None
        self.WEIGHTS_DIR_PATH = None
                
        self.LOCAL_YOLOS_DIR_PATH = None
        self.LOCAL_CFG_DIR_PATH = None
        
        
        
        self.tmp_imgs = './tmp_images/'
        self.tmp_labels = './tmp_labels/'
        
        self.BASE_PATH = os.path.dirname(os.path.realpath(__file__))
        self.CFG_DIR_PATH = os.path.join( self.BASE_PATH , 'Train/cfg')
        
        self.darknet_path = os.path.join( self.BASE_PATH, "darknet_TWCC" )

    def initial(self):

        # 創建資料夾
        if self.LOCAL_CFG_DIR_PATH != None:
            self.LOCAL_CFG_DIR_PATH = os.path.abspath(self.LOCAL_CFG_DIR_PATH)
            if os.path.exists(self.LOCAL_CFG_DIR_PATH):
                shutil.rmtree(self.LOCAL_CFG_DIR_PATH)
            os.makedirs(self.LOCAL_CFG_DIR_PATH, exist_ok=True)

        if self.tmp_imgs != None:
            self.tmp_imgs = os.path.abspath(self.tmp_imgs)
            if os.path.exists(self.tmp_imgs):
                shutil.rmtree(self.tmp_imgs)
            os.makedirs(self.tmp_imgs, exist_ok=True)

        if self.tmp_labels != None:
            self.tmp_labels = os.path.abspath(self.tmp_labels)
            if os.path.exists(self.tmp_labels):
                shutil.rmtree(self.tmp_labels)
            os.makedirs(self.tmp_labels, exist_ok=True)
        
        if self.LOCAL_YOLOS_DIR_PATH != None:
            self.LOCAL_YOLOS_DIR_PATH = os.path.abspath(self.LOCAL_YOLOS_DIR_PATH)
            if os.path.exists(self.LOCAL_YOLOS_DIR_PATH):
                shutil.rmtree(self.LOCAL_YOLOS_DIR_PATH)
            os.makedirs(self.LOCAL_YOLOS_DIR_PATH, exist_ok=True)


        if self.IMAGES_DIR_PATH != None:
            self.IMAGES_DIR_PATH = os.path.abspath(self.IMAGES_DIR_PATH)
            os.makedirs(self.IMAGES_DIR_PATH, exist_ok=True)
        
        if self.LABELS_DIR_PATH != None:
            self.LABELS_DIR_PATH = os.path.abspath(self.LABELS_DIR_PATH)
            os.makedirs(self.LABELS_DIR_PATH, exist_ok=True)

        if self.WEIGHTS_DIR_PATH != None:
            self.WEIGHTS_DIR_PATH = os.path.abspath(self.WEIGHTS_DIR_PATH)
            os.makedirs(self.WEIGHTS_DIR_PATH, exist_ok=True)

        if self.CFG_DIR_PATH != None:
            self.CFG_DIR_PATH = os.path.abspath(self.CFG_DIR_PATH)
            os.makedirs(self.CFG_DIR_PATH, exist_ok=True)

    def train(self, pretrain_weight = True):

        if pretrain_weight == True:
            gdown.download('https://drive.google.com/uc?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp', os.path.join(self.BASE_PATH, 'yolov4.conv.137'), quiet=False)
            pretrain_weight = os.path.join(self.BASE_PATH, 'yolov4.conv.137')

        elif pretrain_weight == None:
            pretrain_weight = ''
        else:
            pretrain_weight = os.path.abspath(pretrain_weight)
        

        os.chdir(self.darknet_path)
        cmd = "./darknet detector train {}/obj.data {}/yolov4-custom.cfg {} -dont_show | grep 'avg loss'".format(self.LOCAL_CFG_DIR_PATH, self.LOCAL_CFG_DIR_PATH, pretrain_weight)
        print(cmd)
        return cmd
            
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
        paths = glob.glob(os.path.join(self.tmp_labels, "*.xml"))
        paths.sort()
        for path in paths:
            with open(path, 'r') as f:
                content = f.read()

            # extract label names
            matches = re.findall(r'<name>([\w_\-\!\*]+)<\/name>', content, flags=0)
            matches.sort()
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


    # ------- FOR Darknet2TF START ---------#
    def set_conf(self, WEIGHTS,OUTPUT,INPUT_SIZE,MODEL_TYPE,CLASSES_FILE):
        # WEIGHTS = './data/yolov4.weights'
        # OUTPUT = './checkpoints/yolov4-416'
        # INPUT_SIZE = 416
        # MODEL_TYPE = 'yolov4'
        # CLASSES_FILE = "./data/classes/coco.names"

        self.FLAGS["weights"] =  WEIGHTS            # path to weights file
        self.FLAGS["output"] = OUTPUT               # path to output
        self.FLAGS["tiny"] = bool(False)            # is yolo-tiny or not
        self.FLAGS["input_size"] = int(INPUT_SIZE)  # define input size of export model
        self.FLAGS["framework"] = "tf"              # define what framework do you want to convert (tf, trt, tflite)
        self.FLAGS["model"] = MODEL_TYPE            # yolov3 or yolov4
        self.FLAGS["classes"] = CLASSES_FILE        # classes defined path. eg: coco.names
        self.FLAGS["score_thres"] = float(0.2)      # define score threshold

    def save_tf(self, WEIGHTS, OUTPUT, INPUT_SIZE=416, MODEL_TYPE="yolov4", CLASSES_FILE = ''):
        if CLASSES_FILE == '': 
            CLASSES_FILE = os.path.join(self.BASE_PATH,"Darknet2tf/data/classes/coco.names")
        else: 
            CLASSES_FILE = os.path.abspath(CLASSES_FILE)

        if WEIGHTS != None:
            WEIGHTS = os.path.abspath(WEIGHTS)
        else:
            assert("You must set WEIGHTS path !")

        if OUTPUT != None:
            OUTPUT = os.path.abspath(OUTPUT)
        else:
            assert("You must set OUTPUT path !")
        
        self.set_conf(WEIGHTS,OUTPUT,INPUT_SIZE,MODEL_TYPE,CLASSES_FILE)

        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.FLAGS)

        input_layer = tf.keras.layers.Input([self.FLAGS["input_size"], self.FLAGS["input_size"], 3])
        feature_maps = YOLO_BASE(input_layer, NUM_CLASS, self.FLAGS["model"], self.FLAGS["tiny"])
        bbox_tensors = []
        prob_tensors = []
        if self.FLAGS["tiny"]:
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    output_tensors = decode(fm, self.FLAGS["input_size"] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.FLAGS["framework"])
                else:
                    output_tensors = decode(fm, self.FLAGS["input_size"] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.FLAGS["framework"])
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
        else:
            for i, fm in enumerate(feature_maps):
                if i == 0:
                    output_tensors = decode(fm, self.FLAGS["input_size"] // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.FLAGS["framework"])
                elif i == 1:
                    output_tensors = decode(fm, self.FLAGS["input_size"] // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.FLAGS["framework"])
                else:
                    output_tensors = decode(fm, self.FLAGS["input_size"] // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, self.FLAGS["framework"])
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
        pred_bbox = tf.concat(bbox_tensors, axis=1)
        pred_prob = tf.concat(prob_tensors, axis=1)
        if self.FLAGS["framework"] == 'tflite':
            pred = (pred_bbox, pred_prob)
        else:
            boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=self.FLAGS["score_thres"], input_shape=tf.constant([self.FLAGS["input_size"], self.FLAGS["input_size"]]))
            pred = tf.concat([boxes, pred_conf], axis=-1)

        tf.compat.v1.reset_default_graph() # 確保model是乾淨的狀態
        model = tf.keras.Model(input_layer, pred)
        utils.load_weights(model, self.FLAGS["weights"], self.FLAGS["model"], self.FLAGS["tiny"])
        model.summary()
        model.save(self.FLAGS["output"])

    # ------- FOR Darknet2TF END ---------#

    # ------- FOR YOLO Predict in START -----#

    def detect(self, WEIGHTS, image_dir = "yolo_pkg/example_imgs", output_dir = "yolo_pkg/results", CLASSES_FILE = "", iou = 0.45, score = 0.25, count_mAP = False, class_result = "map_file/pred_file" , true_label_path = None, gt_file = "map_file/GT_file", VOC_output = False, VOC_path = "VOC_result"):

        arg = EasyDict()
        arg.framework   = 'tf'                          # (tf, tflite, trt)
        arg.weights     = WEIGHTS                       # path to weights file
        arg.size        = 416                           # resize images to
        arg.tiny        = False                         # yolo or yolo-tiny
        arg.model       = 'yolov4'                      # yolov3 or yolov4
        arg.image       = image_dir                     # path to input image dir
        arg.output      = output_dir                    # path to output image dir
        arg.iou         = iou                           # iou threshold
        arg.score       = score                         # score threshold
        if CLASSES_FILE == '': 
            CLASSES_FILE = os.path.join(self.BASE_PATH,"Darknet2tf/data/classes/coco.names")
        arg.classes     = CLASSES_FILE                  # classes defined path. eg: coco.names
        if count_mAP == True:
            assert true_label_path != None , "要計算mAP需要給Ground True label 檔案的路徑"
            arg.true_label_path = true_label_path
            arg.count_mAP = count_mAP
            arg.class_result = class_result
            if not os.path.isdir(class_result):
                os.makedirs(class_result)
            if not os.path.isdir(gt_file):
                os.makedirs(gt_file)
        else:
            arg.count_mAP = False
        
        arg.VOC_output = VOC_output
        arg.VOC_path = VOC_path

        if VOC_output:
            if not os.path.isdir(arg.VOC_path):
                os.makedirs(arg.VOC_path)


        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        import cv2

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(arg)
        input_size = arg.size
        image_dir_path = arg.image

        print("loading Model ...")
        # saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        # infer = saved_model_loaded.signatures['serving_default']
        infer = tf.keras.models.load_model(arg.weights, compile=False)

        imgs = os.listdir(image_dir_path)
        # out_list = []
        if not os.path.isdir(arg.output):
            os.mkdir(arg.output)
        
        for img in imgs:
            if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'png' :
                print(img)
        
                original_image = cv2.imread(image_dir_path + "/" + img)
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

                # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
                image_data = cv2.resize(original_image, (input_size, input_size))
                image_data = image_data / 255.
                # image_data = image_data[np.newaxis, ...].astype(np.float32)

                images_data = []
                for i in range(1):
                    images_data.append(image_data)
                images_data = np.asarray(images_data).astype(np.float32)

                batch_data = tf.constant(images_data)
                # pred_bbox = infer(batch_data)
                pred_bbox = infer.predict(batch_data)
                # print(type(pred_bbox))
                # print(pred_bbox.shape)
                # print(pred_bbox)
                # for key, value in pred_bbox.items():
                    # boxes = value[:, :, 0:4]
                    # pred_conf = value[:, :, 4:]

                boxes = pred_bbox[:,:,0:4]
                pred_conf = pred_bbox[:,:,4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=arg.iou,
                    score_threshold=arg.score
                )
                pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
                image , outputfile = utils.draw_bbox(original_image, pred_bbox, arg.classes)

                # image = utils.draw_bbox(image_data*255, pred_bbox)
                image = Image.fromarray(image.astype(np.uint8))
                # image.show()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
                cv2.imwrite( os.path.join(arg.output , img), image)

                if arg.VOC_output:
                    height , width , depth = image.shape
                    self.create_VOC_file(arg.VOC_path, img.split(".")[0], width, height, depth, outputfile)
                                            

                if arg.count_mAP :
                    with open( os.path.join( arg.class_result, img.split(".")[0] + ".txt") , "w" , encoding='UTF-8') as f :
                        for line in outputfile:
                            f.write(" ".join(line))
                            f.write("\n")
                        f.close()

        if arg.count_mAP :
            print("--- convert yolo to voc format ---")
            convert_yolo_coordinates_to_voc(arg.classes, arg.true_label_path, arg.image, gt_file)
            print("--- mAP : ---")
            MAP = mAP(  gt_file, arg.class_result, arg.image)
            MAP.run()


    # ------- FOR YOLO Predict in TF END-----#
    def formatter(self, elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="    ")

    def create_root(self, filename, width, height, depth):
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "{}".format(filename)
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        return root


    def create_object_annotation(self, root, voc_labels):
        for voc_label in voc_labels:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = voc_label[0]
            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(voc_label[2])
            ET.SubElement(bbox, "ymin").text = str(voc_label[3])
            ET.SubElement(bbox, "xmax").text = str(voc_label[4])
            ET.SubElement(bbox, "ymax").text = str(voc_label[5])
        return root


    def create_VOC_file(self, VOC_path, filename, width, height, depth, voc_labels):
        root = self.create_root(filename, width, height, depth)
        root = self.create_object_annotation(root, voc_labels)
        with open("{}/{}.xml".format(VOC_path, filename), "w") as f:
            f.write(self.formatter(root))
            f.close()
