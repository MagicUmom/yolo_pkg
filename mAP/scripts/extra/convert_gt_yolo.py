import sys
import os
import glob
import cv2


def convert_yolo_coordinates_to_voc(CLASSES_FILE, GT_PATH, IMG_PATH, CLASS_RESULT_DIR):
  def count_gt(x_c_n, y_c_n, width_n, height_n, img_width, img_height):
    ## remove normalization given the size of the image
    x_c = float(x_c_n) * img_width
    y_c = float(y_c_n) * img_height
    width = float(width_n) * img_width
    height = float(height_n) * img_height
    ## compute half width and half height
    half_width = width / 2
    half_height = height / 2
    ## compute left, top, right, bottom
    ## in the official VOC challenge the top-left pixel in the image has coordinates (1;1)
    left = int(x_c - half_width) + 1
    top = int(y_c - half_height) + 1
    right = int(x_c + half_width) + 1
    bottom = int(y_c + half_height) + 1
    return left, top, right, bottom

  # read the class_list.txt to a list
  with open(CLASSES_FILE) as f:
    obj_list = f.readlines()
  ## remove whitespace characters like `\n` at the end of each line
    obj_list = [x.strip() for x in obj_list]
  ## e.g. first object in the list
  #print(obj_list[0])

  # change directory to the one with the files to be changed
  # GT_PATH = os.path.join(parent_path, 'input','ground-truth')
  #print(GT_PATH)
  # os.chdir(GT_PATH)

  # old files (YOLO format) will be moved to a new folder (backup/)
  ## create the backup dir if it doesn't exist already
  # if not os.path.exists("backup"):
  #   os.makedirs("backup")

  # create VOC format files
  img_list = [ img for img in os.listdir(IMG_PATH) if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'png' ]
  if len(img_list) == 0:
    print("Error: image not found")
    sys.exit()

  for image in img_list:
    image_name = image.split(".")[0]
    tmp_file = None

    for fname in os.listdir(GT_PATH):
      if fname.startswith(image_name):
        img =  cv2.imread( os.path.join(IMG_PATH, image))
        ## get image width and height
        img_height, img_width = img.shape[:2]
        tmp_file = image_name + ".txt"
        break
    else:
      ## image not found
      print("Error: Ground True File NOT found, corresponding to " + image_name)
      sys.exit()
    # 2. open txt file lines to a list
    with open(os.path.join(GT_PATH, tmp_file)) as f:
      content = f.readlines()
    ## remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    # 3. move old file (YOLO format) to backup
    # os.rename(tmp_file, "backup/" + tmp_file)
    # 4. create new file (VOC format)
    # print(content)
    # print(os.path.join(CLASS_RESULT_DIR, tmp_file))
    with open( os.path.join(CLASS_RESULT_DIR, tmp_file) ,"w") as new_f:
      for line in content:
        ## split a line by spaces.
        ## "c" stands for center and "n" stands for normalized
        obj_id, x_c_n, y_c_n, width_n, height_n = line.split()
        obj_name = obj_list[int(obj_id)]
        # print(obj_name, obj_id)
        left, top, right, bottom = count_gt(x_c_n, y_c_n, width_n, height_n, img_width, img_height)
        ## add new line to file
        #print(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom))
        new_f.write(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')
  print("--- Conversion completed! ---")
