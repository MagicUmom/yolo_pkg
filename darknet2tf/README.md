# Fork From [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) @2021/05/10
- For AIF 內部學員使用

### Prerequisites
* Tensorflow 2.3.0rc0
* OpenCV 4

# tensorflow-yolov4-tflite
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

YOLOv4, YOLOv4-tiny Implemented in Tensorflow 2.0. 
Convert YOLO v4, YOLOv3, YOLO tiny .weights to .pb, .tflite and trt format for tensorflow, tensorflow lite, tensorRT.

Download yolov4.weights file: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT


# Quick Start

## 使用官方的權重
```bash
# 下載 yolov4.weights 權重
# 使用 gdown 下載，如果沒有此套件請使用 pip install gdown
gdown --id 1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT -O ./data/

# 將 darknet權重 轉換成 tf2的權重
python save_model.py --weights=./data/yolov4.weights --output=./checkpoints/yolov4-416 --input_size=416 --model yolov4

# 使用預測
python detect.py --weights=./checkpoints/yolov4-416 --image=./pred_data --output=./pred_result

```


## 使用自己在 Darknet 上訓練的權重 
```bash
# 將原先的 obj.names 複製到 data/classes/ 底下，如果使用課堂上的步驟，應該會在 yolov4_train/cfg/obj.names
# ** -----記得要自己改路徑!! ----- *
cp ../yolov4_train/cfg/obj.names ./data/classes/
# ** -----記得要自己改路徑!! ----- *


# 將原先訓練好的 Darknet權重檔 複製過來這個資料夾下
# ** -----記得要自己改路徑!! ----- *
cp ../yolov4_train/weights_after_train/yolov4-custom_last.weights ./
# ** -----記得要自己改路徑!! ----- *


# 將自己訓練的 darknet權重 轉換成 tf2的權重
python save_model.py --weights=./yolov4-custom_last.weights --output=./checkpoints/yolov4-custom_last --classes=./data/classes/obj.names --input_size=416 --model=yolov4

# 使用預測，需要預測的圖片放進去 pred_data，結果會出現在 pred_result
python detect.py --weights=./checkpoints/yolov4-custom_last --classes=./data/classes/obj.names --image=./pred_data --output=./pred_result

```

# 注意事項

  * 注意要使用自己的訓練的權重就要把 *權重* 及 *obj.names* 都當參數餵進去模型轉換的參數
  * 預測的圖片請放到 pred_data，結果會出現在 pred_result
  * 預測的圖片在程式碼內吃資料夾下附檔名為 .jpg 及 .png (小寫)，如果需要修改，請自行到 detect.py 下修改
  * 要得到預測框可以到 detect.py 及 core/utils內的 draw_bbox 內找尋需要的地方做修改

# References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  
   My project is inspired by these previous fantastic YOLOv3 implementations:
  * [Yolov3 tensorflow](https://github.com/YunYang1994/tensorflow-yolov3)
  * [Yolov3 tf2](https://github.com/zzh8829/yolov3-tf2)
