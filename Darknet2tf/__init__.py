# from . import Darknet2tf
# from . import save_model
def transfer( weights="./data/yolov4.weights", output="./checkpoints/yolov4-416", input_size=416, model="yolov4"):
    return ("python save_model.py --weights=./data/yolov4.weights --output=./checkpoints/yolov4-416 --input_size=416 --model yolov4")