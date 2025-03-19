
from ultralytics import YOLO

import utils


model = YOLO("yolo11n.pt")
metrics = model.val(data="coco128.yaml")

yolo_metrics = utils.yolo_metrics(metrics)
print(yolo_metrics)

size = utils.get_model_size(model)
print('model size: {:.3f}MB'.format(size))
