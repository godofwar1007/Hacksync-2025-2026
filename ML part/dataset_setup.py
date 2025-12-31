
from google.colab import drive  # connecting the ggl dirve
drive.mount('/content/drive')

!pip install ultralytics roboflow    # installing roboflow and ultralytics

import ultralytics
ultralytics.checks()  # this part checks if gpu is available


----

!nvidia-smi  # shows current gpu setup and configs 

----

!yolo task=detect mode=train model=yolov8n.pt data={dataset.location}/data.yaml epochs=50 imgsz=640 plots=True # training the model and the hyperparams.

