
from google.colab import drive  # connecting the ggl dirve
drive.mount('/content/drive')

!pip install ultralytics roboflow    # installing roboflow and ultralytics

import ultralytics
ultralytics.checks()  # this part checks if gpu is available

