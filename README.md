# ONNX-CSFlow-Optical-Flow-Estimation

Python scripts for performing optical flow estimation using the CSFlow model in ONNX.

- Original Video [https://youtu.be/3wdsE1UgP6k](https://youtu.be/3wdsE1UgP6k)

  ![gif](https://user-images.githubusercontent.com/53618876/178768743-59ee104d-af7e-4a91-8b4a-551818a030ef.gif)


- Estimation on the KITTI dataset images from OpticalFlowToolkit example [frame1](https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame1.png) and [frame2](https://github.com/liruoteng/OpticalFlowToolkit/blob/master/data/example/KITTI/frame2.png)

  ![kitti](https://user-images.githubusercontent.com/53618876/178753729-cf335065-1b6d-4e81-9d03-dd77078b294e.jpg)


## Requirements
- opencv-python
- onnxruntime or onnxruntime-gpu

### for image_flow_estimation.py
```bash
python3 -m pip install imread-from-url
```

### for video_flow_estimation.py
```bash
python3 -m pip install youtube_dl
python3 -m pip install git+https://github.com/zizo-pro/pafy@b8976f22c19e4ab5515cacbfae0a3970370c102b
```

## ONNX model
Download the models from [PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/272_CSFlow) and save them into the **[models](https://github.com/fateshelled/ONNX-CSFlow-Optical-Flow-Estimation/tree/main/models)** folder.

## DEMO
- Image inference:
```bash
python3 image_flow_estimation.py
```

- Video inference:
```bash
python3 video_flow_estimation.py
```

- Webcam inference:
```bash
python webcam_flow_estimation.py
```

## References:

- Base Code: https://github.com/ibaiGorordo/ONNX-RAFT-Optical-Flow-Estimation
- CSFlow model: https://github.com/MasterHow/CSFlow
- PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
- OpticalFlowToolkit toolkit: https://github.com/liruoteng/OpticalFlowToolkit
- Original paper: https://arxiv.org/abs/2202.00909
