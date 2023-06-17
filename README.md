# ByteTrack with YOLOv5 (integrating YOLOv5 detector with Bytetrack)

## Installation
### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```


# Usage example 
cd <BYTETRACK-HOME>
python tools/yolo5.py -f exps/yolox_s.py -c pretrained/yolov5s.pt --fuse  --num_classes 3
