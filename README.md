# Room-Number-Recognition
Room number recognition service for *indoor self-driving car*s. This service analyzes all contours in every single frame, finds the *Room Number Plate*, and extracts the number from the plate.

## Branches
- `master`: stable version for release (with tag named like v1.7.4)
- `develop`: branch for development
- `hotfix`: fix bugs occurred at a stable version

## Training
### Checker Model

### SVHN Model
SVHN Model is a CNN model based on [YOLOv5-small](https://github.com/ultralytics/yolov5) model, which is trained with [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)(format 2).
For more information, visit this repository: [icns-distributed-cloud/YOLOv5-SVHN](https://github.com/icns-distributed-cloud/YOLOv5-SVHN)

## Project Structure
```
/
├── docs/
├── app/
└── README.md 
```
- `docs`: documentation files
- `app`: main service

## Environment
- Training
    - CentOS 7.5
    - Nvidia GTX 1080 Ti * 8ea
    - Python 3.8
    - PyTorch 1.7.0
- Service
    - Raspberry Pi 4 Model B
    - Raspbian Buster
    - Python 3.8
    - OpenCV 4.4.0

## Quickstart
```
$ git clone https://github.com/icns-distributed-cloud/Room-Number-Recognition
$ cd Room-Number-Recognition/app
$ pip3 install -r requirements.txt
$ python3 main.py --config=config.json
```