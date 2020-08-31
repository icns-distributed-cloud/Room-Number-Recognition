# Room-Number-Recognition
Room number recognition service for *indoor self-driving car*s. This service analyzes all contours in every single frame, finds the *Room Number Plate*, and extracts the number from the plate.

**under development**

## Branches
- `master`: stable version for release (with tag named like v1.7.4)
- `develop`: branch for development
- `feature`: development for new feature
- `hotfix`: fix bugs occurred at a stable version

## Project Structure
```
.
├── docs/
├── app/
├── train/
│   └── datasets/
└── README.md 
```
- `docs`: documentation files
- `app`: main service
- `train`: directory for training models
- `train/datasets`: datasets for training

## Environment
- Training
    - CentOS 7.5
    - Nvidia GTX 1080 Ti
    - Python 3.7
    - Tensorflow 1.15.2
    - Keras 2.3.1
- Service
    - Raspberry Pi 4 Model B
    - Raspbian Buster
    - Go 1.15.0
    - OpenCV 4.2.0

## Todo
- [v] Complete the Dockerfile
- [v] Change implementation from Python to Go.