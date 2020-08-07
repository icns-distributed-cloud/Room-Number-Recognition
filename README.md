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
- Raspberry Pi 4 Model B
- Raspbian Buster
- Python 3.7
- OpenCV 4.2.0
- Tensorflow 1.15.2
- Keras 2.3.1

## Todo
- [ ] Complete the Dockerfile
- [ ] Change implementation from Python to Go.