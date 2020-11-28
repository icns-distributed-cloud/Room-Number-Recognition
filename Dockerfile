FROM freckie/rpi-opencv-pytorch-python:3.8-buster

WORKDIR /app
RUN git clone https://github.com/icns-distributed-cloud/room-number-recognition

WORKDIR /app/room-number-recognition/app
RUN pip3 install -r requirements.txt
RUN pip3 uninstall pillow && pip3 install pillow
