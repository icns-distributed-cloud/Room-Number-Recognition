FROM arm32v7/python:3.7-slim-buster

LABEL maintainer="Myung-Hyun Kim freckie@frec.kr"

ENV DEVICE_NUMBER 0
ENV PADDING_SIZE 5

RUN mkdir /app
WORKDIR /app

# Install dependencies
RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y wget \
    git \
    pkg-config \
    libhdf5-103 \
    libhdf5-dev \
    libhdf5-serial-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-setuptools \
    python3-scipy

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y python3-h5py

# libjasper for OpenCV
RUN git clone https://github.com/jasperproject/jasper-client.git jasper && \
    chmod +x jasper/jasper.py && \
    pip3 install --upgrade setuptools && \ 
    pip3 install -r jasper/client/requirements.txt

# Install Tensorflow
RUN pip3 install six wheel mock
RUN wget https://github.com/Qengineering/Tensorflow-Raspberry-Pi/raw/master/tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl && \
    pip3 install tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl

# Install Keras
RUN pip3 install cython \
    keras==2.3.1

# Install OpenCV
RUN wget https://github.com/dltpdn/opencv-for-rpi/releases/download/4.2.0_buster_pi3b/opencv4.2.0.deb.tar && \
    tar -xvf opencv4.2.0.deb.tar && \
    apt-get install -y ./OpenCV*.deb

ADD . /app/

CMD [ "python3", "/app/main.py", "--device=$DEVICE_NUMBER", "--padding-size=$PADDING_SIZE" ]
