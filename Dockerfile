FROM tensorflow/tensorflow:2.3.0rc0-gpu-jupyter

COPY requirements-gpu.txt /tmp/requirements.txt

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN pip install -r /tmp/requirements.txt
RUN rm /tmp/requirements.txt