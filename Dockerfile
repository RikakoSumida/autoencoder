FROM pytorch/pytorch:latest
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git tmux libopencv-dev wget zip tree libsndfile1
RUN pip install fastprogress tensorboard opencv-python scikit-learn pandas librosa