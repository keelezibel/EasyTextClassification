FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV TZ='Asia/Singapore' \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y sox libsox-fmt-mp3 libsndfile1 ffmpeg python3 python3-pip git

ARG APPDIR=/app
WORKDIR $APPDIR

ARG PIP_TRUSTED_HOST="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

COPY ./requirements.txt ./
RUN pip install --verbose -r requirements.txt --no-cache-dir

# copy application source
COPY . $APPDIR/

# Run the application:
ENTRYPOINT [ "/bin/sh" ]
