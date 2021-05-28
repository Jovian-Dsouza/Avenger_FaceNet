FROM nvidia/cuda:10.2-base-ubuntu18.04

LABEL maintainer="Jovian Dsouza <dsouzajovian123@gmail.com>"

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one && \
    apt-get clean && rm -rf /var/lib/apt/lists/* 
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
ENV PATH="/root/miniconda3/bin:${PATH}"

RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

RUN conda install --yes numpy scikit-learn tqdm facenet-pytorch \
                       pillow 

COPY models /models
COPY openface.pth /openface.pth
COPY loadOpenFace.py /loadOpenFace.py
COPY SpatialCrossMapLRN_temp.py /SpatialCrossMapLRN_temp.py

COPY train.py /train.py
COPY test.py /test.py

# docker run -v <our local dataset path>:/dataset jovain19/pytorch:latest bash