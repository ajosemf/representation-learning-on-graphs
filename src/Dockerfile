FROM tensorflow/tensorflow:1.5.0-py3
LABEL author="ajosemf@gmail.com"

RUN apt-get update && apt-get install -y \
    cmake \
    wget \
    && apt clean \
    && cd /home \
    && mkdir rep_learning \
    && cd rep_learning \
    && wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz \
    && tar xzf metis-5.1.0.tar.gz \
    && cd metis-5.1.0 \
    && make config shared=1 prefix=~/.local/ \
    && make install \
    && rm -rf metis-5.1.0

ENV METIS_DLL=/root/.local/lib/libmetis.so

WORKDIR /home/rep_learning
