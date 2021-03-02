ARG MEAD=mead2-pytorch-gpu
ARG VERSION=latest

FROM meadml/${MEAD}:${VERSION}

WORKDIR /usr/audio8

COPY . /usr/audio8
RUN apt-get update && apt-get install -y libasound2-dev libsndfile-dev
RUN python3.6 -m pip install soundfile
RUN python3.6 -m pip install editdistance
RUN python3.6 -m pip install scipy
