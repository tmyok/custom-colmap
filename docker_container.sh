#!/bin/sh
# -*- coding: utf-8 -*-

docker run \
    -it \
    --rm \
    --gpus all \
    --shm-size=32g \
    --name custom-colmap \
    --volume $(pwd)/:/home/work/ \
    --workdir /home/work/ \
    tmyok/pytorch:custom-colmap-image