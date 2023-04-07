#!/bin/sh

docker run --gpus all --runtime nvidia --rm -p 3000:3000 -p 8000:8000 adelbertc/vida:latest
