#!/usr/bin/env bash
set -ex

CUDA_VERSION_MAJOR_MINOR="10.2"

docker build \
	--build-arg CUDA_VERSION_MAJOR_MINOR=${CUDA_VERSION_MAJOR_MINOR} \
	-t "${D2HUB_IMAGE}" . -f docker/Dockerfile

