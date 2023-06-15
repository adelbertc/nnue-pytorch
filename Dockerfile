FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND="noninteractive"
ENV PYTHONUNBUFFERED=1
RUN apt update && apt install -y \
  build-essential \
  cmake \
  curl \
  ffmpeg \
  gcc \
  libsm6 \
  libxext6 \
  libgl1 \
  python3-pip \
  vim \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir --upgrade pip

# Ensure CUDART libraries can be found are on the library path.
ARG CUDART_PATH=/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=${CUDART_PATH}:${LD_LIBRARY_PATH}
# RUN ln -s ${CUDART_PATH}/libcudart.so.12 ${CUDART_PATH}/libcudart.so
# RUN ln -s ${CUDART_PATH}/libnvrtc.so.12 ${CUDART_PATH}/libnvrtc.so
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /service

# Install server wheel.
COPY octoai_sdk-0.1.0-py3-none-any.whl  .
RUN python3 -m pip install octoai_sdk-0.1.0-py3-none-any.whl

# TODO(akm): Remove this after CLI depends on inference SDK.
RUN python3 -m pip install httpx

# Install Poetry
# https://python-poetry.org/
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy remaining template contents.
COPY app .

# Workaround for .dockerignore not being respected
RUN rm -r build/
RUN rm -r .venv/

RUN sh compile_data_loader.bat
RUN /root/.local/bin/poetry lock
RUN /root/.local/bin/poetry install
CMD ["/root/.local/bin/poetry", "run", "python3", "-m", "octoai.server", "--service-module", "service", "run"]
# /root/.local/bin/poetry run python3 -m octoai.server --service-module service run