FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

SHELL ["/bin/bash", "-c"]

RUN apt update

########################
# Install dependencies #
########################

RUN DEBIAN_FRONTEND=noninteractive TZ=America/Los_Angeles apt install -y tzdata

RUN apt install -y --no-install-recommends \
	build-essential \
	cmake \
	curl \
	gcc \
	git \
	python3 \
	python3-dev \
	python3-distutils \
	python3-tk \
	unzip \
	vim

# Install Node.js 18 from NodeSource
# https://github.com/nodesource/distributions#deb
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
	apt install -y --no-install-recommends nodejs

RUN useradd -ms /bin/bash -G sudo vida

# Install Bun JS runtime
# https://bun.sh/
RUN curl -fsSL https://bun.sh/install | bash
# RUN . ~/.bashrc

# Install Poetry
# https://python-poetry.org/
RUN curl -sSL https://install.python-poetry.org | python3 -
# ENV PATH="~/.local/bin:$PATH"

########################
# Clone vida
########################

WORKDIR /runtime
RUN git clone --branch profiler https://github.com/adelbertc/nnue-pytorch.git vida

WORKDIR /runtime/vida

########################
# Build vida
########################

RUN echo "export PATH=~/.bun/bin:~/.local/bin:$PATH" >> ~/.bashrc
RUN cat ~/.bashrc

RUN sh compile_data_loader.bat
RUN ~/.local/bin/poetry install
RUN PATH=~/.bun/bin:$PATH ~/.local/bin/poetry run pc init

EXPOSE 3000
EXPOSE 8000

CMD ["/root/.local/bin/poetry", "run", "pc", "run"]

