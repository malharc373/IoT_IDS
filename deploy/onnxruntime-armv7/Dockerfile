# Native armv7l (armhf) build environment for onnxruntime, run under QEMU user emulation.
# Debian trixie matches Raspberry Pi OS trixie: glibc 2.41, gcc 14.2 (GLIBCXX <= 3.4.33),
# Python 3.13.5, numpy 2.2.4.
FROM --platform=linux/arm/v7 debian:trixie
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates patch \
      python3 python3-dev python3-numpy python3-setuptools python3-wheel \
      python3-packaging python3-pip python3-venv \
      python3-protobuf python3-sympy python3-flatbuffers python3-coloredlogs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY build.sh /work/build.sh
CMD ["bash", "/work/build.sh"]
