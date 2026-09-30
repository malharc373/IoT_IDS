#!/usr/bin/env bash
# Runs INSIDE the linux/arm/v7 container. Build tree lives in /work/vol (named volume).
set -euo pipefail
TAG=${ORT_TAG:-v1.23.2}
JOBS=${JOBS:-6}
STAGE=${STAGE:-all}   # mlas | all
cd /work/vol
if [ ! -d onnxruntime ]; then
  git clone --depth 1 --branch "$TAG" --recurse-submodules --shallow-submodules \
      https://github.com/microsoft/onnxruntime.git
fi
cd onnxruntime
git log -1 --format='%H %d'

export CFLAGS="-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard"
export CXXFLAGS="$CFLAGS"

COMMON=(--config MinSizeRel --build_dir /work/vol/build --build_wheel --parallel "$JOBS"
        --skip_tests --compile_no_warning_as_error --allow_running_as_root
        --skip_submodule_sync
        --cmake_generator Ninja
        --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
                              CMAKE_C_FLAGS="$CFLAGS" CMAKE_CXX_FLAGS="$CXXFLAGS")

if [ "$STAGE" = mlas ]; then
  python3 tools/ci_build/build.py "${COMMON[@]}" --update
  cmake --build /work/vol/build/MinSizeRel --target onnxruntime_mlas -j "$JOBS"
  echo "MLAS_OK"
else
  python3 tools/ci_build/build.py "${COMMON[@]}" --update --build
  ls -l /work/vol/build/MinSizeRel/dist/
  mkdir -p /work/vol/out && cp /work/vol/build/MinSizeRel/dist/*.whl /work/vol/out/
  echo "BUILD_OK"
fi
