#!/bin/bash
set -euo pipefail

RUN_TEST_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${RUN_TEST_DIR}/build"
SETUP_SCRIPT="${RUN_TEST_DIR}/../local/setup-adios2-prodm.sh"
MPIRUN=${MPIRUN:-$(command -v mpirun.mpich || command -v mpirun)}
MPI_NP=${MPI_NP:-2}

if [ -f "${SETUP_SCRIPT}" ]; then
    . "${SETUP_SCRIPT}"
fi

mkdir -p "${BUILD_DIR}"

echo "[build]"
"${RUN_TEST_DIR}/compile.sh"

echo "[test_serial]"
"${BUILD_DIR}/test_serial"

echo "[test_single]"
"${BUILD_DIR}/test_single"

echo "[test_small]"
"${BUILD_DIR}/test_small"

echo "[test_mpi]"
"${MPIRUN}" -n "${MPI_NP}" "${BUILD_DIR}/test_mpi"
