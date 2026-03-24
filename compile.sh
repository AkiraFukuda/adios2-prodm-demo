#!/bin/bash
set -e

SERIAL_CXX=${SERIAL_CXX:-$(command -v c++ || command -v g++)}
MPI_CXX=${MPI_CXX:-$(command -v mpicxx.mpich || command -v mpicxx)}
SERIAL_CXXFLAGS=$(adios2-config --cxx-flags -s)
SERIAL_LDFLAGS=$(adios2-config --cxx-libs -s)
MPI_CXXFLAGS=$(adios2-config --cxx-flags -m)
MPI_LDFLAGS=$(adios2-config --cxx-libs -m)

$SERIAL_CXX -std=c++11 test_serial.cpp -o build/test_serial ${SERIAL_CXXFLAGS} ${SERIAL_LDFLAGS}
$MPI_CXX -std=c++11 test_single.cpp -o build/test_single ${MPI_CXXFLAGS} ${MPI_LDFLAGS}
$MPI_CXX -std=c++11 test_small.cpp -o build/test_small ${MPI_CXXFLAGS} ${MPI_LDFLAGS}
$MPI_CXX -std=c++11 test_mpi.cpp -o build/test_mpi ${MPI_CXXFLAGS} ${MPI_LDFLAGS}
