#include <mpi.h>
#include <adios2.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>  // std::getenv
#include <string>
#include <unistd.h>

namespace
{
bool FileExists(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    return static_cast<bool>(ifs);
}

std::string DefaultDatasetPath()
{
    std::string home = std::getenv("HOME") ? std::getenv("HOME") : ".";
    const std::string legacyHomeData = home + "/data_example/Uf48.bin.f32";
    if (FileExists(legacyHomeData))
    {
        return legacyHomeData;
    }

    char exePath[4096] = {};
    const ssize_t len = readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
    if (len > 0)
    {
        std::string exeDir(exePath, static_cast<std::size_t>(len));
        const std::size_t slash = exeDir.find_last_of('/');
        if (slash != std::string::npos)
        {
            exeDir.resize(slash);
            const std::string bundledNear = exeDir + "/../datasets-ProDM/Uf48.100x500x500.bin.f32";
            if (FileExists(bundledNear))
            {
                return bundledNear;
            }

            const std::string bundledFar = exeDir + "/../../datasets-ProDM/Uf48.100x500x500.bin.f32";
            if (FileExists(bundledFar))
            {
                return bundledFar;
            }
        }
    }

    return legacyHomeData;
}
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 1 && rank == 0)
    {
        std::cerr << "[Info] MPI size = " << size << std::endl;
    }

    std::string binPath = DefaultDatasetPath();

    if (argc > 1)
    {
        binPath = argv[1];
    }

    // 全局尺寸：100 x 500 x 500 float32
    const std::size_t NZ = 100;
    const std::size_t NY = 500;
    const std::size_t NX = 500;
    const std::size_t N  = NZ * NY * NX;
    const std::size_t expectedBytes = N * sizeof(float);

    // ------------------ 1. Z 向域分解 -------------------
    const std::size_t baseNZ = NZ / size;
    const std::size_t remNZ  = NZ % size;

    // 前 remNZ 个 rank 多一个切片
    std::size_t localNZ = baseNZ + (static_cast<std::size_t>(rank) < remNZ ? 1 : 0);

    // 该 rank 起始 Z 位置
    std::size_t startZ = rank * baseNZ + (static_cast<std::size_t>(rank) < remNZ ? rank : remNZ);

    if (localNZ == 0)
    {
        if (rank == 0)
        {
            std::cerr << "Error: MPI size is larger than NZ, some ranks have no data." << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const std::size_t localN = localNZ * NY * NX;

    if (rank == 0)
    {
        std::cout << "Global dims : NZ = " << NZ << ", NY = " << NY << ", NX = " << NX << std::endl;
    }
    std::cout << "Rank " << rank << " localNZ = " << localNZ
              << ", startZ = " << startZ
              << ", localN = " << localN << std::endl;

    // ------------------ 2. 用 MPI-IO 并行读取原始二进制文件 -------------------
    std::vector<float> dataLocal(localN, 0.0f);

    MPI_File fh;
    MPI_Status status;

    int mpierr = MPI_File_open(MPI_COMM_WORLD,
                               binPath.c_str(),
                               MPI_MODE_RDONLY,
                               MPI_INFO_NULL,
                               &fh);
    if (mpierr != MPI_SUCCESS)
    {
        if (rank == 0)
        {
            std::cerr << "Error: Cannot open file " << binPath << " with MPI-IO" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 检查文件大小（只在 rank 0 打印）
    MPI_Offset fileSizeBytes = 0;
    MPI_File_get_size(fh, &fileSizeBytes);

    if (rank == 0)
    {
        if (fileSizeBytes != static_cast<MPI_Offset>(expectedBytes))
        {
            std::cerr << "Warning: File size mismatch: fileSize = "
                      << fileSizeBytes
                      << ", expected = " << expectedBytes << " bytes" << std::endl;
        }
        std::cout << "Read (MPI-IO): " << binPath
                  << " (global elements = " << N << ")" << std::endl;
    }

    // 设置视图，按 float 元素访问（这样 offset 就是以 float 为单位）
    MPI_File_set_view(fh,
                      0,
                      MPI_FLOAT,
                      MPI_FLOAT,
                      const_cast<char*>("native"),
                      MPI_INFO_NULL);

    // 当前 rank 在文件中的起始元素偏移（以 float 为单位）
    const std::size_t offsetElems = startZ * NY * NX;

    mpierr = MPI_File_read_at_all(fh,
                                  static_cast<MPI_Offset>(offsetElems),
                                  dataLocal.data(),
                                  static_cast<int>(localN),
                                  MPI_FLOAT,
                                  &status);
    if (mpierr != MPI_SUCCESS)
    {
        std::cerr << "Error: MPI_File_read_at_all failed on rank " << rank << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_File_close(&fh);

    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------ 3. ADIOS2 并行写入 -------------------
    adios2::ADIOS adios(MPI_COMM_WORLD);

    const std::string fname = "test_prodm_3d.bp";

    {
        auto io = adios.DeclareIO("io_prodm_write");

        adios2::Dims shape{NZ, NY, NX};
        adios2::Dims start{startZ, 0, 0};
        adios2::Dims count{localNZ, NY, NX};

        auto var = io.DefineVariable<float>("field", shape, start, count);

        adios2::Operator prodmOp =
            adios.DefineOperator("op_prodm", "prodm", {{"accuracy", "1e-4"}});

        var.AddOperation(prodmOp, {});

        auto engine = io.Open(fname, adios2::Mode::Write);

        if (rank == 0)
            std::cout << "Start writing ProDM BP file (parallel): " << fname << std::endl;

        engine.BeginStep();
        engine.Put(var, dataLocal.data());
        engine.EndStep();

        engine.Close();

        if (rank == 0)
            std::cout << "Write complete: " << fname << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------ 4. ADIOS2 并行读取并验证 -------------------
    {
        auto io = adios.DeclareIO("io_prodm_read");
        auto engine = io.Open(fname, adios2::Mode::Read);

        std::vector<float> recLocal(localN, 0.0f);

        while (engine.BeginStep() == adios2::StepStatus::OK)
        {
            auto var = io.InquireVariable<float>("field");
            if (!var)
            {
                if (rank == 0)
                    std::cerr << "Error: Cannot find variable 'field' " << std::endl;
                engine.EndStep();
                break;
            }

            adios2::Dims start{startZ, 0, 0};
            adios2::Dims count{localNZ, NY, NX};
            var.SetSelection({start, count});

            engine.Get(var, recLocal.data(), adios2::Mode::Sync);
            engine.EndStep();
        }

        engine.Close();

        // 每个 rank 计算本地误差
        double local_max_abs_err = 0.0;
        long double local_mse_sum = 0.0L;

        for (std::size_t i = 0; i < localN; ++i)
        {
            double e  = static_cast<double>(recLocal[i]) - static_cast<double>(dataLocal[i]);
            double ae = std::abs(e);
            if (ae > local_max_abs_err)
                local_max_abs_err = ae;
            local_mse_sum += static_cast<long double>(e) * static_cast<long double>(e);
        }

        // 聚合到 rank 0
        double global_max_abs_err = 0.0;
        long double global_mse_sum = 0.0L;

        MPI_Reduce(&local_max_abs_err, &global_max_abs_err,
                   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_mse_sum, &global_mse_sum,
                   1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            long double mse_ld = global_mse_sum / static_cast<long double>(N);

            std::cout << "Test Completed (parallel).\n";
            std::cout << "Max abs error = " << global_max_abs_err << "\n";
            std::cout << "MSE          = " << static_cast<double>(mse_ld) << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
