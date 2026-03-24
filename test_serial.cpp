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
    const std::string homeData = home + "/data/Uf48.100x500x500.bin.f32";
    if (FileExists(homeData))
    {
        return homeData;
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

    return homeData;
}
}

int main(int argc, char **argv)
{
    std::string binPath = DefaultDatasetPath();

    if (argc > 1)
    {
        binPath = argv[1];
    }

    // 100 x 500 x 500 float32
    const std::size_t NZ = 100;
    const std::size_t NY = 500;
    const std::size_t NX = 500;
    const std::size_t N  = NZ * NY * NX;
    const std::size_t expectedBytes = N * sizeof(float);

    std::vector<float> data;
    {
        std::ifstream ifs(binPath, std::ios::binary);
        if (!ifs)
        {
            std::cerr << "Error: Cannot open file " << binPath << std::endl;
            return 1;
        }

        // Read
        ifs.seekg(0, std::ios::end);
        std::streamsize fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        if (fileSize != static_cast<std::streamsize>(expectedBytes))
        {
            std::cerr << "Warning: File size not match: "
                      << "fileSize = " << fileSize
                      << ", expected = " << expectedBytes << " bytes" << std::endl;
        }

        data.resize(N);
        if (!ifs.read(reinterpret_cast<char*>(data.data()), expectedBytes))
        {
            std::cerr << "Error: Fail to read file: " << binPath << std::endl;
            return 1;
        }

        std::cout << "Read: " << binPath
                  << " (elements = " << N << ")" << std::endl;
    }

    const std::string fname = "data/test_prodm_serial.bp";

    // ------------------ Write -------------------
    {
        adios2::ADIOS adios;
        auto io = adios.DeclareIO("io_prodm_write");

        adios2::Dims shape{NZ, NY, NX};
        adios2::Dims start{0, 0, 0};
        adios2::Dims count{NZ, NY, NX};

        auto var = io.DefineVariable<float>("field", shape, start, count);

        adios2::Operator prodmOp =
            adios.DefineOperator("op_prodm", "prodm", {});

        var.AddOperation(prodmOp, {});

        auto engine = io.Open(fname, adios2::Mode::Write);

        std::cout << "Start writing ProDM BP file: " << fname << std::endl;

        engine.BeginStep();
        engine.Put(var, data.data());
        engine.EndStep();

        engine.Close();

        std::cout << "Write complete: " << fname << std::endl;
    }

    // ------------------ Read -------------------
    {
        adios2::ADIOS adios;
        auto io = adios.DeclareIO("io_prodm_read");
        auto engine = io.Open(fname, adios2::Mode::Read);

        std::vector<float> rec(N, 0.0f);

        while (engine.BeginStep() == adios2::StepStatus::OK)
        {
            auto var = io.InquireVariable<float>("field");
            if (!var)
            {
                std::cerr << "Error: Cannot find 'field' " << std::endl;
                break;
            }

            adios2::Operator prodmOp =
                adios.DefineOperator("op_prodm", "prodm", {{"accuracy", "1e-3"}});

            var.AddOperation(prodmOp, {});

            var.SetSelection({{0, 0, 0}, {NZ, NY, NX}});
            engine.Get(var, rec.data(), adios2::Mode::Sync);
            engine.EndStep();
        }

        engine.Close();

        double max_abs_err = 0.0;
        long double mse = 0.0L;

        for (std::size_t i = 0; i < N; ++i)
        {
            double e = static_cast<double>(rec[i]) - static_cast<double>(data[i]);
            double ae = std::abs(e);
            if (ae > max_abs_err)
                max_abs_err = ae;
            mse += static_cast<long double>(e) * static_cast<long double>(e);
        }
        mse /= static_cast<long double>(N);

        std::cout << "Test Completed.\n";
        std::cout << "Max abs error = " << max_abs_err << "\n";
        std::cout << "MSE          = " << static_cast<double>(mse) << "\n";
    }

    return 0;
}
