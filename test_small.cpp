#include <mpi.h>
#include <adios2.h>
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const std::string fname = "test_prodm_small.bp";

    const std::size_t NX = 16;
    std::vector<double> data(NX);
    for (std::size_t i = 0; i < NX; ++i)
    {
        data[i] = std::sin(0.1 * i) + std::cos(0.2 * i);
    }

    // ------------------ 写 -------------------
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        auto io = adios.DeclareIO("io_prodm_write");

        adios2::Dims shape{NX};
        adios2::Dims start{0};
        adios2::Dims count{NX};

        auto var = io.DefineVariable<double>("x", shape, start, count);

        adios2::Operator prodmOp =
            adios.DefineOperator("op_prodm", "prodm", {});

        var.AddOperation(prodmOp, {});

        auto engine = io.Open(fname, adios2::Mode::Write);

        engine.BeginStep();
        engine.Put(var, data.data());
        engine.EndStep();

        engine.Close();

        if (rank == 0)
            std::cout << "ProDM write finished: " << fname << std::endl;
    }

    // ------------------ 读 -------------------
    {
        adios2::ADIOS adios(MPI_COMM_WORLD);
        auto io = adios.DeclareIO("io_prodm_read");
        auto engine = io.Open(fname, adios2::Mode::Read);

        std::vector<double> rec(NX, 0.0);

        while (engine.BeginStep() == adios2::StepStatus::OK)
        {
            auto var = io.InquireVariable<double>("x");
            if (!var)
            {
                if (rank == 0)
                    std::cerr << "Variable x not found!" << std::endl;
                break;
            }

            adios2::Operator prodmOp =
                adios.DefineOperator("op_prodm", "prodm", {{"accuracy", "1e-4"}});

            var.AddOperation(prodmOp, {});
            var.SetSelection({{0}, {NX}});
            engine.Get(var, rec.data(), adios2::Mode::Sync);
            engine.EndStep();
        }

        engine.Close();

        double max_abs_err = 0.0;
        double mse = 0.0;
        for (std::size_t i = 0; i < NX; ++i)
        {
            double e = rec[i] - data[i];
            max_abs_err = std::max(max_abs_err, std::abs(e));
            mse += e * e;
        }
        mse /= static_cast<double>(NX);

        if (rank == 0)
        {
            std::cout << "ProDM test finished.\n";
            std::cout << "Max abs error = " << max_abs_err << "\n";
            std::cout << "MSE          = " << mse << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
