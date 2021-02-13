#include "source/linear_elasticity/include/linear_elasticity.h"
#include "source/nonlinear_elasticity/include/nonlinear_elasticity.h"

int
main(int argc, char **argv)
{
  using namespace dealii;

#ifdef DEAL_II_WITH_MPI
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
#endif

  try
    {
      deallog.depth_console(0);
      static const unsigned int n_threads = MultithreadInfo::n_threads();

      // Query adapter and deal.II info
      const std::string adapter_info =
        GIT_SHORTREV == std::string("") ?
          "unknown" :
          (GIT_SHORTREV + std::string(" on branch ") + GIT_BRANCH);
      const std::string dealii_info =
        DEAL_II_GIT_SHORTREV == std::string("") ?
          "unknown" :
          (DEAL_II_GIT_SHORTREV + std::string(" on branch ") +
           DEAL_II_GIT_BRANCH);

      std::cout
        << "-----------------------------------------------------------------------------"
        << std::endl
        << "--     . running with " << n_threads << " thread"
        << (n_threads == 1 ? "" : "s") << std::endl;

      std::cout << "--     . adapter revision " << adapter_info << std::endl;
      std::cout << "--     . deal.II " << DEAL_II_PACKAGE_VERSION
                << " (revision " << dealii_info << ")" << std::endl;
      std::cout
        << "-----------------------------------------------------------------------------"
        << std::endl
        << std::endl;


      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[2];
      else
        parameter_file = "nonlinear_elasticity.prm";

      // Extract case path for the output directory
      size_t      pos = parameter_file.find_last_of("/");
      std::string case_path =
        std::string::npos == pos ? "" : parameter_file.substr(0, pos + 1);

      std::string solver_type = argv[1];
      // Dimension is determinded via cmake -DDIM
      if (solver_type == "-nonlinear") // nonlinear
        {
          Nonlinear_Elasticity::Solid<DIM> solid(case_path);
          solid.run();
        }
      else if (solver_type == "-linear") // linear
        {
          Linear_Elasticity::ElastoDynamics<DIM> elastic_solver(case_path);
          elastic_solver.run();
        }
      else
        AssertThrow(false, ExcMessage("Unknown solver specified. "))
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
