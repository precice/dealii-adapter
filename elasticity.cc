#include <adapter/parameters.h>
#include <sys/stat.h>

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

      // Store the name of the parameter file
      const std::string parameter_file = argc > 1 ? argv[1] : "parameters.prm";

      // Create output directory for the current case
      // Query output directory from the parameter file
      ParameterHandler prm;
      Parameters::Time time;
      time.add_output_parameters(prm);
      prm.parse_input(parameter_file, "", true);

      // force trailing / so we can handle everything in loop
      std::string pathname = time.output_folder;
      if (pathname[pathname.size() - 1] != '/')
        pathname += '/';

      size_t pre = 0;
      size_t pos;
      mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
      while ((pos = pathname.find_first_of('/', pre)) != std::string::npos)
        {
          const std::string subdir = pathname.substr(0, pos++);
          pre                      = pos;

          // if leading '/', first string is 0 length
          if (subdir.size() == 0)
            continue;

          int mkdir_return_value;
          if ((mkdir_return_value = mkdir(subdir.c_str(), mode)) &&
              (errno != EEXIST))
            {
              AssertThrow(mkdir_return_value,
                          ExcMessage("Can't create: " + pathname));
              return mkdir_return_value;
            }
        }

      // Query solver type from the parameter file
      Parameters::Solver solver;
      solver.add_output_parameters(prm);
      prm.parse_input(parameter_file, "", true);

      if (solver.model == "neo-Hookean") // nonlinear
        {
          Nonlinear_Elasticity::Solid<DIM> solid(parameter_file);
          solid.run();
        }
      else if (solver.model == "linear") // linear
        {
          Linear_Elasticity::ElastoDynamics<DIM> elastic_solver(parameter_file);
          elastic_solver.run();
        }
      else
        AssertThrow(false, ExcNotImplemented());
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
