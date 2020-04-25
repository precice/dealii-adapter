#ifndef parameter_handling_h
#define parameter_handling_h

#include <deal.II/base/parameter_handler.h>

namespace adapter
{
  using namespace dealii;

  // TODO: Merge some structs together
  namespace Parameters
  {
    struct FESystem
    {
      unsigned int poly_degree;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "1",
                          Patterns::Integer(0),
                          "Polynomial degree of the FE system");
      }
      prm.leave_subsection();
    }

    void
    FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
      }
      prm.leave_subsection();
    }


    struct Materials
    {
      double nu;
      double mu;
      double rho;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Poisson's ratio",
                          "0.3",
                          Patterns::Double(-1.0, 0.5),
                          "Poisson's ratio");

        prm.declare_entry("Shear modulus",
                          "0.4225e6",
                          Patterns::Double(),
                          "Shear modulus");

        prm.declare_entry("rho", "1000", Patterns::Double(), "rho");
      }
      prm.leave_subsection();
    }

    void
    Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        nu  = prm.get_double("Poisson's ratio");
        mu  = prm.get_double("Shear modulus");
        rho = prm.get_double("rho");
      }
      prm.leave_subsection();
    }


    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type",
                          "CG",
                          Patterns::Selection("CG|Direct"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual",
                          "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry(
          "Max iteration multiplier",
          "1",
          Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");
      }
      prm.leave_subsection();
    }

    void
    LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin           = prm.get("Solver type");
        tol_lin            = prm.get_double("Residual");
        max_iterations_lin = prm.get_double("Max iteration multiplier");
      }
      prm.leave_subsection();
    }


    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void
    NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f             = prm.get_double("Tolerance force");
        tol_u             = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }


    struct Time
    {
      double delta_t;
      double end_time;
      int    output_interval;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time step size",
                          "0.1",
                          Patterns::Double(),
                          "Time step size");

        prm.declare_entry("Output interval",
                          "1",
                          Patterns::Integer(),
                          "Output interval");
      }
      prm.leave_subsection();
    }

    void
    Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time        = prm.get_double("End time");
        delta_t         = prm.get_double("Time step size");
        output_interval = prm.get_integer("Output interval");
      }
      prm.leave_subsection();
    }


    struct NewmarkParameters
    {
      double beta;
      double gamma;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    NewmarkParameters::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Newmark parameters");
      {
        prm.declare_entry("beta", "0.25", Patterns::Double(0, 0.5), "beta");

        prm.declare_entry("gamma", "0.5", Patterns::Double(0, 1), "gamma");
      }
      prm.leave_subsection();
    }

    void
    NewmarkParameters::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Newmark parameters");
      {
        beta  = prm.get_double("beta");
        gamma = prm.get_double("gamma");
      }
      prm.leave_subsection();
    }



    struct PreciceConfiguration
    {
      std::string  scenario;
      bool         enable_precice;
      std::string  config_file;
      std::string  participant;
      std::string  mesh_name;
      std::string  read_data_name;
      std::string  write_data_name;
      unsigned int interface_mesh_id;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void
    PreciceConfiguration::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("precice configuration");
      {
        prm.declare_entry("Scenario",
                          "FSI3",
                          Patterns::Selection("FSI3|PF"),
                          "Cases: FSI3 or PF for perpendicular flap");
        prm.declare_entry(
          "Enable precice",
          "true",
          Patterns::Bool(),
          "Whether preCICE is used for coupling to another solver");
        prm.declare_entry("precice config-file",
                          "precice-config.xml",
                          Patterns::Anything(),
                          "Name of the precice configuration file");
        prm.declare_entry(
          "Participant",
          "dealiisolver",
          Patterns::Anything(),
          "Name of the participant in the precice-config.xml file");
        prm.declare_entry(
          "Mesh name",
          "dealii-mesh-nodes",
          Patterns::Anything(),
          "Name of the coupling mesh in the precice-config.xml file");
        prm.declare_entry(
          "Read data name",
          "received-data",
          Patterns::Anything(),
          "Name of the read data in the precice-config.xml file");
        prm.declare_entry(
          "Write data name",
          "calculated-data",
          Patterns::Anything(),
          "Name of the write data in the precice-config.xml file");
        prm.declare_entry(
          "Interface mesh ID",
          "1",
          Patterns::Integer(0),
          "Boundary mesh ID of the coupling interface in deal.II");
      }
      prm.leave_subsection();
    }

    void
    PreciceConfiguration::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("precice configuration");
      {
        scenario          = prm.get("Scenario");
        enable_precice    = prm.get_bool("Enable precice");
        config_file       = prm.get("precice config-file");
        participant       = prm.get("Participant");
        mesh_name         = prm.get("Mesh name");
        read_data_name    = prm.get("Read data name");
        write_data_name   = prm.get("Write data name");
        interface_mesh_id = prm.get_integer("Interface mesh ID");
      }
      prm.leave_subsection();
    }



    struct AllParameters : public FESystem,
                           public Materials,
                           public LinearSolver,
                           public NonlinearSolver,
                           public Time,
                           public NewmarkParameters,
                           public PreciceConfiguration

    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);

      // Optional, if we want to print all parameters in the beginning of the
      // simulation
      //      prm.print_parameters(std::cout,ParameterHandler::Text);
    }

    void
    AllParameters::declare_parameters(ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
      NewmarkParameters::declare_parameters(prm);
      PreciceConfiguration::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
      NewmarkParameters::parse_parameters(prm);
      PreciceConfiguration::parse_parameters(prm);
    }
  } // namespace Parameters
} // namespace adapter

#endif
