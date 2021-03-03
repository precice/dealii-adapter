#include <adapter/parameters.h>

namespace Parameters
{
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
                        Patterns::Integer(0),
                        "Write results every x timesteps");
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

  void
  System::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("System properties");
    {
      prm.declare_entry("Shear modulus",
                        "0.4225e6",
                        Patterns::Double(),
                        "Shear modulus");

      prm.declare_entry("Poisson's ratio",
                        "0.3",
                        Patterns::Double(-1.0, 0.5),
                        "Poisson's ratio");

      prm.declare_entry("rho", "1000", Patterns::Double(0.0), "density");

      prm.declare_entry("body forces",
                        "0,0,0",
                        Patterns::List(Patterns::Double()),
                        "body forces x,y,z");
    }
    prm.leave_subsection();
  }

  void
  System::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("System properties");
    {
      mu  = prm.get_double("Shear modulus");
      nu  = prm.get_double("Poisson's ratio");
      rho = prm.get_double("rho");
      const std::vector<std::string> body_forces_input =
        Utilities::split_string_list(prm.get("body forces"));
      for (uint d = 0; d < 3; ++d)
        body_force[d] = Utilities::string_to_double(body_forces_input[d]);
    }
    prm.leave_subsection();

    lambda = 2 * mu * nu / (1 - 2 * nu);
  }


  void
  Solver::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear solver");
    {
      prm.declare_entry("Solver type",
                        "Direct",
                        Patterns::Selection("CG|Direct"),
                        "Linear solver: CG or Direct");

      prm.declare_entry("Residual",
                        "1e-6",
                        Patterns::Double(0.0),
                        "Linear solver residual (scaled by residual norm)");

      prm.declare_entry(
        "Max iteration multiplier",
        "1",
        Patterns::Double(0.0),
        "Linear solver iterations (multiples of the system matrix size)");

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
  Solver::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solver");
    {
      type_lin           = prm.get("Solver type");
      tol_lin            = prm.get_double("Residual");
      max_iterations_lin = prm.get_double("Max iteration multiplier");

      max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
      tol_f             = prm.get_double("Tolerance force");
      tol_u             = prm.get_double("Tolerance displacement");
    }
    prm.leave_subsection();
  }


  void
  Discretization::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      prm.declare_entry("Polynomial degree",
                        "3",
                        Patterns::Integer(0),
                        "Polynomial degree of the FE system");

      prm.declare_entry("theta",
                        "0.5",
                        Patterns::Double(0, 1),
                        "Time integration scheme");

      prm.declare_entry("beta",
                        "0.25",
                        Patterns::Double(0, 0.5),
                        "Newmark beta");

      prm.declare_entry("gamma",
                        "0.5",
                        Patterns::Double(0, 1),
                        "Newmark gamma");
    }
    prm.leave_subsection();
  }

  void
  Discretization::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      poly_degree = prm.get_integer("Polynomial degree");
      theta       = prm.get_double("theta");
      beta        = prm.get_double("beta");
      gamma       = prm.get_double("gamma");
    }
    prm.leave_subsection();
  }


  void
  PreciceAdapterConfiguration::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("precice configuration");
    {
      prm.declare_entry("Scenario",
                        "FSI3",
                        Patterns::Selection("FSI3|PF"),
                        "Cases: FSI3 or PF for perpendicular flap");
      prm.declare_entry("precice config-file",
                        "precice-config.xml",
                        Patterns::Anything(),
                        "Name of the precice configuration file");
      prm.declare_entry(
        "Participant name",
        "dealiisolver",
        Patterns::Anything(),
        "Name of the participant in the precice-config.xml file");
      prm.declare_entry(
        "Mesh name",
        "dealii-mesh",
        Patterns::Anything(),
        "Name of the coupling mesh in the precice-config.xml file");
      prm.declare_entry("Read data name",
                        "received-data",
                        Patterns::Anything(),
                        "Name of the read data in the precice-config.xml file");
      prm.declare_entry(
        "Write data name",
        "calculated-data",
        Patterns::Anything(),
        "Name of the write data in the precice-config.xml file");
      prm.declare_entry("Flap location",
                        "0.0",
                        Patterns::Double(-3, 3),
                        "PF x-location");
    }
    prm.leave_subsection();
  }

  void
  PreciceAdapterConfiguration::parse_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("precice configuration");
    {
      scenario         = prm.get("Scenario");
      config_file      = prm.get("precice config-file");
      participant_name = prm.get("Participant name");
      mesh_name        = prm.get("Mesh name");
      read_data_name   = prm.get("Read data name");
      write_data_name  = prm.get("Write data name");
      flap_location    = prm.get_double("Flap location");
    }
    prm.leave_subsection();
    // Look at the specific type of read data
    if ((read_data_name.find("Stress") == 0))
      data_consistent = true;
    else if ((read_data_name.find("Force") == 0))
      data_consistent = false;
    else
      AssertThrow(
        false,
        ExcMessage(
          "Unknown read data type. Please use 'Force' or 'Stress' in the read data naming."));
  }

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
    Solver::declare_parameters(prm);
    Discretization::declare_parameters(prm);
    System::declare_parameters(prm);
    Time::declare_parameters(prm);
    PreciceAdapterConfiguration::declare_parameters(prm);
  }

  void
  AllParameters::parse_parameters(ParameterHandler &prm)
  {
    Solver::parse_parameters(prm);
    Discretization::parse_parameters(prm);
    System::parse_parameters(prm);
    Time::parse_parameters(prm);
    PreciceAdapterConfiguration::parse_parameters(prm);
  }
} // namespace Parameters
