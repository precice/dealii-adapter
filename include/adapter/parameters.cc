#include <adapter/parameters.h>

namespace Parameters
{
  void
  Time::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time");
    {
      prm.add_parameter("End time", end_time, "End time", Patterns::Double());

      prm.add_parameter("Time step size",
                        delta_t,
                        "Time step size",
                        Patterns::Double());

      prm.add_parameter("Output interval",
                        output_interval,
                        "Write results every x timesteps",
                        Patterns::Integer(0));
      prm.add_parameter("Output folder",
                        output_folder,
                        "Output folder",
                        Patterns::Anything());
    }
    prm.leave_subsection();
  }


  void
  System::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("System properties");
    {
      prm.add_parameter("Shear modulus",
                        mu,
                        "Shear modulus",
                        Patterns::Double());

      prm.add_parameter("Poisson's ratio",
                        nu,
                        "Poisson's ratio",
                        Patterns::Double(-1.0, 0.5));

      prm.add_parameter("rho", rho, "density", Patterns::Double(0.0));

      prm.add_parameter("body forces",
                        body_force,
                        "body forces x,y,z",
                        Patterns::List(Patterns::Double()));
    }
    prm.leave_subsection();
  }


  void
  Solver::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Solver");
    {
      prm.add_parameter("Model",
                        model,
                        "Structural model to be used: linear or neo-Hookean",
                        Patterns::Selection("linear|neo-Hookean"));

      prm.add_parameter("Solver type",
                        type_lin,
                        "Linear solver: CG or Direct",
                        Patterns::Selection("CG|Direct"));

      prm.add_parameter(
        "Residual",
        tol_lin,
        "CG solver residual (multiplied by residual norm, ignored if Model == linear)",
        Patterns::Double(0.0));

      prm.add_parameter(
        "Max iteration multiplier",
        max_iterations_lin,
        "Max CG solver iterations (multiples of the system matrix size)",
        Patterns::Double(0.0));

      prm.add_parameter(
        "Max iterations Newton-Raphson",
        max_iterations_NR,
        "Number of Newton-Raphson iterations allowed (ignored if Model == linear)",
        Patterns::Integer(0));

      prm.add_parameter(
        "Tolerance force",
        tol_f,
        "Force residual tolerance for non-linear iteration (ignored if Model == linear)",
        Patterns::Double(0.0));

      prm.add_parameter(
        "Tolerance displacement",
        tol_u,
        "Displacement error tolerance for non-linear iteration (ignored if Model == linear)",
        Patterns::Double(0.0));
    }
    prm.leave_subsection();
  }


  void
  Discretization::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Discretization");
    {
      prm.add_parameter("Polynomial degree",
                        poly_degree,
                        "Polynomial degree of the FE system",
                        Patterns::Integer(0));

      prm.add_parameter("theta",
                        theta,
                        "Time integration scheme",
                        Patterns::Double(0, 1));

      prm.add_parameter("beta", beta, "Newmark beta", Patterns::Double(0, 0.5));

      prm.add_parameter("gamma",
                        gamma,
                        "Newmark gamma",
                        Patterns::Double(0, 1));
    }
    prm.leave_subsection();
  }


  void
  PreciceAdapterConfiguration::add_output_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("precice configuration");
    {
      prm.add_parameter("Scenario",
                        scenario,
                        "Cases: FSI3 or PF for perpendicular flap",
                        Patterns::Selection("FSI3|PF"));

      prm.add_parameter("precice config-file",
                        config_file,
                        "Name of the precice configuration file",
                        Patterns::Anything());

      prm.add_parameter(
        "Participant name",
        participant_name,
        "Name of the participant in the precice-config.xml file",
        Patterns::Anything());

      prm.add_parameter(
        "Mesh name",
        mesh_name,
        "Name of the coupling mesh in the precice-config.xml file",
        Patterns::Anything());

      prm.add_parameter("Read data name",
                        read_data_name,
                        "Name of the read data in the precice-config.xml file",
                        Patterns::Anything());

      prm.add_parameter("Write data name",
                        write_data_name,
                        "Name of the write data in the precice-config.xml file",
                        Patterns::Anything());

      prm.add_parameter("Flap location",
                        flap_location,
                        "PF x-location",
                        Patterns::Double(-3, 3));
    }
    prm.leave_subsection();
  }


  AllParameters::AllParameters(const std::string &input_file)
  {
    ParameterHandler prm;

    Solver::add_output_parameters(prm);
    Discretization::add_output_parameters(prm);
    System::add_output_parameters(prm);
    Time::add_output_parameters(prm);
    PreciceAdapterConfiguration::add_output_parameters(prm);

    prm.parse_input(input_file);

    lambda = 2 * mu * nu / (1 - 2 * nu);

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

    // Optional, if we want to print all parameters in the beginning of the
    // simulation
    //      prm.print_parameters(std::cout,ParameterHandler::Text);
  }
} // namespace Parameters
