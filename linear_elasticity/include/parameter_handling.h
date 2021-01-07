#ifndef parameter_handling_h
#define parameter_handling_h

#include <deal.II/base/parameter_handler.h>

namespace Linear_Elasticity
{
  using namespace dealii;

  /**
   * This class declares all parameters, which can be specified in the parameter
   * file. The subsection abut preCICE configurations is directly interlinked
   * to the Adapter class.
   */
  namespace Parameters
  {
    // TODO: Add more parameters
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

    /**
     * @brief Discretization: Specifies parameters for time integration by a
     *        theta scheme and polynomial degree of the FE system
     */
    struct Discretization
    {
      double       theta;
      unsigned int poly_degree;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Discretization::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Discretization");
      {
        prm.declare_entry("theta",
                          "0.5",
                          Patterns::Double(0, 1),
                          "Time integration scheme");

        prm.declare_entry("Polynomial degree",
                          "3",
                          Patterns::Integer(0),
                          "Polynomial degree of the FE system");
      }
      prm.leave_subsection();
    }

    void
    Discretization::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Discretization");
      {
        theta       = prm.get_double("theta");
        poly_degree = prm.get_integer("Polynomial degree");
      }
      prm.leave_subsection();
    }

    /**
     * @brief The System struct keeps material properties and body force contributions
     */
    struct System
    {
      double               mu;
      double               lambda;
      double               rho;
      Tensor<1, 3, double> body_force;


      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    System::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("System properties");
      {
        prm.declare_entry("mu", "0.5e6", Patterns::Double(), "mu");

        prm.declare_entry("lambda", "2e6", Patterns::Double(), "lambda");

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
        mu     = prm.get_double("mu");
        lambda = prm.get_double("lambda");
        rho    = prm.get_double("rho");
        const std::vector<std::string> body_forces_input =
          Utilities::split_string_list(prm.get("body forces"));
        for (uint d = 0; d < 3; ++d)
          body_force[d] = Utilities::string_to_double(body_forces_input[d]);
      }
      prm.leave_subsection();
    }

    /**
     * @brief LinearSolver: Specifies linear solver properties
     */
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

    /**
     * @brief PreciceAdapterConfiguration: Specifies preCICE related information.
     *        A lot of these information need to be consistent with the
     *        precice-config.xml file.
     */
    struct PreciceAdapterConfiguration
    {
      std::string scenario;
      std::string config_file;
      std::string participant_name;
      std::string mesh_name;
      std::string read_data_name;
      std::string write_data_name;
      double flap_location;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


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
          "Flap location",
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
        flap_location    = prm.get_double("Flap x-location");
      }
      prm.leave_subsection();
    }



    struct AllParameters : public LinearSolver,
                           public Discretization,
                           public System,
                           public Time,
                           public PreciceAdapterConfiguration

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
      LinearSolver::declare_parameters(prm);
      Discretization::declare_parameters(prm);
      System::declare_parameters(prm);
      Time::declare_parameters(prm);
      PreciceAdapterConfiguration::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      LinearSolver::parse_parameters(prm);
      Discretization::parse_parameters(prm);
      System::parse_parameters(prm);
      Time::parse_parameters(prm);
      PreciceAdapterConfiguration::parse_parameters(prm);
    }
  } // namespace Parameters
} // namespace Linear_Elasticity

#endif
