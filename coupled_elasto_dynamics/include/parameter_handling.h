#ifndef parameter_handling_h
#define parameter_handling_h

#include <deal.II/base/parameter_handler.h>

namespace Adapter
{
  using namespace dealii;

  /**
   * This class declares all parameters, which can be specified in the parameter
   * file. The subsection abut preCICE configurations is directly interlinked
   * to the CouplingFunctions class.
   */
  namespace Parameters
  {
    // TODO: Add more parameters
    struct Time
    {
      double delta_t;
      double end_time;
      double theta;
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

        prm.declare_entry("theta",
                          "0.5",
                          Patterns::Double(0, 1),
                          "Time integration scheme");
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
        theta           = prm.get_double("theta");
      }
      prm.leave_subsection();
    }

    struct Materials
    {
      double mu;
      double lambda;
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
        prm.declare_entry("mu", "0.5e6", Patterns::Double(), "mu");

        prm.declare_entry("lambda", "2e6", Patterns::Double(), "lambda");

        prm.declare_entry("density", "1000", Patterns::Double(0.0), "density");
      }
      prm.leave_subsection();
    }

    void
    Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        mu     = prm.get_double("mu");
        lambda = prm.get_double("lambda");
        rho    = prm.get_double("density");
      }
      prm.leave_subsection();
    }

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
      }
      prm.leave_subsection();
    }



    struct AllParameters : public FESystem,
                           public Materials,
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
      FESystem::declare_parameters(prm);
      Materials::declare_parameters(prm);
      Time::declare_parameters(prm);
      PreciceAdapterConfiguration::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Materials::parse_parameters(prm);
      Time::parse_parameters(prm);
      PreciceAdapterConfiguration::parse_parameters(prm);
    }
  } // namespace Parameters
} // namespace Adapter

#endif
