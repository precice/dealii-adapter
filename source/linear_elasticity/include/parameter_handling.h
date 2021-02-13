#pragma once

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
      double      flap_location;
      bool        data_consistent;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


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
  } // namespace Parameters
} // namespace Linear_Elasticity
