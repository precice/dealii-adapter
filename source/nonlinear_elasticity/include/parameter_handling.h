#pragma once

#include <deal.II/base/parameter_handler.h>

namespace Nonlinear_Elasticity
{
  using namespace dealii;

  /**
   * This class declares all parameters, which can be specified in the parameter
   * file. The subsection abut preCICE configurations is directly interlinked
   * to the Adapter class.
   */
  namespace Parameters
  {
    /**
     * @brief System: Specifies system properties
     */
    struct System
    {
      double               nu;
      double               mu;
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
     * @brief NonlinearSolver: Specifies nonlinear solver properties
     */
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


    /**
     * @brief Time: Specifies simulation time properties
     */
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
     * @brief Discretization: Specifies parameters for time integration by an
     *        implicit Newmark scheme and polynomial degree of the FE system
     */
    struct Discretization
    {
      double       beta;
      double       gamma;
      unsigned int poly_degree;


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

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    struct AllParameters : public System,
                           public LinearSolver,
                           public NonlinearSolver,
                           public Time,
                           public Discretization,
                           public PreciceAdapterConfiguration

    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


  } // namespace Parameters
} // namespace Nonlinear_Elasticity
