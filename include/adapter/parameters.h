#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

/**
 * This class declares all parameters, which can be specified in the parameter
 * file. The subsection abut preCICE configurations is directly interlinked
 * to the Adapter class.
 */
namespace Parameters
{
  using namespace dealii;
  /**
   * @brief Time: Specifies simulation time properties
   */
  struct Time
  {
    double end_time;
    double delta_t;
    int    output_interval;

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
    double               nu;
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
  struct Solver
  {
    std::string  type_lin;
    double       tol_lin;
    double       max_iterations_lin;
    unsigned int max_iterations_NR;
    double       tol_f;
    double       tol_u;

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
    unsigned int poly_degree;
    // For the linear elastic model (theta-scheme)
    double theta;
    // For the nonlinear elastic model (Newmark)
    double beta;
    double gamma ;

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


  struct AllParameters : public Solver,
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

#endif // PARAMETERS_H
