#ifndef NONLINEAR_ELASTICITY_H
#define NONLINEAR_ELASTICITY_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/transformations.h>

#include <adapter/adapter.h>
#include <adapter/time_handler.h>

#include <fstream>
#include <iostream>

#include "compressible_neo_hook_material.h"
#include "parameter_handling.h"
#include "postprocessor.h"

namespace Nonlinear_Elasticity
{
  using namespace dealii;

  // PointHistory class offers a method for storing data at the quadrature
  // points. Here each quadrature point holds a pointer to a material
  // description
  template <int dim, typename NumberType>
  class PointHistory
  {
  public:
    PointHistory()
    {}

    virtual ~PointHistory()
    {}

    void
    setup_lqp(const Parameters::AllParameters &parameters)
    {
      material.reset(
        new Material_Compressible_Neo_Hook_One_Field<dim, NumberType>(
          parameters.mu, parameters.nu, parameters.rho));
    }

    // Strain energy
    NumberType
    get_Psi(const NumberType &                         det_F,
            const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return material->get_Psi(det_F, b_bar);
    }
    // Kirchhoff stress
    SymmetricTensor<2, dim, NumberType>
    get_tau(const NumberType &                         det_F,
            const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return material->get_tau(det_F, b_bar);
    }
    // Tangent
    SymmetricTensor<4, dim, NumberType>
    get_Jc(const NumberType &                         det_F,
           const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return material->get_Jc(det_F, b_bar);
    }
    // Density
    NumberType
    get_rho() const
    {
      return material->get_rho();
    }

  private:
    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, NumberType>>
      material;
  };


  // Forward declarations for classes that will perform assembly of the
  // linearized system
  template <int dim, typename NumberType>
  struct Assembler_Base;
  template <int dim, typename NumberType>
  struct Assembler;

  // The Solid class is the central class in that it represents the problem at
  // hand. It follows the usual scheme in that all it really has is a
  // constructor, destructor and a run() function that dispatches all the work
  // to private functions of this class:
  template <int dim, typename NumberType = double>
  class Solid
  {
  public:
    Solid(const std::string &case_path);

    virtual ~Solid();

    void
    run();

  private:
    // Generates grid and sets boundary IDs, which are needed for different BCs
    void
    make_grid();
    // Set up the finite element system to be solved
    void
    system_setup();

    // Several functions to assemble the system and right hand side matrices
    // using multithreading. Each of them comes as a wrapper function, one that
    // is executed to do the work in the WorkStream model on one cell, and one
    // that copies the work done on this one cell into the global object that
    // represents it
    void
    assemble_system(const BlockVector<double> &solution_delta,
                    const BlockVector<double> &acceleration,
                    const BlockVector<double> &external_stress);

    // We use a separate data structure to perform the assembly. It needs access
    // to some low-level data, so we simply befriend the class instead of
    // creating a complex interface to provide access as necessary.
    friend struct Assembler_Base<dim, NumberType>;
    friend struct Assembler<dim, NumberType>;

    // Apply Dirichlet boundary conditions on the displacement field
    void
    make_constraints(const int &it_nr);

    // Create and update the quadrature points. Here, no data needs to be copied
    // into a global object, so the copy_local_to_global function is empty:
    void
    setup_qph();

    // Solve for the displacement using a Newton-Raphson method.
    void
    solve_nonlinear_timestep(BlockVector<double> &solution_delta);

    std::pair<unsigned int, double>
    solve_linear_system(BlockVector<double> &newton_update);

    // Solution retrieval
    BlockVector<double>
    get_total_solution(const BlockVector<double> &solution_delta) const;

    // Update fnuctions for time dependent variables according to Newmarks
    // scheme
    void
    update_acceleration(BlockVector<double> &displacement_delta);

    void
    update_velocity(BlockVector<double> &displacement_delta);

    void
    update_old_variables();

    // Post-processing and writing data to file :
    void
    output_results() const;

    const Parameters::AllParameters parameters;

    double vol_reference;
    double vol_current;

    Triangulation<dim> triangulation;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim, NumberType>>
      quadrature_point_history;

    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;

    // Description of how the block-system is arranged. There is just 1 block,
    // that contains a vector DOF u. This is a legacy of the original work
    // (step-44)
    static const unsigned int n_blocks          = 1;
    static const unsigned int n_components      = dim;
    static const unsigned int first_u_component = 0;

    enum
    {
      u_dof = 0
    };

    std::vector<types::global_dof_index> dofs_per_block;

    const QGauss<dim>     qf_cell;
    const QGauss<dim - 1> qf_face;
    const unsigned int    n_q_points;
    const unsigned int    n_q_points_f;
    // Interface ID, which is later assigned to the mesh region for coupling
    // It is chosen arbotrarily
    const unsigned int boundary_interface_id;

    // Newmark parameters
    // Coefficients, which are needed for time dependencies
    const double alpha_1 =
      1. / (parameters.beta * std::pow(parameters.delta_t, 2));
    const double alpha_2 = 1. / (parameters.beta * parameters.delta_t);
    const double alpha_3 = (1 - (2 * parameters.beta)) / (2 * parameters.beta);
    const double alpha_4 =
      parameters.gamma / (parameters.beta * parameters.delta_t);
    const double alpha_5 = 1 - (parameters.gamma / parameters.beta);
    const double alpha_6 =
      (1 - (parameters.gamma / (2 * parameters.beta))) * parameters.delta_t;

    // Read body force from parameter file
    const Tensor<1, 3, double> body_force = parameters.body_force;

    // Clamped boundary ID to be used consistently
    const unsigned int clamped_boundary_id          = 1;
    const unsigned int out_of_plane_clamped_mesh_id = 8;

    // ..and store the directory, in order to output the result files there
    const std::string case_path;

    AffineConstraints<double> constraints;
    BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> tangent_matrix;
    BlockVector<double>       system_rhs;
    BlockVector<double>       total_displacement;
    BlockVector<double>       total_displacement_old;
    BlockVector<double>       velocity;
    BlockVector<double>       velocity_old;
    BlockVector<double>       acceleration;
    BlockVector<double>       acceleration_old;

    // Alias to collect all time dependent variables in a single vector
    // This is directly passed to the Adapter routine in order to
    // store these variables for implicit couplings.
    std::vector<BlockVector<double> *> state_variables;

    // Global vector, which keeps all contributions of the Fluid participant
    // i.e. stress data for assembly. This vector is filled properly in the
    // Adapter
    BlockVector<double> external_stress;

    // In order to measure some timings
    mutable TimerOutput timer;

    // The main adapter objects: The time class keeps track of the current time
    // and time steps. The Adapter class includes all functionalities for
    // coupling via preCICE. Look at the documentation of the class for more
    // information.
    Adapter::Time time;
    Adapter::Adapter<dim, BlockVector<double>, Parameters::AllParameters>
      adapter;

    // Then define a number of variables to store norms and update norms and
    // normalisation factors.
    struct Errors
    {
      Errors()
        : u(1.0)
      {}

      void
      reset()
      {
        u = 1.0;
      }
      void
      normalise(const Errors &val)
      {
        if (val.u != 0.0)
          u /= val.u;
      }

      double u;
    };

    Errors error_residual, error_residual_0, error_residual_norm, error_update,
      error_update_0, error_update_norm;

    // Methods to calculate erros
    void
    get_error_residual(Errors &error_residual);

    void
    get_error_update(const BlockVector<double> &newton_update,
                     Errors &                   error_update);

    // Print information to screen during simulation
    static void
    print_conv_header();

    void
    print_conv_footer();
  };

} // namespace Nonlinear_Elasticity

#endif // NONLINEAR_ELASTICITY_H
