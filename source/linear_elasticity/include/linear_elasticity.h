#ifndef LINEAR_ELASTICITY_H
#define LINEAR_ELASTICITY_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <adapter/adapter.h>
#include <adapter/parameters.h>
#include <adapter/time_handler.h>

#include <fstream>
#include <iostream>

#include "postprocessor.h"

// The Linear_Elasticity case includes a linear elastic material with a one-step
// theta time integration
namespace Linear_Elasticity
{
  using namespace dealii;

  template <int dim>
  class ElastoDynamics
  {
  public:
    ElastoDynamics(const std::string &parameter_file);

    ~ElastoDynamics();
    // As usual in dealii, the run function covers the main time loop of the
    // system
    void
    run();

  private:
    // Create the mesh and set boundary IDs for different boundary conditions
    void
    make_grid();

    // Set up the FE system and allocate data structures
    void
    setup_system();

    // Compute time invariant matrices e.g. stiffness matrix and mass matrix
    void
    assemble_system();

    // Assemble the Neumann contribution i.e. the coupling data obtained from
    // the Fluid participant
    void
    assemble_rhs();

    void
    assemble_consistent_loading();

    // Solve the linear system
    void
    solve();

    // Update the displacement according to the theta scheme
    void
    update_displacement();

    // Output results to vtk files
    void
    output_results() const;

    // Paramter class parsing all user specific input parameters
    const Parameters::AllParameters parameters;

    // Boundary IDs, reserved for the respectve application
    unsigned int       clamped_mesh_id;
    unsigned int       out_of_plane_clamped_mesh_id;
    const unsigned int interface_boundary_id;

    // Dealii typical objects
    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;
    FESystem<dim>        fe;
    MappingQGeneric<dim> mapping;
    const unsigned int   quad_order;

    AffineConstraints<double> hanging_node_constraints;

    // Matrices used during computations
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> stepping_matrix;

    // Time dependent variables
    Vector<double> old_velocity;
    Vector<double> velocity;
    Vector<double> old_displacement;
    Vector<double> displacement;
    Vector<double> old_stress;
    Vector<double> stress;
    Vector<double> system_rhs;

    // Body forces e.g. gravity. Values are specified in the input file
    const bool     body_force_enabled;
    Vector<double> body_force_vector;

    // In order to measure some timings
    mutable TimerOutput timer;

    // The main adapter objects: The time class keeps track of the current time
    // and time steps. The Adapter class includes all functionalities for
    // coupling via preCICE. Look at the documentation of the class for more
    // information.
    Adapter::Time                              time;
    Adapter::Adapter<dim, dim, Vector<double>> adapter;

    // Alias for all time dependent variables, which should be saved/reloaded
    // in case of an implicit coupling. This vector is directly used in the
    // Adapter class
    std::vector<Vector<double> *> state_variables;
  };
} // namespace Linear_Elasticity

#endif // LINEAR_ELASTICITY_H
