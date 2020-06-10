#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
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

#include <fstream>
#include <iostream>

#include "../adapter/adapter.h"
#include "../adapter/time.h"
#include "include/compressible_neo_hook_material.h"
#include "include/parameter_handling.h"
#include "include/postprocessor.h"

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
    const unsigned int clamped_boundary_id = 1;

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


  // Constructor initializes member variables and reads the parameter file
  template <int dim, typename NumberType>
  Solid<dim, NumberType>::Solid(const std::string &case_path)
    : parameters(
        Parameters::AllParameters(case_path + "nonlinear_elasticity.prm"))
    , vol_reference(0.0)
    , vol_current(0.0)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , degree(parameters.poly_degree)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , // displacement
    dof_handler_ref(triangulation)
    , dofs_per_cell(fe.dofs_per_cell)
    , u_fe(first_u_component)
    , dofs_per_block(n_blocks)
    , qf_cell(parameters.poly_degree + 2)
    , qf_face(parameters.poly_degree + 2)
    , n_q_points(qf_cell.size())
    , n_q_points_f(qf_face.size())
    , boundary_interface_id(7)
    , case_path(case_path)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , time(parameters.end_time, parameters.delta_t)
    , adapter(parameters, boundary_interface_id)
  {}

  // Destructor clears the DoFHandler
  template <int dim, typename NumberType>
  Solid<dim, NumberType>::~Solid()
  {
    dof_handler_ref.clear();
  }


  // As deal typical, the run function starts the calculation
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::run()
  {
    // First, set up a grid and the FE system, as usual
    make_grid();
    system_setup();
    output_results();

    // Initialize preCICE before starting the time loop
    // Here, all information concerning the coupling is passed to preCICE
    adapter.initialize(dof_handler_ref, total_displacement, external_stress);

    BlockVector<NumberType> solution_delta(dofs_per_block);

    // Start the time loop. Steering is done by preCICE itself
    while (adapter.precice.isCouplingOngoing())
      {
        // If we have an implicit coupling, we need to save data before
        // advancing in time in order to restore it later
        adapter.save_current_state_if_required(state_variables, time);

        solution_delta = 0.0;

        time.increment();

        // Solve a the system using the Newton-Raphson algorithm
        solve_nonlinear_timestep(solution_delta);
        total_displacement += solution_delta;

        // Update time dependent variables afterwards
        update_acceleration(solution_delta);
        update_velocity(solution_delta);
        update_old_variables();

        // We are interested in some timings. Here, we measure, how much time we
        // spent through coupling. In case of a parallel coupling schemes, we
        // can directly see the load balancing
        timer.enter_subsection("Advance adapter");
        // ... and pass the coupling data to preCICE, in this case displacement
        // (write data) and stress (read data)
        adapter.advance(total_displacement,
                        external_stress,
                        time.get_delta_t());

        timer.leave_subsection("Advance adapter");

        // Restore the old state, if our implicit time step is not yet converged
        adapter.reload_old_state_if_required(state_variables, time);

        // ...and output results, if the coupling time step has converged
        if (adapter.precice.isTimeWindowComplete() &&
            time.get_timestep() % parameters.output_interval == 0)
          output_results();
      }

    // finalizes preCICE and finishes the simulation
    adapter.precice.finalize();
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::make_grid()
  {
    // Assert here, since dimension information is not available in parameter
    // class and the input is parsed as List
    Assert(
      (dim == 2 && body_force[2] == 0) || dim == 3,
      ExcMessage(
        "Setting body forces in z-direction for a two dimensional simulation has no effect"));

    const std::string testcase(parameters.scenario);

    Point<dim>   point_bottom, point_tip;
    unsigned int id_flap_long_bottom, id_flap_long_top, id_flap_short_bottom,
      id_flap_short_top, n_x, n_y;

    // Assertion is done via a input pattern in the parameter class
    if (testcase == "PF")
      { // flap_perp
        point_bottom =
          dim == 3 ? Point<dim>(-0.05, 0, 0) : Point<dim>(-0.05, 0);
        point_tip = dim == 3 ? Point<dim>(0.05, 1, 0.3) : Point<dim>(0.05, 1);

        // IDs for PF
        id_flap_long_bottom  = 0; // x direction
        id_flap_long_top     = 1;
        id_flap_short_bottom = 2; // y direction
        id_flap_short_top    = 3;

        n_x = 5;
        n_y = 30;
      }
    else // FSI3, don't use condition to avoid wmaybe unitialized warning
      {
        point_bottom = dim == 3 ? Point<dim>(0.24899, 0.19, -0.005) :
                                  Point<dim>(0.24899, 0.19);
        point_tip =
          dim == 3 ? Point<dim>(0.6, 0.21, 0.005) : Point<dim>(0.6, 0.21);

        // IDs for FSI3/CSM2
        id_flap_long_bottom  = 2; // x direction
        id_flap_long_top     = 3;
        id_flap_short_bottom = 0; // y direction
        id_flap_short_top    = 1;

        n_x = 25;
        n_y = 2;
      }

    std::vector<unsigned int> repetitions({n_x, n_y});

    // Generate the mesh
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/ true);


    // refine all cells global_refinement times
    const unsigned int global_refinement = 0;
    triangulation.refine_global(global_refinement);


    // Cell iterator for boundary conditions

    // The boundary ID for Neumann BCs is stored globally to
    // avoid errors.
    // Note, the selected IDs are arbitrarily chosen. They just need to be
    // unique
    const unsigned int neumann_boundary_id = boundary_interface_id;
    // ...and for clamped boundaries. The ID needs to be consistent with the one
    // set in make_constarints. We decided to set one globally, which is reused
    // in make_constraints
    const unsigned int clamped_id = clamped_boundary_id;
    // Not apparent in this cases
    // This might be useful in case you want to overwrite/delete default IDs
    //    const unsigned int do_nothing_boundary_id = 2;

    // Finally, set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true)
          {
            if (face->boundary_id() == id_flap_short_bottom)
              face->set_boundary_id(clamped_id);
            else if (face->boundary_id() == id_flap_long_bottom ||
                     face->boundary_id() == id_flap_long_top ||
                     face->boundary_id() == id_flap_short_top)
              face->set_boundary_id(neumann_boundary_id);
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary "
                                     "condition?"))
          }
    // Check, whether the given IDs are mutually exclusive
    Assert(
      clamped_id != neumann_boundary_id,
      ExcMessage(
        "Boundary IDs must not be the same, for different boundary types."));
    Assert(boundary_interface_id == adapter.deal_boundary_interface_id,
           ExcMessage("Wrong interface ID in the Adapter."));

    vol_reference = GridTools::volume(triangulation);
    vol_current   = vol_reference;
    std::cout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::system_setup()
  {
    timer.enter_subsection("Setup system");

    std::vector<unsigned int> block_component(n_components,
                                              u_dof); // Displacement

    // The DOF handler is then initialised and we renumber the grid in an
    // efficient manner. We also record the number of DOFs per block.
    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);
    DoFRenumbering::component_wise(dof_handler_ref, block_component);
    dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler_ref, block_component);

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: "
              << triangulation.n_active_cells()
              << "\n\t Polynomial degree: " << parameters.poly_degree
              << "\n\t Number of degrees of freedom: "
              << dof_handler_ref.n_dofs() << std::endl;

    tangent_matrix.clear();
    {
      const types::global_dof_index n_dofs_u = dofs_per_block[u_dof];

      BlockDynamicSparsityPattern csp(n_blocks, n_blocks);

      csp.block(u_dof, u_dof).reinit(n_dofs_u, n_dofs_u);
      csp.collect_sizes();

      Table<2, DoFTools::Coupling> coupling(n_components, n_components);
      for (unsigned int ii = 0; ii < n_components; ++ii)
        for (unsigned int jj = 0; jj < n_components; ++jj)
          coupling[ii][jj] = DoFTools::always;
      DoFTools::make_sparsity_pattern(
        dof_handler_ref, coupling, csp, constraints, false);
      sparsity_pattern.copy_from(csp);
    }

    // Setup the sparsity pattern and tangent matrixI
    tangent_matrix.reinit(sparsity_pattern);

    // We then set up storage vectors. Here, one vector for each time dependent
    // variable is needed
    system_rhs.reinit(dofs_per_block);
    system_rhs.collect_sizes();

    total_displacement.reinit(dofs_per_block);
    total_displacement.collect_sizes();

    // Copy initialization
    total_displacement_old.reinit(total_displacement);
    velocity.reinit(total_displacement);
    velocity_old.reinit(total_displacement);
    // TODO: Estimate acc properly in case of body forces
    acceleration.reinit(total_displacement);
    acceleration_old.reinit(total_displacement);
    external_stress.reinit(total_displacement);

    // Alias: Container, which holds references for all time dependent variables
    // to enable a compact notation
    state_variables = std::vector({&total_displacement,
                                   &total_displacement_old,
                                   &velocity,
                                   &velocity_old,
                                   &acceleration,
                                   &acceleration_old});

    setup_qph();

    timer.leave_subsection();
  }

  // Firstly the actual QPH data objects are created. This must be done only
  // once the grid is refined to its finest level.

  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const std::vector<std::shared_ptr<PointHistory<dim, NumberType>>> lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters);
      }
  }


  // The next function is the driver method for the Newton-Raphson scheme. At
  // its top we create a new vector to store the current Newton update step,
  // reset the error storage objects and print solver header.
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::solve_nonlinear_timestep(
    BlockVector<double> &solution_delta)
  {
    std::cout << std::endl
              << "Timestep " << time.get_timestep() << " @ " << std::fixed
              << time.current() << "s" << std::endl;

    BlockVector<double> newton_update(dofs_per_block);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    // We now perform a number of Newton iterations to iteratively solve the
    // nonlinear problem. Since the problem is fully nonlinear and we are
    // using a full Newton method, the data stored in the tangent matrix and
    // right-hand side vector is not reusable and must be cleared at each
    // Newton step.
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
      {
        std::cout << " " << std::setw(2) << newton_iteration << " "
                  << std::flush;

        make_constraints(newton_iteration);
        // Acceleration is evaluated at t_n+1 and therefore updated in each
        // lineraized step
        update_acceleration(solution_delta);

        assemble_system(solution_delta, acceleration, external_stress);

        // Residual error = rhs error
        get_error_residual(error_residual);

        if (newton_iteration == 0)
          error_residual_0 = error_residual;

        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        // Check absolute errors for dynamic cases as well, since there might be
        // situations, with small or even no deformations in coupled setups
        if (newton_iteration > 0 &&
              (error_update_norm.u <= parameters.tol_u &&
               error_residual_norm.u <= parameters.tol_f) ||
            (error_update.u <= 1e-15 && error_residual.u <= 5e-9))
          {
            std::cout << " CONVERGED! " << std::endl;
            print_conv_footer();

            break;
          }

        // Solve the system
        const std::pair<unsigned int, double> lin_solver_output =
          solve_linear_system(newton_update);

        // Update errors
        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;

        // We can now determine the normalised Newton update error, and perform
        // the actual update of the solution increment for the current time
        // step, update all quadrature point information pertaining to this new
        // displacement and stress state and continue iterating:
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        solution_delta += newton_update;

        std::cout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
                  << std::scientific << lin_solver_output.first << "  "
                  << lin_solver_output.second << "  " << error_residual_norm.u
                  << "  " << error_residual.u << "  "
                  << "  " << error_update_norm.u << "  " << error_update.u
                  << "  " << std::endl;
      }

    AssertThrow(newton_iteration < parameters.max_iterations_NR,
                ExcMessage("No convergence in nonlinear solver!"));
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::print_conv_header()
  {
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "    SOLVER STEP    "
              << " |  LIN_IT   LIN_RES    RES_NORM   "
              << "RES_ABS      U_NORM    "
              << " U_ABS " << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;
  }


  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::print_conv_footer()
  {
    error_residual.normalise(error_residual_0);
    error_update.normalise(error_update_0);

    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "Relative errors:" << std::endl
              << "Displacement:\t" << error_update.u << std::endl
              << "Residual: \t" << error_residual.u << std::endl
              << "v / V_0:\t" << vol_current << " / " << vol_reference
              << std::endl;
  }


  // Determine the true residual error for the problem. That is, determine the
  // error in the residual for the unconstrained degrees of freedom. Note that
  // to do so, we need to ignore constrained DOFs by setting the residual in
  // these vector components to zero.
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::get_error_residual(Errors &error_residual)
  {
    BlockVector<double> error_res(dofs_per_block);

    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_res(i) = system_rhs(i);

    error_residual.u = error_res.block(u_dof).l2_norm();
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::get_error_update(
    const BlockVector<double> &newton_update,
    Errors &                   error_update)
  {
    BlockVector<double> error_ud(dofs_per_block);
    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        error_ud(i) = newton_update(i);

    error_update.u = error_ud.block(u_dof).l2_norm();
  }


  // Returns the total displacement needed during assembly
  template <int dim, typename NumberType>
  BlockVector<double>
  Solid<dim, NumberType>::get_total_solution(
    const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(total_displacement);
    solution_total += solution_delta;
    return solution_total;
  }


  // Update the acceleration according to Newmarks method
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::update_acceleration(
    BlockVector<double> &displacement_delta)
  {
    acceleration.equ(alpha_1, displacement_delta);
    acceleration.add(-alpha_2, velocity_old, -alpha_3, acceleration_old);
  }


  // Update the velocity according to Newmarks method
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::update_velocity(
    BlockVector<double> &displacement_delta)
  {
    velocity.equ(alpha_4, displacement_delta);
    velocity.add(alpha_5, velocity_old, alpha_6, acceleration_old);
  }


  // Update variables related to old time steps
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::update_old_variables()

  {
    total_displacement_old = total_displacement;
    velocity_old           = velocity;
    acceleration_old       = acceleration;
  }



  template <int dim, typename NumberType>
  struct Assembler_Base
  {
    virtual ~Assembler_Base()
    {}

    // Here we deal with the tangent matrix assembly structures. The PerTaskData
    // object stores local contributions.
    struct PerTaskData_ASM
    {
      const Solid<dim, NumberType> *       solid;
      FullMatrix<double>                   cell_matrix;
      Vector<double>                       cell_rhs;
      std::vector<types::global_dof_index> local_dof_indices;

      PerTaskData_ASM(const Solid<dim, NumberType> *solid)
        : solid(solid)
        , cell_matrix(solid->dofs_per_cell, solid->dofs_per_cell)
        , cell_rhs(solid->dofs_per_cell)
        , local_dof_indices(solid->dofs_per_cell)
      {}

      void
      reset()
      {
        cell_matrix = 0.0;
        cell_rhs    = 0.0;
      }
    };

    // On the other hand, the ScratchData object stores the larger objects such
    // as the shape-function values array (Nx) and a shape function gradient and
    // symmetric gradient vector which we will use during the assembly.
    struct ScratchData_ASM
    {
      const BlockVector<double> &             solution_total;
      const BlockVector<double> &             acceleration;
      std::vector<Tensor<2, dim, NumberType>> solution_grads_u_total;
      std::vector<Tensor<1, dim, NumberType>> local_acceleration;

      const BlockVector<double> &external_stress;

      FEValues<dim>     fe_values_ref;
      FEFaceValues<dim> fe_face_values_ref;

      std::vector<std::vector<Tensor<2, dim, NumberType>>> grad_Nx;
      std::vector<std::vector<SymmetricTensor<2, dim, NumberType>>>
        symm_grad_Nx;

      std::vector<std::vector<Tensor<1, dim, NumberType>>> shape_value;

      ScratchData_ASM(const FiniteElement<dim> & fe_cell,
                      const QGauss<dim> &        qf_cell,
                      const UpdateFlags          uf_cell,
                      const QGauss<dim - 1> &    qf_face,
                      const UpdateFlags          uf_face,
                      const BlockVector<double> &solution_total,
                      const BlockVector<double> &acceleration,
                      const BlockVector<double> &external_stress)
        : solution_total(solution_total)
        , acceleration(acceleration)
        , solution_grads_u_total(qf_cell.size())
        , local_acceleration(qf_cell.size())
        , external_stress(external_stress)
        , fe_values_ref(fe_cell, qf_cell, uf_cell)
        , fe_face_values_ref(fe_cell, qf_face, uf_face)
        , grad_Nx(qf_cell.size(),
                  std::vector<Tensor<2, dim, NumberType>>(
                    fe_cell.dofs_per_cell))
        , symm_grad_Nx(qf_cell.size(),
                       std::vector<SymmetricTensor<2, dim, NumberType>>(
                         fe_cell.dofs_per_cell))
        , shape_value(qf_cell.size(),
                      std::vector<Tensor<1, dim, NumberType>>(
                        fe_cell.dofs_per_cell))
      {}

      ScratchData_ASM(const ScratchData_ASM &rhs)
        : solution_total(rhs.solution_total)
        , acceleration(rhs.acceleration)
        , solution_grads_u_total(rhs.solution_grads_u_total)
        , local_acceleration(rhs.local_acceleration)
        , external_stress(rhs.external_stress)
        , fe_values_ref(rhs.fe_values_ref.get_fe(),
                        rhs.fe_values_ref.get_quadrature(),
                        rhs.fe_values_ref.get_update_flags())
        , fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
                             rhs.fe_face_values_ref.get_quadrature(),
                             rhs.fe_face_values_ref.get_update_flags())
        , grad_Nx(rhs.grad_Nx)
        , symm_grad_Nx(rhs.symm_grad_Nx)
        , shape_value(rhs.shape_value)
      {}

      void
      reset()
      {
        const unsigned int n_q_points = fe_values_ref.get_quadrature().size();
        const unsigned int n_dofs_per_cell = fe_values_ref.dofs_per_cell;
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            Assert(grad_Nx[q_point].size() == n_dofs_per_cell,
                   ExcInternalError());
            Assert(symm_grad_Nx[q_point].size() == n_dofs_per_cell,
                   ExcInternalError());

            solution_grads_u_total[q_point] = Tensor<2, dim, NumberType>();
            for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
              {
                grad_Nx[q_point][k] = Tensor<2, dim, NumberType>();
                symm_grad_Nx[q_point][k] =
                  SymmetricTensor<2, dim, NumberType>();
                shape_value[q_point][k] = Tensor<1, dim, NumberType>();
              }
          }
      }
    };
    // Due to the C++ specialization rules, we need one more level of
    // indirection in order to define the assembly routine for all different
    // number. The next function call is specialized for each NumberType, but to
    // prevent having to specialize the whole class along with it we have
    // inlined the definition of the other functions that are common to all
    // implementations.
    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data)
    {
      assemble_system_tangent_residual_one_cell(cell, scratch, data);
      assemble_neumann_contribution_one_cell(cell, scratch, data);
    }

    // This function adds the local contribution to the system matrix.
    void
    copy_local_to_global_ASM(const PerTaskData_ASM &data)
    {
      const AffineConstraints<double> &constraints = data.solid->constraints;
      BlockSparseMatrix<double> &      tangent_matrix =
        const_cast<Solid<dim, NumberType> *>(data.solid)->tangent_matrix;
      BlockVector<double> &system_rhs =
        const_cast<Solid<dim, NumberType> *>(data.solid)->system_rhs;

      constraints.distribute_local_to_global(data.cell_matrix,
                                             data.cell_rhs,
                                             data.local_dof_indices,
                                             tangent_matrix,
                                             system_rhs);
    }

    // This function needs to exist in the base class for Workstream to work
    // with a reference to the base class.
  protected:
    virtual void
    assemble_system_tangent_residual_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator & /*cell*/,
      ScratchData_ASM & /*scratch*/,
      PerTaskData_ASM & /*data*/)
    {
      AssertThrow(false, ExcPureFunctionCalled());
    }

    // Next we assemble the Neumann contribution. We first check to see, if the
    // cell face exists on a boundary on which the stress is applied i.e. in our
    // case the coupling boundary.
    void
    assemble_neumann_contribution_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data)
    {
      const unsigned int & n_q_points_f      = data.solid->n_q_points_f;
      const unsigned int & dofs_per_cell     = data.solid->dofs_per_cell;
      const FESystem<dim> &fe                = data.solid->fe;
      const unsigned int & u_dof             = data.solid->u_dof;
      const FEValuesExtractors::Vector &u_fe = data.solid->u_fe;
      const unsigned int &interf_id = data.solid->boundary_interface_id;

      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true && face->boundary_id() == interf_id)
          {
            scratch.fe_face_values_ref.reinit(cell, face);

            // Initialize vector for values at each quad point
            std::vector<Tensor<1, dim, NumberType>> local_stress(n_q_points_f);

            // Then, we extract the proper values from the global
            // external_stress vector, which has already been filled with the
            // coupling data
            scratch.fe_face_values_ref[u_fe].get_function_values(
              scratch.external_stress, local_stress);

            for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                 ++f_q_point)
              {
                // In the next step, we perform a pull_back operation, since our
                // Fluid participant usually works with ALE methods and the
                // structure solver here assembles everything in reference
                // coniguration
                const Tensor<2, dim, NumberType> F =
                  Physics::Elasticity::Kinematics::F(
                    scratch.solution_grads_u_total[f_q_point]);

                const Tensor<1, dim, NumberType> referential_stress =
                  Physics::Transformations::Covariant::pull_back(
                    local_stress[f_q_point], F);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int i_group =
                      fe.system_to_base_index(i).first.first;

                    if (i_group == u_dof)
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        const double Ni =
                          scratch.fe_face_values_ref.shape_value(i, f_q_point);
                        const double JxW =
                          scratch.fe_face_values_ref.JxW(f_q_point);

                        data.cell_rhs(i) +=
                          (Ni * referential_stress[component_i]) * JxW;
                      }
                  }
              }
          }
    }
  };

  template <int dim>
  struct Assembler<dim, double> : Assembler_Base<dim, double>
  {
    typedef double NumberType;
    using typename Assembler_Base<dim, NumberType>::ScratchData_ASM;
    using typename Assembler_Base<dim, NumberType>::PerTaskData_ASM;

    virtual ~Assembler()
    {}

    virtual void
    assemble_system_tangent_residual_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data)
    {
      // Aliases for data referenced from the Solid class
      const unsigned int & n_q_points           = data.solid->n_q_points;
      const unsigned int & dofs_per_cell        = data.solid->dofs_per_cell;
      const FESystem<dim> &fe                   = data.solid->fe;
      const unsigned int & u_dof                = data.solid->u_dof;
      const FEValuesExtractors::Vector &u_fe    = data.solid->u_fe;
      const double &                    alpha_1 = data.solid->alpha_1;

      // Define const force vector for gravity
      const Tensor<1, 3, double> body_force = data.solid->body_force;

      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>>
        lqph = const_cast<const Solid<dim, NumberType> *>(data.solid)
                 ->quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      // We first need to find the solution gradients at quadrature points
      // inside the current cell and then we update each local QP using the
      // displacement gradient:
      scratch.fe_values_ref[u_fe].get_function_gradients(
        scratch.solution_total, scratch.solution_grads_u_total);

      scratch.fe_values_ref[u_fe].get_function_values(
        scratch.acceleration, scratch.local_acceleration);

      // Const in the whole domain, so we call it once for each cell
      const double rho = lqph[0]->get_rho();

      // Now we build the local cell stiffness matrix. Since the global and
      // local system matrices are symmetric, we can exploit this property by
      // building only the lower half of the local matrix and copying the values
      // to the upper half.
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
          // In doing so, we first extract some configuration dependent
          // variables from our QPH history objects for the current quadrature
          // point.
          // Get kinematic variables
          const Tensor<2, dim, NumberType> &grad_u =
            scratch.solution_grads_u_total[q_point];

          const Tensor<1, dim, NumberType> &acc =
            scratch.local_acceleration[q_point];

          const Tensor<2, dim, NumberType> F =
            Physics::Elasticity::Kinematics::F(grad_u);
          const NumberType                 det_F = determinant(F);
          const Tensor<2, dim, NumberType> F_bar =
            Physics::Elasticity::Kinematics::F_iso(F);
          const SymmetricTensor<2, dim, NumberType> b_bar =
            Physics::Elasticity::Kinematics::b(F_bar);
          const Tensor<2, dim, NumberType> F_inv = invert(F);
          Assert(det_F > NumberType(0.0), ExcInternalError());


          // Update scratch data
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              const unsigned int k_group =
                fe.system_to_base_index(k).first.first;

              if (k_group == u_dof)
                {
                  scratch.grad_Nx[q_point][k] =
                    scratch.fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                  scratch.symm_grad_Nx[q_point][k] =
                    symmetrize(scratch.grad_Nx[q_point][k]);
                  scratch.shape_value[q_point][k] =
                    scratch.fe_values_ref[u_fe].value(k, q_point);
                }
              else
                Assert(k_group <= u_dof, ExcInternalError());
            }

          // Get material contributions
          const SymmetricTensor<2, dim, NumberType> tau =
            lqph[q_point]->get_tau(det_F, b_bar);
          const SymmetricTensor<4, dim, NumberType> Jc =
            lqph[q_point]->get_Jc(det_F, b_bar);
          const Tensor<2, dim, NumberType> tau_ns(tau);


          // Aliases for readability
          const std::vector<SymmetricTensor<2, dim>> &symm_grad_Nx =
            scratch.symm_grad_Nx[q_point];
          const std::vector<Tensor<2, dim>> &grad_Nx = scratch.grad_Nx[q_point];
          const double JxW = scratch.fe_values_ref.JxW(q_point);
          const std::vector<Tensor<1, dim, NumberType>> &shape_value =
            scratch.shape_value[q_point];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;
              const unsigned int i_group =
                fe.system_to_base_index(i).first.first;

              // Residual assembly
              if (i_group == u_dof)
                {
                  // Geometrical stress and body force contribution
                  data.cell_rhs(i) -=
                    ((symm_grad_Nx[i] * tau) -
                     (body_force[component_i] * rho *
                      scratch.fe_values_ref.shape_value(i, q_point))) *
                    JxW;
                  // Mass matrix contribution with acceleration
                  // Cannot be merged with tangent assembly, since there, the
                  // matrix symmetry is utilized i.e. only the upper half is
                  // assembled
                  for (uint j = 0; j < dofs_per_cell; ++j)
                    data.cell_rhs(i) -= shape_value[i] * rho * shape_value[j] *
                                        acc[component_i] * JxW;
                }
              else
                Assert(i_group <= u_dof, ExcInternalError());

              // Tangent assembly
              for (unsigned int j = 0; j <= i; ++j)
                {
                  const unsigned int component_j =
                    fe.system_to_component_index(j).first;
                  const unsigned int j_group =
                    fe.system_to_base_index(j).first.first;

                  if ((i_group == j_group) && (i_group == u_dof))
                    {
                      // The material contribution:
                      data.cell_matrix(i, j) +=
                        symm_grad_Nx[i] * Jc * symm_grad_Nx[j] * JxW;

                      // Geometrical stress and mass matrix contributions
                      if (component_i == component_j)
                        {
                          data.cell_matrix(i, j) +=
                            (grad_Nx[i][component_i] * tau_ns *
                               grad_Nx[j][component_j] +
                             shape_value[i][component_i] * rho * alpha_1 *
                               shape_value[j][component_j]) *
                            JxW;
                        }
                    }
                  else
                    Assert((i_group <= u_dof) && (j_group <= u_dof),
                           ExcInternalError());
                }
            }
        }

      // Copy triangular matrix
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
          data.cell_matrix(i, j) = data.cell_matrix(j, i);
    }
  };

  // Since we use TBB for assembly, we simply setup a copy of the data
  // structures required for the process and pass them, along with the memory
  // addresses of the assembly functions to the WorkStream object for
  // processing. Note that we must ensure that the matrix is reset before any
  // assembly operations can occur.
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::assemble_system(
    const BlockVector<double> &solution_delta,
    const BlockVector<double> &acceleration,
    const BlockVector<double> &external_stress)
  {
    timer.enter_subsection("Assemble linear system");
    std::cout << " ASM " << std::flush;

    tangent_matrix = 0.0;
    system_rhs     = 0.0;

    const UpdateFlags uf_cell(update_values | update_gradients |
                              update_JxW_values);
    const UpdateFlags uf_face(update_values | update_JxW_values);

    const BlockVector<double> solution_total(
      get_total_solution(solution_delta));
    typename Assembler_Base<dim, NumberType>::PerTaskData_ASM per_task_data(
      this);
    typename Assembler_Base<dim, NumberType>::ScratchData_ASM scratch_data(
      fe,
      qf_cell,
      uf_cell,
      qf_face,
      uf_face,
      solution_total,
      acceleration,
      external_stress);

    Assembler<dim, NumberType> assembler;

    WorkStream::run(dof_handler_ref.begin_active(),
                    dof_handler_ref.end(),
                    static_cast<Assembler_Base<dim, NumberType> &>(assembler),
                    &Assembler_Base<dim, NumberType>::assemble_system_one_cell,
                    &Assembler_Base<dim, NumberType>::copy_local_to_global_ASM,
                    scratch_data,
                    per_task_data);

    timer.leave_subsection();
  }

  // The constraints for this problem are simple to describe. However, since we
  // are dealing with an iterative Newton method, it should be noted that any
  // displacement constraints should only be specified at the zeroth iteration
  // and subsequently no additional contributions are to be made since the
  // constraints are already exactly satisfied.
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::make_constraints(const int &it_nr)
  {
    std::cout << " CST " << std::flush;

    if (it_nr > 1)
      return;
    constraints.clear();
    const bool apply_dirichlet_bc = (it_nr == 0);


    {
      // Fix in every direction
      const int boundary_id = clamped_boundary_id;

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(
                                                   n_components),
                                                 constraints,
                                                 fe.component_mask(u_fe));
      else
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(
                                                   n_components),
                                                 constraints,
                                                 fe.component_mask(u_fe));
    }

    if (dim == 3)
      {
        // The FEValuesExtractors allow to fix only a certain direction, in this
        // case the z-direction
        const int                        boundary_id = clamped_boundary_id;
        const FEValuesExtractors::Scalar z_displacement(2);

        if (apply_dirichlet_bc == true)
          VectorTools::interpolate_boundary_values(
            dof_handler_ref,
            boundary_id,
            ZeroFunction<dim>(n_components),
            constraints,
            fe.component_mask(z_displacement));
        else
          VectorTools::interpolate_boundary_values(
            dof_handler_ref,
            boundary_id,
            ZeroFunction<dim>(n_components),
            constraints,
            fe.component_mask(z_displacement));
      }

    constraints.close();
  }

  // Finally, solve the system
  template <int dim, typename NumberType>
  std::pair<unsigned int, double>
  Solid<dim, NumberType>::solve_linear_system(
    BlockVector<double> &newton_update)
  {
    BlockVector<double> A(dofs_per_block);
    BlockVector<double> B(dofs_per_block);

    unsigned int lin_it  = 0;
    double       lin_res = 0.0;

    {
      timer.enter_subsection("Linear solver");
      std::cout << " SLV " << std::flush;
      if (parameters.type_lin == "CG")
        {
          const int solver_its = tangent_matrix.block(u_dof, u_dof).m() *
                                 parameters.max_iterations_lin;
          const double tol_sol =
            parameters.tol_lin * system_rhs.block(u_dof).l2_norm();

          SolverControl solver_control(solver_its, tol_sol);

          GrowingVectorMemory<Vector<double>> GVM;
          SolverCG<Vector<double>>            solver_CG(solver_control, GVM);

          // TODO: Change to different one
          PreconditionSelector<SparseMatrix<double>, Vector<double>>
            preconditioner("ssor", .65);
          preconditioner.use_matrix(tangent_matrix.block(u_dof, u_dof));

          solver_CG.solve(tangent_matrix.block(u_dof, u_dof),
                          newton_update.block(u_dof),
                          system_rhs.block(u_dof),
                          preconditioner);

          lin_it  = solver_control.last_step();
          lin_res = solver_control.last_value();
        }
      else if (parameters.type_lin == "Direct")
        {
          SparseDirectUMFPACK A_direct;
          A_direct.initialize(tangent_matrix.block(u_dof, u_dof));
          A_direct.vmult(newton_update.block(u_dof), system_rhs.block(u_dof));

          lin_it  = 1;
          lin_res = 0.0;
        }
      else
        Assert(parameters.type_lin == "Direct" || parameters.type_lin == "CG",
               ExcMessage("Linear solver type not implemented"));

      timer.leave_subsection();
    }

    constraints.distribute(newton_update);

    return std::make_pair(lin_it, lin_res);
  }


  // Write ouput files
  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::output_results() const
  {
    timer.enter_subsection("Output results");
    DataOut<dim> data_out;

    // Note: There is at least paraView v 5.5 needed to visualize this output
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler_ref);
    // Postprocessed data is provided by the Postprocessor
    Postprocessor<dim> postprocessor;
    data_out.add_data_vector(total_displacement, postprocessor);

    // To visualize everything on a displaced grid
    Vector<double> soln(total_displacement.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      soln(i) = total_displacement(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);

    data_out.build_patches(q_mapping, degree, DataOut<dim>::curved_boundary);

    std::ostringstream filename;
    filename << case_path << "solution-"
             << time.get_timestep() / parameters.output_interval << ".vtk";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);
    timer.leave_subsection("Output results");
  }

} // namespace Nonlinear_Elasticity

int
main(int argc, char **argv)
{
  using namespace Nonlinear_Elasticity;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      deallog.depth_console(0);

      constexpr unsigned int dimension = 2;

      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[1];
      else
        parameter_file = "nonlinear_elasticity.prm";

      // Extract case path for the output directory
      size_t      pos = parameter_file.find_last_of("/");
      std::string case_path =
        std::string::npos == pos ? "" : parameter_file.substr(0, pos + 1);

      Solid<dimension> solid(case_path);
      solid.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
