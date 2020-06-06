#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
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

#include <fstream>
#include <iostream>

#include "../adapter/adapter.h"
#include "../adapter/time.h"
#include "include/parameter_handling.h"
#include "include/postprocessor.h"
#include "precice/SolverInterface.hpp"

// The Linear_Elasticity case includes a linear elastic material with a one-step
// theta time integration
namespace Linear_Elasticity
{
  using namespace dealii;

  template <int dim>
  class ElastoDynamics
  {
  public:
    ElastoDynamics(const std::string &case_path);
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
    Adapter::Time                                                    time;
    Adapter::Adapter<dim, Vector<double>, Parameters::AllParameters> adapter;

    // Alias for all time dependent variables, which should be saved/reloaded
    // in case of an implicit coupling. This vector is directly used in the
    // Adapter class
    std::vector<Vector<double> *> state_variables;
    // for the output directory
    const std::string case_path;
  };



  // Constructor
  template <int dim>
  ElastoDynamics<dim>::ElastoDynamics(const std::string &case_path)
    : parameters(case_path + "linear_elasticity.prm")
    , interface_boundary_id(6)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , mapping(MappingQGeneric<dim>(parameters.poly_degree))
    , quad_order(parameters.poly_degree + 1)
    , body_force_enabled(parameters.body_force.norm() > 1e-15)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , time(parameters.end_time, parameters.delta_t)
    , adapter(parameters, interface_boundary_id)
    , case_path(case_path)
  {}



  // Destructor
  template <int dim>
  ElastoDynamics<dim>::~ElastoDynamics()
  {
    dof_handler.clear();
  }



  template <int dim>
  void
  ElastoDynamics<dim>::make_grid()
  {
    uint n_x, n_y, n_z;

    // Both preconfigured cases consist of a rectangle
    Point<dim> point_bottom;
    Point<dim> point_tip;

    // boundary IDs are obtained through colorize = true
    uint id_flap_long_bottom, id_flap_long_top, id_flap_short_bottom,
      id_flap_short_top, id_flap_out_of_plane_bottom, id_flap_out_of_plane_top;

    // Hron & Turek FSI3 case
    if (parameters.scenario == "FSI3")
      {
        // FSI 3
        n_x          = 18;
        n_y          = 3;
        n_z          = 1;
        point_bottom = dim == 3 ? Point<dim>(0.24899, 0.19, -0.005) :
                                  Point<dim>(0.24899, 0.19);
        point_tip =
          dim == 3 ? Point<dim>(0.6, 0.21, 0.005) : Point<dim>(0.6, 0.21);

        // IDs for FSI3
        id_flap_long_bottom  = 2; // x direction
        id_flap_long_top     = 3;
        id_flap_short_bottom = 0; // y direction
        id_flap_short_top    = 1;
      }
    else
      {
        // Flap_perp case
        n_x = 3;
        n_y = 18;
        n_z = 1;
        point_bottom =
          dim == 3 ? Point<dim>(-0.05, 0, 0) : Point<dim>(-0.05, 0);
        point_tip = dim == 3 ? Point<dim>(0.05, 1, 0.3) : Point<dim>(0.05, 1);

        // IDs for PF
        id_flap_long_bottom  = 0; // x direction
        id_flap_long_top     = 1;
        id_flap_short_bottom = 2; // y direction
        id_flap_short_top    = 3;
      }

    // Same for both scenarios, only relevant for quasi-2D
    id_flap_out_of_plane_bottom = 4; // z direction
    id_flap_out_of_plane_top    = 5;

    // Vector of dim values denoting the number of cells to generate in that
    // direction
    std::vector<unsigned int> repetitions(dim);
    repetitions[0] = n_x;
    repetitions[1] = n_y;
    if (dim == 3)
      repetitions[2] = n_z;

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/ true);

    // Refine all cells global_refinement times
    const unsigned int global_refinement = 0;
    triangulation.refine_global(global_refinement);

    // Set the desired IDs for clamped boundaries and out_of_plane clamped
    // boundaries. The interface ID (refering to the coupling) is specified in
    // the Constructor, since it is needed by the Constructor of the Adapter
    // class.
    clamped_mesh_id              = 0;
    out_of_plane_clamped_mesh_id = 4;

    // The IDs must not be the same:
    std::string error_message(
      "The interface_id cannot be the same as the clamped one");
    Assert(clamped_mesh_id != interface_boundary_id, ExcMessage(error_message));
    Assert(out_of_plane_clamped_mesh_id != interface_boundary_id,
           ExcMessage(error_message));
    Assert(interface_boundary_id == adapter.deal_boundary_interface_id,
           ExcMessage("Wrong interface ID in the Adapter specified"));

    // Iterate over all cells and set the IDs
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true)
          {
            // Boundaries for the interface
            if (face->boundary_id() == id_flap_short_top ||
                face->boundary_id() == id_flap_long_bottom ||
                face->boundary_id() == id_flap_long_top)
              face->set_boundary_id(interface_boundary_id);
            // Boundaries clamped in all directions
            else if (face->boundary_id() == id_flap_short_bottom)
              face->set_boundary_id(clamped_mesh_id);
            // Boundaries clamped out-of-plane (z) direction
            else if (face->boundary_id() == id_flap_out_of_plane_bottom ||
                     face->boundary_id() == id_flap_out_of_plane_top)
              face->set_boundary_id(out_of_plane_clamped_mesh_id);
          }
  }



  template <int dim>
  void
  ElastoDynamics<dim>::setup_system()
  {
    // This follows the usual dealii steps
    dof_handler.distribute_dofs(fe);
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    hanging_node_constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    // Initialize relevant matrices
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);
    stepping_matrix.reinit(sparsity_pattern);

    // Initialize all vectors
    old_velocity.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());

    old_displacement.reinit(dof_handler.n_dofs());
    displacement.reinit(dof_handler.n_dofs());

    system_rhs.reinit(dof_handler.n_dofs());
    old_stress.reinit(dof_handler.n_dofs());
    stress.reinit(dof_handler.n_dofs());

    body_force_vector.reinit(dof_handler.n_dofs());

    std::cout << "Triangulation:"
              << "\n\t Number of active cells: "
              << triangulation.n_active_cells()
              << "\n\t Polynomial degree: " << parameters.poly_degree
              << "\n\t Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    // Define alias for time dependent variables as described above
    state_variables = {
      &old_velocity, &velocity, &old_displacement, &displacement, &old_stress};

    // loads at time 0
    // TODO: Check, if initial conditions should be set at the beginning
    old_stress = 0.0;
  }



  template <int dim>
  void
  ElastoDynamics<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(quad_order);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    // Lame constants
    Functions::ConstantFunction<dim> lambda(parameters.lambda),
      mu(parameters.mu);

    // Assemble the stiffness matrix according to a linear material law using
    // the lame paramters
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        // Next we get the values of the coefficients at the quadrature
        // points.
        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);


        // Then assemble the entries of the local stiffness matrix
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const unsigned int component_j =
                  fe.system_to_component_index(j).first;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                  {
                    cell_matrix(i, j) +=
                      // the first term is (lambda d_i u_i, d_j v_j) + (mu d_i
                      // u_j, d_j v_i).
                      (                                                  //
                        (fe_values.shape_grad(i, q_point)[component_i] * //
                         fe_values.shape_grad(j, q_point)[component_j] * //
                         lambda_values[q_point])                         //
                        +                                                //
                        (fe_values.shape_grad(i, q_point)[component_j] * //
                         fe_values.shape_grad(j, q_point)[component_i] * //
                         mu_values[q_point])                             //
                        +                                                //
                        // the second term is (mu nabla u_i, nabla v_j).
                        ((component_i == component_j) ?        //
                           (fe_values.shape_grad(i, q_point) * //
                            fe_values.shape_grad(j, q_point) * //
                            mu_values[q_point]) :              //
                           0)                                  //
                        ) *                                    //
                      fe_values.JxW(q_point);                  //
                  }
              }
          }


        // The transfer from local degrees of freedom into the global matrix
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              stiffness_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i, j));
          }
      }


    // Here, we use the MatrixCreator to create a mass matrix, which is constant
    // through the whole simulation
    {
      Functions::ConstantFunction<dim> rho_f(parameters.rho);

      MatrixCreator::create_mass_matrix(
        mapping, dof_handler, QGauss<dim>(quad_order), mass_matrix, &rho_f);
    }

    // Then, we save the system_matrix, which is needed every timestep
    stepping_matrix.copy_from(stiffness_matrix);

    stepping_matrix *= time.get_delta_t() * time.get_delta_t() *
                       parameters.theta * parameters.theta;

    stepping_matrix.add(1, mass_matrix);

    hanging_node_constraints.condense(stepping_matrix);

    // Calculate contribution of gravity and store them in gravitational_force
    if (body_force_enabled)
      {
        Vector<double> bf_vector(dim);
        for (uint d = 0; d < dim; ++d)
          bf_vector[d] = parameters.rho * parameters.body_force[d];

        // Create a constant function object
        Functions::ConstantFunction<dim> bf_function(bf_vector);

        // Create the contribution to the right-hand side vector
        VectorTools::create_right_hand_side(mapping,
                                            dof_handler,
                                            QGauss<dim>(quad_order),
                                            bf_function,
                                            body_force_vector);
      }
  }


  // Process RHS assembly, which is the coupling data (stress) in this case
  template <int dim>
  void
  ElastoDynamics<dim>::assemble_rhs()
  {
    timer.enter_subsection("Assemble rhs");

    // Initialize all objects as usual
    system_rhs = 0.0;

    // Quadrature formula for integration over faces (dim-1)
    QGauss<dim - 1> face_quadrature_formula(quad_order);

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.dofs_per_cell;
    const unsigned int n_face_q_points = face_quadrature_formula.size();

    Vector<double>                       cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    // In order to get the local fe values
    std::vector<Vector<double>> local_stress(n_face_q_points,
                                             Vector<double>(dim));

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;

        // Assemblw the right-hand side force vector each timestep
        // by applying contributions only on the coupling interface
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true &&
              face->boundary_id() == interface_boundary_id)
            {
              fe_face_values.reinit(cell, face);
              // Extract relevant data from the global stress vector by using
              // 'get_function_values()'
              // In contrast to the nonlinear solver, no pull back is performed.
              // The equilibrium is stated in reference configuration, but only
              // valid for very small deformations
              fe_face_values.get_function_values(stress, local_stress);

              for (unsigned int f_q_point = 0; f_q_point < n_face_q_points;
                   ++f_q_point)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    const unsigned int component_i =
                      fe.system_to_component_index(i).first;

                    cell_rhs(i) += fe_face_values.shape_value(i, f_q_point) *
                                   local_stress[f_q_point][component_i] *
                                   fe_face_values.JxW(f_q_point);
                  }
            }

        // Local dofs to global
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    // Update time dependent variables related to the previous time step t_n
    old_velocity     = velocity;
    old_displacement = displacement;

    // Add contribution of body forces, if necessary
    if (body_force_enabled)
      system_rhs.add(1, body_force_vector);

    // Assemble global RHS:
    // RHS=(M-theta*(1-theta)*delta_t^2*K)*V_n - delta_t*K* D_n +
    // delta_t*theta*F_n+1 + delta_t*(1-theta)*F_n

    // tmp vector to store intermediate results
    Vector<double> tmp;
    tmp.reinit(dof_handler.n_dofs());

    tmp = system_rhs;

    system_rhs *= time.get_delta_t() * parameters.theta;
    system_rhs.add(time.get_delta_t() * (1 - parameters.theta), old_stress);
    old_stress = tmp;

    mass_matrix.vmult(tmp, old_velocity);
    system_rhs.add(1, tmp);

    stiffness_matrix.vmult(tmp, old_velocity);
    system_rhs.add(-parameters.theta * time.get_delta_t() * time.get_delta_t() *
                     (1 - parameters.theta),
                   tmp);

    stiffness_matrix.vmult(tmp, old_displacement);
    system_rhs.add(-time.get_delta_t(), tmp);

    hanging_node_constraints.condense(system_rhs);

    // Copy the system_matrix every timestep, since applying the BC deletes
    // certain rows and columns
    system_matrix = 0.0;
    system_matrix.copy_from(stepping_matrix);

    // Set Dirichlet BCs:
    // clamped in all directions
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             clamped_mesh_id,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);
    if (dim == 3)
      {
        const FEValuesExtractors::Scalar z_component(2);
        // clamped out_of_plane
        VectorTools::interpolate_boundary_values(
          dof_handler,
          out_of_plane_clamped_mesh_id,
          Functions::ZeroFunction<dim>(dim),
          boundary_values,
          fe.component_mask(z_component));
      }

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       velocity,
                                       system_rhs);

    timer.leave_subsection("Assemble rhs");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::solve()
  {
    timer.enter_subsection("Solve system");

    uint   lin_it  = 1;
    double lin_res = 0.0;

    // Solve the linear system either using an iterative CG solver with SSOR or
    // a direct solver using UMFPACK
    if (parameters.type_lin == "CG")
      {
        std::cout << "\t CG solver: " << std::endl;

        const int solver_its =
          system_matrix.m() * parameters.max_iterations_lin;
        const double tol_sol = parameters.tol_lin * system_rhs.l2_norm();

        SolverControl         solver_control(solver_its, tol_sol);
        GrowingVectorMemory<> GVM;
        SolverCG<>            solver_CG(solver_control, GVM);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        solver_CG.solve(system_matrix, velocity, system_rhs, preconditioner);

        lin_it  = solver_control.last_step();
        lin_res = solver_control.last_value();
      }
    else if (parameters.type_lin == "Direct")
      {
        std::cout << "\t Direct solver: " << std::endl;

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(velocity, system_rhs);
      }
    else
      Assert(parameters.type_lin == "Direct" || parameters.type_lin == "CG",
             ExcNotImplemented());

    // assert divergence
    Assert(velocity.linfty_norm() < 1e4, ExcMessage("Linear system diverged"));
    std::cout << "\t     No of iterations:\t" << lin_it
              << "\n \t     Final residual:\t" << lin_res << std::endl;
    hanging_node_constraints.distribute(velocity);

    timer.leave_subsection("Solve system");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::update_displacement()
  {
    // D_n+1= D_n + delta_t*theta* V_n+1 + delta_t*(1-theta)* V_n
    displacement.add(time.get_delta_t() * parameters.theta, velocity);
    displacement.add(time.get_delta_t() * (1 - parameters.theta), old_velocity);
  }



  template <int dim>
  void
  ElastoDynamics<dim>::output_results() const
  {
    timer.enter_subsection("Output results");
    DataOut<dim> data_out;

    // Note: There is at least paraView v 5.5 needed to visualize this output
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);

    // The postprocessor class computes straines and passes the displacement to
    // the output
    Postprocessor<dim> postprocessor;
    data_out.add_data_vector(displacement, postprocessor);

    // visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(parameters.poly_degree,
                                    dof_handler,
                                    displacement);
    data_out.build_patches(q_mapping,
                           parameters.poly_degree,
                           DataOut<dim>::curved_boundary);

    // Check, if the output directory exists
    std::ifstream output_directory(case_path + "dealii_output");
    Assert(
      output_directory,
      ExcMessage(
        "Unable to find the output directory. "
        "By default, this program stores result files in a directory called dealii_output. "
        "This needs to be located in your case directory, where the parameter file is located as well."));

    // Store all files in a seperate folder called dealii_ouput
    std::ofstream output(
      case_path + "dealii_output/solution-" +
      std::to_string(time.get_timestep() / parameters.output_interval) +
      ".vtk");
    data_out.write_vtk(output);
    std::cout << "\t Output written to solution-" +
                   std::to_string(time.get_timestep() /
                                  parameters.output_interval) +
                   ".vtk \n"
              << std::endl;
    timer.leave_subsection("Output results");
  }



  template <int dim>
  void
  ElastoDynamics<dim>::run()
  {
    // In the beginning, we create the mesh and set up the data structures
    make_grid();
    setup_system();
    output_results();
    assemble_system();

    // Then, we initialize preCICE i.e. we pass our mesh and coupling
    // information to preCICE
    adapter.initialize(dof_handler, displacement, stress);

    // Then, we start the time loop. The loop itself is steered by preCICE. This
    // line replaces the usual 'while( time < end_time)'
    while (adapter.precice.isCouplingOngoing())
      {
        // In case of an implicit coupling, we need to store time dependent
        // data, in order to reload it later. The decision, whether it is
        // necessary to store the data is handled by preCICE as well
        adapter.save_current_state_if_required(state_variables, time);

        // Afterwards, we start the actual time step computation
        time.increment();

        std::cout << std::endl
                  << "Timestep " << time.get_timestep() << " @ " << std::fixed
                  << time.current() << "s" << std::endl;

        // Assemble the time dependent contribution obtained from the Fluid
        // participant
        assemble_rhs();

        // ...and solver the system
        solve();

        // Update time dependent data according to the theta-scheme
        update_displacement();

        // Then, we exchange data with other participants. Most of the work is
        // done in the adapter: We just need to pass both data vectors with
        // coupling data to the adapter. In case of FSI, 'displacement' is the
        // data we calculate and pass to preCICE and 'stress' is the (global)
        // vector filled by preCICE/ the Fluid participant.
        // Depending on the coupling scheme, we need to wait here for other
        // participant to finish their time step. Therefore, we measure the
        // timings around this functionality
        timer.enter_subsection("Advance adapter");
        adapter.advance(displacement, stress, time.get_delta_t());
        timer.leave_subsection("Advance adapter");

        // Next, we reload the data we have previosuly stored in the beginning
        // of the time loop. This is only relevant for implicit couplings and
        // preCICE steeres the reloading depending on the specific
        // configuration.
        adapter.reload_old_state_if_required(state_variables, time);

        // At last, we ask preCICE, whether this coupling time step (= time
        // window in preCICE terms) is finished and write the result files
        if (adapter.precice.isTimeWindowComplete() &&
            time.get_timestep() % parameters.output_interval == 0)
          output_results();
      }

    // After the time loop, we finalize the coupling i.e. terminate
    // communication etc.
    adapter.precice.finalize();
  }
} // namespace Linear_Elasticity

int
main(int argc, char **argv)
{
  using namespace Linear_Elasticity;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[1];
      else
        parameter_file = "linear_elasticity.prm";

      // Extract case path for the output directory
      size_t      pos = parameter_file.find_last_of("/");
      std::string case_path =
        std::string::npos == pos ? "" : parameter_file.substr(0, pos + 1);

      ElastoDynamics<DIM> elastic_solver(case_path);
      elastic_solver.run();
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
