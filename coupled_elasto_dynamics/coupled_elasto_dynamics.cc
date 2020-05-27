#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

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

#include "include/coupling_functions.h"
#include "include/parameter_handling.h"
#include "include/time.h"
#include "precice/SolverInterface.hpp"

namespace Linear_Elasticity
{
  using namespace dealii;

  // evaluate strains as a tensor in the data out object
  template <int dim>
  class StrainPostprocessor : public DataPostprocessorTensor<dim>
  {
  public:
    StrainPostprocessor()
      : DataPostprocessorTensor<dim>("strain", update_gradients)
    {}
    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &               computed_quantities) const
    {
      AssertDimension(input_data.solution_gradients.size(),
                      computed_quantities.size());
      for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
        {
          AssertDimension(computed_quantities[p].size(),
                          (Tensor<2, dim>::n_independent_components));
          for (unsigned int d = 0; d < dim; ++d)
            for (unsigned int e = 0; e < dim; ++e)
              computed_quantities[p]
                                 [Tensor<2, dim>::component_to_unrolled_index(
                                   TableIndices<2>(d, e))] =
                                   (input_data.solution_gradients[p][d][e] +
                                    input_data.solution_gradients[p][e][d]) /
                                   2;
        }
    }
  };


  template <int dim>
  class CoupledElastoDynamics
  {
  public:
    CoupledElastoDynamics(const std::string &case_path);
    ~CoupledElastoDynamics();
    void
    run();

  private:
    void
    make_grid();
    void
    setup_system();
    void
    assemble_system();
    void
    assemble_rhs();
    void
    solve();
    void
    update_displacement();
    void
    output_results() const;


    Adapter::Parameters::AllParameters parameters;

    // grid related variables
    Triangulation<dim> triangulation;
    unsigned int       clamped_mesh_id;
    unsigned int       out_of_plane_clamped_mesh_id;

    Adapter::Time   time;
    DoFHandler<dim> dof_handler;

    FESystem<dim>        fe;
    MappingQGeneric<dim> mapping;
    const unsigned int   quad_order;

    AffineConstraints<double> hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> system_matrix;
    SparseMatrix<double> stepping_matrix;

    Vector<double> old_velocity;
    Vector<double> velocity;
    Vector<double> old_displacement;
    Vector<double> displacement;
    Vector<double> old_forces;
    Vector<double> forces;
    Vector<double> system_rhs;

    bool           compute_gravity;
    Vector<double> gravitational_force;
    double         gravity_value;
    int            gravity_direction;

    Adapter::PreciceDealCoupling::CouplingFunctions<dim, Vector<double>>
      coupling_functions;

    std::vector<Vector<double> *> state_variables;
    // for the output directory
    const std::string case_path;
  };


  // constructor
  template <int dim>
  CoupledElastoDynamics<dim>::CoupledElastoDynamics(
    const std::string &case_path)
    : parameters(case_path + "parameters.prm")
    , time(parameters.end_time, parameters.delta_t)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , mapping(MappingQGeneric<dim>(parameters.poly_degree))
    , quad_order(parameters.poly_degree + 1)
    , coupling_functions(parameters)
    , case_path(case_path)
  {}

  // destructor
  template <int dim>
  CoupledElastoDynamics<dim>::~CoupledElastoDynamics()
  {
    dof_handler.clear();
  }

  template <int dim>
  void
  CoupledElastoDynamics<dim>::make_grid()
  {
    std::cout << "  Create mesh: " << std::endl;

    uint n_x, n_y, n_z;

    Point<dim> point_bottom;
    Point<dim> point_tip;

    // boundary IDs are obtained through colorize = true
    uint id_flap_long_bottom, id_flap_long_top, id_flap_short_bottom,
      id_flap_short_top, id_flap_out_of_plane_bottom, id_flap_out_of_plane_top;


    if (parameters.scenario == "FSI3")
      {
        // FSI 3
        n_x          = 30;
        n_y          = 5;
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
    else if (parameters.scenario == "PF")
      {
        // flap_perp
        n_x = 5;
        n_y = 30;
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
    else
      {
        std::cout
          << "Selected scenario is not preconfigured. Options are FSI3 and PF. "
          << "You might want to configure your own case in the make_grid function"
          << std::endl;
      }

    // same for both scenarios
    id_flap_out_of_plane_bottom = 4; // z direction
    id_flap_out_of_plane_top    = 5;

    // vector of dim values denoting the number of cells to generate in that
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

    // refine all cells global_refinement times
    const unsigned int global_refinement = 0;

    triangulation.refine_global(global_refinement);

    std::cout << "\t Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    // set the desired IDs for clamped boundaries and out_of_plane clamped
    // boundaries interface ID is set in the parameter file
    clamped_mesh_id              = 0;
    out_of_plane_clamped_mesh_id = 4;
    const unsigned int neumann_id =
      coupling_functions.deal_boundary_interface_id;
    // the IDs must not be the same:
    std::string error_message(
      "The interface_id cannot be the same as the clamped one");
    Assert(clamped_mesh_id != neumann_id, ExcMessage(error_message));
    Assert(out_of_plane_clamped_mesh_id != neumann_id,
           ExcMessage(error_message));


    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true)
          {
            // boundaries for the interface
            if (face->boundary_id() == id_flap_short_top ||
                face->boundary_id() == id_flap_long_bottom ||
                face->boundary_id() == id_flap_long_top)
              face->set_boundary_id(neumann_id);
            // boundaries clamped in all directions
            else if (face->boundary_id() == id_flap_short_bottom)
              face->set_boundary_id(clamped_mesh_id);
            // boundaries clamped out-of-plane (z) direction
            else if (face->boundary_id() == id_flap_out_of_plane_bottom ||
                     face->boundary_id() == id_flap_out_of_plane_top)
              face->set_boundary_id(out_of_plane_clamped_mesh_id);
          }
  }

  template <int dim>
  void
  CoupledElastoDynamics<dim>::setup_system()
  {
    std::cout << "  Setup system: " << std::endl;

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

    std::cout << "\t Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);
    stepping_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(mapping,
                                      dof_handler,
                                      QGauss<dim>(quad_order),
                                      mass_matrix);
    mass_matrix *= parameters.rho;

    old_velocity.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());

    old_displacement.reinit(dof_handler.n_dofs());
    displacement.reinit(dof_handler.n_dofs());

    system_rhs.reinit(dof_handler.n_dofs());
    old_forces.reinit(dof_handler.n_dofs());
    forces.reinit(dof_handler.n_dofs());

    gravitational_force.reinit(dof_handler.n_dofs());


    state_variables = {
      &old_velocity, &velocity, &old_displacement, &displacement, &old_forces};

    // loads at time 0
    // TODO: Check, if initial conditions should be set at the beginning
    old_forces = 0.0;

    // const value of gravity (e.g. 9.81) and its direction (x (0),y(1) or z(2))
    compute_gravity   = false;
    gravity_value     = -2;
    gravity_direction = 1;
  }


  template <int dim>
  void
  CoupledElastoDynamics<dim>::assemble_system()
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


    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;

        fe_values.reinit(cell);

        // next we get the values of the coefficients at the quadrature
        // points.
        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);


        // then assemble the entries of the local stiffness matrix
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


        // the transfer from local degrees of freedom into the global matrix
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              stiffness_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i, j));
          }
      }

    // To save the system_matrix
    stepping_matrix.copy_from(stiffness_matrix);

    stepping_matrix *= time.get_delta_t() * time.get_delta_t() *
                       parameters.theta * parameters.theta;

    stepping_matrix.add(1, mass_matrix);

    hanging_node_constraints.condense(stepping_matrix);

    // Calculate contribution of gravity and store them in gravitational_force
    if (compute_gravity)
      {
        Vector<double> gravity_vector(dim);
        // assign the specified values, rho * g is assumed
        gravity_vector[gravity_direction] = parameters.rho * gravity_value;

        // create a constant function object
        Functions::ConstantFunction<dim, double> gravity_function(
          gravity_vector);

        // create the contribution to the right-hand side vector
        VectorTools::create_right_hand_side(mapping,
                                            dof_handler,
                                            QGauss<dim>(quad_order),
                                            gravity_function,
                                            gravitational_force);
      }
  }


  template <int dim>
  void
  CoupledElastoDynamics<dim>::assemble_rhs()
  {
    std::cout << "\t Assemble system " << std::endl;
    system_rhs = 0.0;

    // quadrature formula for integration over faces (dim-1)
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
    std::vector<Vector<double>> local_stress(n_face_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;

        // assembling the right-hand side force vector each timestep
        // by applying contributions from the coupling interface
        for (const auto &face : cell->face_iterators())
          if (face->at_boundary() == true &&
              face->boundary_id() ==
                coupling_functions.deal_boundary_interface_id)
            {
              fe_face_values.reinit(cell, face);
              // Extract from global stress vector
              // In contrast to the nonlinear solver, no pull back is performed.
              // The equilibrium is stated in reference configuration, but only
              // valid for very small deformations
              fe_face_values.get_function_values(forces, local_stress);

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

        // local dofs to global
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    // update variables time dependent variables
    old_velocity     = velocity;
    old_displacement = displacement;

    // add contribution of gravitational forces
    if (compute_gravity)
      system_rhs.add(1, gravitational_force);

    // assemble global RHS:
    // RHS=(M-theta*(1-theta)*delta_t^2*K)*V_n - delta_t*K* D_n +
    // delta_t*theta*F_n+1 + delta_t*(1-theta)*F_n

    // tmp vector to store intermediate results
    Vector<double> tmp;
    tmp.reinit(dof_handler.n_dofs());

    tmp = system_rhs;

    system_rhs *= time.get_delta_t() * parameters.theta;
    system_rhs.add(time.get_delta_t() * (1 - parameters.theta), old_forces);
    old_forces = tmp;

    mass_matrix.vmult(tmp, old_velocity);
    system_rhs.add(1, tmp);

    stiffness_matrix.vmult(tmp, old_velocity);
    system_rhs.add(-parameters.theta * time.get_delta_t() * time.get_delta_t() *
                     (1 - parameters.theta),
                   tmp);

    stiffness_matrix.vmult(tmp, old_displacement);
    system_rhs.add(-time.get_delta_t(), tmp);

    hanging_node_constraints.condense(system_rhs);

    // copy the system_matrix every timestep, since applying the BC deletes
    // certain rows and columns
    system_matrix = 0.0;
    system_matrix.copy_from(stepping_matrix);

    // set Dirichlet BC
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
  }

  template <int dim>
  void
  CoupledElastoDynamics<dim>::solve()
  {
    const std::string solver_type = "Direct";

    uint   lin_it;
    double lin_res;

    if (solver_type == "CG")
      {
        std::cout << "\t CG solver: " << std::endl;
        SolverControl solver_control(1000, 1e-12);
        SolverCG<>    cg(solver_control);

        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.2);

        cg.solve(system_matrix, velocity, system_rhs, preconditioner);

        lin_it  = solver_control.last_step();
        lin_res = solver_control.last_value();
      }
    else if (solver_type == "Direct")
      {
        std::cout << "\t Direct solver: " << std::endl;

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(velocity, system_rhs);

        lin_it  = 1;
        lin_res = 0.0;
      }
    else
      Assert(solver_type == "Direct" || solver_type == "CG",
             ExcNotImplemented());

    // assert divergence
    Assert(velocity.linfty_norm() < 1e4, ExcMessage("Linear system diverged"));
    std::cout << "\t     No of iterations:\t" << lin_it
              << "\n \t     Final residual:\t" << lin_res << std::endl;
    hanging_node_constraints.distribute(velocity);
  }

  template <int dim>
  void
  CoupledElastoDynamics<dim>::update_displacement()
  {
    // D_n+1= D_n + delta_t*theta* V_n+1 + delta_t*(1-theta)* V_n
    displacement.add(time.get_delta_t() * parameters.theta, velocity);
    displacement.add(time.get_delta_t() * (1 - parameters.theta), old_velocity);
  }


  template <int dim>
  void
  CoupledElastoDynamics<dim>::output_results() const
  {
    // compute the strains save it in the outputfiles
    StrainPostprocessor<dim> strain_u;

    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(displacement,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    data_out.add_data_vector(displacement, strain_u);

    // visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(parameters.poly_degree,
                                    dof_handler,
                                    displacement);
    data_out.build_patches(q_mapping, parameters.poly_degree);

    // check, if the output directory exists
    std::ifstream output_directory(case_path + "dealii_output");
    Assert(
      output_directory,
      ExcMessage(
        "Unable to find the output directory. "
        "By default, this program stores result files in a directory called dealii_output. "
        "This needs to be located in your case directory, where the parameter file is located as well."));

    // store all files in a seperate folder called dealii_ouput
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
  }



  template <int dim>
  void
  CoupledElastoDynamics<dim>::run()
  {
    make_grid();
    setup_system();
    output_results();
    time.increment();

    assemble_system();
    coupling_functions.initialize_precice(dof_handler, displacement, forces);

    while (coupling_functions.precice.isCouplingOngoing())
      {
        std::cout << std::endl
                  << "Timestep " << time.get_timestep() << " @ "
                  << time.current() << "s" << std::endl;

        coupling_functions.save_current_state_if_required(state_variables,
                                                          time);

        assemble_rhs();

        solve();

        update_displacement();

        coupling_functions.advance_precice(displacement,
                                           forces,
                                           time.get_delta_t());

        time.increment();

        coupling_functions.reload_old_state_if_required(state_variables, time);

        if (coupling_functions.precice.isTimeWindowComplete() &&
            time.get_timestep() % parameters.output_interval == 0)
          output_results();
      }
    coupling_functions.precice.finalize();
  }
} // namespace Linear_Elasticity

int
main(int argc, char **argv)
{
  try
    {
      std::cout << "--------------------------------------------------\n"
                << "             Running deal.ii solver \n"
                << "--------------------------------------------------\n"
                << std::endl;

      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[1];
      else
        parameter_file = "parameters.prm";

      // Extract case path for the output directory
      size_t      pos = parameter_file.find_last_of("/");
      std::string case_path =
        std::string::npos == pos ? "" : parameter_file.substr(0, pos + 1);

      Linear_Elasticity::CoupledElastoDynamics<DIM> elastic_solver(case_path);
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
