#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <adapter/adapter.h>
#include <adapter/parameters.h>
#include <adapter/time_handler.h>

#include <fstream>
#include <iostream>


namespace Heat_Transfer
{
  using namespace dealii;


  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation(const std::string &parameter_file);
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
    solve_time_step();
    void
    output_results() const;

    Triangulation<dim>              triangulation;
    const Parameters::AllParameters parameters;
    const types::boundary_id        interface_boundary_id = 0;
    const types::boundary_id        dirichlet_boundary_id = 2;
    FE_Q<dim>                       fe;
    DoFHandler<dim>                 dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> heat_flux;
    Vector<double> old_solution;
    Vector<double> system_rhs;
    Vector<double> tmp;
    Vector<double> forcing_terms;

    mutable TimerOutput                      timer;
    Adapter::Time                            time;
    Adapter::Adapter<dim, 1, Vector<double>> adapter;

    std::vector<Vector<double> *> state_variables;

    const double theta;
    const double alpha;
    const double beta;
  };


  template <int dim>
  class AnalyticSolution : public Function<dim>
  {
  public:
    AnalyticSolution(const double alpha, const double beta)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override
    {
      (void)component;
      AssertIndexRange(component, 1);
      const double time = this->get_time();
      return 1 + (p[0] * p[0]) + (alpha * p[1] * p[1]) + (beta * time);
    }

  private:
    const double alpha;
    const double beta;
  };



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide(const double alpha, const double beta)
      : Function<dim>()
      , alpha(alpha)
      , beta(beta)
    {}

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return beta - 2 - (2 * alpha);
    }

  private:
    const double alpha;
    const double beta;
  };



  template <int dim>
  HeatEquation<dim>::HeatEquation(const std::string &parameter_file)
    : parameters(parameter_file)
    , fe(parameters.poly_degree)
    , dof_handler(triangulation)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
    , time(parameters.end_time, parameters.delta_t)
    , adapter(parameters, interface_boundary_id)
    , theta(1)
    , alpha(3)
    , beta(1.3)
  {}



  template <int dim>
  void
  HeatEquation<dim>::make_grid()
  {
    GridGenerator::hyper_rectangle(triangulation,
                                   Point<dim>{1, 0},
                                   Point<dim>{2, 1},
                                   true);
    for (const auto &cell : triangulation.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() == true)
          {
            // Boundaries for the dirichlet boundary
            if (face->boundary_id() != 0)
              face->set_boundary_id(dirichlet_boundary_id);
          }

    const unsigned int global_refinement = 4;
    triangulation.refine_global(global_refinement);
    AssertThrow(interface_boundary_id == adapter.deal_boundary_interface_id,
                ExcMessage("Wrong interface ID in the Adapter specified"));
  }



  template <int dim>
  void
  HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);

    solution.reinit(dof_handler.n_dofs());
    heat_flux.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    tmp.reinit(solution.size());
    forcing_terms.reinit(solution.size());
  }



  template <int dim>
  void
  HeatEquation<dim>::assemble_system()
  {
    assemble_rhs();
    system_matrix.copy_from(mass_matrix);
    system_matrix.add(theta * time.get_delta_t(), laplace_matrix);

    constraints.condense(system_matrix, system_rhs);
    {
      AnalyticSolution<dim> boundary_values_function(alpha, beta);
      boundary_values_function.set_time(time.current());

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler,
                                               dirichlet_boundary_id,
                                               boundary_values_function,
                                               boundary_values);

      MatrixTools::apply_boundary_values(boundary_values,
                                         system_matrix,
                                         solution,
                                         system_rhs);
    }
  }



  template <int dim>
  void
  HeatEquation<dim>::assemble_rhs()
  {
    mass_matrix.vmult(system_rhs, old_solution);

    laplace_matrix.vmult(tmp, old_solution);
    system_rhs.add(-(1 - theta) * time.get_delta_t(), tmp);

    RightHandSide<dim> rhs_function(alpha, beta), rhs_function_old(alpha, beta);
    rhs_function.set_time(time.current());
    rhs_function_old.set_time(time.current() - time.get_delta_t());


    Assert(fe.n_components() == rhs_function.n_components,
           ExcDimensionMismatch(fe.n_components(), rhs_function.n_components));
    Assert(forcing_terms.size() == dof_handler.n_dofs(),
           ExcDimensionMismatch(forcing_terms.size(), dof_handler.n_dofs()));

    UpdateFlags update_flags =
      UpdateFlags(update_values | update_quadrature_points | update_JxW_values);

    const QGauss<dim>     quadrature(fe.degree + 1);
    const QGauss<dim - 1> f_quadrature(fe.degree + 1);
    FEValues<dim>         fe_values(StaticMappingQ1<dim>::mapping,
                            fe,
                            quadrature,
                            update_flags);
    FEFaceValues<dim>     fe_f_values(StaticMappingQ1<dim>::mapping,
                                  fe,
                                  f_quadrature,
                                  update_flags);

    const unsigned int dofs_per_cell   = fe_values.dofs_per_cell,
                       n_q_points      = fe_values.n_quadrature_points,
                       n_face_q_points = f_quadrature.size();

    std::vector<types::global_dof_index> dofs(dofs_per_cell);
    Vector<double>                       cell_vector(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();

    std::vector<double> rhs_values(n_q_points);
    std::vector<double> rhs_values_old(n_q_points);
    std::vector<double> local_flux(n_face_q_points);

    for (; cell != endc; ++cell)
      {
        fe_values.reinit(cell);

        const std::vector<double> &weights = fe_values.get_JxW_values();
        rhs_function.value_list(fe_values.get_quadrature_points(), rhs_values);
        rhs_function_old.value_list(fe_values.get_quadrature_points(),
                                    rhs_values_old);

        cell_vector = 0;
        for (unsigned int point = 0; point < n_q_points; ++point)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_vector(i) +=
                ((rhs_values[point] * time.get_delta_t() * theta) +
                 (rhs_values_old[point] * time.get_delta_t() * (1 - theta))) *
                fe_values.shape_value(i, point) * weights[point];
            }

        for (unsigned int face : cell->face_indices())
          if (cell->face(face)->at_boundary() &&
              (cell->face(face)->boundary_id() == interface_boundary_id))
            {
              fe_f_values.reinit(cell, face);
              fe_f_values.get_function_values(heat_flux, local_flux);

              for (unsigned int f_q_point = 0; f_q_point < n_face_q_points;
                   ++f_q_point)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    cell_vector(i) -= fe_f_values.shape_value(i, f_q_point) *
                                      time.get_delta_t() *
                                      local_flux[f_q_point] *
                                      fe_f_values.JxW(f_q_point);
                  }
            }
        cell->get_dof_indices(dofs);
        constraints.distribute_local_to_global(cell_vector, dofs, system_rhs);
      } // end cell loop
  }



  template <int dim>
  void
  HeatEquation<dim>::solve_time_step()
  {
    timer.enter_subsection("solve system");

    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    constraints.distribute(solution);

    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
    timer.leave_subsection("solve system");
  }



  template <int dim>
  void
  HeatEquation<dim>::output_results() const
  {
    timer.enter_subsection("output results");
    std::cout << "Writing solution to " << time.get_timestep() << std::endl;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "Temperature");

    data_out.build_patches();

    data_out.set_flags(
      DataOutBase::VtkFlags(time.current(), time.get_timestep()));

    const std::string filename =
      "solution-" + Utilities::int_to_string(time.get_timestep(), 3) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
    timer.leave_subsection("output results");
  }



  template <int dim>
  void
  HeatEquation<dim>::run()
  {
    make_grid();

    setup_system();

    AnalyticSolution<dim> initial_condition(alpha, beta);
    initial_condition.set_time(0);
    VectorTools::interpolate(dof_handler, initial_condition, old_solution);
    solution  = old_solution;
    heat_flux = 0;
    output_results();

    adapter.initialize(dof_handler, solution, heat_flux);
    state_variables = {&solution, &old_solution, &system_rhs};

    while (adapter.precice.isCouplingOngoing())
      {
        adapter.save_current_state_if_required(state_variables, time);

        time.increment();


        std::cout << std::endl
                  << "Timestep " << time.get_timestep() << " @ " << std::fixed
                  << time.current() << "s" << std::endl;

        assemble_system();
        solve_time_step();
        timer.enter_subsection("Advance adapter");
        adapter.advance(solution, heat_flux, time.get_delta_t());
        timer.leave_subsection("Advance adapter");

        adapter.reload_old_state_if_required(state_variables, time);

        if (adapter.precice.isTimeWindowComplete())
          {
            old_solution = solution;
            if (time.get_timestep() % parameters.output_interval == 0)
              output_results();
          }
      }
  }
} // namespace Heat_Transfer
