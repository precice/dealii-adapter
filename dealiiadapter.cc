/*---------------------------------------------------------------------------*\
   deal.II solver for dynamic linear elastic problems
   with preCICE-adapter for partitiond FSI simulations

   Copyright (c) 2018

   Build on anextension of the step-8 tutorial program of the deal.II library
   See also the README.md
\*---------------------------------------------------------------------------*/

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/fe/mapping_q_eulerian.h>

#include <fstream>
#include <iostream>

namespace adapter
{
  using namespace dealii;

    namespace Parameters
    {
    //TODO: Add more parameters
    struct Time
    {
      double delta_t;
      double end_time;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1",
                          Patterns::Double(),
                          "End time");

        prm.declare_entry("Time step size", "0.1",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

    struct AllParameters :
      public Time

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
    }

    void AllParameters::declare_parameters(ParameterHandler &prm)
    {
      Time::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler &prm)
    {
      Time::parse_parameters(prm);
    }


    }

//Class for the simulation time
class Time
{
public:
  Time (const double time_end,
        const double delta_t)
    :
    timestep(0),
    time_current(0.0),
    time_end(time_end),
    delta_t(delta_t)
  {}

  virtual ~Time()
  {}

  double current() const
  {
    return time_current;
  }
  double end() const
  {
    return time_end;
  }
  double get_delta_t() const
  {
    return delta_t;
  }
  unsigned int get_timestep() const
  {
    return timestep;
  }
  void increment()
  {
    time_current += delta_t;
    ++timestep;
  }

private:
  unsigned int timestep;
  double       time_current;
  const double time_end;
  const double delta_t;
};


  template <int dim>
  class ElasticProblem
  {
  public:
    ElasticProblem(const std::string &input_file);
    ~ElasticProblem();
    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;


    Parameters::AllParameters parameters;

    Triangulation<dim> triangulation;

    Time                      time;
    DoFHandler<dim>    dof_handler;

    FESystem<dim> fe;

    AffineConstraints<double> hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double> solution;
    Vector<double> system_rhs;
  };


  template <int dim>
  void right_hand_side(const typename DoFHandler<dim>::active_cell_iterator &cell,
                       const std::vector<Point<dim>> &points,
                       std::vector<Tensor<1, dim>> &  values)
  {

     // Assertion for the right size of the vector
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));
    Assert(dim >= 2, ExcNotImplemented());



    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {

        //TODO: Parametrize
        // x-direction=values[point_n][0]
          values[point_n][0] = 0.0;

        // y-direction=values[point_n][1]
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
              if (cell->face(face)->at_boundary() == true
                      && cell->face(face)->boundary_id() == 3)
                  values[point_n][1] = 16*1000*-2;//16 cells 1000 density 2 gravity
              else
                  values[point_n][1] = 0.0;
      }
  }

  //Constructor
  template <int dim>
  ElasticProblem<dim>::ElasticProblem(const std::string &input_file)
    : parameters(input_file)
    , time(parameters.end_time, parameters.delta_t)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(1), dim)
  {}

  //Destructor
  template <int dim>
  ElasticProblem<dim>::~ElasticProblem()
  {
    dof_handler.clear();
  }

  //This testcase refers to the CSM benchmark of Hron and Turek
  template <int dim>
    void ElasticProblem<dim>::make_grid()
  {
        std::vector< std::vector< double > > stepsize( dim );

        std::vector<double> x_direction  (5);//number of cells in this direction
        std::vector<double> y_direction  (1);//number of cells in this direction

        //length of cells in x-direction
        for (int i=0; i<5; i++)
            x_direction[i]=(0.6-0.24899)/5;

        //length of cells in y-direction
        y_direction[0]=0.02;

        stepsize[0] = x_direction;
        stepsize[1] = y_direction;


        GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                  stepsize,
                                       (dim==3 ? Point<dim>(0.24899, 0.19, 0.05) : Point<dim>(0.24899, 0.19)),
                                       (dim==3 ? Point<dim>(0.6, 0.21, 0.05) : Point<dim>(0.6, 0.21)),
                                       true);

        //TODO: Add refinement to parameter class
        triangulation.refine_global(4);


   }

  template <int dim>
  void ElasticProblem<dim>::setup_system()
  {
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

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }


  template <int dim>
  void ElasticProblem<dim>::assemble_system()
  {
    QGauss<dim> quadrature_formula(2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // As was shown in previous examples as well, we need a place where to
    // store the values of the coefficients at all the quadrature points on a
    // cell. In the present situation, we have two coefficients, lambda and
    // mu.
    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);


    //Lame constants
    Functions::ConstantFunction<dim> lambda(2e6), mu(0.5e6);

    // Like the two constant functions above, the function
    // right_hand_side is called once per cell.
    std::vector<Tensor<1, dim>> rhs_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;

        fe_values.reinit(cell);

        // Next we get the values of the coefficients at the quadrature
        // points. Likewise for the right hand side:
        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);
        right_hand_side(cell, fe_values.get_quadrature_points(), rhs_values);


        // Then assemble the entries of the local stiffness matrix and right
        // hand side vector.
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
                      // The first term is (lambda d_i u_i, d_j v_j) + (mu d_i
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
                        // The second term is (mu nabla u_i, nabla v_j).


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

        // Assembling the right hand side

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              cell_rhs(i) += fe_values.shape_value(i, q_point) *
                             rhs_values[q_point][component_i] *
                             fe_values.JxW(q_point);
          }

        // The transfer from local degrees of freedom into the global matrix
        // and right hand side vector

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));

            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    hanging_node_constraints.condense(system_matrix);
    hanging_node_constraints.condense(system_rhs);


    //0 refers to the boundary_id
    //TODO: Parametrize
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);
  }



  template <int dim>
  void ElasticProblem<dim>::solve()
  {
    SolverControl solver_control(1000, 1e-12);
    SolverCG<>    cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, solution, system_rhs, preconditioner);

    hanging_node_constraints.distribute(solution);
  }



  //TODO: This function is a relict and is currently ignored
  //      Check single steps and remove it completly
  template <int dim>
  void ElasticProblem<dim>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(2),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    triangulation.execute_coarsening_and_refinement();
  }


  template <int dim>
  void ElasticProblem<dim>::output_results(const unsigned int cycle) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 1:
          solution_names.emplace_back("displacement");
          break;
        case 2:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          break;
        case 3:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          solution_names.emplace_back("z_displacement");
          break;
        default:
          Assert(false, ExcNotImplemented());
      }


    //1 refers to the polynomial degree
    //Visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(1, dof_handler, solution);

    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches(q_mapping, 1);

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }


 template <int dim>
  void ElasticProblem<dim>::run()
  {

         make_grid();

         setup_system();

        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        assemble_system();
        solve();

        //TODO: Setup the right order of the functions for the time loop
        while(time.current() <= time.end())
        {
            output_results(time.current());

            time.increment();

        }
  }
} //end namespace adapter

int main()
{
  try
    {

      const unsigned int dim = 2;

      adapter::ElasticProblem<dim> elastic_solver("parameters.prm");
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
