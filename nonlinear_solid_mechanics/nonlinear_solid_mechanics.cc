/* ---------------------------------------------------------------------
 * Copyright (c) 2020 by the preCICE authors
 *
 * This file is part of the dealii-adapter for the coupling library
 * preCICE. Parts of this program are based on deal.II tutorial programs.
 *
 * This adapter is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. The full text of the license can
 * be found in the file LICENSE in the precice/dealii-adapter repository.
 * Parts of this program are based on the previous work of Jean-Paul Pelteret
 * and Andrew McBride in their deal.II code gallery program 'Quasi-Static
 * Finite-Strain Compressible Elasticity'
 * ---------------------------------------------------------------------
 *
 * Author: David Schneider 2020
 */
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

#include "include/coupling_functions.h"
#include "include/parameter_handling.h"
#include "include/time.h"
#include "precice/SolverInterface.hpp"

namespace adapter
{
  using namespace dealii;

  template <int dim, typename NumberType>
  class Material_Compressible_Neo_Hook_One_Field
  {
  public:
    Material_Compressible_Neo_Hook_One_Field(const double mu,
                                             const double nu,
                                             const double rho)
      : kappa((2.0 * mu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu)))
      , c_1(mu / 2.0)
      , rho(rho)
    {
      Assert(kappa > 0, ExcInternalError());
    }

    ~Material_Compressible_Neo_Hook_One_Field()
    {}

    NumberType
    get_Psi(const NumberType &                         det_F,
            const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return get_Psi_vol(det_F) + get_Psi_iso(b_bar);
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau(const NumberType &                         det_F,
            const SymmetricTensor<2, dim, NumberType> &b_bar)
    {
      return get_tau_vol(det_F) + get_tau_iso(b_bar);
    }

    SymmetricTensor<4, dim, NumberType>
    get_Jc(const NumberType &                         det_F,
           const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return get_Jc_vol(det_F) + get_Jc_iso(b_bar);
    }

    NumberType
    get_rho() const
    {
      return rho;
    }

  private:
    const double kappa;
    const double c_1;
    const double rho;

    NumberType
    get_Psi_vol(const NumberType &det_F) const
    {
      return (kappa / 4.0) * (det_F * det_F - 1.0 - 2.0 * std::log(det_F));
    }

    NumberType
    get_Psi_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return c_1 * (trace(b_bar) - dim);
    }

    NumberType
    get_dPsi_vol_dJ(const NumberType &det_F) const
    {
      return (kappa / 2.0) * (det_F - 1.0 / det_F);
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_vol(const NumberType &det_F) const
    {
      return NumberType(get_dPsi_vol_dJ(det_F) * det_F) *
             Physics::Elasticity::StandardTensors<dim>::I;
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return Physics::Elasticity::StandardTensors<dim>::dev_P *
             get_tau_bar(b_bar);
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau_bar(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return 2.0 * c_1 * b_bar;
    }

    NumberType
    get_d2Psi_vol_dJ2(const NumberType &det_F) const
    {
      return ((kappa / 2.0) * (1.0 + 1.0 / (det_F * det_F)));
    }

    SymmetricTensor<4, dim, NumberType>
    get_Jc_vol(const NumberType &det_F) const
    {
      return det_F *
             ((get_dPsi_vol_dJ(det_F) + det_F * get_d2Psi_vol_dJ2(det_F)) *
                Physics::Elasticity::StandardTensors<dim>::IxI -
              (2.0 * get_dPsi_vol_dJ(det_F)) *
                Physics::Elasticity::StandardTensors<dim>::S);
    }

    SymmetricTensor<4, dim, NumberType>
    get_Jc_iso(const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      const SymmetricTensor<2, dim> tau_bar = get_tau_bar(b_bar);
      const SymmetricTensor<2, dim> tau_iso = get_tau_iso(b_bar);
      const SymmetricTensor<4, dim> tau_iso_x_I =
        outer_product(tau_iso, Physics::Elasticity::StandardTensors<dim>::I);
      const SymmetricTensor<4, dim> I_x_tau_iso =
        outer_product(Physics::Elasticity::StandardTensors<dim>::I, tau_iso);
      const SymmetricTensor<4, dim> c_bar = get_c_bar();

      return (2.0 / dim) * trace(tau_bar) *
               Physics::Elasticity::StandardTensors<dim>::dev_P -
             (2.0 / dim) * (tau_iso_x_I + I_x_tau_iso) +
             Physics::Elasticity::StandardTensors<dim>::dev_P * c_bar *
               Physics::Elasticity::StandardTensors<dim>::dev_P;
    }

    SymmetricTensor<4, dim, double>
    get_c_bar() const
    {
      return SymmetricTensor<4, dim>();
    }
  };


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

    NumberType
    get_Psi(const NumberType &                         det_F,
            const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return material->get_Psi(det_F, b_bar);
    }

    SymmetricTensor<2, dim, NumberType>
    get_tau(const NumberType &                         det_F,
            const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return material->get_tau(det_F, b_bar);
    }

    SymmetricTensor<4, dim, NumberType>
    get_Jc(const NumberType &                         det_F,
           const SymmetricTensor<2, dim, NumberType> &b_bar) const
    {
      return material->get_Jc(det_F, b_bar);
    }

    NumberType
    get_rho() const
    {
      return material->get_rho();
    }

  private:
    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, NumberType>>
      material;
  };



  template <int dim, typename NumberType>
  struct Assembler_Base;
  template <int dim, typename NumberType>
  struct Assembler;

  template <int dim, typename NumberType = double>
  class Solid
  {
  public:
    Solid(const std::string &case_path);

    virtual ~Solid();

    void
    run();

  private:
    void
    make_grid();

    void
    system_setup();

    void
    assemble_system(const BlockVector<double> &solution_delta,
                    const BlockVector<double> &acceleration,
                    const BlockVector<double> &external_stress);

    friend struct Assembler_Base<dim, NumberType>;
    friend struct Assembler<dim, NumberType>;

    void
    make_constraints(const int &it_nr);

    void
    setup_qph();

    void
    solve_nonlinear_timestep(BlockVector<double> &solution_delta);

    std::pair<unsigned int, double>
    solve_linear_system(BlockVector<double> &newton_update);

    BlockVector<double>
    get_total_solution(const BlockVector<double> &solution_delta) const;

    void
    update_acceleration(BlockVector<double> &displacement_delta);

    void
    update_velocity(BlockVector<double> &displacement_delta);

    void
    update_old_variables();

    void
    output_results() const;

    const Parameters::AllParameters parameters;

    double vol_reference;
    double vol_current;

    Triangulation<dim> triangulation;

    Time        time;
    TimerOutput timer;

    CellDataStorage<typename Triangulation<dim>::cell_iterator,
                    PointHistory<dim, NumberType>>
      quadrature_point_history;

    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;

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

    // Alias
    std::vector<BlockVector<double> *> state_variables;

    // container for data exchange with precice
    BlockVector<double> external_stress;

    PreciceDealCoupling::CouplingFunctions<dim, BlockVector<double>>
      coupling_functions;

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

    void
    get_error_residual(Errors &error_residual);

    void
    get_error_update(const BlockVector<double> &newton_update,
                     Errors &                   error_update);

    static void
    print_conv_header();

    void
    print_conv_footer();
  };



  template <int dim, typename NumberType>
  Solid<dim, NumberType>::Solid(const std::string &case_path)
    : parameters(Parameters::AllParameters(case_path + "parameters.prm"))
    , vol_reference(0.0)
    , vol_current(0.0)
    , triangulation(Triangulation<dim>::maximum_smoothing)
    , time(parameters.end_time, parameters.delta_t)
    , timer(std::cout, TimerOutput::summary, TimerOutput::wall_times)
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
    , coupling_functions(parameters)
  {}

  template <int dim, typename NumberType>
  Solid<dim, NumberType>::~Solid()
  {
    dof_handler_ref.clear();
  }


  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::run()
  {
    make_grid();
    system_setup();
    output_results();
    time.increment();

    coupling_functions.initialize_precice(dof_handler_ref,
                                          total_displacement,
                                          external_stress);


    BlockVector<NumberType> solution_delta(dofs_per_block);
    while (coupling_functions.precice.isCouplingOngoing())
      {
        solution_delta = 0.0;

        // TODO: Place it right and check expected behavior of the state
        // variables
        coupling_functions.save_current_state(state_variables, time);

        solve_nonlinear_timestep(solution_delta);
        total_displacement += solution_delta;

        update_acceleration(solution_delta);
        update_velocity(solution_delta);
        update_old_variables();

        coupling_functions.advance_precice(total_displacement,
                                           external_stress,
                                           time.get_delta_t());
        time.increment();

        coupling_functions.reload_old_state(state_variables, time);

        if (coupling_functions.precice.isTimeWindowComplete() &&
            time.get_timestep() % parameters.output_interval == 0)
          output_results();
      }

    coupling_functions.precice.finalize();
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::make_grid()
  {
    const std::string testcase("PF");

    Point<dim>   point_bottom, point_tip;
    unsigned int id_flap_long_bottom, id_flap_long_top, id_flap_short_bottom,
      id_flap_short_top, n_x, n_y;

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
    else if (testcase == "FSI3")
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

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/ true);


    // refine all cells global_refinement times
    const unsigned int global_refinement = 0;
    triangulation.refine_global(global_refinement);


    // Cell iterator for boundary conditions
    const unsigned int clamped_boundary_id = 1;
    //    const unsigned int do_nothing_boundary_id = 2;
    const unsigned int neumann_boundary_id = parameters.interface_mesh_id;

    typename Triangulation<dim>::active_cell_iterator cell = triangulation
                                                               .begin_active(),
                                                      endc =
                                                        triangulation.end();
    for (; cell != endc; ++cell)
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() == true)
          {
            if (cell->face(face)->boundary_id() == id_flap_short_bottom)
              cell->face(face)->set_boundary_id(clamped_boundary_id);
            else if (cell->face(face)->boundary_id() == id_flap_long_bottom ||
                     cell->face(face)->boundary_id() == id_flap_long_top ||
                     cell->face(face)->boundary_id() == id_flap_short_top)
              cell->face(face)->set_boundary_id(neumann_boundary_id);
            else
              AssertThrow(false,
                          ExcMessage("Unknown boundary id, did "
                                     "you set a boundary "
                                     "condition?"))
          }

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

    tangent_matrix.reinit(sparsity_pattern);

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

    // Alias
    state_variables = std::vector({&total_displacement,
                                   &total_displacement_old,
                                   &velocity,
                                   &velocity_old,
                                   &acceleration,
                                   &acceleration_old});

    setup_qph();

    timer.leave_subsection();
  }


  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    quadrature_point_history.initialize(triangulation.begin_active(),
                                        triangulation.end(),
                                        n_q_points);

    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation.begin_active();
         cell != triangulation.end();
         ++cell)
      {
        const std::vector<std::shared_ptr<PointHistory<dim, NumberType>>> lqph =
          quadrature_point_history.get_data(cell);
        Assert(lqph.size() == n_q_points, ExcInternalError());

        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          lqph[q_point]->setup_lqp(parameters);
      }
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::solve_nonlinear_timestep(
    BlockVector<double> &solution_delta)
  {
    std::cout << std::endl
              << "Timestep " << time.get_timestep() << " @ " << time.current()
              << "s" << std::endl;

    BlockVector<double> newton_update(dofs_per_block);

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
      {
        std::cout << " " << std::setw(2) << newton_iteration << " "
                  << std::flush;

        make_constraints(newton_iteration);
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

        const std::pair<unsigned int, double> lin_solver_output =
          solve_linear_system(newton_update);

        // Update errors
        get_error_update(newton_update, error_update);
        if (newton_iteration == 0)
          error_update_0 = error_update;

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



  template <int dim, typename NumberType>
  BlockVector<double>
  Solid<dim, NumberType>::get_total_solution(
    const BlockVector<double> &solution_delta) const
  {
    BlockVector<double> solution_total(total_displacement);
    solution_total += solution_delta;
    return solution_total;
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::update_acceleration(
    BlockVector<double> &displacement_delta)
  {
    acceleration.equ(alpha_1, displacement_delta);
    acceleration.add(-alpha_2, velocity_old, -alpha_3, acceleration_old);
  }



  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::update_velocity(
    BlockVector<double> &displacement_delta)
  {
    velocity.equ(alpha_4, displacement_delta);
    velocity.add(alpha_5, velocity_old, alpha_6, acceleration_old);
  }



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

    void
    assemble_system_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData_ASM &                                     scratch,
      PerTaskData_ASM &                                     data)
    {
      assemble_system_tangent_residual_one_cell(cell, scratch, data);
      assemble_neumann_contribution_one_cell(cell, scratch, data);
    }

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

  protected:
    virtual void
    assemble_system_tangent_residual_one_cell(
      const typename DoFHandler<dim>::active_cell_iterator & /*cell*/,
      ScratchData_ASM & /*scratch*/,
      PerTaskData_ASM & /*data*/)
    {
      AssertThrow(false, ExcPureFunctionCalled());
    }

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
      const unsigned int &interf_id = data.solid->parameters.interface_mesh_id;

      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() == true &&
            cell->face(face)->boundary_id() == interf_id)
          {
            scratch.fe_face_values_ref.reinit(cell, face);

            // Initialize vector for values at each quad point
            std::vector<Tensor<1, dim, NumberType>> local_stress(n_q_points_f);

            scratch.fe_face_values_ref[u_fe].get_function_values(
              scratch.external_stress, local_stress);

            for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                 ++f_q_point)
              {
                const Tensor<2, dim, NumberType> F =
                  Physics::Elasticity::Kinematics::F(
                    scratch.solution_grads_u_total[f_q_point]);

                const Tensor<1, dim, NumberType> referential_stress =
                  Physics::Transformations::Contravariant::pull_back(
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
      const unsigned int & n_q_points           = data.solid->n_q_points;
      const unsigned int & dofs_per_cell        = data.solid->dofs_per_cell;
      const FESystem<dim> &fe                   = data.solid->fe;
      const unsigned int & u_dof                = data.solid->u_dof;
      const FEValuesExtractors::Vector &u_fe    = data.solid->u_fe;
      const double &                    alpha_1 = data.solid->alpha_1;

      data.reset();
      scratch.reset();
      scratch.fe_values_ref.reinit(cell);
      cell->get_dof_indices(data.local_dof_indices);

      const std::vector<std::shared_ptr<const PointHistory<dim, NumberType>>>
        lqph = const_cast<const Solid<dim, NumberType> *>(data.solid)
                 ->quadrature_point_history.get_data(cell);
      Assert(lqph.size() == n_q_points, ExcInternalError());

      scratch.fe_values_ref[u_fe].get_function_gradients(
        scratch.solution_total, scratch.solution_grads_u_total);

      scratch.fe_values_ref[u_fe].get_function_values(
        scratch.acceleration, scratch.local_acceleration);

      // Const in the whole domain, so we call it once for each cell
      const double rho = lqph[0]->get_rho();

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        {
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

          // TODO: Add to parameter file
          // Define const force vector for gravity
          const Tensor<1, dim> body_force({0, 0. * rho});

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
                     (body_force[component_i] *
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
      const int boundary_id = 1;

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
        const int                        boundary_id = 2;
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
        Assert(false, ExcMessage("Linear solver type not implemented"));

      timer.leave_subsection();
    }

    constraints.distribute(newton_update);

    return std::make_pair(lin_it, lin_res);
  }

  template <int dim, typename NumberType>
  void
  Solid<dim, NumberType>::output_results() const
  {
    DataOut<dim> data_out;

    // Note: There is at least paraView v 5.5 needed to visualize this output
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(total_displacement,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<double> soln(total_displacement.size());
    for (unsigned int i = 0; i < soln.size(); ++i)
      soln(i) = total_displacement(i);
    MappingQEulerian<dim> q_mapping(degree, dof_handler_ref, soln);
    data_out.build_patches(q_mapping, degree);

    std::ostringstream filename;
    filename << "solution-" << time.get_timestep() << ".vtk";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtk(output);
  }

} // namespace adapter

int
main(int argc, char **argv)
{
  using namespace adapter;
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
        parameter_file = "parameters.prm";

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
