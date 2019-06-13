/* ---------------------------------------------------------------------
 * Copyright (c) 2018 - 2019 by the preCICE authors
 *
 * This file is part of the dealii-adapter for the coupling library
 * preCICE. Parts of this program are based on deal.II tutorial programs.
 *
 * This adapter is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version. The full text of the license can
 * be found in the file LICENSE in the precice/dealii-adapter repository.
 * ---------------------------------------------------------------------
 *
 * Author: David Schneider 2018,2019
 */
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

// include for preCICE
#include "precice/SolverInterface.hpp"

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
    double theta;
    int    output_interval;

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

        prm.declare_entry("Output interval", "1",
                          Patterns::Integer(0),
                          "Write results every x timesteps");

        prm.declare_entry("theta", "0.5",
                          Patterns::Double(0,1),
                          "Time integration scheme");

    }
    prm.leave_subsection();
}

void Time::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Time");
    {
        end_time = prm.get_double("End time");
        delta_t = prm.get_double("Time step size");
        output_interval = prm.get_integer("Output interval");
        theta = prm.get_double("theta");
    }
    prm.leave_subsection();
}

struct Materials
{
    double mu;
    double lambda;
    double rho;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};

void Materials::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Material properties");
    {
        prm.declare_entry("mu", "0.5e6",
                          Patterns::Double(),
                          "mu");

        prm.declare_entry("lambda", "2e6",
                          Patterns::Double(),
                          "lambda");

        prm.declare_entry("density", "1000",
                          Patterns::Double(0.0),
                          "density");
    }
    prm.leave_subsection();
}

void Materials::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Material properties");
    {
        mu = prm.get_double("mu");
        lambda = prm.get_double("lambda");
        rho = prm.get_double("density");
    }
    prm.leave_subsection();
}

struct FESystem
{
    unsigned int poly_degree;
    unsigned int quad_order;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};


void FESystem::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Finite element system");
    {
        prm.declare_entry("Polynomial degree", "1",
                          Patterns::Integer(0),
                          "Polynomial degree of the FE system");

        prm.declare_entry("Quadrature order", "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
    }
    prm.leave_subsection();
}

void FESystem::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Finite element system");
    {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order = prm.get_integer("Quadrature order");
    }
    prm.leave_subsection();
}

struct precice_configuration
{
    std::string scenario;
    bool        enable_precice;
    std::string config_file;
    std::string participant;
    std::string node_mesh;
    std::string face_mesh;
    std::string read_data_name;
    std::string write_data_name;
    unsigned int interface_mesh_id;

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);
};


void precice_configuration::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("precice configuration");
    {
        prm.declare_entry("Scenario", "FSI3",
                          Patterns::Selection("FSI3|PF"),
                          "Cases: FSI3 or PF for perpendicular flap");
        prm.declare_entry("Enable precice", "true",
                          Patterns::Bool(),
                          "Whether preCICE is used for coupling to another solver");
        prm.declare_entry("precice config-file", "precice-config.xml",
                          Patterns::Anything(),
                          "Name of the precice configuration file");
        prm.declare_entry("Participant", "dealiisolver",
                          Patterns::Anything(),
                          "Name of the participant in the precice-config.xml file");
        prm.declare_entry("Node mesh name", "dealii-mesh-nodes",
                          Patterns::Anything(),
                          "Name of the node based coupling mesh in the precice-config.xml file");
        prm.declare_entry("Face mesh name", "dealii-mesh-faces",
                          Patterns::Anything(),
                          "Name of the face based coupling mesh in the precice-config.xml file");
        prm.declare_entry("Read data name", "received-data",
                          Patterns::Anything(),
                          "Name of the read data in the precice-config.xml file");
        prm.declare_entry("Write data name", "calculated-data",
                          Patterns::Anything(),
                          "Name of the write data in the precice-config.xml file");
        prm.declare_entry("Interface mesh ID", "1",
                          Patterns::Integer(0),
                          "Boundary mesh ID of the coupling interface in deal.II");
    }
    prm.leave_subsection();
}

void precice_configuration::parse_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("precice configuration");
    {
        scenario = prm.get("Scenario");
        enable_precice = prm.get_bool("Enable precice");
        config_file = prm.get("precice config-file");
        participant = prm.get("Participant");
        node_mesh = prm.get("Node mesh name");
        face_mesh = prm.get("Face mesh name");
        read_data_name = prm.get("Read data name");
        write_data_name = prm.get("Write data name");
        interface_mesh_id = prm.get_integer("Interface mesh ID");
    }
    prm.leave_subsection();
}


struct AllParameters :
        public Time,
        public Materials,
        public FESystem,
        public precice_configuration
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
    Materials::declare_parameters(prm);
    FESystem::declare_parameters(prm);
    precice_configuration::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler &prm)
{
    Time::parse_parameters(prm);
    Materials::parse_parameters(prm);
    FESystem::parse_parameters(prm);
    precice_configuration::parse_parameters(prm);
}

}
//evaluate strains as a tensor in the data out object
template <int dim>
class StrainPostprocessor : public DataPostprocessorTensor<dim>
{
public:
    StrainPostprocessor ()
        :
          DataPostprocessorTensor<dim> ("strain",
                                        update_gradients)
    {}
    virtual
    void
    evaluate_vector_field (const DataPostprocessorInputs::Vector<dim> &input_data,
                           std::vector<Vector<double> >               &computed_quantities) const
    {
        AssertDimension (input_data.solution_gradients.size(),
                         computed_quantities.size());
        for (unsigned int p=0; p<input_data.solution_gradients.size(); ++p)
        {
            AssertDimension (computed_quantities[p].size(),
                             (Tensor<2,dim>::n_independent_components));
            for (unsigned int d=0; d<dim; ++d)
                for (unsigned int e=0; e<dim; ++e)
                    computed_quantities[p][Tensor<2,dim>::component_to_unrolled_index(TableIndices<2>(d,e))]
                            = (input_data.solution_gradients[p][d][e]
                               +
                               input_data.solution_gradients[p][e][d]) / 2;
        }
    }
};


// Class for the simulation time
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
    {
        n_timesteps=time_end/delta_t;

        if((time_end/delta_t)-n_timesteps>1e-12)
        {
            ++n_timesteps;
            std::cerr<< "  Warning: Timestep size is not a multiple of the end time.\n"
                     << "           Simulation will be terminated at t = "
                     <<n_timesteps*delta_t
                    << "\n"
                    <<std::endl;
        }
    }

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

    unsigned int get_n_timesteps() const
    {
        return n_timesteps;
    }

    void increment()
    {
        time_current += delta_t;
        ++timestep;
    }

    void restore()
    {
        time_current -= delta_t;
        --timestep;
    }

private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
    unsigned int n_timesteps;
};


template <int dim>
class CoupledElastoDynamics
{
public:
    CoupledElastoDynamics(const std::string &case_path);
    ~CoupledElastoDynamics();
    void run();

private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void assemble_rhs();
    void solve();
    void update_displacement();
    void output_results(const unsigned int timestep) const;
    void compute_timesteps();
    // preCICE related functions
    void initialize_precice();
    void advance_precice();
    void extract_relevant_displacements(std::vector<double>& precice_displacements);
    void save_old_state();
    void reload_old_state();


    Parameters::AllParameters   parameters;

    // grid related variables
    Triangulation<dim> triangulation;
    unsigned int   global_refinement;
    unsigned int   clamped_mesh_id;
    unsigned int   out_of_plane_clamped_mesh_id;

    Time                      time;
    DoFHandler<dim>    dof_handler;

    FESystem<dim>               fe;
    MappingQGeneric<dim>   mapping;

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

    // variables for implicit coupling
    Vector<double> old_state_old_velocity;
    Vector<double> old_state_velocity;
    Vector<double> old_state_old_displacement;
    Vector<double> old_state_displacement;
    Vector<double> old_state_old_forces;

    // preCICE related initializations
    int            node_mesh_id;
    int            face_mesh_id;
    int            forces_data_id;
    int            displacements_data_id;
    int            n_interface_nodes;
    int            n_interface_faces;

    std::vector<double>      precice_forces;
    std::vector<double>      precice_displacements;
    std::vector<double>      interface_nodes_positions;
    std::vector<int>         interface_nodes_ids;
    std::vector<double>      interface_faces_positions;
    std::vector<int>         interface_faces_ids;


    IndexSet                 coupling_dofs;

    // preCICE API
    precice::SolverInterface precice;

    // for the output directory
    std::string              case_path;
};


// constructor
template <int dim>
CoupledElastoDynamics<dim>::CoupledElastoDynamics(const std::string &case_path)
    : parameters(case_path + "parameters.prm")
    , time(parameters.end_time, parameters.delta_t)
    , dof_handler(triangulation)
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , mapping(MappingQGeneric<dim>(parameters.poly_degree))
    , precice(parameters.participant,0,1)
    , case_path(case_path)
{}

// destructor
template <int dim>
CoupledElastoDynamics<dim>::~CoupledElastoDynamics()
{
    dof_handler.clear();
}

template <int dim>
void CoupledElastoDynamics<dim>::make_grid()
{
    std::cout<<"  Create mesh: "<<std::endl;

    uint n_x, n_y, n_z;

    Point<dim>  point_bottom;
    Point<dim>  point_tip;

    // boundary IDs are obtained through colorize = true
    uint         id_flap_long_bottom;
    uint            id_flap_long_top;
    uint        id_flap_short_bottom;
    uint           id_flap_short_top;
    uint id_flap_out_of_plane_bottom;
    uint    id_flap_out_of_plane_top;


    if (parameters.scenario == "FSI3")
    {
        // FSI 3
        n_x = 30;
        n_y = 5;
        n_z = 1;
        point_bottom = dim==3 ? Point<dim>(0.24899, 0.19, -0.005) : Point<dim>(0.24899, 0.19);
        point_tip    = dim==3 ? Point<dim>(0.6, 0.21, 0.005) : Point<dim>(0.6, 0.21);

        // IDs for FSI3
        id_flap_long_bottom = 2; //x direction
        id_flap_long_top = 3;
        id_flap_short_bottom = 0; //y direction
        id_flap_short_top = 1;
    }
    else if(parameters.scenario == "PF")
    {
        // flap_perp
        n_x = 5;
        n_y = 30;
        n_z = 1;
        point_bottom = dim==3 ? Point<dim>(-0.05, 0, 0) : Point<dim>(-0.05, 0);
        point_tip    = dim==3 ? Point<dim>(0.05, 1, 0.3) : Point<dim>(0.05, 1);

        // IDs for PF
        id_flap_long_bottom = 0; //x direction
        id_flap_long_top = 1;
        id_flap_short_bottom = 2; //y direction
        id_flap_short_top = 3;
    }
    else
    {
        std::cout<<"Selected scenario is not preconfigured. Options are FSI3 and PF. "
                <<"You might want to configure your own case in the make_grid function"
               <<std::endl;
    }

    // same for both scenarios
    id_flap_out_of_plane_bottom = 4; //z direction
    id_flap_out_of_plane_top = 5;

    // vector of dim values denoting the number of cells to generate in that direction
    std::vector< unsigned int > repetitions(dim);
    repetitions[0] = n_x;
    repetitions[1] = n_y;
    if ( dim==3 )
        repetitions[2] = n_z;

    // refine all cells global_refinement times
    global_refinement = 0;


    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              point_bottom,
                                              point_tip,
                                              /*colorize*/true);

    triangulation.refine_global(global_refinement);

    std::cout << "\t Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    // set the desired IDs for clamped boundaries and out_of_plane clamped boundaries
    // interface ID is set in the parameter file
    clamped_mesh_id = 0;
    out_of_plane_clamped_mesh_id = 4;

    // the IDs must not be the same:
    std::string error_message("The interface_id cannot be the same as the clamped one");
    Assert(clamped_mesh_id != parameters.interface_mesh_id, ExcMessage(error_message));
    Assert(out_of_plane_clamped_mesh_id != parameters.interface_mesh_id, ExcMessage(error_message));


    // count simultaniously the relevant coupling faces
    n_interface_faces = 0;

    typename Triangulation<dim>::active_cell_iterator cell =
            triangulation.begin_active(), endc = triangulation.end();
    for (; cell != endc; ++cell)
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if (cell->face(face)->at_boundary() ==  true)
            {
                // boundaries for the interface
                if(cell->face(face)->boundary_id () == id_flap_short_top
                        || cell->face(face)->boundary_id () == id_flap_long_bottom
                        || cell->face(face)->boundary_id () == id_flap_long_top)
                {
                    cell->face(face)->set_boundary_id(parameters.interface_mesh_id);
                    ++n_interface_faces;
                }
                // boundaries clamped in all directions
                else if (cell->face(face)->boundary_id () == id_flap_short_bottom)
                {
                    cell->face(face)->set_boundary_id(clamped_mesh_id);
                }
                // boundaries clamped out-of-plane (z) direction
                else if( cell->face(face)->boundary_id () == id_flap_out_of_plane_bottom
                         || cell->face(face)->boundary_id () ==id_flap_out_of_plane_top)
                {
                    cell->face(face)->set_boundary_id(out_of_plane_clamped_mesh_id);
                }
            }
        }
}

template <int dim>
void CoupledElastoDynamics<dim>::setup_system()
{
    std::cout<<"  Setup system: "<<std::endl;

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

    MatrixCreator::create_mass_matrix (mapping,
                                       dof_handler, QGauss<dim>(parameters.quad_order),
                                       mass_matrix);
    mass_matrix*=parameters.rho;

    old_velocity.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());

    old_displacement.reinit(dof_handler.n_dofs());
    displacement.reinit(dof_handler.n_dofs());

    system_rhs.reinit(dof_handler.n_dofs());
    old_forces.reinit(dof_handler.n_dofs());
    forces.reinit(dof_handler.n_dofs());

    gravitational_force.reinit(dof_handler.n_dofs());

    // loads at time 0
    // TODO: Check, if initial conditions should be set at the beginning
    old_forces=0.0;

    // const value of gravity (e.g. 9.81) and its direction (x (0),y(1) or z(2))
    compute_gravity     = false;
    gravity_value       = -2;
    gravity_direction   =  1;
}


template <int dim>
void CoupledElastoDynamics<dim>::assemble_system()
{
    QGauss<dim> quadrature_formula(parameters.quad_order);

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
    Functions::ConstantFunction<dim> lambda(parameters.lambda), mu(parameters.mu);


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

    stepping_matrix*=time.get_delta_t()*time.get_delta_t()*parameters.theta*parameters.theta;

    stepping_matrix.add(1,mass_matrix);

    hanging_node_constraints.condense(stepping_matrix);

    //Calculate contribution of gravity and store them in gravitational_force
    if (compute_gravity)
    {
        Vector<double> gravity_vector(dim);
        // assign the specified values, rho * g is assumed
        gravity_vector[gravity_direction] = parameters.rho * gravity_value;

        // create a constant function object
        Functions::ConstantFunction<dim, double> gravity_function(gravity_vector);

        // create the contribution to the right-hand side vector
        VectorTools::create_right_hand_side( mapping,
                                             dof_handler, QGauss<dim>(parameters.quad_order),
                                             gravity_function, gravitational_force);
    }
}


template <int dim>
void CoupledElastoDynamics<dim>::assemble_rhs()
{
    std::cout<<"\t Assemble system "<<std::endl;
    system_rhs=0.0;

    // quadrature formula for integration over faces (dim-1)
    QGauss<dim-1> face_quadrature_formula(parameters.quad_order);

    FEFaceValues<dim> fe_face_values(mapping,
                                     fe,
                                     face_quadrature_formula,
                                     update_values |
                                     update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell    = fe.dofs_per_cell;
    const unsigned int n_face_q_points  = face_quadrature_formula.size();

    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // in order to convert the array precice forces in a vector format
    Tensor<1, dim>      spatial_force_vector;
    double              surface_area;

    // looks for the correct value in the array obtained by preCICE
    int force_iterator  = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_rhs    = 0;

        // assembling the right-hand side force vector each timestep
        // by applying contributions from the coupling interface
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
            if (cell->face(face)->at_boundary() == true &&
                    cell->face(face)->boundary_id() == parameters.interface_mesh_id)
            {
                fe_face_values.reinit(cell, face);

                // get face area dA, in order to apply Nansons formula
                surface_area = cell->face(face)->measure();

                // store coupling data in a traction vector
                // obtained forces are (according to OpenFOAM) measured in the deformed configuration.
                // calculate needed traction from forces according to Nansons formula:
                // we obtain as coupling data: t*da = f and use t_0 = t * da/dA = f/dA
                for(uint jj = 0; jj < dim; ++jj)
                    spatial_force_vector[jj] = precice_forces[force_iterator * dim + jj]/surface_area;

                ++force_iterator;

                for (unsigned int face_q_point = 0; face_q_point < n_face_q_points; ++face_q_point)
                {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const unsigned int component_i =
                                fe.system_to_component_index(i).first;

                        cell_rhs(i) += fe_face_values.shape_value(i, face_q_point) *
                                spatial_force_vector[component_i] *
                                fe_face_values.JxW(face_q_point);
                    }
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
    old_velocity=velocity;
    old_displacement=displacement;

    // add contribution of gravitational forces
    if (compute_gravity)
        system_rhs.add(1,gravitational_force);

    // assemble global RHS:
    // RHS=(M-theta*(1-theta)*delta_t^2*K)*V_n - delta_t*K* D_n + delta_t*theta*F_n+1 + delta_t*(1-theta)*F_n

    // tmp vector to store intermediate results
    Vector<double> tmp;
    tmp.reinit(dof_handler.n_dofs());

    tmp=system_rhs;

    system_rhs*=time.get_delta_t() * parameters.theta;
    system_rhs.add(time.get_delta_t()*(1-parameters.theta), old_forces);
    old_forces=tmp;

    mass_matrix.vmult(tmp, old_velocity);
    system_rhs.add(1,tmp);

    stiffness_matrix.vmult(tmp, old_velocity);
    system_rhs.add(-parameters.theta*time.get_delta_t()*time.get_delta_t()*(1-parameters.theta), tmp);

    stiffness_matrix.vmult(tmp, old_displacement);
    system_rhs.add(-time.get_delta_t(), tmp);

    hanging_node_constraints.condense(system_rhs);

    // copy the system_matrix every timestep, since applying the BC deletes certain rows and columns
    system_matrix=0.0;
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
        VectorTools::interpolate_boundary_values(dof_handler,
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
void CoupledElastoDynamics<dim>::solve()
{
    std::cout<<"\t CG solver: "<<std::endl;
    SolverControl solver_control(1000, 1e-12);
    SolverCG<>    cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, velocity, system_rhs, preconditioner);

    // assert divergence
    Assert(velocity.linfty_norm()<1e4, ExcMessage("Linear system diverged"));
    std::cout<<"\t     No of iterations:\t"<<solver_control.last_step()
            <<"\n \t     Final residual:\t"<<solver_control.last_value()<<std::endl;
    hanging_node_constraints.distribute(velocity);
}

template <int dim>
void CoupledElastoDynamics<dim>::update_displacement()
{
    // D_n+1= D_n + delta_t*theta* V_n+1 + delta_t*(1-theta)* V_n
    displacement.add( time.get_delta_t() * parameters.theta, velocity);
    displacement.add( time.get_delta_t() * ( 1 - parameters.theta), old_velocity);

}


template <int dim>
void CoupledElastoDynamics<dim>::output_results(const unsigned int timestep) const
{
    // compute the strains save it in the outputfiles
    StrainPostprocessor<dim> strain_u;

    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(displacement,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    data_out.add_data_vector(displacement, strain_u);

    // visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(parameters.poly_degree, dof_handler, displacement);
    data_out.build_patches(q_mapping, parameters.poly_degree);

    // check, if the output directory exists
    std::ifstream output_directory (case_path + "dealii_output");
    Assert(output_directory, ExcMessage("Unable to find the output directory. "
                                        "By default, this program stores result files in a directory called dealii_output. "
                                        "This needs to be located in your case directory, where the parameter file is located as well."));

    // store all files in a seperate folder called dealii_ouput
    std::ofstream output(case_path + "dealii_output/solution-" + std::to_string(timestep) + ".vtk");
    data_out.write_vtk(output);
    std::cout<< "\t Output written to solution-" + std::to_string(timestep) + ".vtk \n" <<std::endl;
}

template <int dim>
void CoupledElastoDynamics<dim>::compute_timesteps()
{

    if(parameters.enable_precice == true)
    {
        while(precice.isCouplingOngoing() &&
              time.get_timestep() < time.get_n_timesteps())
        {

            time.increment();

            std::cout << "  Time = " <<time.current()
                      << " at timestep " << time.get_timestep()
                      << " of " <<time.get_n_timesteps()
                      << std::endl;

            if (precice.isActionRequired(precice::constants::actionWriteIterationCheckpoint())){
                save_old_state();
                precice.fulfilledAction(precice::constants::actionWriteIterationCheckpoint());
            }

            assemble_rhs();

            solve();

            update_displacement();

            advance_precice();

            if (precice.isActionRequired(precice::constants::actionReadIterationCheckpoint())){
                reload_old_state();
                precice.fulfilledAction(precice::constants::actionReadIterationCheckpoint());
            }

            if(precice.isTimestepComplete()
                    && time.get_timestep() % parameters.output_interval == 0)
                output_results(time.get_timestep());

        }

        precice.finalize();
    }
    else
        while(time.get_timestep() < time.get_n_timesteps())
        {

            time.increment();

            std::cout << "  Time = " <<time.current()
                      << " at timestep " << time.get_timestep()
                      << " of " <<time.get_n_timesteps()
                      << std::endl;

            assemble_rhs();

            solve();

            update_displacement();

            if( time.get_timestep() % parameters.output_interval == 0)
                output_results(time.get_timestep());
        }
}

template <int dim>
void CoupledElastoDynamics<dim>::initialize_precice()
{

    if(parameters.enable_precice == true)
    {
        // read the precice configuration file to configure coupling features at run-time
        precice.configure(parameters.config_file);

        // assert matching dimensions between deal.ii and precice
        // only valid for the current adapter setup
        // TODO: Adapt for quasi-2D cases (#5)
        Assert(dim == precice.getDimensions(),
               ExcDimensionMismatch(dim, precice.getDimensions()));

        // get precice specific IDs from precice
        node_mesh_id = precice.getMeshID(parameters.node_mesh);
        face_mesh_id = precice.getMeshID(parameters.face_mesh);
        forces_data_id  = precice.getDataID(parameters.read_data_name, face_mesh_id);
        displacements_data_id = precice.getDataID(parameters.write_data_name, node_mesh_id);
    }

    // initialization is also needed for uncoupled simulation
    // get the number of interface nodes from deal.ii
    std::set<types::boundary_id> couplingBoundary;
    couplingBoundary.insert(parameters.interface_mesh_id);
    const FEValuesExtractors::Scalar x_displacement(0);

    DoFTools::extract_boundary_dofs(dof_handler, fe.component_mask(x_displacement), coupling_dofs, couplingBoundary);
    n_interface_nodes = coupling_dofs.n_elements();//TODO: Adapt

    std::cout << "\t Number of coupling nodes:     "
              << n_interface_nodes << std::endl;

    precice_displacements.resize(dim * n_interface_nodes);
    interface_nodes_positions.resize(dim * n_interface_nodes);
    interface_nodes_ids.resize(n_interface_nodes);

    // same for the face based mesh
    // number of coupling faces already obtained in the make_grid function
    precice_forces.resize(dim * n_interface_faces);

    // set a constant value for each direction in case of an uncoupled simulation
    if (parameters.enable_precice == false)
        for (uint it = 0; it  < precice_forces.size()/dim; it++)
        {
            precice_forces[it*dim]= 0;
            precice_forces[it*dim+1]= 0;
            if(dim == 3)
                precice_forces[it*dim+2]= 0;
        }

    interface_faces_positions.resize(dim * n_interface_faces);
    interface_faces_ids.resize(n_interface_faces);

    std::cout << "\t Number of coupling faces:     "
              << n_interface_faces << std::endl;


    if(parameters.enable_precice == true)
    {
        // get the coordinates of the interface nodes from deal.ii
        std::map<types::global_dof_index, Point<dim>> support_points;
        // TODO: Check mapping: Maybe add the mapping object to get the coordinates of higher order shape functions
        DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, support_points);
        // support_points contains now the coordinates of all dofs
        // in the next step, the relevant coordinates are extracted using the extracted coupling_dofs
        int node_position_iterator=0;
        for (auto element=coupling_dofs.begin(); element!=coupling_dofs.end(); ++element)
        {
            for(int jj=0; jj<dim; ++jj)
                interface_nodes_positions[node_position_iterator * dim + jj] = support_points[*element][jj];

            ++node_position_iterator;
        }

        // pass node coordinates to precice
        precice.setMeshVertices(node_mesh_id, n_interface_nodes, interface_nodes_positions.data(), interface_nodes_ids.data());

        // get the coordinates of the face centers and store them in interface_faces_positions
        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler.begin_active(),
                endc = dof_handler.end();

        int face_position_iterator = 0;
        for (; cell!=endc; ++cell)
            for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
            {
                if (cell->face(face)->at_boundary()
                        &&
                        (cell->face(face)->boundary_id() == parameters.interface_mesh_id))
                {
                    for(int jj=0; jj<dim; ++jj)
                        interface_faces_positions[face_position_iterator * dim + jj] = cell->face(face)->center()[jj];

                    ++face_position_iterator;
                }
            }


        // pass face coordinates to precice
        precice.setMeshVertices(face_mesh_id, n_interface_faces, interface_faces_positions.data(), interface_faces_ids.data());

        precice.initialize();

        // write initial writeData to preCICE
        if (precice.isActionRequired(precice::constants::actionWriteInitialData()))
        {
            // store initial write_data for precice in precice_displacements
            extract_relevant_displacements(precice_displacements);

            precice.writeBlockVectorData(displacements_data_id, n_interface_nodes, interface_nodes_ids.data(), precice_displacements.data());
            precice.fulfilledAction(precice::constants::actionWriteInitialData());

            precice.initializeData();
        }

        // read initial readData from preCICE for the first time step
        if (precice.isReadDataAvailable())
            precice.readBlockVectorData(forces_data_id, n_interface_faces, interface_faces_ids.data(), precice_forces.data());
    }
}

template <int dim>
void CoupledElastoDynamics<dim>::advance_precice()
{
    if(precice.isWriteDataRequired(time.get_delta_t()))
    {
        extract_relevant_displacements(precice_displacements);
        precice.writeBlockVectorData(displacements_data_id, n_interface_nodes, interface_nodes_ids.data(), precice_displacements.data());
    }

    precice.advance(time.get_delta_t());

    if(precice.isReadDataAvailable())
    {
        precice.readBlockVectorData(forces_data_id, n_interface_faces, interface_faces_ids.data(), precice_forces.data());
    }
}

template <int dim>
void CoupledElastoDynamics<dim>::extract_relevant_displacements(std::vector<double>& precice_displacements)
{
    int data_iterator = 0;
    for (auto element=coupling_dofs.begin(); element!=coupling_dofs.end(); ++element)
    {
        for(int jj=0; jj<dim; ++jj)
            precice_displacements[data_iterator * dim + jj] = displacement[*element+jj];

        ++data_iterator;
    }
}

template <int dim>
void CoupledElastoDynamics<dim>::save_old_state()
{
    // store current state for implict coupling
    old_state_velocity          = velocity;
    old_state_old_velocity      = old_velocity;
    old_state_displacement      = displacement;
    old_state_old_displacement  = old_displacement;
    old_state_old_forces        = old_forces;
}

template <int dim>
void CoupledElastoDynamics<dim>::reload_old_state()
{
    // load old state for implicit coupling
    velocity            = old_state_velocity;
    old_velocity        = old_state_old_velocity;
    displacement        = old_state_displacement;
    old_displacement    = old_state_old_displacement;
    old_forces          = old_state_old_forces;

    time.restore();
}

template <int dim>
void CoupledElastoDynamics<dim>::run()
{
    make_grid();

    setup_system();

    output_results(time.get_timestep());

    assemble_system();

    initialize_precice();

    compute_timesteps();

}
} // end namespace adapter

int main(int argc, char **argv)
{
    try
    {
        std::cout << "--------------------------------------------------\n"
                  << "             Running deal.ii solver \n"
                  << "--------------------------------------------------\n"
                  <<std::endl;

        const unsigned int dim = 3;

        std::string parameter_file;
        if (argc > 1)
            parameter_file = argv[1];
        else
            parameter_file = "parameters.prm";

        // Extract path for the output directory for the output directory
        size_t pos = parameter_file.find_last_of("/");
        std::string case_path = std::string::npos == pos ? "" : parameter_file.substr(0, pos+1);

        adapter::CoupledElastoDynamics<dim> elastic_solver(case_path);
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
