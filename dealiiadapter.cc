/*---------------------------------------------------------------------------*\
   deal.II solver for linear elasto-dynamics with
   preCICE-adapter for partitiond FSI simulations

   Copyright (c) 2018

   Build on an extension of the step-8 tutorial program of the deal.II library
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
        prm.declare_entry("Polynomial degree", "2",
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
    void assemble_rhs();
    void solve();
    void update_displacement();
    void output_results(const unsigned int timestep) const;
    //precice related functions
    void initialize_precice();
    void advance_precice();
    void extract_relevant_displacements(std::vector<double>& precice_displacements);
    void apply_precice_forces(std::vector<double>& precice_forces);
    void save_old_state();
    void reload_old_state();


    Parameters::AllParameters parameters;

    Triangulation<dim> triangulation;

    Time                      time;
    DoFHandler<dim>    dof_handler;

    FESystem<dim> fe;

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

    // variables for implicit coupling
    Vector<double> old_state_old_velocity;
    Vector<double> old_state_old_displacement;
    Vector<double> old_state_old_forces;
    Vector<double> old_state_forces;


    //precice related initializations
    int             node_mesh_id;
    int             face_mesh_id;
    int             forces_data_id;
    int             displacements_data_id;
    int             n_interface_nodes;
    int             n_interface_faces;

    std::vector<double>     precice_forces;
    std::vector<double>     precice_displacements;
    std::vector<double>     interface_nodes_positions;
    std::vector<int>        interface_nodes_ids;
    std::vector<double>     interface_faces_positions;
    std::vector<int>        interface_faces_ids;



    IndexSet                coupling_dofs;

    precice::SolverInterface         precice;
};


template <int dim>// to be removed
void right_hand_side(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     const std::vector<Point<dim>> &points,
                     std::vector<Tensor<1, dim>> &values,
                     unsigned int boundary_id)
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
                    && cell->face(face)->boundary_id() == boundary_id)
            {  values[point_n][1] = 2*1000*-2;//2 cells (refinement) 1000 density 2 gravity
                break;
            }
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
    , fe(FE_Q<dim>(parameters.poly_degree), dim)
    , precice(parameters.participant,0,1)
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
    std::cout<<"  Create mesh: "<<std::endl;

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
    triangulation.refine_global(1);

    std::cout << "\t Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    // Boundary 0 is clamped, 4 and 5 (out-of-plane) are only relevant for 3D
    // count simultaniously the relevant coupling faces
    n_interface_faces = 0;

    typename Triangulation<dim>::active_cell_iterator cell =
            triangulation.begin_active(), endc = triangulation.end();
    for (; cell != endc; ++cell)
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        {
            if (cell->face(face)->at_boundary() ==  true
                    && cell->face(face)->boundary_id ()!=0
                    && cell->face(face)->boundary_id ()!=4
                    && cell->face(face)->boundary_id ()!=5)
            {
                cell->face(face)->set_boundary_id(parameters.interface_mesh_id);
                ++n_interface_faces;
            }
        }
}

template <int dim>
void ElasticProblem<dim>::setup_system()
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

    MatrixCreator::create_mass_matrix (dof_handler, QGauss<dim>(2),
                                       mass_matrix);
    mass_matrix*=parameters.rho;

    old_velocity.reinit(dof_handler.n_dofs());
    velocity.reinit(dof_handler.n_dofs());
    old_displacement.reinit(dof_handler.n_dofs());
    displacement.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
    old_forces.reinit(dof_handler.n_dofs());
    forces.reinit(dof_handler.n_dofs());
    // Loads at time 0
    // TODO: Check, if initial conditions should be set at the beginning
    old_forces=0.0;
}


template <int dim>
void ElasticProblem<dim>::assemble_system()
{
    QGauss<dim> quadrature_formula(parameters.quad_order);

    FEValues<dim> fe_values(fe,
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

    // To save the system_matrix
    stepping_matrix.copy_from(stiffness_matrix);

    stepping_matrix*=time.get_delta_t()*time.get_delta_t()*parameters.theta*parameters.theta;

    stepping_matrix.add(1,mass_matrix);

    hanging_node_constraints.condense(stepping_matrix);

}


template <int dim>
void ElasticProblem<dim>::assemble_rhs()
{
    std::cout<<"\t Assemble system "<<std::endl;
    system_rhs=0.0;

    QGauss<dim> quadrature_formula(parameters.quad_order);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // In order to convert the array precice forces in a vector format
    Tensor<1, dim>      force_vector;
    // Looks for the right value to assign to the vector
    int force_iterator  = 0;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_rhs    = 0;
        fe_values.reinit(cell);

        // Assembling the right hand side force vector
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
            if (cell->face(face)->at_boundary() == true &&
                    cell->face(face)->boundary_id() == parameters.interface_mesh_id)
            {
                // store coupling data in a vector
                for(uint jj = 0; jj < dim; ++jj)
                    force_vector[jj] = precice_forces[force_iterator * dim + jj];

                ++force_iterator;

                for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                        const unsigned int component_i =
                                fe.system_to_component_index(i).first;

                        cell_rhs(i) += fe_values.shape_value(i, q_point) *
                                force_vector[component_i] *
                                fe_values.JxW(q_point);
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

    // Update variables
    old_velocity=velocity;
    old_displacement=displacement;


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

    // Copy the system_matrix every timestep, since applying the BC deletes certain rows and columns
    system_matrix=0.0;
    system_matrix.copy_from(stepping_matrix);


    // 0 refers to the boundary_id
    // TODO: Parametrize
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       velocity,
                                       system_rhs);

}

template <int dim>
void ElasticProblem<dim>::solve()
{
    std::cout<<"\t CG solver: "<<std::endl;
    SolverControl solver_control(1000, 1e-12);
    SolverCG<>    cg(solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    cg.solve(system_matrix, velocity, system_rhs, preconditioner);

    //Assert divergence
    Assert(velocity.linfty_norm()<1e4, ExcMessage("Linear system diverged"));
    std::cout<<"\t     No of iterations:\t"<<solver_control.last_step()
            <<"\n \t     Final residual:\t"<<solver_control.last_value()<<std::endl;
    hanging_node_constraints.distribute(velocity);
}

template <int dim>
void ElasticProblem<dim>::update_displacement()
{
    // D_n+1= D_n + delta_t*theta* V_n+1 + delta_t*(1-theta)* V_n
    displacement.add( time.get_delta_t() * parameters.theta, velocity);
    displacement.add( time.get_delta_t() * ( 1 - parameters.theta), old_velocity);

}


template <int dim>
void ElasticProblem<dim>::output_results(const unsigned int timestep) const
{
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


    //Visualize the displacements on a displaced grid
    MappingQEulerian<dim> q_mapping(parameters.poly_degree, dof_handler, displacement);
    data_out.build_patches(q_mapping, parameters.poly_degree);

    std::ofstream output("solution-" + std::to_string(timestep) + ".vtk");
    data_out.write_vtk(output);
    std::cout<< "\t Output written to solution-" + std::to_string(timestep) + ".vtk \n" <<std::endl;
}

template <int dim>
void ElasticProblem<dim>::initialize_precice()
{

    // get precice specific IDs from precice
    precice.configure(parameters.config_file);

    // Assert matching dimensions between deal.ii and precice
    // Only valid for the current adapter setup
    // TODO: Adapt for quasi-2D cases (#5)
    Assert(dim == precice.getDimensions(),
           ExcDimensionMismatch(dim, precice.getDimensions()));

    node_mesh_id = precice.getMeshID(parameters.node_mesh);
    face_mesh_id = precice.getMeshID(parameters.face_mesh);
    forces_data_id  = precice.getDataID(parameters.read_data_name, face_mesh_id);
    displacements_data_id = precice.getDataID(parameters.write_data_name, node_mesh_id);


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


    // get the coordinates of the interface nodes from deal.ii
    std::map<types::global_dof_index, Point<dim>> support_points;
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

    // same for the face based mesh
    // number of coupling faces already obtained in the make_grid function
    precice_forces.resize(dim * n_interface_faces);
    interface_faces_positions.resize(dim * n_interface_faces);
    interface_faces_ids.resize(n_interface_faces);

    std::cout << "\t Number of coupling faces:     "
              << n_interface_faces << std::endl;


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

                //locally_owned_face_indices[coupling_face_iter]=cell->face_index(f);
                //store the index of the coupling face
                //this is necessary to assign the right face with the right value using multithreading
                ++face_position_iterator;
            }
        }


    // pass face coordinates to precice
    precice.setMeshVertices(face_mesh_id, n_interface_faces, interface_faces_positions.data(), interface_faces_ids.data());

    precice.initialize();

    //Write initial writeData to preCICE
    if (precice.isActionRequired(precice::constants::actionWriteInitialData()))
    {
        // store initial write_data for precice in precice_displacements
        extract_relevant_displacements(precice_displacements);

        precice.writeBlockVectorData(displacements_data_id, n_interface_nodes, interface_nodes_ids.data(), precice_displacements.data());
        precice.fulfilledAction(precice::constants::actionWriteInitialData());

        precice.initializeData();
    }


    //Read initial readData from preCICE for the first time step
    if (precice.isReadDataAvailable())
        precice.readBlockVectorData(forces_data_id, n_interface_faces, interface_faces_ids.data(), precice_forces.data());

    // this function is currently unnecessary (and wrong), since the precice forces are directly used in the assembly function
    //    apply_precice_forces(precice_forces);

}

template <int dim>
void ElasticProblem<dim>::advance_precice()
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

        // this function is currently unnecessary (and wrong), since the precice forces are directly used in the assembly function
        //        apply_precice_forces(precice_forces);
    }
}

template <int dim>
void ElasticProblem<dim>::extract_relevant_displacements(std::vector<double>& precice_displacements)
{
    int data_iterator = 0;
    for (auto element=coupling_dofs.begin(); element!=coupling_dofs.end(); ++element)
    {
        for(int jj=0; jj<dim; ++jj)
            precice_displacements[data_iterator * dim + jj] = displacement[*element+jj];

        ++data_iterator;
    }

}

// this function just stores the force data at the respective position in the force vector,
// which is later used in the assemble_rhs function to build the RHS
template <int dim>
void ElasticProblem<dim>::apply_precice_forces(std::vector<double>& precice_forces)
{
    forces = 0;
    int data_iterator = 0;
    for (auto element=coupling_dofs.begin(); element!=coupling_dofs.end(); ++element)
    {
        for(int jj=0; jj<dim; ++jj)
            forces[*element + jj] = precice_forces[data_iterator * dim + jj];

        ++data_iterator;
    }

}

template <int dim>
void ElasticProblem<dim>::save_old_state()
{
    // Store current state for implict coupling
    if (precice.isActionRequired(precice::constants::actionWriteIterationCheckpoint()))
    {
        old_state_old_velocity = old_velocity;
        old_state_old_displacement = old_displacement;
        old_state_old_forces = old_forces;

        precice.fulfilledAction(precice::constants::actionWriteIterationCheckpoint());
    }

}

template <int dim>
void ElasticProblem<dim>::reload_old_state()
{
    // Load old state for implicit coupling
    if (precice.isActionRequired(precice::constants::actionReadIterationCheckpoint()))
    {
        velocity = old_velocity;
        old_velocity = old_state_old_velocity;
        displacement = old_displacement;
        old_displacement = old_state_old_displacement;
        old_forces = old_state_old_forces;

        time.restore();

        precice.fulfilledAction(precice::constants::actionReadIterationCheckpoint());
    }

}

template <int dim>
void ElasticProblem<dim>::run()
{
    make_grid();

    setup_system();

    output_results(time.get_timestep());

    assemble_system();

    initialize_precice();

    while(precice.isCouplingOngoing() &&
          time.get_timestep() < time.get_n_timesteps())
    {

        time.increment();

        std::cout << "  Time = " <<time.current()
                  << " at timestep " << time.get_timestep()
                  << " of " <<time.get_n_timesteps()
                  << std::endl;

        save_old_state();

        assemble_rhs();

        solve();

        update_displacement();

        advance_precice();

        reload_old_state();

        if(precice.isTimestepComplete())
            output_results(time.get_timestep());

    }

    precice.finalize();
}
} //end namespace adapter

int main()
{
    try
    {
        std::cout << "--------------------------------------------------\n"
                  << "             Running deal.ii solver \n"
                  << "--------------------------------------------------\n"
                  <<std::endl;

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
