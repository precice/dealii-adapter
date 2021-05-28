#ifndef ADAPTER_H
#define ADAPTER_H

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <adapter/dof_tools_extension.h>
#include <adapter/time_handler.h>
#include <precice/SolverInterface.hpp>


namespace Adapter
{
  using namespace dealii;

  /**
   * The Adapter class keeps all functionalities to couple deal.II to other
   * solvers with preCICE i.e. data structures are set up, necessary information
   * is passed to preCICE etc.
   */
  template <int dim, int data_dim, typename VectorType>
  class Adapter
  {
  public:
    /**
     * @brief      Constructor, which sets up the precice Solverinterface
     *
     * @param[in]  parameters Parameter class, which hold the data specified
     *             in the parameters.prm file
     * @param[in]  deal_boundary_interface_id Boundary ID of the triangulation,
     *             which is associated with the coupling interface.
     */
    template <typename ParameterClass>
    Adapter(const ParameterClass &parameters,
            const unsigned int    deal_boundary_interface_id);

    /**
     * @brief      Initializes preCICE and passes all relevant data to preCICE
     *
     * @param[in]  dof_handler Initialized dof_handler
     * @param[in]  deal_to_precice Data, which should be given to preCICE and
     *             exchanged with other participants. Wether this data is
     *             required already in the beginning depends on your
     *             individual configuration and preCICE determines it
     *             automatically. In many cases, this data will just represent
     *             your initial condition.
     * @param[out] precice_to_deal Data, which is received from preCICE/ from
     *             other participants. Wether this data is useful already in
     *             the beginning depends on your individual configuration and
     *             preCICE determines it automatically. In many cases, this
     *             data will just represent the initial condition of other
     *             participants.
     */
    void
    initialize(const DoFHandler<dim> &dof_handler,
               const VectorType &     deal_to_precice,
               VectorType &           precice_to_deal);

    /**
     * @brief      Advances preCICE after every timestep, converts data formats
     *             between preCICE and dealii
     *
     * @param[in]  deal_to_precice Same data as in @p initialize_precice() i.e.
     *             data, which should be given to preCICE after each time step
     *             and exchanged with other participants.
     * @param[out] precice_to_deal Same data as in @p initialize_precice() i.e.
     *             data, which is received from preCICE/other participants
     *             after each time step and exchanged with other participants.
     * @param[in]  computed_timestep_length Length of the timestep used by
     *             the solver.
     */
    void
    advance(const VectorType &deal_to_precice,
            VectorType &      precice_to_deal,
            const double      computed_timestep_length);

    /**
     * @brief      Saves current state of time dependent variables in case of an
     *             implicit coupling
     *
     * @param[in]  state_variables Vector containing all variables to store as
     *             reference
     *
     * @note       This function only makes sense, if it is used with
     *             @p reload_old_state_if_required. Therefore, the order, in which the
     *             variables are passed into the vector must be the same for
     *             both functions.
     * @note       The absolute time has no impact on the computation, but on the output.
     *             Therefore, we call here in the @p Time class a method to store the
     *             current time and reload it later. This is necessary, in
     *             case your solver is subcycling.
     */
    void
    save_current_state_if_required(
      const std::vector<VectorType *> &state_variables,
      Time &                           time_class);

    /**
     * @brief      Reloads the previously stored variables in case of an implicit
     *             coupling. The current implementation supports subcycling,
     *             i.e. previously refers o the last time
     *             @p save_current_state_if_required() has been called.
     *
     * @param[out] state_variables Vector containing all variables to reload
     *             as reference
     *
     * @note       This function only makes sense, if the state variables have been
     *             stored by calling @p save_current_state_if_required. Therefore,
     *             the order, in which the variables are passed into the
     *             vector must be the same for both functions.
     */
    void
    reload_old_state_if_required(std::vector<VectorType *> &state_variables,
                                 Time &                     time_class);

    /**
     * @brief public precice solverinterface
     */

    precice::SolverInterface precice;

    // Boundary ID of the deal.II mesh, associated with the coupling
    // interface. The variable is public and should be used during grid
    // generation, but is also involved during system assembly. The only thing,
    // one needs to make sure is, that this ID is not given to another part of
    // the boundary e.g. clamped one.
    const unsigned int deal_boundary_interface_id;

  private:
    // preCICE related initializations
    // These variables are specified and read from the parameter file
    const std::string mesh_name;
    const std::string read_data_name;
    const std::string write_data_name;

    // To be adjusted for MPI parallelized codes
    static constexpr unsigned int this_mpi_process = 0;
    static constexpr unsigned int n_mpi_processes  = 1;

    // These IDs are given by preCICE during initialization
    int mesh_id;
    int read_data_id;
    int write_data_id;
    int n_interface_nodes;

    // Dof IndexSets of the global deal.II vectors, containing relevant
    // coupling dof indices
    IndexSet coupling_dofs_x_comp;
    IndexSet coupling_dofs_y_comp;
    IndexSet coupling_dofs_z_comp;

    // Data containers which are passed to preCICE in an appropriate preCICE
    // specific format
    std::vector<int>    interface_nodes_ids;
    std::vector<double> read_data;
    std::vector<double> write_data;

    // Container to store time dependent data in case of an implicit coupling
    std::vector<VectorType> old_state_data;
    double                  old_time_value;

    /**
     * @brief format_deal_to_precice Formats a global deal.II vector of type
     *        VectorType to a std::vector for preCICE. This functions is only
     *        used internally in the class and should not be called in the
     *        solver.
     *
     * @param[in] deal_to_precice Global deal.II vector of VectorType. The
     *            result (preCICE specific vector) is stored in the class in
     *            the variable 'write_data'.
     *
     * @note  The order, in which preCICE obtains data from the solver, needs
     *        to be consistent with the order of the initially passed vertices
     *        coordinates.
     */
    void
    format_deal_to_precice(const VectorType &deal_to_precice);

    /**
     * @brief format_precice_to_deal Takes the std::vector obtained by preCICE
     *        in 'read_data' and inserts the values to the right position in
     *        the global deal.II vector of size n_global_dofs. This is the
     *        opposite functionality as @p foramt_precice_to_deal(). This
     *        functions is only used internally in the class and should not
     *        be called in the solver.
     *
     * @param[out] precice_to_deal Global deal.II vector of VectorType and
     *             size n_global_dofs.
     *
     * @note  The order, in which preCICE obtains data from the solver, needs
     *        to be consistent with the order of the initially passed vertices
     *        coordinates.
     */
    void
    format_precice_to_deal(VectorType &precice_to_deal) const;
  };



  template <int dim, int data_dim, typename VectorType>
  template <typename ParameterClass>
  Adapter<dim, data_dim, VectorType>::Adapter(
    const ParameterClass &parameters,
    const unsigned int    deal_boundary_interface_id)
    : precice(parameters.participant_name,
              parameters.config_file,
              this_mpi_process,
              n_mpi_processes)
    , deal_boundary_interface_id(deal_boundary_interface_id)
    , mesh_name(parameters.mesh_name)
    , read_data_name(parameters.read_data_name)
    , write_data_name(parameters.write_data_name)
  {}



  template <int dim, int data_dim, typename VectorType>
  void
  Adapter<dim, data_dim, VectorType>::initialize(
    const DoFHandler<dim> &dof_handler,
    const VectorType &     deal_to_precice,
    VectorType &           precice_to_deal)
  {
    AssertThrow(
      dim == precice.getDimensions(),
      ExcMessage("The dimension of your solver needs to be consistent with the "
                 "dimension specified in your precice-config file. In case you "
                 "run one of the tutorials, the dimension can be specified via "
                 "cmake -D DIM=dim ."));

    AssertThrow(dim > 1, ExcNotImplemented());

    // get precice specific IDs from precice and store them in the class
    // they are later needed for data transfer
    mesh_id       = precice.getMeshID(mesh_name);
    read_data_id  = precice.getDataID(read_data_name, mesh_id);
    write_data_id = precice.getDataID(write_data_name, mesh_id);


    // get the number of interface nodes from deal.II
    // Therefore, we extract one component of the vector valued dofs and store
    // them in an IndexSet
    std::set<types::boundary_id> couplingBoundary;
    couplingBoundary.insert(deal_boundary_interface_id);

    const FEValuesExtractors::Scalar x_displacement(0);

    DoFTools::extract_boundary_dofs(dof_handler,
                                    dof_handler.get_fe().component_mask(
                                      x_displacement),
                                    coupling_dofs_x_comp,
                                    couplingBoundary);

    // The dofs related to the y-component are needed as well. See also
    // comment below, why this is necessary.
    if (data_dim > 1)
      {
        const FEValuesExtractors::Scalar y_displacement(1);

        DoFTools::extract_boundary_dofs(dof_handler,
                                        dof_handler.get_fe().component_mask(
                                          y_displacement),
                                        coupling_dofs_y_comp,
                                        couplingBoundary);
      }
    if (data_dim > 2)
      {
        const FEValuesExtractors::Scalar z_displacement(2);
        DoFTools::extract_boundary_dofs(dof_handler,
                                        dof_handler.get_fe().component_mask(
                                          z_displacement),
                                        coupling_dofs_z_comp,
                                        couplingBoundary);
      }

    n_interface_nodes = coupling_dofs_x_comp.n_elements();

    std::cout << "\t Number of coupling nodes:     " << n_interface_nodes
              << std::endl;

    // Set up a vector to pass the node positions to preCICE. Each node is
    // specified once. One needs to specify in the precice-config.xml, whether
    // the data is vector valued or not.
    std::vector<double> interface_nodes_positions(dim * n_interface_nodes);

    // Set up the appropriate size of the data container needed for data
    // exchange. Here, we deal with a vector valued problem for read and write
    // data namely displacement and forces. Therefore, we need dim entries per
    // vertex
    write_data.resize(data_dim * n_interface_nodes);
    read_data.resize(data_dim * n_interface_nodes);
    interface_nodes_ids.resize(n_interface_nodes);

    // get the coordinates of the interface nodes from deal.ii
    std::map<types::global_dof_index, Point<dim>> support_points;

    // We use here a simple Q1 mapping. In case one has more complex
    // geomtries, you might want to change this to a higher order mapping.
    // We only need to map the first component for a dim dimensional problem

    DoFTools::map_boundary_dofs_to_support_points(
      StaticMappingQ1<dim>::mapping,
      dof_handler,
      support_points,
      dof_handler.get_fe().component_mask(x_displacement),
      deal_boundary_interface_id);

    // support_points contains now the coordinates of all dofs
    // in the next step, the relevant coordinates are extracted using the
    // IndexSet with the extracted coupling_dofs.

    // preCICE expects all data in the format [x0, y0, z0, x1, y1 ...]
    int node_position_iterator = 0;
    for (auto element : coupling_dofs_x_comp)
      {
        for (int i = 0; i < dim; ++i)
          interface_nodes_positions[node_position_iterator * dim + i] =
            support_points[element][i];

        ++node_position_iterator;
      }

    // pass node coordinates to precice
    precice.setMeshVertices(mesh_id,
                            n_interface_nodes,
                            interface_nodes_positions.data(),
                            interface_nodes_ids.data());

    // Initialize preCICE internally
    precice.initialize();

    // write initial writeData to preCICE if required
    if (precice.isActionRequired(precice::constants::actionWriteInitialData()))
      {
        // store initial write_data for precice in write_data
        format_deal_to_precice(deal_to_precice);
        if (data_dim > 1)
          {
            precice.writeBlockVectorData(write_data_id,
                                         n_interface_nodes,
                                         interface_nodes_ids.data(),
                                         write_data.data());
          }
        else
          {
            Assert(data_dim == 1, ExcInternalError());
            precice.writeBlockScalarData(write_data_id,
                                         n_interface_nodes,
                                         interface_nodes_ids.data(),
                                         write_data.data());
          }
        precice.markActionFulfilled(
          precice::constants::actionWriteInitialData());
      }

    precice.initializeData();

    // read initial readData from preCICE if required for the first time step
    if (precice.isReadDataAvailable())
      {
        if (data_dim > 1)
          {
            precice.readBlockVectorData(read_data_id,
                                        n_interface_nodes,
                                        interface_nodes_ids.data(),
                                        read_data.data());
          }
        else
          {
            Assert(data_dim == 1, ExcInternalError());
            precice.readBlockScalarData(read_data_id,
                                        n_interface_nodes,
                                        interface_nodes_ids.data(),
                                        read_data.data());
          }
        format_precice_to_deal(precice_to_deal);
      }
  }



  template <int dim, int data_dim, typename VectorType>
  void
  Adapter<dim, data_dim, VectorType>::advance(
    const VectorType &deal_to_precice,
    VectorType &      precice_to_deal,
    const double      computed_timestep_length)
  {
    // This is essentially the same as during initialization
    // We have already all IDs and just need to convert our obtained data to
    // the preCICE compatible 'write_data' vector, which is done in the
    // format_deal_to_precice function. All this is of course only done in
    // case write data is required.
    if (precice.isWriteDataRequired(computed_timestep_length))
      {
        format_deal_to_precice(deal_to_precice);

        if (data_dim > 1)
          {
            precice.writeBlockVectorData(write_data_id,
                                         n_interface_nodes,
                                         interface_nodes_ids.data(),
                                         write_data.data());
          }
        else
          {
            Assert(data_dim == 1, ExcInternalError());
            precice.writeBlockScalarData(write_data_id,
                                         n_interface_nodes,
                                         interface_nodes_ids.data(),
                                         write_data.data());
          }
      }

    // Here, we need to specify the computed time step length and pass it to
    // preCICE
    precice.advance(computed_timestep_length);

    // Here, we obtain data from another participant. Again, we insert the
    // data in our global vector by calling format_precice_to_deal
    if (precice.isReadDataAvailable())
      {
        if (data_dim > 1)
          {
            precice.readBlockVectorData(read_data_id,
                                        n_interface_nodes,
                                        interface_nodes_ids.data(),
                                        read_data.data());
          }
        else
          {
            Assert(data_dim == 1, ExcInternalError());
            precice.readBlockScalarData(read_data_id,
                                        n_interface_nodes,
                                        interface_nodes_ids.data(),
                                        read_data.data());
          }

        format_precice_to_deal(precice_to_deal);
      }
  }



  template <int dim, int data_dim, typename VectorType>
  void
  Adapter<dim, data_dim, VectorType>::format_deal_to_precice(
    const VectorType &deal_to_precice)
  {
    // Assumption: x index is in the same position as y index in each IndexSet
    // In general, higher order support points in the element are first
    // ordered in the x component. An IndexSet for the first component might
    // look like this: [1] [3456] [11] for a 7th order 1d interface/2d cell.
    // Therefore, an index for the respective x component dof is not always
    // followed by an index on the same position for the y component

    auto x_comp = coupling_dofs_x_comp.begin();
    auto y_comp = coupling_dofs_y_comp.begin();
    auto z_comp = coupling_dofs_z_comp.begin();

    for (int i = 0; i < n_interface_nodes; ++i)
      {
        AssertIndexRange(data_dim * i, write_data.size());
        AssertIndexRange(*x_comp, deal_to_precice.size());

        write_data[data_dim * i] = deal_to_precice[*x_comp];
        ++x_comp;
        if (data_dim > 1)
          {
            write_data[(data_dim * i) + 1] = deal_to_precice[*y_comp];
            ++y_comp;
          }
        if (data_dim == 2)
          {
            write_data[(data_dim * i) + 2] = deal_to_precice[*z_comp];
            ++z_comp;
          }
      }
  }



  template <int dim, int data_dim, typename VectorType>
  void
  Adapter<dim, data_dim, VectorType>::format_precice_to_deal(
    VectorType &precice_to_deal) const
  {
    // This is the opposite direction as above. See comment there.
    auto x_comp = coupling_dofs_x_comp.begin();
    auto y_comp = coupling_dofs_y_comp.begin();
    auto z_comp = coupling_dofs_z_comp.begin();

    for (int i = 0; i < n_interface_nodes; ++i)
      {
        AssertIndexRange(data_dim * i, read_data.size());
        AssertIndexRange(*x_comp, precice_to_deal.size());
        precice_to_deal[*x_comp] = read_data[data_dim * i];
        ++x_comp;
        if (data_dim > 1)
          {
            precice_to_deal[*y_comp] = read_data[(data_dim * i) + 1];
            ++y_comp;
          }
        if (data_dim > 2)
          {
            precice_to_deal[*z_comp] = read_data[(data_dim * i) + 2];
            ++z_comp;
          }
      }
  }



  template <int dim, int data_dim, typename VectorType>
  void
  Adapter<dim, data_dim, VectorType>::save_current_state_if_required(
    const std::vector<VectorType *> &state_variables,
    Time &                           time_class)
  {
    // First, we let preCICE check, whether we need to store the variables.
    // Then, the data is stored in the class
    if (precice.isActionRequired(
          precice::constants::actionWriteIterationCheckpoint()))
      {
        old_state_data.resize(state_variables.size());

        for (uint i = 0; i < state_variables.size(); ++i)
          old_state_data[i] = *(state_variables[i]);

        old_time_value = time_class.current();

        precice.markActionFulfilled(
          precice::constants::actionWriteIterationCheckpoint());
      }
  }



  template <int dim, int data_dim, typename VectorType>
  void
  Adapter<dim, data_dim, VectorType>::reload_old_state_if_required(
    std::vector<VectorType *> &state_variables,
    Time &                     time_class)
  {
    // In case we need to reload a state, we just take the internally stored
    // data vectors and write then in to the input data
    if (precice.isActionRequired(
          precice::constants::actionReadIterationCheckpoint()))
      {
        Assert(state_variables.size() == old_state_data.size(),
               ExcMessage(
                 "state_variables are not the same as previously saved."));

        for (uint i = 0; i < state_variables.size(); ++i)
          *(state_variables[i]) = old_state_data[i];

        // Here, we expect the time class to offer an option to specify a
        // given time value.
        time_class.set_absolute_time(old_time_value);

        precice.markActionFulfilled(
          precice::constants::actionReadIterationCheckpoint());
      }
  }
} // namespace Adapter

#endif // ADAPTER_H
