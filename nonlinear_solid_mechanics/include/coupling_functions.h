#ifndef COUPLING_FUNCTIONS_H
#define COUPLING_FUNCTIONS_H

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <precice/SolverInterface.hpp>

#include "parameter_handling.h"

namespace adapter
{
  using namespace dealii;

  namespace PreciceDealCoupling
  {
    template <int dim, typename VectorType>
    class CouplingFunctions
    {
    public:
      /**
       * @brief      Constructor, which sets up the precice Solverinterface
       *
       * @param[in]  parameters Parameter class, which hold the data specified
       *             in the parameters.prm file
       */
      CouplingFunctions(const Parameters::AllParameters &parameters);

      /**
       * @brief     Initializes preCICE and passes all relevant data to preCICE
       *
       * @param[in] dof_handler Initialized dof_handler
       * @param[in] coupling_data Data, which should be given to preCICE and
       *            exchanged with other participants. If this data is
       *            required already in the beginning depends on your individual
       *            configuration and preCICE determines it automatically.
       *            In many cases, this data will just represent your initial
       *            condition.
       */
      void
      initialize_precice(const DoFHandler<dim> &dof_handler,
                         const VectorType &     coupling_data);

      void
      advance_precice(const VectorType &coupling_data,
                      const double      computedTimestepLength);

      void
      extract_relevant_displacements(const VectorType &coupling_data);

      /**
       * @brief      Saves time dependent variables in case of an implicit coupling
       *
       * @param[in]  state_variables vector containing all variables to store
       *
       * @note       This function only makes sense, if it is used with
       *             @p reload_old_state. Therefore, the order, in which the
       *             variables are passed into the vector must be the same for
       *             both functions.
       */
      void
      save_old_state(const std::vector<VectorType> &state_variables);

      /**
       * @brief      Reloads the previously stored variables in case of an implicit
       *             coupling
       *
       * @param[out] state_variables vector containing all variables to reload
       *
       * @note       This function only makes sense, if the state variables have been
       *             stored by calling @p save_old_state. Therefore, the order, in
       *             which the variables are passed into the vector must be the
       * same for both functions.
       */
      void
      reload_old_state(std::vector<VectorType> &state_variables) const;


    private:
      // preCICE related initializations
      const unsigned int deal_boundary_id;
      const bool         enable_precice;
      const std::string  config_file;
      const std::string  participant;
      const std::string  mesh_name;
      const std::string  read_data_name;
      const std::string  write_data_name;

      int precice_mesh_id;
      int n_interface_nodes;
      int forces_data_id;
      int displacements_data_id;

      precice::SolverInterface precice;

      IndexSet coupling_dofs;

      std::vector<int>    interface_nodes_ids;
      std::vector<double> precice_forces;
      std::vector<double> precice_displacements;

      std::vector<VectorType> old_state_data;
    };



    template <int dim, typename VectorType>
    CouplingFunctions<dim, VectorType>::CouplingFunctions(
      const Parameters::AllParameters &parameters)
      : deal_boundary_id(parameters.interface_mesh_id)
      , enable_precice(parameters.enable_precice)
      , config_file(parameters.config_file)
      , participant(parameters.participant)
      , mesh_name(parameters.mesh_name)
      , read_data_name(parameters.read_data_name)
      , write_data_name(parameters.write_data_name)
      , precice(participant,
                config_file,
                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    {}



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::initialize_precice(
      const DoFHandler<dim> &dof_handler,
      const VectorType &     coupling_data)
    {
      Assert(dim == precice.getDimensions(),
             ExcDimensionMismatch(dim, precice.getDimensions()));

      // get precice specific IDs from precice
      precice_mesh_id = precice.getMeshID(mesh_name);
      forces_data_id  = precice.getDataID(read_data_name, precice_mesh_id);
      displacements_data_id =
        precice.getDataID(write_data_name, precice_mesh_id);


      // get the number of interface nodes from deal.ii
      std::set<types::boundary_id> couplingBoundary;
      couplingBoundary.insert(deal_boundary_id);

      DoFTools::extract_boundary_dofs(dof_handler,
                                      ComponentMask(),
                                      coupling_dofs,
                                      couplingBoundary);

      n_interface_nodes = coupling_dofs.n_elements() / dim;

      std::cout << "\t Number of coupling nodes:     " << n_interface_nodes
                << std::endl;

      std::vector<double> interface_nodes_positions(dim * n_interface_nodes);

      precice_displacements.resize(dim * n_interface_nodes);
      interface_nodes_ids.resize(n_interface_nodes);

      // number of coupling faces already obtained in the make_grid function
      precice_forces.resize(dim * n_interface_nodes);

      // get the coordinates of the interface nodes from deal.ii
      std::map<types::global_dof_index, Point<dim>> support_points;

      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                           dof_handler,
                                           support_points);
      // support_points contains now the coordinates of all dofs
      // in the next step, the relevant coordinates are extracted using the
      // extracted coupling_dofs

      // New set containing only once
      IndexSet coupling_dofs_x_comp;
      // Extract again, since vector valued FE
      const FEValuesExtractors::Scalar x_displacement(0);

      DoFTools::extract_boundary_dofs(dof_handler,
                                      dof_handler.get_fe().component_mask(
                                        x_displacement),
                                      coupling_dofs_x_comp,
                                      couplingBoundary);

      int node_position_iterator = 0;
      for (auto element = coupling_dofs_x_comp.begin();
           element != coupling_dofs_x_comp.end();
           ++element)
        {
          for (int jj = 0; jj < dim; ++jj)
            interface_nodes_positions[node_position_iterator * dim + jj] =
              support_points[*element][jj];

          ++node_position_iterator;
        }

      // pass node coordinates to precice
      precice.setMeshVertices(precice_mesh_id,
                              n_interface_nodes,
                              interface_nodes_positions.data(),
                              interface_nodes_ids.data());


      precice.initialize();

      // write initial writeData to preCICE
      if (precice.isActionRequired(
            precice::constants::actionWriteInitialData()))
        {
          // store initial write_data for precice in precice_displacements
          extract_relevant_displacements(coupling_data);

          precice.writeBlockVectorData(displacements_data_id,
                                       n_interface_nodes,
                                       interface_nodes_ids.data(),
                                       precice_displacements.data());

          precice.markActionFulfilled(
            precice::constants::actionWriteInitialData());

          precice.initializeData();
        }

      // read initial readData from preCICE for the first time step
      if (precice.isReadDataAvailable())
        precice.readBlockVectorData(forces_data_id,
                                    n_interface_nodes,
                                    interface_nodes_ids.data(),
                                    precice_forces.data());
    }


    // Currently, port from precice forces to VectorType is missing
    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::advance_precice(
      const VectorType &coupling_data,
      const double      computedTimestepLength)
    {
      if (precice.isWriteDataRequired(computedTimestepLength))
        {
          extract_relevant_displacements(coupling_data);
          precice.writeBlockVectorData(displacements_data_id,
                                       n_interface_nodes,
                                       interface_nodes_ids.data(),
                                       precice_displacements.data());
        }

      precice.advance(computedTimestepLength);

      if (precice.isReadDataAvailable())
        {
          precice.readBlockVectorData(forces_data_id,
                                      n_interface_nodes,
                                      interface_nodes_ids.data(),
                                      precice_forces.data());
        }
    }


    // TODO: Check this again, especially operator[] for BlockVectors
    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::extract_relevant_displacements(
      const VectorType &coupling_data)
    {
      int data_iterator = 0;
      for (auto element = coupling_dofs.begin(); element != coupling_dofs.end();
           ++element)
        {
          precice_displacements[data_iterator] = coupling_data[*element];

          ++data_iterator;
        }
    }


    // TODO: What about time? Maybe add an additional container
    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::save_old_state(
      const std::vector<VectorType> &state_variables)
    {
      old_state_data = state_variables;
    }



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::reload_old_state(
      std::vector<VectorType> &state_variables) const
    {
      state_variables = old_state_data;
    }
  } // namespace PreciceDealCoupling
} // namespace adapter

#endif // COUPLING_FUNCTIONS_H
