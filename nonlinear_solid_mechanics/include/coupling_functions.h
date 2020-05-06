#ifndef COUPLING_FUNCTIONS_H
#define COUPLING_FUNCTIONS_H

#include <deal.II/base/exceptions.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <precice/SolverInterface.hpp>

#include "parameter_handling.h"
#include "time.h"

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
       * @brief      Initializes preCICE and passes all relevant data to preCICE
       *
       * @param[in]  dof_handler Initialized dof_handler
       * @param[in]  deal_to_precice Data, which should be given to preCICE and
       *             exchanged with other participants. Wether this data is
       *             required already in the beginning depends on your
       *             individual configuration and preCICE determines it
       *             automatically. In many cases, this data will just represent
       *             your initial condition.
       * @param[out] precice_to_deal
       */
      void
      initialize_precice(const DoFHandler<dim> &dof_handler,
                         const VectorType &     deal_to_precice,
                         VectorType &           precice_to_deal);

      /**
       * @brief      Advances preCICE after every timestep, converts data formats
       *             between preCICE and dealii
       *
       * @param[in]  deal_to_precice
       * @param[out] precice_to_deal
       * @param[in]  computedTimestepLength
       */
      void
      advance_precice(const VectorType &deal_to_precice,
                      VectorType &      precice_to_deal,
                      const double      computed_timestep_length);

      /**
       * @brief      Saves time dependent variables in case of an implicit coupling
       *
       * @param[in]  state_variables vector containing all variables to store
       *
       * @note       This function only makes sense, if it is used with
       *             @p reload_old_state_if_required. Therefore, the order, in which the
       *             variables are passed into the vector must be the same for
       *             both functions.
       * @note       The absolute time has no impact on the computation, but on the output.
       *             Therefore, we call here in the @p Time class a method to store the
       *             current time and reload it later. This is necessary for
       *             subcycling.
       */
      void
      save_current_state_if_required(
        const std::vector<VectorType *> &state_variables,
        Time &                           time_class);

      /**
       * @brief      Reloads the previously stored variables in case of an implicit
       *             coupling
       *
       * @param[out] state_variables vector containing all variables to reload
       *
       * @note       This function only makes sense, if the state variables have been
       *             stored by calling @p save_current_state_if_required. Therefore, the order, in
       *             which the variables are passed into the vector must be the
       *             same for both functions.
       */
      void
      reload_old_state_if_required(std::vector<VectorType *> &state_variables,
                                   Time &                     time_class);

      /**
       * @brief public precice solverinterface
       */

      precice::SolverInterface precice;

    private:
      // preCICE related initializations
      const unsigned int deal_boundary_id;
      const bool         enable_precice;
      const std::string  mesh_name;
      const std::string  read_data_name;
      const std::string  write_data_name;

      int precice_mesh_id;
      int n_interface_nodes;
      int ptd_data_id;
      int dtp_data_id;

      IndexSet coupling_dofs_x_comp;
      IndexSet coupling_dofs_y_comp;

      std::vector<int>    interface_nodes_ids;
      std::vector<double> precice_ptd;
      std::vector<double> precice_dtp;

      std::vector<VectorType> old_state_data;
      double                  old_time_value;

      void
      format_deal_to_precice(const VectorType &deal_to_precice);

      void
      format_precice_to_deal(VectorType &precice_to_deal) const;
    };



    template <int dim, typename VectorType>
    CouplingFunctions<dim, VectorType>::CouplingFunctions(
      const Parameters::AllParameters &parameters)
      : precice(parameters.participant_name,
                parameters.config_file,
                Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
      , deal_boundary_id(parameters.interface_mesh_id)
      , enable_precice(parameters.enable_precice)
      , mesh_name(parameters.mesh_name)
      , read_data_name(parameters.read_data_name)
      , write_data_name(parameters.write_data_name)
    {}



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::initialize_precice(
      const DoFHandler<dim> &dof_handler,
      const VectorType &     deal_to_precice,
      VectorType &           precice_to_deal)
    {
      Assert(dim == precice.getDimensions(),
             ExcDimensionMismatch(dim, precice.getDimensions()));

      // get precice specific IDs from precice
      precice_mesh_id = precice.getMeshID(mesh_name);
      ptd_data_id     = precice.getDataID(read_data_name, precice_mesh_id);
      dtp_data_id     = precice.getDataID(write_data_name, precice_mesh_id);


      // get the number of interface nodes from deal.ii
      std::set<types::boundary_id> couplingBoundary;
      couplingBoundary.insert(deal_boundary_id);

      // Extract again, since vector valued FE
      const FEValuesExtractors::Scalar x_displacement(0);

      DoFTools::extract_boundary_dofs(dof_handler,
                                      dof_handler.get_fe().component_mask(
                                        x_displacement),
                                      coupling_dofs_x_comp,
                                      couplingBoundary);

      // Extract again, since vector valued FE
      const FEValuesExtractors::Scalar y_displacement(1);

      DoFTools::extract_boundary_dofs(dof_handler,
                                      dof_handler.get_fe().component_mask(
                                        y_displacement),
                                      coupling_dofs_y_comp,
                                      couplingBoundary);

      n_interface_nodes = coupling_dofs_x_comp.n_elements();

      std::cout << "\t Number of coupling nodes:     " << n_interface_nodes
                << std::endl;

      std::vector<double> interface_nodes_positions(dim * n_interface_nodes);

      precice_dtp.resize(dim * n_interface_nodes);
      precice_ptd.resize(dim * n_interface_nodes);
      interface_nodes_ids.resize(n_interface_nodes);

      // get the coordinates of the interface nodes from deal.ii
      std::map<types::global_dof_index, Point<dim>> support_points;

      DoFTools::map_dofs_to_support_points(MappingQ1<dim>(),
                                           dof_handler,
                                           support_points);
      // support_points contains now the coordinates of all dofs
      // in the next step, the relevant coordinates are extracted using the
      // extracted coupling_dofs. Since we deal with a vector valued problem,
      // an IndexSet holding one component is sufficient.

      int node_position_iterator = 0;
      for (auto element : coupling_dofs_x_comp)
        {
          for (int i = 0; i < dim; ++i)
            interface_nodes_positions[node_position_iterator * dim + i] =
              support_points[element][i];

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
          // store initial write_data for precice in precice_dtp
          format_deal_to_precice(deal_to_precice);

          precice.writeBlockVectorData(dtp_data_id,
                                       n_interface_nodes,
                                       interface_nodes_ids.data(),
                                       precice_dtp.data());

          precice.markActionFulfilled(
            precice::constants::actionWriteInitialData());

          precice.initializeData();
        }

      // read initial readData from preCICE for the first time step
      if (precice.isReadDataAvailable())
        {
          precice.readBlockVectorData(ptd_data_id,
                                      n_interface_nodes,
                                      interface_nodes_ids.data(),
                                      precice_ptd.data());

          format_precice_to_deal(precice_to_deal);
        }
    }



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::advance_precice(
      const VectorType &deal_to_precice,
      VectorType &      precice_to_deal,
      const double      computed_timestep_length)
    {
      if (precice.isWriteDataRequired(computed_timestep_length))
        {
          format_deal_to_precice(deal_to_precice);
          precice.writeBlockVectorData(dtp_data_id,
                                       n_interface_nodes,
                                       interface_nodes_ids.data(),
                                       precice_dtp.data());
        }

      precice.advance(computed_timestep_length);

      if (precice.isReadDataAvailable())
        {
          precice.readBlockVectorData(ptd_data_id,
                                      n_interface_nodes,
                                      interface_nodes_ids.data(),
                                      precice_ptd.data());

          format_precice_to_deal(precice_to_deal);
        }
    }



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::format_deal_to_precice(
      const VectorType &deal_to_precice)
    {
      // Assumption: x index is in the same position as y index in each IndexSet
      // In general, higher order support points in the element are first
      // ordered in the x component. An IndexSet for the first component might
      // look like this: [1] [3456] [11] for a 7th order 1d interface/2d cell.
      // Therefore, an index for the respective x component dof is not always
      // followed by an index on the same position for the y component
      // TODO: It works fine, but there might be a more elegant way to implement
      // the iterator over both sets
      auto x_comp = coupling_dofs_x_comp.begin();
      auto y_comp = coupling_dofs_y_comp.begin();
      for (int i = 0; i < n_interface_nodes; ++i)
        {
          precice_dtp[2 * i]       = deal_to_precice[*x_comp];
          precice_dtp[(2 * i) + 1] = deal_to_precice[*y_comp];
          ++x_comp;
          ++y_comp;
        }
    }



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::format_precice_to_deal(
      VectorType &precice_to_deal) const
    {
      auto x_comp = coupling_dofs_x_comp.begin();
      auto y_comp = coupling_dofs_y_comp.begin();
      for (int i = 0; i < n_interface_nodes; ++i)
        {
          precice_to_deal[*x_comp] = precice_ptd[2 * i];
          precice_to_deal[*y_comp] = precice_ptd[(2 * i) + 1];
          ++x_comp;
          ++y_comp;
        }
    }



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::save_current_state_if_required(
      const std::vector<VectorType *> &state_variables,
      Time &                           time_class)
    {
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



    template <int dim, typename VectorType>
    void
    CouplingFunctions<dim, VectorType>::reload_old_state_if_required(
      std::vector<VectorType *> &state_variables,
      Time &                     time_class)
    {
      if (precice.isActionRequired(
            precice::constants::actionReadIterationCheckpoint()))
        {
          Assert(state_variables.size() == old_state_data.size(),
                 ExcMessage(
                   "state_variables are not the same as previously saved."));

          for (uint i = 0; i < state_variables.size(); ++i)
            *(state_variables[i]) = old_state_data[i];

          time_class.set_absolute_time(old_time_value);

          precice.markActionFulfilled(
            precice::constants::actionReadIterationCheckpoint());
        }
    }
  } // namespace PreciceDealCoupling
} // namespace adapter

#endif // COUPLING_FUNCTIONS_H
