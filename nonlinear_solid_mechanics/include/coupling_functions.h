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
    template <int dim, typename Number = double>
    class CouplingFunctions
    {
    public:
      CouplingFunctions(const Parameters::AllParameters &parameters);

      void
      initialize_precice(const DoFHandler<dim> &dof_handler);

      void
      advance_precice();

      void
      extract_relevant_displacements();

      void
      save_old_state();

      void
      reload_old_state();


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
      // TODO: Check, if this might be better a BlockVector
      std::vector<Number> precice_forces;
      std::vector<Number> precice_displacements;
    };


    template <int dim, typename Number>
    CouplingFunctions<dim, Number>::CouplingFunctions(
      const Parameters::AllParameters &parameters)
      : deal_boundary_id(parameters.interface_mesh_id)
      , enable_precice(parameters.enable_precice)
      , config_file(parameters.config_file)
      , participant(parameters.participant)
      , mesh_name(parameters.mesh_name)
      , read_data_name(parameters.read_data_name)
      , write_data_name(parameters.write_data_name)
      , precice(participant, config_file, /*proc_id*/ 0, /*n_procs*/ 1)
    {}


    template <int dim, typename Number>
    void
    CouplingFunctions<dim, Number>::initialize_precice(
      const DoFHandler<dim> &dof_handler)
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

      std::vector<Number> interface_nodes_positions(dim * n_interface_nodes);

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
          extract_relevant_displacements();

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



    template <int dim, typename Number>
    void
    CouplingFunctions<dim, Number>::advance_precice()
    {}



    template <int dim, typename Number>
    void
    CouplingFunctions<dim, Number>::extract_relevant_displacements()
    {}



    template <int dim, typename Number>
    void
    CouplingFunctions<dim, Number>::save_old_state()
    {}



    template <int dim, typename Number>
    void
    CouplingFunctions<dim, Number>::reload_old_state()
    {}
  } // namespace PreciceDealCoupling
} // namespace adapter

#endif // COUPLING_FUNCTIONS_H
