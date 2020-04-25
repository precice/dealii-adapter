#ifndef COUPLING_FUNCTIONS_H
#define COUPLING_FUNCTIONS_H

#include <deal.II/base/exceptions.h>

#include <deal.II/fe/fe.h>

#include <precice/SolverInterface.hpp>

#include "parameter_handling.h"

namespace adapter
{
  using namespace dealii;

  namespace PreciceDealCoupling
  {
    template <int dim>
    class CouplingFunctions
    {
    public:
      CouplingFunctions(const Parameters::AllParameters &parameters);

      void
      initialize_precice();

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
      const unsigned int interface_mesh_id;
      const bool         enable_precice;
      const std::string  config_file;
      const std::string  participant;
      const std::string  mesh_name;
      const std::string  read_data_name;
      const std::string  write_data_name;

      unsigned int n_interface_nodes;
      unsigned int forces_data_id;
      unsigned int displacements_data_id;

      precice::SolverInterface precice;
    };


    template <int dim>
    CouplingFunctions<dim>::CouplingFunctions(
      const Parameters::AllParameters &parameters)
      : interface_mesh_id(parameters.interface_mesh_id)
      , enable_precice(parameters.enable_precice)
      , config_file(parameters.config_file)
      , participant(parameters.participant)
      , mesh_name(parameters.mesh_name)
      , read_data_name(parameters.read_data_name)
      , write_data_name(parameters.write_data_name)
      , precice(participant, config_file, /*proc_id*/ 0, /*n_procs*/ 1)
    {}


    template <int dim>
    void
    CouplingFunctions<dim>::initialize_precice()
    {}



    template <int dim>
    void
    CouplingFunctions<dim>::advance_precice()
    {}



    template <int dim>
    void
    CouplingFunctions<dim>::extract_relevant_displacements()
    {}



    template <int dim>
    void
    CouplingFunctions<dim>::save_old_state()
    {}



    template <int dim>
    void
    CouplingFunctions<dim>::reload_old_state()
    {}
  } // namespace PreciceDealCoupling
} // namespace adapter

#endif // COUPLING_FUNCTIONS_H
