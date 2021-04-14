#pragma once

#include <deal.II/numerics/data_postprocessor.h>

// Calculate Strains and pass displacement to the dof_handler
// TODO: Strains are symmetric. Currently all tensor components are processed
namespace Linear_Elasticity
{
  using namespace dealii;
  /**
   *  The Postprocessor class is used to handle the generation of output in the
   *  main class. Here, strains are computed and processed with the displacement
   *  intto the output file. Therefore, the deal.II DataPostprocessor interface
   *  is used.
   */
  template <int dim>
  class Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor();

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string>
    get_names() const override;

    virtual std::vector<
      DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual UpdateFlags
    get_needed_update_flags() const override;
  };


  template <int dim>
  Postprocessor<dim>::Postprocessor()
  {}


  template <int dim>
  void
  Postprocessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    const unsigned int n_evaluation_points = input_data.solution_values.size();

    Assert(n_evaluation_points == input_data.solution_gradients.size(),
           ExcInternalError());
    Assert(computed_quantities.size() == n_evaluation_points,
           ExcInternalError());
    Assert(input_data.solution_values[0].size() == dim, ExcInternalError());

    Assert(computed_quantities[0].size() == ((dim * dim) + dim),
           ExcInternalError());

    for (unsigned int p = 0; p < n_evaluation_points; ++p)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            computed_quantities[p][d] = input_data.solution_values[p][d];
            for (unsigned int e = 0; e < dim; ++e)
              computed_quantities[p]
                                 [Tensor<2, dim>::component_to_unrolled_index(
                                    TableIndices<2>(d, e)) +
                                  dim] =
                                   (input_data.solution_gradients[p][d][e] +
                                    input_data.solution_gradients[p][e][d]) /
                                   2;
          }
      }
  }



  template <int dim>
  std::vector<std::string>
  Postprocessor<dim>::get_names() const
  {
    std::vector<std::string> names;
    for (unsigned int d = 0; d < dim; ++d)
      names.emplace_back("displacement");

    static const char suffixes[] = {'x', 'y', 'z'};

    const std::string strain_name = "strain";
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        names.emplace_back(strain_name + '_' + suffixes[d] + suffixes[e]);

    return names;
  }

  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  Postprocessor<dim>::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation;
    for (unsigned int d = 0; d < dim; ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int d = 0; d < (dim * dim); ++d)
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }



  template <int dim>
  UpdateFlags
  Postprocessor<dim>::get_needed_update_flags() const
  {
    return update_values | update_gradients;
  }

} // namespace Linear_Elasticity
