#ifndef COMPRESSIBLE_NEO_HOOK_MATERIAL_H
#define COMPRESSIBLE_NEO_HOOK_MATERIAL_H

#include <deal.II/physics/elasticity/standard_tensors.h>

namespace Neo_Hook_Solid
{
  using namespace dealii;
  /**
   * The Material_Compressible_Neo_Hook_One_Field class is nearly the same as in
   * the original work. The density has been added as additional parameter,
   * which is needed for time dependent problems.
   */
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
} // namespace Neo_Hook_Solid
#endif // COMPRESSIBLE_NEO_HOOK_MATERIAL_H
