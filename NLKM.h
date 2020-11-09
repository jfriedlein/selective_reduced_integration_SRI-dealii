#ifndef NLKM_H
#define NLKM_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>

using namespace dealii;

namespace NLKM
{
	template <int dim>
	SymmetricTensor<2,dim> get_stress_S_vol( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S )
	{
		SymmetricTensor<2,dim> RCG_C = symmetrize( transpose(F)*F );
		return 1./3. * ( stress_S * RCG_C ) * invert(RCG_C);
	}
	
	/**
	 * @note We don't use the Lagrangian tangent defined as \f$ C = 2 \frac{\partial S}{\partial C} \f$ here,
	 * but only the derivative \f$ \frac{\partial S}{\partial C} \f$. Hence, make sure you incorporate the factor of
	 * 2 in your assembly routine. If you want to use the Lagrangian tangent
	 * you would have to multiply \a Tangent by 0.5 and the overall returned result by 2.
	 * @param F
	 * @param stress_S
	 * @param Tangent
	 * @return
	 */
	template <int dim>
	SymmetricTensor<4,dim> get_dKxS_dC( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const SymmetricTensor<4,dim> &Tangent )
	{
		SymmetricTensor<2,dim> RCG_C = symmetrize( transpose(F)*F );
		SymmetricTensor<2,dim> RCGinv_Cinv = invert(RCG_C);
		//SymmetricTensor<4,dim> K = 1./3. * outer_product( RCGinv_Cinv, RCG_C );
		return 1./3. * (
						  outer_product( RCGinv_Cinv, ( RCG_C * Tangent + stress_S*identity_tensor<dim>()) )
						  + (RCG_C*stress_S) * Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F)
					   );
	}
}

#endif // NLKM_H
