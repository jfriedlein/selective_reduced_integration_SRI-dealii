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
	
	
	template <int dim>
	SymmetricTensor<4,dim> get_dKxS_dC( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const SymmetricTensor<4,dim> &Tangent )
	{
		SymmetricTensor<2,dim> RCG_C = symmetrize( transpose(F)*F );
		SymmetricTensor<2,dim> RCGinv_Cinv = invert(RCG_C);
		//SymmetricTensor<4,dim> K = 1./3. * outer_product( RCGinv_Cinv, RCG_C );
		return 2. * 1./3. * (
														outer_product( RCGinv_Cinv, ( RCG_C * 0.5 * Tangent + stress_S*identity_tensor<dim>()) )
														+ (RCG_C*stress_S) * Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F)
													  );
	}
}

#endif // NLKM_H
