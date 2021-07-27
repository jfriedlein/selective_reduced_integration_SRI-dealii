#ifndef NLKM_H
#define NLKM_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>

using namespace dealii;

namespace NLKM
{
	/**
	 * Rename this fnc, because it can also be used for others tensors besides the stress
	 * @param F
	 * @param stress_S
	 * @return
	 */
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
	SymmetricTensor<4,dim> get_dKxS_dC( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const SymmetricTensor<4,dim> &dS_dC )
	{
		SymmetricTensor<2,dim> RCG_C = symmetrize( transpose(F)*F );
		SymmetricTensor<2,dim> RCGinv_Cinv = invert(RCG_C);
		//SymmetricTensor<4,dim> K = 1./3. * outer_product( RCGinv_Cinv, RCG_C );
		return 1./3. * (
						  outer_product( RCGinv_Cinv, ( RCG_C * dS_dC + stress_S*identity_tensor<dim>()) )
						  + (RCG_C*stress_S) * Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F)
					   );
	}
	// @todo-optimize Improve this routine, maybe use above results for dS_dC
	template <int dim>
	Tensor<4,dim> get_dKxS_dF( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const Tensor<4,dim> &dS_dF )
	{
		SymmetricTensor<2,dim> RCG_C = symmetrize( transpose(F)*F );
		SymmetricTensor<2,dim> RCGinv_Cinv = invert(RCG_C);
		Tensor<2,dim> F_inv = invert(F);
		Tensor<4,dim> d_Finv_dF = StandardTensors::dFinv_dF( F_inv, true );
		Tensor<4,dim> d_FT_inv_d_F = StandardTensors::dFTinv_dF( F );
		SymmetricTensor<4,dim> K = 1./3. * outer_product( RCGinv_Cinv, RCG_C );

 		return 1./3. * (
 							//( contract<1,0>( d_Finv_dF, transpose(F_inv) ) + F_inv * d_FT_inv_d_F ) * (RCG_C*stress_S)
 						    ( RCG_C*stress_S ) *
 							double_contract<2,0,3,1>( Tensor<4,dim>(Physics::Elasticity::StandardTensors<dim>::dC_inv_dC(F)),
 													  StandardTensors::dC_dF(F))
							+
							double_contract<2,0,3,1> (
														Tensor<4,dim> (outer_product( RCGinv_Cinv, stress_S )) ,
														StandardTensors::dC_dF(F)
													 )
					   )
 				+ double_contract<2,0,3,1>( Tensor<4,dim>(K), dS_dF );
	}
}

#endif // NLKM_H
