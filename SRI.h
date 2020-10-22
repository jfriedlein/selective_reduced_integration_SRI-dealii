#ifndef SRI_H
#define SRI_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>

#include "NLKM.h"

using namespace dealii;

/**
 * Namespace summarising function for selective reduced integration (SRI)
 */
namespace SRI
{
	/**
	 * Return whether we currently shall assemble the first part (k<n_q_points)
	 * or whether we already did that and now will assemble the reduced integrated
	 * quadrature points.
	 */
	bool assemble_first_part( const unsigned int k, const unsigned int n_q_points )
	{
		if ( k<n_q_points ) // assemble first_part
			return true;
		else // assemble second part
			return false;
	}

	/**
	 * Return the normal stress as a tensor with zero shear stresses
	 */
	template <int dim>
	SymmetricTensor<2,dim> get_normal_part( const SymmetricTensor<2,dim> &SymTen )
	{
		SymmetricTensor<2,dim> normal_part;
		for ( unsigned m=0; m<dim ; m++ )
			normal_part[m][m] = SymTen[m][m];
		return normal_part;
	}
	
	
	/**
	 * Return the Tangent that corresponds to a stress tensor with only normal stresses
	 */
	template <int dim>
	SymmetricTensor<4,dim> get_normal_part( const SymmetricTensor<4,dim> &SymTen )
	{
		SymmetricTensor<4,dim> normal_part;
		for ( unsigned m=0; m<dim ; m++ )
			for ( unsigned o=0; o<dim ; o++ )
				for ( unsigned p=0; p<dim ; p++ )
					normal_part[m][m][o][p] = SymTen[m][m][o][p];
		return normal_part;
	}
	
	
	/**
	 * Return the second stress part (either volumetric or normal stress)
	 * @todo Catch not implemented SRI_type in a more general location
	 */
	template <int dim>
	SymmetricTensor<2,dim> second_part ( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, enums::enum_SRI_type SRI_type )
	{
		if ( SRI_type == enums::vol_dev )
			return NLKM::get_stress_S_vol(F, stress_S);
		else if ( SRI_type == enums::normal_shear )
			return get_normal_part(stress_S);
	}
	
	
	/**
	 * Return the first stress part (either deviatoric or shear stress) as the
	 * difference between the total stress and the second part
	 */
	template <int dim>
	SymmetricTensor<2,dim> first_part ( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, enums::enum_SRI_type SRI_type )
	{
			return stress_S - second_part( F, stress_S, SRI_type );
	}
	
	
	/**
	 * Return the second tangent part
	 */
	template <int dim>
	SymmetricTensor<4,dim> second_part ( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const SymmetricTensor<4,dim> &Tangent, enums::enum_SRI_type SRI_type )
	{
		if ( SRI_type == enums::vol_dev )
			return NLKM::get_dKxS_dC( F, stress_S, Tangent);
		else if ( SRI_type == enums::normal_shear )
			return get_normal_part(Tangent);
	}
	
	
	/**
	 * Return the first tangent part
	 */
	template <int dim>
	SymmetricTensor<4,dim> first_part ( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const SymmetricTensor<4,dim> &Tangent, enums::enum_SRI_type SRI_type )
	{
			return Tangent - second_part( F, stress_S, Tangent, SRI_type );
	}
	
	
	/**
	 * Return the stress part defined by the input arguments \a k and \a n_q_points
	 */
	template <int dim>
	SymmetricTensor<2,dim> part ( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S,
								  enums::enum_SRI_type SRI_type, const unsigned int k, const unsigned int n_q_points )
	{
		if ( assemble_first_part(k,n_q_points) ) // first
			return first_part( F, stress_S, SRI_type );
		else
			return second_part( F, stress_S, SRI_type );
	}
	
	
	/**
	 * Return the tangent part defined by the input arguments \a k and \a n_q_points
	 */
	template <int dim>
	SymmetricTensor<4,dim> part ( const Tensor<2,dim> &F, const SymmetricTensor<2,dim> &stress_S, const SymmetricTensor<4,dim> &Tangent,
										enums::enum_SRI_type SRI_type, const unsigned int k, const unsigned int n_q_points )
	{
		if ( assemble_first_part(k,n_q_points) ) // first
			return first_part( F, stress_S, Tangent, SRI_type );
		else
			return second_part( F, stress_S, Tangent, SRI_type );
	}
	

	/**
	 * The output \a fe_values_part is a pointer to a reference (?)
	 * @todo Find a better way, in case there is one.
	 */
	template<int dim>
	void init_fe_k ( FEValues<dim> &fe_values_first_part, FEValues<dim> &fe_values_second_part, const unsigned int k, const unsigned int n_q_points,
					 FEValues<dim> *(&fe_values_part), unsigned int &k_rel )
	{
		if ( assemble_first_part(k,n_q_points) ) // first part (FI)
		{
			fe_values_part = &fe_values_first_part;
			k_rel = k;
		}
		else // second part (RI)
		{
			fe_values_part = &fe_values_second_part;
			k_rel = k-n_q_points;
		}
	}
}

#endif // SRI_H
