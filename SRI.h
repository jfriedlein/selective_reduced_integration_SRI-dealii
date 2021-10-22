#ifndef SRI_H
#define SRI_H

#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/elasticity/standard_tensors.h>
#include <iostream>

#include "NLKM.h"

using namespace dealii;

namespace enums
{
	/**
	* @note Why do we name it vol-dev and shear-normal? Even though shear-normal
	* sounds unusual it makes more sense here, because the first name always describes
	* the part that causes problems (locks) and thus needs to be integrated with a reduced order.
	*/
    enum enum_SRI_type
	 {
		vol_dev_split = 0,
		shear_normal_split = 1
	 };
}

/**
 * Namespace summarising function for selective reduced integration (SRI)
 * - vol-dev split: RI for volumetric part and FuI for deviatoric part
 * - shear-normal split: RI for shear part (shear-locking) and FuI for normal part
 * @todo-optimize Check use of equation (11) from [https://journals.pan.pl/Content/84498/PDF/04_paper.pdf] by  SUCHOCKI
 */
namespace SRI
{
	/**
	 * For SRI we append the solution_grads by the gradient at the reduced integrated quadrature points
	 * Afterwards the size of the \a solution_grad_u vector is (n_q_points+n_q_points_RI).
	 * @todo: avoid using the FEextractor maybe use fe_values[u_fe] as input
	 */
	template<int dim>
	void prepare_solGrads ( const unsigned int n_q_points_RI, FEValues<dim> &fe_values_ref_RI,
							const Vector<double> &current_solution, std::vector< Tensor<2,dim> > &solution_grads_u )
	 {
		// We evaluate the gradients also at the RI quadrature points ...
		 std::vector< Tensor<2,dim> > solution_grads_u_RI(n_q_points_RI);
		 fe_values_ref_RI[(FEValuesExtractors::Vector) 0].get_function_gradients(current_solution,solution_grads_u_RI);
		// ... and append the existing vector of gradients by the additional gradients
		 solution_grads_u.insert(solution_grads_u.end(), solution_grads_u_RI.begin(), solution_grads_u_RI.end());
	 }

	/**
	 * Prepare the solutions gradients and the scalar field for a two-field problem
	 * @param n_q_points_RI
	 * @param fe_values_ref_RI
	 * @param current_solution
	 * @param solution_grads_u
	 * @param phi_n1
	 * @param solution_grads_phi
	 */
	template<int dim>
	void prepare_sol2_solGrads ( const unsigned int n_q_points_RI, FEValues<dim> &fe_values_ref_RI,
								 const Vector<double> &current_solution, std::vector< Tensor<2,dim> > &solution_grads_u,
								 std::vector<double> &phi_n1, std::vector<Tensor<1,dim> > &solution_grads_phi )
	{
		std::vector< double > phi_n1_RI ( n_q_points_RI );
		fe_values_ref_RI[(FEValuesExtractors::Scalar) dim].get_function_values(current_solution, phi_n1_RI);
		phi_n1.insert(phi_n1.end(), phi_n1_RI.begin(), phi_n1_RI.end());

		std::vector< Tensor<2,dim> > solution_grads_u_RI(n_q_points_RI);
		fe_values_ref_RI[(FEValuesExtractors::Vector) 0].get_function_gradients(current_solution,solution_grads_u_RI);
		solution_grads_u.insert(solution_grads_u.end(), solution_grads_u_RI.begin(), solution_grads_u_RI.end());

		std::vector<Tensor<1,dim> > solution_grads_phi_RI(n_q_points_RI);
		fe_values_ref_RI[(FEValuesExtractors::Scalar) dim].get_function_gradients(current_solution, solution_grads_phi_RI);
		solution_grads_phi.insert(solution_grads_phi.end(), solution_grads_phi_RI.begin(), solution_grads_phi_RI.end());
	 }

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
	SymmetricTensor<2,3> get_shear_part( const SymmetricTensor<2,3> &SymTen )
	{
		SymmetricTensor<2,3> shear_part (SymTen);
		// Loop over the diagonal (normal) entries and set them to zero.
		// The remaining shear entries are unchanged from the above initialisation.
		for ( unsigned m=0; m<3 ; m++ )
			shear_part[m][m] = 0;
		return shear_part;
	}
	
	
	/**
	 * Return the Tangent that corresponds to a stress tensor with only normal stresses
	 * Templated by TangentTensorClass, so we can use this with symmetric and unsymmetric fourth order tensors
	 * @todo Add some asserts to ensure that the size of input tensor is okay
	 */
	template <class TangentTensorClass>
	TangentTensorClass get_shear_part( const TangentTensorClass &TangentTen )
	{
		TangentTensorClass shear_part (TangentTen);
		// Loop over the normal stress entries (first two indices [m][m] and
		// set all below entries [o][p] to zero for these related normal stresses
		for ( unsigned m=0; m<3 ; m++ )
			for ( unsigned o=0; o<3 ; o++ )
				for ( unsigned p=0; p<3 ; p++ )
					shear_part[m][m][o][p] = 0;
		return shear_part;
	}
	
	
	/**
	 * Return the second stress part (either volumetric or normal stress)
	 * @todo Catch not implemented SRI_type in a more general location
	 */
	SymmetricTensor<2,3> second_part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, enums::enum_SRI_type SRI_type )
	{
		if ( SRI_type == enums::vol_dev_split )
			return NLKM::get_stress_S_vol(F, stress_S);
		else if ( SRI_type == enums::shear_normal_split )
			return get_shear_part(stress_S);
		else
			AssertThrow(false, ExcMessage("SRI<< This kind of split is not implemented. Check the available options in SRI.h."));
	}
	
	
	/**
	 * Return the first stress part (either deviatoric or shear stress) as the
	 * difference between the total stress and the second part
	 */
	SymmetricTensor<2,3> first_part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, enums::enum_SRI_type SRI_type )
	{
			return stress_S - second_part( F, stress_S, SRI_type );
	}
	
	
//	template <class TangentTensorClass>
//	TangentTensorClass test ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, const TangentTensorClass &Tangent )
//	{
//
//		std::cout << "exectued" << std::endl;
//		return TangentTensorClass ();
//	}


	/**
	 * Return the second tangent part
	 */
	SymmetricTensor<4,3> second_part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, const SymmetricTensor<4,3> &Tangent, enums::enum_SRI_type SRI_type )
	{
		if ( SRI_type == enums::vol_dev_split )
			return NLKM::get_dKxS_dC( F, stress_S, Tangent);
		else if ( SRI_type == enums::shear_normal_split )
			return get_shear_part(Tangent);
		else
			AssertThrow(false, ExcMessage("SRI<< This kind of split is not implemented. Check the available options in SRI.h."));
	}
	Tensor<4,3> second_part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, const Tensor<4,3> &Tangent, enums::enum_SRI_type SRI_type )
	{
		if ( SRI_type == enums::vol_dev_split )
			return NLKM::get_dKxS_dF( F, stress_S, Tangent);
		else if ( SRI_type == enums::shear_normal_split )
			return get_shear_part(Tangent);
		else
			AssertThrow(false, ExcMessage("SRI<< This kind of split is not implemented. Check the available options in SRI.h."));
	}
	
	/**
	 * Return the first tangent part
	 */
	template <class TangentTensorClass>
	TangentTensorClass first_part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, const TangentTensorClass &Tangent, enums::enum_SRI_type SRI_type )
	{
			return Tangent - second_part( F, stress_S, Tangent, SRI_type );
	}
	
	
	/**
	 * Return the stress part defined by the input arguments \a k and \a n_q_points
	 */
	template <int dim>
	SymmetricTensor<2,dim> part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S,
								  enums::enum_SRI_type SRI_type, const unsigned int k, const unsigned int n_q_points )
	{
		if ( assemble_first_part(k,n_q_points) ) // first
			return extract_dim<dim>( first_part( F, stress_S, SRI_type ) );
		else
			return extract_dim<dim>( second_part( F, stress_S, SRI_type ) );
	}
	
	
	/**
	 * Return the tangent part defined by the input arguments \a k and \a n_q_points
	 */
	template <int dim>
	SymmetricTensor<4,dim> part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, const SymmetricTensor<4,3> &Tangent,
										enums::enum_SRI_type SRI_type, const unsigned int k, const unsigned int n_q_points )
	{
		if ( assemble_first_part(k,n_q_points) ) // first
			return extract_dim<dim>( first_part( F, stress_S, Tangent, SRI_type ) );
		else
			return extract_dim<dim>( second_part( F, stress_S, Tangent, SRI_type ) );
	}
	/**
	 * Because Tensor<4,3> is entered and Tensor<4,dim> is output, the standard template<class T> did not work, so I copied this single function
	 * @param F
	 * @param stress_S
	 * @param Tangent
	 * @param SRI_type
	 * @param k
	 * @param n_q_points
	 * @return
	 */
	template <int dim>
	Tensor<4,dim> part ( const Tensor<2,3> &F, const SymmetricTensor<2,3> &stress_S, const Tensor<4,3> &Tangent,
										enums::enum_SRI_type SRI_type, const unsigned int k, const unsigned int n_q_points )
	{
		if ( assemble_first_part(k,n_q_points) ) // first
			return extract_dim<dim>( first_part( F, stress_S, Tangent, SRI_type ) );
		else
			return extract_dim<dim>( second_part( F, stress_S, Tangent, SRI_type ) );
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
