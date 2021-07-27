
// Besides all your other includes, you now also include the SRI file,
// for instance as follows: \n
// Selective Reduced Integration (SRI)
#include "../selective_reduced_integration_SRI-dealii/SRI.h"

...

// Then we go on and extent the deal.II typical main class, here named \a Solid
template <int dim>
class Solid
{
	...
	
	// Later we need certain values, e.g. the shape gradients for the displacement
	// components, so we define the following extractor \a u_fe
	 const FEValuesExtractors::Vector u_fe;	// extractor for the dim displacement components

	// Here we delcare your typical QGauss quadrature rule \a qf_cell
	// for the integration over the cell. We add another QGauss rule named \a qf_cell_RI.
	// that will describe the reduced integration (RI). Furthermore, it is nice to add
	// the \a n_q_points_RI that stores the number of QPs for the reduced integrations
	// (for Q1SR elements that is always 1, but maybe you want to try Q2SR at some point)
	 const QGauss<dim>                qf_cell;
	 const QGauss<dim>                qf_cell_RI;
	 const unsigned int               n_q_points;
	 const unsigned int				 n_q_points_RI;

	// A flag to decide whether we want to use SRI (actually I always use a parameter
	// in the prm file to change element formulations)
	 const bool SRI_active = true;
	 
	// You can decide whether you want to do a volumetric-deviatoric split "vol_dev_split" to alleviate
	// volumetric locking or a shear-normal split "shear_normal_split" to counteract shear locking. For this
	// an enumerator was declared inside the function in the namespace \a enums
	 enums::enum_SRI_type SRI_type = enums::vol_dev_split;
	
	 ...
}


// Constructor
template <int dim>
Solid<dim>::Solid( ... )
:
...
// Here we choose a standard \a FE_Q (just so you know)
fe(	FE_Q<dim>(degree), dim), 	// displacement
u_fe(0),
// In the constructor for the above main class, we now also have to 
// initialise the new variables, which we do as follows
qf_cell( degree +1 )
qf_cell_RI( degree +1 -1 ),
n_q_points (qf_cell.size()),
n_q_points_RI ( SRI_active ? (qf_cell_RI.size()) : 0 ),
...
{
}


// Assemble one-field finite strain over material configuration \n
// We emphasis the relevant changes in the comment by a leading [SRI]
template <int dim>
void Solid<dim>::assemble_system_fstrain( /*output-> tangent_matrix, system_rhs*/ )
{
	// FEValues and FaceValues to compute quantities on quadrature points for our finite
	// element space including mapping from the real cell.
	// There are no requirements on this \a \fe_values_ref and \fe_vace_values_ref for SRI,
	// so you can keep your standard code for these.
	 FEValues<dim> fe_values_ref ( 	fe,//The used FiniteElement
									qf_cell,//The quadrature rule for the cell
									update_values| //UpdateFlag for shape function values
									update_gradients| //shape function gradients
									update_JxW_values|  //transformed quadrature weights multiplied with Jacobian of transformation
									update_quadrature_points );

	 FEFaceValues<dim> fe_face_values_ref ( fe,
											qf_face, //The quadrature for face quadrature points
											update_values|
											update_gradients|
											update_normal_vectors| //compute normal vector for face
											update_JxW_values|
											update_quadrature_points|
											update_jacobians );

	// [SRI] In addition for SRIs we define the reduced integration rule
	 FEValues<dim> fe_values_ref_RI (	fe,//The used FiniteElement
									 	qf_cell_RI,//The quadrature rule for the cell
										update_values | //UpdateFlag for shape function values
										update_gradients | //shape function gradients
										update_JxW_values );  //transformed quadrature weights multiplied with Jacobian of transformation


	// Quantities to store the local rhs and matrix contribution
	 FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	 Vector<double> cell_rhs (dofs_per_cell);
	// Vector with the indices (global) of the local dofs
	 std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	// Compute the current, total solution, i.e. starting value of
	// current load step and current solution_delta
	 Vector<double> current_solution = get_total_solution(this->solution_delta);

	// Tangents class and Tangent members
	 Tangent_groups_u<dim> Tangents;
	 SymmetricTensor<4,dim> Tangent;
	 SymmetricTensor<2,dim> Tangent_theta;

	// Iterators to loop over all active cells
	 typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
													endc = dof_handler_ref.end();

	for(;cell!=endc;++cell)
	{
		// Reset the local rhs and matrix for every cell
		 cell_matrix=0.0;
		 cell_rhs=0.0;
		// Reinit the FEValues instance for the current cell, i.e.
		// compute the values for the current cell
		 fe_values_ref.reinit(cell);

		// [SRI] Also reinit the RI rule for SRI
		 if ( SRI_active )
			fe_values_ref_RI.reinit(cell);

		// Vector to store the gradients of the solution at
		// n_q_points quadrature points
		 std::vector<Tensor<2,dim> > solution_grads_u(n_q_points);
		// Fill the previous vector using get_function_gradients
		 fe_values_ref[u_fe].get_function_gradients(current_solution,solution_grads_u);

		// [SRI] Prepare the solutions gradients for the RI rule. Herein we extend
		// the given vector of solution gradients by the additional gradients at the reduced
		// integration quadrature points.
		 if ( SRI_active )
			 SRI::prepare_solGrads( n_q_points_RI, fe_values_ref_RI, current_solution, solution_grads_u );

		// [SRI] k_rel is the relative qp-counter used for everything related to
		// FEValues objects and needed for SRI
		 unsigned int k_rel = 0;

		// Write the global indices of the local dofs of the current cell
		 cell->get_dof_indices(local_dof_indices);

		// Get the QPH for the QPs of the current cell, all stored in one vector of pointers
		 const std::vector< std::shared_ptr< PointHistory<dim> > > lqph = quadrature_point_history.get_data(cell);

		// [SRI] Loop over all quadrature points of the cell \n
		// For SRI the variable \a n_q_points_RI contains the number of QPs for reduced integration
		// over which we also loop. Else (FuI, RI, Fbar) this number is zero, so we don't loop
		// over these additional QPs
		for( unsigned int k=0; k < (n_q_points+n_q_points_RI); ++k )
		{
			// Compute the deformation gradient from the solution gradients. Note that the
			// vector of solutions gradients has been, in case of SRI, been extended by the
			// values at the RI QPs, so we have to use the full QP counter \a k.
			// @note If you want to work with 2D or even axisymmetry, you would need to
			// modify the deformation gradient now and cross some more t's (see https://github.com/jfriedlein/2D_axial-symmetry_plane-strain_dealii).
			//
			 Tensor<2,dim> DeformationGradient = (Tensor<2, dim>(StandardTensors::I<dim>()) + solution_grads_u[k]);

			// [SRI] SRI (does no harm to the remaining ELFORMs) \n
			// We initalise the \a fe_values_part pointer with the currently needed
			// FEValues quantity. For the first \a n_q_points quadrature points we
			// assign the standard \a fe_values_ref and in the remaining \a n_q_points_RI
			// we use the \a fe_values_ref_RI for the reduced integration
			 FEValues<dim> *fe_values_part = nullptr;
			 SRI::init_fe_k( /*input->*/ fe_values_ref, fe_values_ref_RI, k, n_q_points,
					 	 	 /*output->*/ fe_values_part, k_rel );
			// [SRI] We declare *_part variables for the stress and the tangent, that
			// eiter contain the deviatoric or volumetric parts
			 SymmetricTensor<2,dim> stress_part;
			 SymmetricTensor<4,dim> Tangent_part;

			// The material model will still return the full stress and tangents, because
			// we call it with the standard deformation gradient. Thus, we also create the
			// full stress and tangent as tensor. (There is certainly a more efficient way to do this.)
			 SymmetricTensor<2,dim> stress_S;
			 SymmetricTensor<4,dim> dS_dC;

			// Now you have to call your material model with the deformation gradient and
			// whatever else, to get your stress and tangent. The following is just a dummy
			// and far from my actual code.
			 elastoplasticity( DeformationGradient, lqph[k], stress_S, dS_dC );

			// [SRI] Extract the desired parts from the stress and tangent, in case we use SRI.
			// Depending on the \a SRI_type, we now do either a volumetric-deviatoric split or
			// a trivial normal-shear split.
			 if ( SRI_active )
			 {
				stress_part = SRI::part<dim>( DeformationGradient, stress_S, SRI_type, k, n_q_points );
				Tangent_part = SRI::part<dim>( DeformationGradient, stress_S, dS_dC, SRI_type, k, n_q_points );
			 }
			 // For full integration, we just copy the full tensors into the *_part variables
			 else
			 {
				stress_part = stress_S;
				Tangent_part = dS_dC;
			 }

			// [SRI] The quadrature weight for the current quadrature point.
			// That seems to be a daily task, but due to the variable FEValues
			// element you have to use the general *_part variable and the relative
			// QP counter \a k_rel
			 const double JxW = (*fe_values_part).JxW(k_rel);

			// Loop over all dof's of the cell
			 for(unsigned int i=0; i < dofs_per_cell; ++i)
			 {
				// [SRI] Assemble system_rhs contribution
				// Here we also access the FEValues, so we replace this by the
				// *_part counterpart together with \a k-rel
				 Tensor<2,dim> grad_X_N_u_i = (*fe_values_part)[u_fe].gradient(i,k_rel);

				// [SRI] The standard residual as shown on GitHub. Note that this is exactly the same
				// as without SRI, only that we replace the PK2 stress by the stress part \stress_part
				// that either contains the volumetric or deviatoric stress (or normal-shear)
				cell_rhs(i) -= ( symmetrize( transpose(DeformationGradient) * grad_X_N_u_i ) * stress_part ) * JxW;

				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					// [SRI] Assemble tangent contribution
					 Tensor<2,dim> grad_X_N_u_j = (*fe_values_part)[u_fe].gradient(j,k_rel);

					// The linearisation of the right Cauchy-Green tensor (You will recall this line
					// when you take a closer look at the F-bar formulation)
					 SymmetricTensor<2,dim> deltaRCG = 2. * symmetrize( transpose(grad_X_N_u_j) * DeformationGradient );

					// Again, the only difference to the standard integration, is that we replace
					// the stress and now also the tangent by the *_part counterparts. That's it!
					cell_matrix(i,j) += (
											symmetrize( transpose(grad_X_N_u_i) * grad_X_N_u_j ) * stress_part
											+
											symmetrize( transpose(DeformationGradient) * grad_X_N_u_i )
											* ( Tangent_part * deltaRCG )
										)
										* JxW;
				} // end for(j)
			 } // end for(i)
		} // end for(k)

		// Copy local to global:
		constraints.distribute_local_to_global(cell_matrix,cell_rhs,
								local_dof_indices,
								tangent_matrix,system_rhs,false);
	} // end for(cell)
} // end assemble_system

