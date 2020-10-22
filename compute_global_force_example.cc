template <int dim>
void Solid<dim>::compute_global_force( )
{
   // sum the force vectors over all the faces of the boundary
	double force_at_undeformed_face = 0.;

   // Compute the force from the nodal components
	double force_via_nodes=0.;
	{
		Vector<double> force_vector_dofs(dof_handler_ref.n_dofs()); // ToDo-assure: should not be a problem for the two-field setup, then the damage dofs should remain zero and are never extracted

		//FEValues and FaceValues to compute quantities on quadrature points for our finite ...
		//	... element space including mapping from the real cell
		FEValues<dim> fe_values_ref (fe,//The used FiniteElement
									qf_cell,//The quadrature rule for the cell
									update_values| //UpdateFlag for shape function values
									update_JxW_values | //transformed quadrature weights multiplied with Jacobian of transformation
									update_gradients |
									update_quadrature_points ); // for the JxW-Values, ToDo: maybe use the data in the QPH
		
		FEValues<dim> fe_values_ref_center (fe,//The used FiniteElement
											qf_center,//The quadrature rule for the cell
											update_values| //UpdateFlag for shape function values
											update_gradients| //shape function gradients
											update_JxW_values);  //transformed quadrature weights multiplied with Jacobian of transformation
		
		//Quantities to store the local rhs and matrix contribution
		 Vector<double> cell_f (dofs_per_cell);

		//Vector with the indicies (global) of the local dofs
		 std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		//Iterators to loop over all active cells
		typename DoFHandler<dim>::active_cell_iterator 	cell = dof_handler_ref.begin_active(),
														endc = dof_handler_ref.end();
		for(;cell!=endc;++cell)
		{
			// Reset the local rhs and matrix for every cell
			 cell_f=0.0;

			// Reinit the FEValues instance for the current cell, i.e.
			// compute the values for the current cell
			 fe_values_ref.reinit(cell);
			 
			if ( parameter.use_SRI )
				 fe_values_ref_center.reinit(cell);
			 
			//Write the global indicies of the local dofs of the current cell
			 cell->get_dof_indices(local_dof_indices);

			// get the QPH for the QPs of the current cell
			 const std::vector<std::shared_ptr<PointHistory<dim> > > lqph = quadrature_point_history.get_data(cell);

			//Vector to store the gradients of the solution at
			//n_q_points quadrature points
			std::vector<Tensor<2,dim> > solution_grads_u(n_q_points);

			// Fill the previous vector using get_function_gradients
			if ( parameter.active_fstrain )
			{
				fe_values_ref[u_fe].get_function_gradients(solution_n,solution_grads_u);
			
				// For SRI we append the solution_grads by the gradient at the reduced integrated quadrature points
				// Afterwards the size of the \a solution_grad_u vector is (n_q_points+n_SRI_qp).
				 if ( parameter.use_SRI )
				 {
					std::vector< Tensor<2,dim> > solution_grads_u_RI(n_SRI_qp);
					fe_values_ref_center[u_fe].get_function_gradients(solution_n,solution_grads_u_RI);
					solution_grads_u.insert(solution_grads_u.end(), solution_grads_u_RI.begin(), solution_grads_u_RI.end());
				 }
			 }
			
			// k_rel is the relative qp-counter used for everything related to FEValues objects
			 unsigned int k_rel = 0;
			
			// Loop over all quadrature points of the cell
			for(unsigned int k=0; k < (n_q_points+n_SRI_qp); ++k)
			{
				// Here we have to use solution_grads_u[k] not k_rel because we've put everything into this vector
				 Tensor<2,dim> DeformationGradient = (Tensor<2, dim>(StandardTensors::I<dim>()) + solution_grads_u[k]);
				
				// SRI (includes FuI)
				 FEValues<dim> *fe_values_part = nullptr;
				 SRI::init_fe_k( /*input->*/ fe_values_ref, fe_values_ref_center, k, n_q_points, /*output->*/ fe_values_part, k_rel );
				 SymmetricTensor<2,dim> stress_part;
				 
				// Get the last PK2 stress from the QP-history
				 SymmetricTensor<2,dim> stress_S = lqph[k]->stress;
				
				// Extract the desired parts from the stress in case we use SRI
				 if ( parameter.use_SRI )
					stress_part = SRI::part( DeformationGradient, stress_S, enums::enum_SRI_type(parameter.SRI_type), k, n_q_points );
				 // For full integration, we just copy the full tensors into the *_part variables
				 else
					stress_part = stress_S;
				
				//The quadrature weight for the current quadrature point
				 const double JxW = get_JxW<dim> (parameter.type_2D, *fe_values_part, k_rel);

				// Loop over all dof's of the cell
				for(unsigned int i=0; i<dofs_per_cell; ++i)
				{
					Tensor<2,dim> shape_gradient_wrt_ref_config_i = (*fe_values_part)[u_fe].gradient(i,k_rel);
					
					// RHS consists of the residuum of the last NR-iteration, which depends on the stress in the last iteration
					if ( parameter.chosenMaterialModel == enums::NeoHook )
					{
						AssertThrow(parameter.use_SRI==false, ExcMessage("Compute global force<< not implemented for SRI on NeoHook"));
						Tensor<2,dim> DeformationGradient =  (Tensor<2,dim> (StandardTensors::I<dim>()) + solution_grads_u[k]);
						NeoHookeanMaterial<dim> material(parameter.lame_mu, parameter.lame_lambda);
						SymmetricTensor<2,dim> stress_S = material.get_2ndPiolaKirchhoffStress(DeformationGradient);
						cell_f(i) += ( symmetrize( transpose(DeformationGradient) * shape_gradient_wrt_ref_config_i ) * stress_S ) * JxW;
					}
					else if ( parameter.active_fstrain )
					{
						Tensor<2,dim> DeformationGradient =  (Tensor<2,dim> (StandardTensors::I<dim>()) + solution_grads_u[k]);
						Tensor<2,dim> shape_gradient_wrt_spt_config_i_u = shape_gradient_wrt_ref_config_i * invert(DeformationGradient);
	
						if ( parameter.chosenMaterialModel == enums::global_damage )
						{
							// ToDo-assure: Do we need the phi-terms in the residuum for the computation of the force?
							cell_f(i) += ( symmetrize( transpose(DeformationGradient) * shape_gradient_wrt_ref_config_i ) * stress_part ) * JxW;
						}
						else
							cell_f(i) += ( symmetrize( transpose(DeformationGradient) * shape_gradient_wrt_ref_config_i ) * stress_part ) * JxW;
					}
					else
					{
						SymmetricTensor<2,dim> sym_shape_gradient_wrt_ref_config_i = symmetrize(shape_gradient_wrt_ref_config_i);
						cell_f(i) += (sym_shape_gradient_wrt_ref_config_i * stress_S ) * JxW;
					}
				}
			}
			if ( parameter.driver==enums::Neumann )
				constraints.distribute_local_to_global( cell_f,
														local_dof_indices,
														force_vector_dofs);
			else if ( parameter.driver==enums::Dirichlet )
				constraints_without_D.distribute_local_to_global( cell_f,
																  local_dof_indices,
																  force_vector_dofs);
		}

		// Extract the force components at the l-dofs
		// ToDo-optimize: Could be done nicer via a pointer set in the above "if"-"else if" referring to either f_hat or u_hat
		if ( parameter.driver==enums::Neumann )
		{
			for ( unsigned int i=0; i<hybridSolver.f_hat.size(); ++i )
				if ( hybridSolver.f_hat[i]!=0 )
					force_via_nodes += force_vector_dofs[i];
		}
		else if ( parameter.driver==enums::Dirichlet )
		{
			for ( unsigned int i=0; i<hybridSolver.u_hat.size(); ++i )
				if ( hybridSolver.u_hat[i]!=0 )
					force_via_nodes += force_vector_dofs[i];
		}
	}

	// Get the displacement of the evaluation point
	 double y_disp;
	 get_y_disp_atPoint(y_disp);

	// Save the global force and y-disp increments to be able to switch from AL back to NR
	 global_force_increment = force_via_nodes - global_force_old;
	 global_force_old = force_via_nodes;

	 y_disp_increment = y_disp - y_disp_old;
	 y_disp_old = y_disp;

	// write force and applied load into a text file
	 std::ofstream write_u_F(pre_txt_directory + "displacement_force.txt", std::ios::app);
	 write_u_F << lambda_i << " , " << force_at_undeformed_face << " , " << force_via_nodes << " , " << y_disp << std::endl;
	 write_u_F.close();
}