// assembly over material configuration - finite strains, displacement dofs only
template <int dim>
void Solid<dim>::assemble_system_fstrain( /*input->*/ const unsigned int &newton_iteration, bool &GG_mode_requested /*output-> tangent_matrix, system_rhs, f_star*/ )
{
	std::cout << " Assemble System" << ((parameter.use_SRI) ? "SRI " : "FuI ") << std::flush;

	MaterialModel<dim> material ( parameter, pre_txt_directory, newton_iteration, current_load_step, unloading );

	//FEValues and FaceValues to compute quantities on quadrature points for our finite
	//element space including mapping from the real cell
	FEValues<dim> fe_values_ref (fe,//The used FiniteElement
								qf_cell,//The quadrature rule for the cell
								update_values| //UpdateFlag for shape function values
								update_gradients| //shape function gradients
								update_JxW_values|  //transformed quadrature weights multiplied with Jacobian of transformation
								update_quadrature_points); /// ToDo: remove this later on

	FEValues<dim> fe_values_ref_center (fe,//The used FiniteElement
										qf_center,//The quadrature rule for the cell
										update_values| //UpdateFlag for shape function values
										update_gradients| //shape function gradients
										update_JxW_values|  //transformed quadrature weights multiplied with Jacobian of transformation
										update_quadrature_points); /// ToDo: remove this later on
	
	FEFaceValues<dim> fe_face_values_ref (fe,
										qf_face, //The quadrature for face quadrature points
										update_values|
										update_normal_vectors| //compute normal vector for face
										update_JxW_values);

	//Quantities to store the local rhs and matrix contribution
	FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
	Vector<double> cell_rhs (dofs_per_cell);
	//Vector with the indicies (global) of the local dofs
	std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
	//Compute the current, total solution, i.e. starting value of
	//current load step and current solution_delta
	Vector<double> current_solution = get_total_solution(this->solution_delta);
	//Iterators to loop over all active cells
	typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
													endc = dof_handler_ref.end();

	for(;cell!=endc;++cell)
	{
		//Reset the local rhs and matrix for every cell
		cell_matrix=0.0;
		cell_rhs=0.0;
		//Reinit the FEValues instance for the current cell, i.e.
		//compute the values for the current cell
		fe_values_ref.reinit(cell);
		fe_values_ref_center.reinit(cell);

		//Vector to store the gradients of the solution at
		//n_q_points quadrature points

		std::vector<Tensor<2,dim> > solution_grads_u(n_q_points);
		fe_values_ref[u_fe].get_function_gradients(current_solution,solution_grads_u);
		
		
		// For SRI we append the solution_grads by the gradient at the reduced integrated quadrature points
	    // Afterwards the size of the \a solution_grad_u vector is (n_q_points+n_SRI_qp).
		 if ( parameter.use_SRI )
		 {
			std::vector< Tensor<2,dim> > solution_grads_u_RI(n_SRI_qp);
			fe_values_ref_center[u_fe].get_function_gradients(current_solution,solution_grads_u_RI);
			solution_grads_u.insert(solution_grads_u.end(), solution_grads_u_RI.begin(), solution_grads_u_RI.end());
		 }
		// k_rel is the relative qp-counter used for everything related to FEValues objects
		 unsigned int k_rel = 0;
		
		 
		//Write the global indices of the local dofs of the current cell
		cell->get_dof_indices(local_dof_indices);

		// Get the QPH for the QPs of the current cell, all stored in one vector of pointers
		 const std::vector<std::shared_ptr<PointHistory<dim> > > lqph = quadrature_point_history.get_data(cell);
		 
		//Loop over all quadrature points of the cell
		for( unsigned int k=0; k < (n_q_points+n_SRI_qp); ++k )
	// material assembly (qconv)
		{
			// Here we have to use solution_grads_u[k] not k_rel because we've put everything into this vector
			 Tensor<2,dim> DeformationGradient = (Tensor<2, dim>(StandardTensors::I<dim>()) + solution_grads_u[k]);
			
			// SRI (includes FuI)
			 FEValues<dim> *fe_values_part = nullptr;
			 SRI::init_fe_k( /*input->*/ fe_values_ref, fe_values_ref_center, k, n_q_points, /*output->*/ fe_values_part, k_rel );
			 SymmetricTensor<2,dim> stress_part;
			 SymmetricTensor<4,dim> Tangent_part;
			
			SymmetricTensor<2,dim> stress_S, Tangent_theta;
			SymmetricTensor<4,dim> Tangent;
			
			double stress_vM;

			Tensor<2,3> DefoGradient_3D = prepare_DefoGrad<dim> (DeformationGradient, parameter.type_2D, *fe_values_part, current_solution, k_rel);
			Tensor<2,3> F_inv = invert(DefoGradient_3D);
			double detF = StrainMeasures::get_DeterminantDefoGrad(DefoGradient_3D,parameter.get_going_mode,GG_mode_requested);
			
			timer.enter_subsection("MatMod");
			switch ( parameter.chosenMaterialModel )
			{
				case elastoplastic:
				{
					double alpha_n = lqph[k]->alpha_n;
					SymmetricTensor<2,3> eps_p_n = lqph[k]->eps_p_n;

					bool triggered_cell = false;

					material.elastoplasticity_fstrain( DefoGradient_3D, stress_S, Tangent, Tangent_theta, alpha_n, eps_p_n, parameter.hardening_type, GG_mode_requested, triggered_cell );

					SymmetricTensor<2,dim> Cauchy_stress = 1./detF * StrainMeasures::push_forward(stress_S,DefoGradient_3D,parameter.get_going_mode,GG_mode_requested);
					
					// Extract the desired parts from the stress in case we use SRI
					 if ( parameter.use_SRI )
					 {
						stress_part = SRI::part( DeformationGradient, stress_S, enums::enum_SRI_type(parameter.SRI_type), k, n_q_points );
						Tangent_part = SRI::part( DeformationGradient, stress_S, Tangent, enums::enum_SRI_type(parameter.SRI_type), k, n_q_points );
					 }
					 // For full integration, we just copy the full tensors into the *_part variables
					 else
					 {
						stress_part = stress_S;
						Tangent_part = Tangent;
					 }
						
					
					stress_vM = compute_vM_stress( Cauchy_stress );	// ToDo-assure: @q von Mises stress from S, Cauchy, Piola???
					lqph[k]->update_elpl_tmp(stress_S, eps_p_n, alpha_n, stress_vM );
				}
				break;
			}
			timer.leave_subsection("MatMod");

			// The quadrature weight for the current quadrature point
			 const double JxW = get_JxW<dim> (parameter.type_2D, *fe_values_part, k_rel);
			 lqph[k]->JxW = JxW;
			 
			// Loop over all dof's of the cell
			 for(unsigned int i=0; i<dofs_per_cell; ++i)
			 {
				//Assemble system_rhs contribution
				Tensor<2,dim> shape_gradient_wrt_ref_config_i = (*fe_values_part)[u_fe].gradient(i,k_rel);
				Tensor<2,dim> shape_gradient_wrt_spt_config_i = shape_gradient_wrt_ref_config_i * F_inv;
				SymmetricTensor<2,dim> sym_shape_gradient_wrt_spt_config_i = symmetrize(shape_gradient_wrt_spt_config_i);
				
				// \f$ f^\text{int} = \boldsymbol{F} \cdot \boldsymbol{S} \cdot \nabla_X \boldsymbol{N} \f$
				cell_rhs(i) -= ( symmetrize( transpose(DefoGradient_3D) * shape_gradient_wrt_ref_config_i ) * stress_part ) * JxW;

				for(unsigned int j=0; j<dofs_per_cell; ++j)
				{
					//Assemble tangent contribution
					Tensor<2,dim> shape_gradient_wrt_ref_config_j = (*fe_values_part)[u_fe].gradient(j,k_rel);
					Tensor<2,dim> shape_gradient_wrt_spt_config_j = shape_gradient_wrt_ref_config_j * F_inv;
					SymmetricTensor<2,dim> sym_shape_gradient_wrt_spt_config_j = symmetrize(shape_gradient_wrt_spt_config_j);
					
					cell_matrix(i,j) += (
											( symmetrize( transpose(shape_gradient_wrt_ref_config_i) * shape_gradient_wrt_ref_config_j ) * stress_part )
											+
											( symmetrize( transpose(DefoGradient_3D) * shape_gradient_wrt_ref_config_i )
											  * (
													Tangent_part
													* symmetrize( transpose(shape_gradient_wrt_ref_config_j) * DefoGradient_3D )
												)
											)
										)
										* JxW;
					
				}
			 }
		}
		
		// ToDo-optimize: move this into its own function and call it from all the assembly routines
		// For Neumann driver we have to add the tractions to the cell_rhs
		if ( parameter.driver == enums::Neumann )
		{
			assemble_Neumann(cell_rhs);
		} // only for Neumann driver

		// Here we assemble f_i, K_01, rho_i and kappa_i
		 if ( parameter.driver==enums::Dirichlet && ( active_solution_method==enums::ArcLength || parameter.hybrid_sol_method ) )
			 hybridSolver.assemble_for_AL_a_hat( cell_rhs, cell_matrix, current_solution, local_dof_indices, constraints_without_D );

		// Copy local to global:
		 assemble_tangent_and_rhs( cell_matrix, cell_rhs, local_dof_indices, newton_iteration  );
	}
}