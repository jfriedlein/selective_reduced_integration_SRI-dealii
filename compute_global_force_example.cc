template <int dim>
void Solid<dim>::compute_global_force( )
{
   // Compute the force from the nodal components
	double force_via_nodes=0.;
	{
		Vector<double> force_vector_dofs(dof_handler_ref.n_dofs()); // ToDo-assure: should not be a problem for the two-field setup, then the damage dofs should remain zero and are never extracted

		// FEValues and FaceValues to compute quantities on quadrature points for our finite ...
		//	... element space including mapping from the real cell
		 FEValues<dim> fe_values_ref (fe,//The used FiniteElement
									qf_cell,//The quadrature rule for the cell
									update_values| //UpdateFlag for shape function values
									update_JxW_values | //transformed quadrature weights multiplied with Jacobian of transformation
									update_gradients ); // for the JxW-Values, ToDo: maybe use the data in the QPH

		 FEValues<dim> fe_values_ref_RI (fe,//The used FiniteElement
											qf_cell_RI,//The quadrature rule for the cell
											update_values| //UpdateFlag for shape function values
											update_gradients| //shape function gradients
											update_JxW_values);  //transformed quadrature weights multiplied with Jacobian of transformation

		// Quantities to store the local rhs and matrix contribution
		 Vector<double> cell_f (dofs_per_cell);

		// Vector with the indicies (global) of the local dofs
		 std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

		// Iterators to loop over all active cells
		 typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_ref.begin_active(),
														endc = dof_handler_ref.end();
		for(;cell!=endc;++cell)
		{
			// Reset the local rhs and matrix for every cell
			 cell_f=0.0;

			// Reinit the FEValues instance for the current cell, i.e.
			// compute the values for the current cell
			 fe_values_ref.reinit(cell);
			 if ( parameter.SRI_active )
				 fe_values_ref_RI.reinit(cell);

			// Write the global indices of the local dofs of the current cell
			 cell->get_dof_indices(local_dof_indices);

			// get the QPH for the QPs of the current cell
			 const std::vector<std::shared_ptr<PointHistory<dim> > > lqph = quadrature_point_history.get_data(cell);

			// Vector to store the gradients of the solution at
			// n_q_points quadrature points
			 std::vector<Tensor<2,dim> > solution_grads_u(n_q_points);

			// For fstrain, fill the previous vector using get_function_gradients
			 if ( parameter.active_fstrain )
			 {
				fe_values_ref[u_fe].get_function_gradients(solution_n,solution_grads_u);

				// For SRI we append the solution_grads by the gradient at the reduced integrated quadrature points
				// Afterwards the size of the \a solution_grad_u vector is (n_q_points+n_SRI_qp).
				 if ( parameter.SRI_active )
				 {
					std::vector< Tensor<2,dim> > solution_grads_u_RI(n_q_points_RI);
					fe_values_ref_RI[u_fe].get_function_gradients(solution_n,solution_grads_u_RI);
					solution_grads_u.insert(solution_grads_u.end(), solution_grads_u_RI.begin(), solution_grads_u_RI.end());
				 }
			 }

			// \a k_rel is the relative qp-counter used for everything related to FEValues objects
			 unsigned int k_rel = 0;

			// Loop over all quadrature points of the cell
			 for(unsigned int k=0; k < (n_q_points+n_q_points_RI);++k)
			 {
				// Compute the deformation gradient, located here to compute it only once for each QP
				// and only for fstrain.
				 Tensor<2,dim> DeformationGradient;
				 if ( parameter.active_fstrain )
				 {
					// Here we have to use solution_grads_u[k] not \a k_rel because we've put everything into this vector
					 DeformationGradient = (Tensor<2, dim>(StandardTensors::I<dim>()) + solution_grads_u[k]);

					// For axisymmetric problems the deformation gradient is modified by the prepare_DefoGrad function.
					// (Also for plane strain, but then we just use the out-of-plane strains as zero.)
					 //Tensor<2,3> DefoGradient_3D = prepare_DefoGrad<dim> (DeformationGradient, parameter.type_2D, fe_values_ref, solution_n, k);
				 }

				// SRI (includes FuI)
				 FEValues<dim> *fe_values_part = nullptr;
				 SRI::init_fe_k( /*input->*/ fe_values_ref, fe_values_ref_RI, k, n_q_points, /*output->*/ fe_values_part, k_rel );
				 SymmetricTensor<2,dim> stress_part;

				// Get the Tangent, stress and history at the QPs from the material model
				// @note Convention: \n
				// We decided to save the PK2 stress S in the PointHistory CONSISTENTLY for
				// all finite strain models. Hence, we expect that \a sigma_n1 contains the \a stress_S
				// or for small strains the sstrain stress measure T
				 SymmetricTensor<2,3> sigma_n1_3D = lqph[k]->stress_3D;
				 SymmetricTensor<2,dim> sigma_n1 = extract_dim<dim>( sigma_n1_3D );

				// Extract the desired parts from the stress in case we use SRI
				 if ( parameter.SRI_active )
					stress_part = SRI::part( DeformationGradient, sigma_n1, enums::enum_SRI_type(parameter.SRI_type), k, n_q_points );
				 // For full integration, we just copy the full tensors into the *_part variables
				 else
					stress_part = sigma_n1;

				// The quadrature weight for the current quadrature point
				 const double JxW = get_JxW<dim> (parameter.type_2D, fe_values_ref, k);

				// Loop over all dof's of the cell
				for(unsigned int i=0; i<dofs_per_cell; ++i)
				{
					Tensor<2,dim> grad_X_N_u_i = fe_values_ref[u_fe].gradient(i,k);

					// Small strain force
					 if ( parameter.active_fstrain==false )
					 {
						// @todo Check LaTeX syntax and possibility to use newcommands such asd'\bT for bold, and \gradX for ...
						// \f$ \cdot \nabla_X^\text{sym} N_u^i : T \f$
						 cell_f(i) += (symmetrize(grad_X_N_u_i) * sigma_n1 ) * JxW;
					 }
					// Finite strain forces
					 else if ( parameter.active_fstrain==true )
					 {
						// \a sigma_n1 contains the PK2 stress, hence the material formulation is used: \n
						// \f$ \[ F^T \cdot \nabla_X N_u^i \]^\text{sym} : S \f$
						 cell_f(i) += ( symmetrize( transpose(DeformationGradient) * grad_X_N_u_i ) * stress_part ) * JxW;
					 }
				}
			}

			// Distribute local to global for each cell
			if ( parameter.driver==enums::Neumann )
				constraints.distribute_local_to_global( cell_f,
														local_dof_indices,
														force_vector_dofs);
			else if ( parameter.driver==enums::Dirichlet )
				constraints_without_D.distribute_local_to_global( cell_f,
																  local_dof_indices,
																  force_vector_dofs);
		}
		// Extract the force components at the l-dofs in the end
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

	// write force and applied load into a text file
	 std::ofstream write_u_F(pre_txt_directory + "displacement_force.txt", std::ios::app);
	 write_u_F << lambda_i << " , " << 0 << " , " << force_via_nodes << " , " << y_disp << std::endl;
	 write_u_F.close();
}
