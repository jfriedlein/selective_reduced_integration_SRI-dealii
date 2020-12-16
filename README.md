# selective_reduced_integration_SRI-dealii

Implementation of helper functions to use selective reduced integration for the assembly of the RHS

For instance, you provide a stress and (stress-strain) tangent, and we extract the volumetric-deviatoric or normal-shear parts from it to integrate the deviatoric or normal part by full integration and the volumetric or shear part via a reduced integration scheme.

## When to use this
Selective reduced integration is one of the easiest ways to alleviate volumetric locking (by vol-dev split) or shear-locking (normal-shear split). Volumetric locking occurs for fully integrated linear elements when they approach incompressibility (e.g. due to large plastic strains), whereas shear locking for fully integrated linear elements occurs for instance in bending. In contrast to reduced integration, which causes hourglassing, SRI renders a proper stiffness matrix.

## Background
In 3D linear elements, for instance, possess 8 quadrature points (QPs). We use the quantities from these standard 8 QPs for the assembly of the deviatoric or normal part of the residual. Additionally, we loop over a 9th QP which is in the center of the Q1 element. The information from the 9th QP is used to assemble the volumetric or shear part.

@todo Add some graphics

Volumetric Locking
- https://www.brown.edu/Departments/Engineering/Courses/En2340/Notes/2017/L9.pdf
- https://dianafea.com/manuals/d943/MatLib/node256.html
- ...
Selective reduced integration
- https://dianafea.com/manuals/d944/ElmLib/node636.html
- ...

## Drawbacks
- add comparison between Q1R, Q2, Q1 and Q1SR regarding accuracy, computation time and efficiency
- add limitations (axisym., anisotropy) and possible extension to remove these
- currently very small convergence region (requires small load steps) - WHY?

## Extensions/Alternatives
- F-bar formulation also works great and will be added in the near future
- B-bar (simple said it's F-bar for small strains)
- Q1P0: already used in deal.II code gallery (@todo add some notes, additional FE fields, ...)
- The standard SRI does not work well for axisymmetry and anisotropy. However, there have been modifications proposed e.g. "GENERALIZATION OF SELECTIVE INTEGRATION PROCEDURES TO ANISOTROPIC AND NONLINEAR MEDIA" by Hughes
- Instead of the additional central QP (or set of reduced integration QPs), we can also use the averaged volumetric stress and tangents from the fully-integrated QPs. These averaged quantities would then replace the values at the center QP and thus eliminate the need for the additional QPs, the call to the material model and the storage of the additional QP-history. This more efficient version has not yet been tested, so information on the speed-up and accuracy cannot be given yet. On the other hand, also keep in mind the advantages of having the additional center QP. For linear elements, for instance, the center possesses superconvergent properties, so it gives us the best (or more provoking: the only proper) gradient thus stress. This precious data could be used for more accurate post-processing and mapping strategies.
- We could also use SRI for higher order elements (Q2SR,...), but why should we?

@todo Check effect of normal-shear split for shear locking in bending (benchmark)

## BUGS
- Currently the code does NOT run in DEBUG mode, because we abuse the reduced integration quadrature formula quite roughly. But it runs fine in RELEASE mode. You can easily remove this in the constructor for qf_center (will be repaired soon)


## DOCU missing!!

## Modifications/Extensions to existing code

0. Assembly routine & compute global force

Be aware of all the uses of *fe_values_part and k_rel! The often go along with each other, e.g. in the gradients.

1. Declarations

```
	const QGauss<dim>                qf_cell;
	const QGauss<dim>                qf_center;

	const QGauss<dim - 1>            qf_face;
	const unsigned int               n_q_points;
	const unsigned int				 n_SRI_qp;
	const unsigned int               n_q_points_f;
```

2. Constructor

```
qf_cell( degree +1 + ( (parameter.reducedIntegration) ? -1 : 0 ) ),	// switch to reduced integration
qf_center( (parameter.use_SRI) ? (degree +1 -1) : 0 ),
qf_face( degree +1 + ( (parameter.reducedIntegration) ? -1 : 0 ) ),
n_q_points (qf_cell.size() ),
n_SRI_qp ( qf_center.size() ),
n_q_points_f (qf_face.size()),
```

3. QP history

Quadrature point history needs to be prepared for total number of quadrature points (n_q_points+n_SRI_qp)


4. History update + general

update_history for each quadrature point (in general check every occurence of n_q_points

## Literature
- "Reduced and selective integration techniques in the finite element analysis of plates" by Hughes et al. https://doi.org/10.1016/0029-5493(78)90184-X
- "On volumetric locking of low-order solid and solid-shell elements for finite elastoviscoplastic deformations and selective reduced integration" by Doll et al. https://www.emerald.com/insight/content/doi/10.1108/02644400010355871/full/html

