# selective_reduced_integration_SRI-dealii

Implementation of helper functions to use selective reduced integration (SRI) for the assembly routine in deal.II

## Requirements
For the code as outlined here to work out of the box you need a deal.II code (it is made for deal.II). But,  you are also very well welcomed to check out the basics and the algorithm, equations, implementation to learn something about SRI and carry it over to other platforms.

If you already know what SRI is (and like it), you can skip the following and go for the implementation further down the page.

## When to use this
Selective reduced integration is one of the easier ways to alleviate volumetric locking (by vol-dev split) or shear-locking (normal-shear split). Volumetric locking occurs for fully integrated linear elements when they approach incompressibility (e.g. due to large plastic strains), whereas shear locking for fully integrated linear elements occurs for instance even in bending. In contrast to reduced integration, which causes hourglassing, SRI renders a proper stiffness matrix.

Furthermore, it is efficient and based on my current experience faster than an F-bar formulation (code see [F-bar element formulation](https://github.com/jfriedlein/F-bar_element_formulation-dealii)).

## Background
In 3D for instance, linear elements possess 8 quadrature points (QPs). We use the quantities from these standard 8 QPs for the assembly of the deviatoric or normal part of the residual and thus also the tangent. Additionally, we loop over a 9th QP which is in the center of the Q1 element. The information from the 9th QP is used to assemble the volumetric or shear part. The center values are "locking-free" and the gradients there exhibit the highest accuracy (superconvergent property). Also take note that you could use SRI for higher-order elements as well, which is supported by the code in this repository.

@todo Add some graphics

Volumetric Locking
- https://www.brown.edu/Departments/Engineering/Courses/En2340/Notes/2017/L9.pdf
- https://dianafea.com/manuals/d943/MatLib/node256.html
- ...

Selective reduced integration
- https://dianafea.com/manuals/d944/ElmLib/node636.html
- ...

## Drawbacks/Extensions
- For the "selective" part in the name, we need the additional quadrature point as outlined int he background section. So, compared to linear elements we need an additional quadrature point, which means additonal computational effort. See the remarks in the "Considerations" section to find a silver lining.
- It is still a linear element, so the quality of the gradients remains "not so good" at every point besides the superconvergent center.
- This standard SRI variant cannot handle axisymmetric problems and anisotropy as stated by . However, there are modifications to handle this ("Generalization of selective integration procedures to anisotropic and non-linear media" by Hughes 1980). (Please let me know whether this actually works.)

## Benchmark
- add comparison between Q1R, Q2, Q1, Fbar and Q1SR regarding accuracy, computation time and efficiency (a teaser: SRI is 10x faster than Q2, and less than 30 % slower than Q1)

 @todo Check effect of normal-shear split for shear locking in bending (benchmark)

## Alternatives (against volumetric locking)
- F-bar formulation also works great and can be found here  [F-bar element formulation](https://github.com/jfriedlein/F-bar_element_formulation-dealii)
- B-bar (simply said: F-bar for small strains)
- Q1P0: already used in deal.II code gallery (@todo add some notes, additional FE fields, ...)

## Considerations
- Instead of the additional central QP (or set of reduced integration QPs), we can also use the averaged volumetric stress and tangents from the fully-integrated QPs. These averaged quantities would then replace the values at the center QP and thus eliminate the need for the additional QPs, the call to the material model and the storage of the additional QP-history. This more efficient version has not yet been tested, so information on the speed-up and accuracy cannot be given yet. On the other hand, also keep in mind the advantages of having the additional center QP. For linear elements, for instance, the center possesses superconvergent properties, so it gives us the best (or more provoking: the only proper) gradient thus stress. This precious data could be used for more accurate post-processing and mapping strategies.
- We could also use SRI for higher order elements (Q2SR,...), but why should we?

## Bugs
- Please tell me in the "Issues" section.

## Implementation/Usage: Modifying and extending your existing code
Back to business, how you can know use SRI in your own code. Again, it is all tailored for deal.II.

This repositories offers you the required helper functions to do this and the overall code extensions you need. In essence, you provide a stress and (stress-strain) tangent, and we extract the volumetric-deviatoric or normal-shear parts from it to integrate the deviatoric or normal part by full integration and the volumetric or shear part via a reduced integration scheme.

In case you have a one-field problem (displacements) your residual might be rather simple, such as

@todo Add equation for standard residual with deformation gradient, shape function gradient and PK2 stress

Then you can dig right into the following details on the implementation. If your residual is more involved, you might firstly take a piece of paper and take a closer look at your residual. SRI wants the deviatoric stress (and tangent) at the standard quadrature points for the residual and adds the missing volumetric stress onto the residual via the center QP. If your residual possesses additional terms that are stress dependent or independent, you know have to decide which parts to integrate fully, integrate by reduced integrations or integrate using all 9 QPs. At best you start with one of the options and see how it goes. This playing around with the residual is one of the things you don't need to do for an F-bar formulation.  


1. Declarations
In your main class (in deal.II it is named for instance "step3" [deal.II step3 tutorial](https://www.dealii.org/current/doxygen/deal.II/step_3.html) where you delcare your typical QGauss quadrature rule `qf_cell` for the integration over the cell, add another QGauss rule named `qf_cell_RI`. The latter will describe the reduced integration (RI). Furthermore, it is nice to add the `n_q_points_RI` that stores the number of QPs for the reduced integrations (for Q1SR elements that is always 1, but maybe you want to try Q2SR at some point)
```
	const QGauss<dim>                qf_cell;
	const QGauss<dim>                qf_cell_RI;

	const unsigned int               n_q_points;
	const unsigned int		 n_q_points_RI;
```

2. Constructor
In the constructor for the above main class, we now also have to  initialise the new variables, which we do as follows
```
...,
qf_cell( degree +1  ),
qf_cell_RI( degree +1 -1 ),
n_q_points (qf_cell.size()),
n_q_points_RI ( SRI_activeI) ? (qf_cell_RI.size()) : 0 ),
...
```
The standard `qf_cell` is initialised as usual, where `degree` denotes the polynomial order that is used for the element (1: linear element, 2: quadratic element, ...). The reduced integration uses one order less, so we init the `qf_cell_RI` with the order of `qf_cell` minus 1. The number of quadrature point for RI is set to its standard value `qf_cell_RI.size())` in case we want to use SRI (boolean flag `SRI_active` is true, else we set this number to zero.

3. Assembly routine
Now the difficult part begins, we extent the assembly routine to be able to use SRI (and still keep the option for standard integration).

@ todo add a mainpage with the assembly routine and the comments

Again, I want to remind you that you think about the correct use of `*fe_values_part` and `k_rel`.The often go along with each other, e.g. in the gradients, so at best check each occurence of the standard `fe_values_ref` and the original `k`. (I emphasise this, because I extended several assembly routines by SRI and that is my most common mistake, even though I created this code itself.)

If you also compute the force at your loading face, e.g. to plot a force-elongation plot, then you also need to extent this routine in case it is based on the residual.

3. QP history
The quadrature point history needs to be prepared for the total number of quadrature points (n_q_points+n_q_points_RI). Also make sure that you properly store all QP values, for instance by checking all occurences of the original `n_q_points`.

That should be it!

## Literature
- "Reduced and selective integration techniques in the finite element analysis of plates" by Hughes et al. https://doi.org/10.1016/0029-5493(78)90184-X
- "On volumetric locking of low-order solid and solid-shell elements for finite elastoviscoplastic deformations and selective reduced integration" by Doll et al. https://www.emerald.com/insight/content/doi/10.1108/02644400010355871/full/html

