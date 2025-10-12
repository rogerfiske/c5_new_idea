---
title: "Taming PINNs: How Hard Constraints Make Neural Networks Obey Physics"
source: "https://medium.com/data-science-collective/taming-pinns-how-hard-constraints-make-neural-networks-obey-physics-7d78e5b9f7a5"
published: 2025-09-30
clipdate: "2025-10-08T12:16:01-07:00"
"created-date": "2025-10-08T12:16:01-07:00"
description: "This article discusses how to improve the training of Physics-Informed Neural Networks (PINNs) by integrating hard constraints that enforce initial and boundary conditions directly into the network architecture, facilitating the solving of partial differential equations in various applications."
---
>[!summary]- Summary


[Sitemap](https://medium.com/sitemap/sitemap.xml)## [Data Science Collective](https://medium.com/data-science-collective?source=post_page---publication_nav-8993e01dcfd3-7d78e5b9f7a5---------------------------------------)

[![Data Science Collective](https://miro.medium.com/v2/resize:fill:76:76/1*0nV0Q-FBHj94Kggq00pG2Q.jpeg)](https://medium.com/data-science-collective?source=post_page---post_publication_sidebar-8993e01dcfd3-7d78e5b9f7a5---------------------------------------)

Advice, insights, and ideas from the Medium data science community

Some people claim to be efficient in multitasking. I am certainly not one of them. For me, having focus on the task is crucial to achieving anything. Despite being acutely aware of this handicap, I often find myself trying to pursue multiple goals at once. Can I blame a [salience network](https://en.wikipedia.org/wiki/Salience_network#:~:text=The%20salience%20network%20\(SN\)%2C%20also%20referred%20to,integration%20of%20sensory%2C%20emotional%2C%20and%20cognitive%20information.) inherited from evolution in the African savannah? One that’s inadequate for a life spent mostly staring at a computer screen? A question for the neurologists.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*v_Y-QL63T_TA9OCidXT-8Q.png)

Figure 1: A constrained neural network. Image by the author, with help from Copilot.

In the [previous post](https://medium.com/data-science-collective/physics-informed-neural-networks-to-solve-the-heat-diffusion-equation-5663d74e24ef), when we introduced the framework of Physics-Informed Neural Networks (PINNs), we defined a composite loss function. This loss function is the weighted sum of three components:

- The initial condition loss
- The boundary loss
- The differential equation residual loss

PINNs are notoriously difficult to train. One of the causes is the **multi-objective optimization** built into the composite loss function. For example, an easy path for the optimizer to lower the total loss could be to minimize the differential equation residual loss, while ignoring the initial and boundary losses. If the weights for the initial and boundary losses are low enough, their loss increases could be more than compensated by the decrease in the differential equation residual loss. Perhaps this problem can be temporarily resolved by adjusting the weights, but nothing guarantees that the optimal weight balance will remain constant throughout the training.

You and I are often faced with a list of tasks to do — goals to achieve. Unless you can do everything at once (I can’t), we need to choose what to concentrate on.

PINNs don’t have the luxury to prioritize one loss component over the others. For a solution to be useful, all loss components must be minimized simultaneously. While training a PINN with a composite loss function, the optimizer is in a situation where it has multiple goals, and it is not allowed to focus on one goal at a time. Shouldn’t we try to make it easier on the optimizer?

## Hard Constraints

To ease the training, it would be desirable to replace the composite loss function with a single loss term. We can achieve this by designing a **neural network architecture that enforces the initial and boundary conditions**. Since the initial and boundary conditions will be “pinned” (no pun intended) by design, we’ll call those “hard constraints”. This is in contrast with the original PINN formulation, where the initial and boundary conditions are favored through loss terms in the composite loss function. These constraints are “soft” in the sense that they are not guaranteed to be satisfied exactly.

Let’s see now how these constraints can be concretely integrated into the neural network architecture.

## Enforced Initial Condition

The initial condition is typically defined by a discrete series of measurements in the space domain. In the case of a 1D + t (one physical dimension plus time) domain, the initial condition would be defined as a series of values along the x-axis: (0 m, 100°C), (0.01 m, 105°C), (0.02 m, 106°C), … (0.3 m, 31°C), for example.

Let’s assume that we can interpolate the value of interest everywhere on the space domain through a function u₀(***x***). In our 1D + t example, u₀(*x*) would return the observation value if *x* belongs to the sample, or a smooth interpolation otherwise, for *x* ∈ \[0 m, 0.3 m\]. A cubic spline function would be perfectly suitable for this interpolation task. We can now incorporate the interpolation function of the initial condition u₀(***x***) in the output of our PINN architecture:

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*6CBbGSp_gx1Hypw41f9CNg.png)

Equation (1) shows one possible way to force u(***x***, 0) = u₀(***x***). β is a constant or a trainable time-decay parameter that will define the transition period between the initial condition regime and the heat diffusion regime. The arbitrary function v(***x***, t) is the output of a neural network, but with an additional twist, as we’ll see.

## Enforced Boundary Condition

If the function v(***x***, t) in equation (1) goes through zero at the boundary, and with the additional assumption that u₀(***x***) satisfies the Dirichlet boundary condition, then u(***x*** ᵩ, t) = u₀(***x*** ᵩ), where ***x*** ᵩ belongs to the boundary.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*KYtnHpuRXUnUBfLCWl_4FQ.png)

Ω(***x***) is a smooth function that outputs zero at the boundary, and a finite non-zero value inside the space domain. We can think of it as a soap bubble on a flat surface.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/0*5IGnWmUkA4NVqWdj)

Figure 2: A soap bubble. Photo by Ashes Sitoula on Unsplash

In the literature \[1\], Ω(*x*) is called the *approximate distance function*, as it behaves like a function that outputs the distance from the nearest boundary. I consider the *boundary bubble function* more visually descriptive, but that’s just a matter of taste.

For the 1D + t case, Ω(*x*) can take the form of a parabola that crosses the spatial limits x\_L and x\_R:

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*rqjjiDtVb0HDfXlsMWffBA.png)

z(x, t) is the output of an arbitrary function. In our case, it will be a neural network.

> The output u(x, t) is not directly the output of a trainable neural network z(x, t). Can we still optimize the neural network parameters through gradient descent?

Yes. The other functions involved in (3) (u₀(***x***), the time-decay function, and the boundary bubble function Ω(*x*)) are all derivable functions with respect to x and t. The deep learning framework will take its gradient into account when back-propagating the loss tensor. For this reason, the exact functional form of the initial condition interpolation, the time decay, and the boundary bubble functions is not critical, as the neural network will adapt to them.

With equation (3) as the filter for the output of a PINN, we are ready to search for the solution of a Partial Differential Equation (PDE).

## Experiment

You can find the code used to generate the results [here](https://github.com/sebastiengilbert73/tutorial_pinn_hard_constraints).

Let’s consider the problem of a thin metal bar whose initial temperature profile is known through a vector of measurements.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*WgPcK8aYotm_QKvSIKziQg.png)

Figure 3: The initial condition, defined by a series of temperature measurements. Image by the author.

The PDE that governs the evolution of the temperature profile over time is the 1D heat diffusion equation:

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*Xrf0WSxZH9rn1Xi-pRMW2Q.png)

We know the [analytical solution to this problem](https://medium.com/data-science/heat-diffusion-in-a-thin-metal-rod-faab655cb02c), so we’ll be able to compare the solution found by the PINN with the analytical solution.

The [PINN architecture](https://github.com/sebastiengilbert73/tutorial_pinn_hard_constraints/blob/main/architectures/constrained1d.py) will be centered around a ResNet with three layers of width 32.

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*6HP4fv62IpnIM-N5EP6nHw.png)

Figure 4: The neural network architecture. Image by the author.

The interpolation function for the initial condition will be a cubic spline, and the boundary bubble function will be a parabola:

In the training program [train.py](https://github.com/sebastiengilbert73/tutorial_pinn_hard_constraints/blob/main/training/train.py), we observe that the loss function is a single term, the differential equation residual loss. The initial condition and boundary losses are satisfied by design, so there is no need to include them in the loss function:

Figure 5 shows the loss evolution for a typical training run:

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*lsR-_xRJjh-MZ6PLFpqQxQ.png)

Figure 5: Loss as a function of epoch. Image by the author.

Animation 1 shows a comparison of the found approximate solution with the analytical solution, over a 10-second simulation:

![](https://miro.medium.com/v2/resize:fit:640/format:webp/1*4FI41KU8jcQb1uCKuiaNvQ.gif)

Animation 1: Comparison of the PINN prediction with the analytical solution. Image by the author.

As expected, the prediction at t=0 corresponds to the interpolation of the initial condition, and the temperature values at x=0 m and x=0.3 m are exactly the set boundary temperatures (i.e., Dirichlet boundary condition).

## Conclusion

We acknowledged the inherent difficulty associated with training a neural network in the context of the multi-objective optimization introduced in the original PINN formulation. We designed simple modifications that force the output of a neural network to match the interpolation function of the initial condition at t=0, and to match the boundary condition when the x-coordinate lies on the domain boundary.

We trained a PINN to solve the 1D+t heat diffusion problem. We obtained good visual confirmation that the PINN indeed learns to satisfy the PDE while being forced to satisfy the initial and boundary conditions.

This approach opens promising avenues for solving PDEs in domains where physical constraints are non-negotiable — from climate modeling to biomedical simulations.

Feel free to experiment with the [code](https://github.com/sebastiengilbert73/tutorial_pinn_hard_constraints). Please let me know if you think it could be useful to solve other PDE problems. Thank you for your time! 🙂

\[1\] Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks, N. Sukumar, Ankit Srivastava, [https://arxiv.org/abs/2104.08426](https://arxiv.org/abs/2104.08426)

[![Data Science Collective](https://miro.medium.com/v2/resize:fill:96:96/1*0nV0Q-FBHj94Kggq00pG2Q.jpeg)](https://medium.com/data-science-collective?source=post_page---post_publication_info--7d78e5b9f7a5---------------------------------------)

[![Data Science Collective](https://miro.medium.com/v2/resize:fill:128:128/1*0nV0Q-FBHj94Kggq00pG2Q.jpeg)](https://medium.com/data-science-collective?source=post_page---post_publication_info--7d78e5b9f7a5---------------------------------------)

[Last published 21 hours ago](https://medium.com/data-science-collective/why-rank-dense-rank-and-row-number-confuse-you-and-how-to-fix-that-18b4b6819699?source=post_page---post_publication_info--7d78e5b9f7a5---------------------------------------)

Advice, insights, and ideas from the Medium data science community

Advisory Engineer, Artificial Intelligence at IBM. [https://www.linkedin.com/in/s%C3%A9bastien-gilbert-69735219/](https://www.linkedin.com/in/s%C3%A9bastien-gilbert-69735219/)

## Responses (2)

Roger Fiske

What are your thoughts?  

```c
Great! Would love to see more, in particular also 2D and 3D problems!
```

```c
Hello, here is the link to the first part of my article series, which explains the holographic universe theory from a different perspective: https://medium.com/@nakikizildag/on-the-similar-worlds-of-pixels-and-atoms-part-1-f57e5b339daaI believe you will find it interesting.
```

## More from Sébastien Gilbert and Data Science Collective

## Recommended from Medium

[

See more recommendations

](https://medium.com/?source=post_page---read_next_recirc--7d78e5b9f7a5---------------------------------------)