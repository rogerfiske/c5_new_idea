---
title: "Coherent state density matrix"
source: "https://chatgpt.com/c/68e7d121-e444-8325-8f73-2e6d30b9d2fc"
published:
clipdate: "2025-10-10T07:45:27-07:00"
"created-date": "2025-10-10T07:45:27-07:00"
description: "Overview of coherent state density matrices, their definitions, properties, and visualizations through Wigner and Q-functions."
---
>[!summary]- Summary


what is 'Coherent state density matrix' and what does it look like can it be visualized

---

Here’s an explanation of what a *coherent-state density matrix* is, what it “looks like” (in various representations), and how one can *visualize* it.

---

## What is a coherent state and its density matrix

1. **Coherent state**  
	In the context of a single bosonic mode (e.g. a mode of the electromagnetic field, or a quantum harmonic oscillator), a *coherent state* $|\alpha\rangle$ is defined as the (normalized) eigenstate of the annihilation operator $\hat a$ :
	$$
	\hat a\,|\alpha\rangle = \alpha\,|\alpha\rangle,
	$$
	where $\alpha$ is a complex number.  
	Equivalently, one can obtain $|\alpha\rangle$ by applying the displacement operator $D(\alpha) = \exp(\alpha \hat a^\dagger - \alpha^* \hat a)$ on the vacuum:
	$$
	|\alpha\rangle = D(\alpha)\,|0\rangle.
	$$
2. **Density matrix of a pure coherent state**  
	If the system is exactly in the coherent state $|\alpha\rangle$ (a *pure state*), then its density operator is
	$$
	\hat \rho = |\alpha\rangle\langle \alpha|.
	$$
	In whatever basis you choose, the matrix elements are
	$$
	\rho_{mn} = \langle m | \hat \rho | n \rangle = \langle m|\alpha\rangle \langle \alpha|n\rangle.
	$$
	If you choose the Fock (number) basis $\{|n\rangle\}$ , you can express $\langle n | \alpha \rangle = e^{-|\alpha|^2/2}\frac{\alpha^n}{\sqrt{n!}}$ . So
	$$
	\rho_{mn} = e^{-|\alpha|^2} \frac{\alpha^n (\alpha^*)^m}{\sqrt{n!\,m!}}.
	$$

Because $|\alpha\rangle\langle \alpha|$ is a *rank-1 projector*, the density matrix of a pure coherent state has maximal coherence—it’s not a classical mixture.

---

## What it “looks like” in phase space or quasi-probability pictures

Often, rather than just looking at matrix elements in the Fock basis, one uses **phase-space quasi-probability distributions** like the Wigner function or the Husimi Q-function to *visualize* quantum states. These are real (or sometimes quasi-probabilistic) functions on a 2D “phase space” (often the complex plane $\alpha$ or the real variables $(x,p)$ ).

- **Wigner function** $W(x,p)$ (or $W(\alpha)$ in complex form) is a quasi-probability distribution that can take negative values. For a coherent state $\rho = |\alpha_0\rangle\langle\alpha_0|$ , the Wigner function is a Gaussian centered at $\alpha_0$ :
	$$
	W(\alpha) \propto \exp\left[-2\,|\alpha - \alpha_0|^2\right].
	$$
	(Precisely, up to normalization)  
	In the $(x,p)$ coordinates, this is a minimum-uncertainty Gaussian “blob” in phase space.  
	(See e.g. phase space / Wigner function treatment in quantum mechanics texts)
- **Husimi Q-function**  
	Another useful function is
	$$
	Q(\alpha) = \frac{1}{\pi}\langle \alpha|\hat \rho|\alpha\rangle.
	$$
	For a coherent state $\rho = |\alpha_0\rangle\langle\alpha_0|$ , that gives
	$$
	Q(\alpha) = \frac{1}{\pi}\,|\langle \alpha|\alpha_0\rangle|^2 = \frac{1}{\pi} \exp(-|\alpha - \alpha_0|^2).
	$$
	That is also a Gaussian (but with a somewhat different width). In quantum optics software (e.g. QuTiP), the Husimi Q-function is often used as a visualization of states given their density matrices.

So the upshot is: in phase space, a coherent state’s density operator “looks like” a displaced Gaussian (a “blob” centered at $\alpha_0$ ).

---

## Visualization in density matrix (matrix elements) form

If you express $\hat \rho = |\alpha\rangle\langle\alpha|$ in a basis (say number states $|n\rangle$ ), then the matrix is full (i.e. nonzero for many off-diagonal elements). The off-diagonal elements are what encode coherence.

A stylized picture: for moderate cutoff in Fock space (say $n = 0,1,\dots,N$ ), the density matrix might look like

$$
\rho_{mn} = e^{-|\alpha|^2}\,\frac{\alpha^n (\alpha^*)^m}{\sqrt{n!\,m!}}.
$$

You could plot:

- Real part of $\rho_{mn}$
- Imaginary part of $\rho_{mn}$
- Magnitude $|\rho_{mn}|$

Often one sees that the diagonals (where $m = n$ ) form a Poisson distribution (for coherent states) and the off-diagonals decay for large $|m-n|$ . You could depict a heatmap or 3D surface.

---

## Mixed coherent-state ensembles / coherence decay

If instead the system is in a **mixture** (statistical ensemble) of coherent states, or undergoes decoherence, then the density matrix becomes more “diagonal” in some basis; the off-diagonals shrink. In extreme cases, it's a classical mixture of coherent states, with no interference terms.

For example, you could have

$$
\hat \rho = \int P(\alpha)\,|\alpha\rangle\langle\alpha|\,d^2\alpha
$$

with some probability distribution $P(\alpha)$ . Then the phase space picture (e.g. Wigner function) is a weighted sum (or convolution) of Gaussians, which could be broader or more smeared out.

---

## Summary

- A coherent state's density matrix (as a *pure state*) is $\hat \rho = |\alpha\rangle\langle\alpha|$ .
- In the Fock basis, $\rho_{mn} = e^{-|\alpha|^2} \frac{\alpha^n (\alpha^*)^m}{\sqrt{n!\,m!}}$ .
- One very useful way to *visualize* it is via the Wigner function or Q-function in phase space, which for a coherent state is a Gaussian “blob” centered at $\alpha$ .
- If there is mixing or decoherence, off-diagonal elements shrink and the visualization becomes more smeared or classical.

If you like, I can also produce a Python code to generate and plot the density matrix, Wigner & Q functions for a coherent state (or a mixture) so you can see it concretely. Do you want me to provide that?