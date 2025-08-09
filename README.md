# Physics-Informed Neural Networks (PINNs) for Differential Equations

Physics-Informed Neural Networks (PINNs) are a class of deep learning models designed to solve differential equations by incorporating physical laws directly into the training process. Instead of relying solely on data, PINNs leverage the underlying mathematical structure of the problem—such as ordinary differential equations (ODEs) and partial differential equations (PDEs)—to guide the learning, incorporating initial and boundary conditions within the loss function. This makes them particularly effective for solving both forward problems (predicting the system’s behavior) and inverse problems (estimating unknown parameters from observed data).

---

## Mathematical Model

Suppose we want to solve a nonlinear partial diferrential equation:

$$
u_{t} + \mathcal{N}[u; \lambda] = 0, \qquad x \in \Omega, \quad t \in [0, T],
$$

where $u(x,t)$ denotes the solution, $\mathcal{N}[·; \lambda]$ is a nonlinear operator parameterized by $\lambda$ and $\Omega$ is a subset of $\mathbb{R}^D$.

The Initial Conditions:

$$
u(x, 0) = u_0(x)
$$

and Boundary Conditions:

$$
\mathcal{B}[u; \lambda] = g(x, t), \qquad \text{where } x \in \partial \Omega, \quad t \in [0, T],
$$

$\mathcal{B}[·;\lambda]$ corresponds to boundary operators such as Dirichlet, Neumann, Robin, or periodic conditions.

The PINN approximates the solution $u(x,t)$ of the differential equation using a neural network $\hat{u}(x, t; \theta)$, where $\theta$ represents the network parameters. 

The network is trained to satisfy, the governing differential equation (PDE), the Initial conditions (ICs), and the Boundary conditions (BCs).

By encoding these constraints into the loss function, the network learns a solution consistent with both the physics and any available data.

---

## Incorporation into the Loss Function

The neural network predicts $\hat{u}(x, t; \theta)$. Using automatic differentiation, the residual of the PDE is computed as:

$$
f(x, t) := \hat{u}_t + \mathcal{N}[\hat{u}; \lambda].
$$

### The PDE residual loss is then:

$$
\mathcal{B}[u; \lambda] = g(x, t), \qquad \text{where } x \in \partial \Omega, \quad t \in [0, T],
$$

$$
\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{PDE}} \left| f(x_i, t_i) \right|^2,
$$

where $\{ (x_i, t_i) \}_{i=1}^{N_{PDE}}$ are collocation points in $\Omega \times [0, T]$.

### The Initial Conditions loss is then:

$$
\mathcal{L}_{IC} = \frac{1}{N_{IC}} \sum_{i=1}^{N_{IC}} \left| \hat{u}(x_i, 0; \theta) - u_0(x_i) \right|^2,
$$

with $\{x_i\}_{i=1}^{N_{IC}} \subset \Omega$. 

### This Boundary Conditions the loss term:

$$
\mathcal{L}_{BC} = \frac{1}{N_{BC}} \sum_{i=1}^{N_{BC}} \left| \hat{u}(x_i, t_i; \theta) - g(x_i, t_i) \right|^2,
$$

where $\{ (x_i, t_i) \}_{i=1}^{N_{BC}} \subset \partial \Omega \times [0, T]$.

### Total Loss Function

The combined loss to minimize is:

$$
\mathcal{L}(\theta, w) = w_{PDE} \mathcal{L}_{PDE} + w_{IC} \mathcal{L}_{IC} + w_{BC} \mathcal{L}_{BC},
$$

where the $w's$ are weights balancing the terms.

---

## Handling Inverse Problems with Data

In inverse problems, unknown parameters or functions within the PDE are inferred by fitting the network not only to the physics but also to observed data points $ \{ (x_d^i, t_d^i, u_d^i) \}$. The data mismatch term is added to the loss:

$$
\mathcal{L}_{data} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left| \hat{u}(x_d^i, t_d^i; \theta) - u_d^i \right|^2.
$$

The total loss for inverse problems becomes:

$$
\mathcal{L}(\theta, w) = w_{PDE} \mathcal{L}_{PDE} + w_{IC} \mathcal{L}_{IC} + w_{BC} \mathcal{L}_{BC} + w_{data} \mathcal{L}_{data}.
$$

Training optimizes both the neural network parameters $\theta$ and the unknown physical parameters, allowing simultaneous solution and parameter identification.

---

## Repository Structure

This repository contains implementations of PINNs that solve ordinary differential equations (ODEs) and partial differential equations (PDEs) using both direct and inverse approaches.

- **GIFs/**: Visualizations and animations illustrating the solutions obtained by the time dependant PINNs.
- **ODEs/**: Examples of ODEs solved using PINNs.
- **PDEs/**: Examples of PDEs including the advection equation, heat equation, and shallow water equations.

## Getting Started

(Here you can add instructions for how to run the code, dependencies, etc.)

PyTorch installation
