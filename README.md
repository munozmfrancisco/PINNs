# Physics-Informed Neural Networks (PINNs) for Differential Equations

Physics-Informed Neural Networks (PINNs) are a class of deep learning models designed to solve differential equations by incorporating physical laws directly into the training process. Instead of relying solely on data, PINNs leverage the underlying mathematical structure of the problem, including ordinary differential equations (ODEs) and partial differential equations (PDEs), to guide the learning, incorporating initial and boundary conditions within the loss function. This makes them particularly effective for solving both forward problems (predicting the system’s behavior) and inverse problems (estimating unknown parameters from observed data).

---

## Mathematical Model

Suppose we want to solve a nonlinear partial differential equation:

$$
u_{t} + \mathcal{N}[u; \lambda] = 0, \qquad \text{where } x \in \Omega, \quad t \in [0, T],
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

The network is trained to satisfy the governing differential equation (PDE), the Initial conditions (ICs), and the Boundary conditions (BCs).

By encoding these constraints into the loss function, the network learns a solution consistent with both the physics and any available data.

---

## Incorporation into the Loss Function

The neural network predicts $\hat{u}(x, t; \theta)$. Using automatic differentiation, the residual of the PDE is computed as:

$$
f(x, t) := \hat{u}_t + \mathcal{N}[\hat{u}; \lambda].
$$

### The PDE residual loss is then:

$$
L_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} \left| f(x^i, t^i) \right|^2,
$$

where $(x, t) \in \Omega \times [0, T]$.

### The Initial Conditions loss is then:

$$
L_{IC} = \frac{1}{N_{IC}} \sum_{i=1}^{N_{IC}} \left| \hat{u}(x^i, 0; \theta) - u_0(x^i) \right|^2,
$$

with $x \in \Omega$. 

### The Boundary Conditions loss term is:

$$
L_{BC} = \frac{1}{N_{BC}} \sum_{i=1}^{N_{BC}} \left| \hat{u}(x^i, t^i; \theta) - g(x^i, t^i) \right|^2,
$$

where $(x, t) \in \partial \Omega \times [0, T]$.

### Total Loss Function

The combined loss to minimize is:

$$
L(\theta, w) = w_{PDE} L_{PDE} + w_{IC} L_{IC} + w_{BC} L_{BC},
$$

where the $w$'s are weights balancing the terms.

---

## Inverse Problems

In inverse problems, unknown parameters or functions within the PDE are inferred by fitting the network not only to the physics but also to observed data points $\{ (x_d^i, t_d^i, u_d^i) \}$. The data mismatch term is added to the loss:

$$
L_{data} = \frac{1}{N_d} \sum_{i=1}^{N_d} \left| \hat{u}(x_d^i, t_d^i; \theta) - u_d^i \right|^2.
$$

### The total loss for inverse problems becomes:

$$
L(\theta, w) = w_{PDE} L_{PDE} + w_{IC} L_{IC} + w_{BC} L_{BC} + w_{data} L_{data}.
$$

Training optimizes both the neural network parameters $\theta$ and the unknown physical parameters, allowing simultaneous solution and parameter identification.

---

## Repository Structure

This repository contains Python implementations of Physics-Informed Neural Networks (PINNs) developed using **PyTorch**, specifically the `torch.nn` module.  

The `torch.nn` module provides the building blocks for defining and training neural networks in PyTorch.

It includes:
- **Layers** (e.g., `nn.Linear`, `nn.Conv2d`) for building network architectures.
- **Activation functions** (e.g., `nn.ReLU`, `nn.Tanh`).
- **Loss functions** (e.g., `nn.MSELoss`) for optimization.
- Tools for creating **custom models** by subclassing `nn.Module`.

This modular design makes it easy to define the PINN architecture as a sequence of fully connected layers with chosen activation functions, while allowing automatic differentiation to enforce the physics constraints.

In addition to PyTorch, the following libraries are used:
- **NumPy** — for efficient numerical operations and data handling.
- **Matplotlib** — for visualization of results, including animations.
- **datetime** — for handling timestamps in output files and logging.
- **math** — for basic mathematical operations outside of PyTorch tensors.
- **SciPy** — used in some examples to compute a high-accuracy *reference solution*, enabling comparison between the PINN prediction and the ground truth.

The PINNs solve both **ordinary differential equations (ODEs)** and **partial differential equations (PDEs)** in forward and inverse problems.

- **GIFs/** — Visualizations and animations illustrating the solutions obtained by the time-dependent PINNs.
- **ODEs/** — Examples of ODEs solved using PINNs.
- **PDEs/** — Examples of PDEs including the advection equation, heat equation, and shallow water equations.

<p align="center">
  <img src="GIFs/Forward/heat.gif" alt="PINN Solution for Heat Equation" width="500"/>
  <br>
  <em>Animation 1. Solution obtained by a PINN for the heat equation.</em>
</p>

As we can see in *Animation 1*, the PINN successfully solves the **heat equation** in a forward problem setting.
[Click here to view the Jupyter Notebook.](PDEs/Forward/heat.ipynb)

---

## Getting Started

### Dependencies Installation

You can install all required dependencies with:

```bash
pip install numpy==1.19.2 scipy==1.5.3 matplotlib==3.3.2 torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2
