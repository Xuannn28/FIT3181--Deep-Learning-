# Deep Learning Assignment

## Assignment 1: Neural Network Foundations & Architectures

Assignment 1 is divided into three parts, each focusing on a distinct phase of model development, theoretical property analysis, or text representation strategy.

### Part 1: Theory & Activation Function Analysis
* **Notebook:** `assignment01_Part1_solution.ipynb`
* **Architectural Overview:** Mathematical derivation, property analysis, manual forward/backward layer tracing, and custom bare-metal optimization implementations tested on real-world datasets.
* **Key Components:**
  * **Activation Function Derivatives & Continuity:** Evaluation of Exponential Linear Unit (ELU) and Gaussian Error Linear Unit (GELU) properties. This includes analytical derivations using calculus product rules, boundary condition analysis for continuity and differentiability at $x = 0$, and setting empirical output ranges.
  * **Feed-Forward Propagation Matrix Tracing:** Step-by-step mathematical verification of a multi-layer feed-forward network. Hidden representations, output logits, and softmax prediction probabilities are calculated sequentially alongside a generalized mathematical proof regarding the non-negativity of Cross-Entropy loss.
  * **Custom Optimization Framework:** Development of structural network blocks (`MyLinear` and `MyFFN`) utilizing low-level tensor operations. Optimization parameters are handled through manual parameter updates, completely bypassing PyTorch's native `torch.optim` module to replicate optimization dynamics directly.
* **Techniques Used:**
  * Exact derivative derivations to analyze continuity, differentiability, and boundary transitions.
  * Functional plotting using Matplotlib and PyTorch profiling to visualize curves and gradient behaviors.
  * Bare-metal implementation of optimization variants including Vanilla Stochastic Gradient Descent (SGD), SGD with Momentum, and AdaGrad via custom tracking state logic.
