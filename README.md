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

### Part 2: Deep Neural Networks (DNN)
* **Notebook:** `assignment01_Part2_solution.ipynb`
* **Architectural Overview:** Design, evaluation, and fine-tuning of Deep Neural Network architectures using the FashionMNIST image classification dataset, incorporating procedural pipelines, hyperparameter tuning pipelines, custom regularized objectives, and advanced neighborhood-ascent optimization concepts.
* **Key Components:**
  * **Dataset Visualization Pipeline:** Implementation of batch-unnormalization routines combined with automated sub-plotting layouts to render multi-class image batches alongside corresponding label annotations.
  * **Modular Trainer & Architecture Design:** Construction of a flexible multi-layer Feed-Forward neural network architecture decoupled from core training hooks, introducing explicit checkpoints to seamlessly handle inference evaluation, metric tracking, and weight persistence across epochs.
  * **Grid Search Hyperparameter Tuning Engine:** Formulation of a combinatorial search loop across discrete grids of layer dimensions and nonlinear activation functions (`ReLU`, `Sigmoid`, `Tanh`), backed by structured validation checkpoints to programmatically isolate the most optimal performance parameters.
* **Techniques Used:**
  * Dataset preparation steps using grayscale image normalization, dynamic split ratios, and automated tensor flattening transformations.
  * Structural model construction via linear matrix mappings and layered activation sequencing.
  * Dynamic grid-search parsing leveraging custom checkpoint and evaluation wrappers.
  * Bare-metal integration of dual forward-backward training loops handling stateful parameter perturbation ($\epsilon$-neighborhood parameter shifts) and weight restoration phases.

 ### Part 3: Convolutional Neural Networks (CNN) and Image Classification
* **Notebook:** `assignment01_Part3_solution.ipynb`
* **Architectural Overview:** Structural engineering, evaluation, and empirical comparison of custom deep convolutional frameworks designed for multi-class animal categorization utilizing an optimized, 20-class subset image repository.
* **Key Components:**
  * **Dynamic Modular CNN Engine (`YourCNN` & `YourBlock`):** Assembly of a customizable, block-based convolutional architecture parametrized to dynamically toggle features such as channel map dimension lists, dynamic dropout rates, custom batch normalization scaling hooks, and $1 \times 1$ downsampled structural residual skip connections.
  * **Global Average Pooling (GAP) Topology:** Integration of bare-metal global spatial reductions (`AdaptiveAvgPool2d((1, 1))`) preceding final linear classification projecting directly to target logit spaces to reduce parametric density and safeguard against dense overfitting points.
  * **Combinatorial Hyperparameter Sweeps:** Implementation of localized tuning frameworks evaluating block combinations against step-decay learning rates to map accuracy boundaries over testing targets.
  * **MixUp Data Augmentation:** Implementation of stochastic convex pixel-blending transformations drawing parameter ratios from continuous $\text{Beta}(\alpha, \alpha)$ distributions to blend paired image vectors and linearize categorical cross-entropy calculations.
  * **CutMix Data Augmentation:** Development of spatial-patch mask replacement routines combining discrete binary box generation (`rand_bbox`) with targeted batch permutations to mix area proportions and enforce holistic texture representations.
* **Techniques Used:**
  * High-dimensional data transformations incorporating color jitter distributions, random axis-flips, and channel pixel normalizations.
  * Structural residual mapping pipelines passing cross-strided dimensionality matches.
  * Multi-target objective loss formulations optimizing fractional continuous label matrices across both MixUp and CutMix training schedules.
