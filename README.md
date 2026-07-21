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
  * **Global Average Pooling (GAP) Topology:** Integration of bare-metal global spatial reductions (`AdaptiveAvgPool2d((1, 1))`) preceding final linear classification projecting directly to target logit spaces to reduce parametric density.
  * **MixUp Data Augmentation:** Implementation of stochastic convex pixel-blending transformations drawing parameter ratios from continuous $\text{Beta}(\alpha, \alpha)$ distributions to blend paired image vectors and linearize categorical cross-entropy calculations.
  * **CutMix Data Augmentation:** Development of spatial-patch mask replacement routines combining discrete binary box generation (`rand_bbox`) with targeted batch permutations to mix area proportions and enforce holistic texture representations.
* **Techniques Used:**
  * High-dimensional data transformations incorporating color jitter distributions, random axis-flips, and channel pixel normalizations.
  * Structural residual mapping pipelines passing cross-strided dimensionality matches.
    
## Assignment 2: RNN, Sequential Modelling & Transformer 

* **Notebook:** `FIT3181_DeepLearning_Assignment2_Official[Main].ipynb`
* **Overview:** Hands-on implementation of sequential neural network architectures, covering low-level Recurrent Neural Network (RNN) dynamics, pre-trained word embeddings for classical text classification, and 1D Convolutional Neural Networks (TextCNN) for sequence feature extraction.

---

### Section 1: Fundamentals in RNNs
* **Architectural Overview:** Bare-metal implementation of a multi-timestep, vanilla Recurrent Neural Network (RNN) cell built completely from scratch using low-level PyTorch tensor operations.
* **Key Components:**
  * **Manual Parameter Initialization:** Explicit declaration of weight matrices ($U, W, V$) and bias vectors ($b, c$) as learnable parameters (`torch.nn.Parameter`).
  * **Recurrent Forward Loop:** Step-by-step state transition calculation across $T$ timesteps using the vanilla RNN equation:
    
    $$h_t = \tanh(X_t U + h_{t-1} W + b)$$
    
  * **Sequence Aggregation & Classification:** Stacking individual timestep hidden states into a unified sequence tensor $(B, L, H)$, extracting the final hidden state representation ($h_{T-1}$), and computing output classification logits.
  * **Manual Backpropagation & SGD:** Execution of `loss.backward()` to compute gradients across all recurrent parameter matrices followed by explicit Stochastic Gradient Descent (SGD) weight updates using `torch.no_grad()`.

---

### Section 2: Deep Learning for Sequential Data

#### Part 1: Using Word2Vec & GloVe for Text Classification
* **Technique:** Feature extraction from unstructured text using pre-trained word vector representations combined with decaying importance weighting mechanisms.
* **Key Components:**
  * **Pre-trained Embeddings:** Integration of the `glove-wiki-gigaword-100` model via `gensim` to map tokens to 100-dimensional dense vectors (handling out-of-vocabulary words via zero-vector fallback).
  * **Weighted Sentence Embeddings:** Generation of sentence-level representations using exponential position decay ($0.9^i$) normalized via Softmax to weight word vectors:
    
    $$\text{Vector}_{\text{sentence}} = \sum_{i=1}^{T} w_i \cdot v_i$$
    
  * **Pipeline Scaling & Modeling:** Feature normalization using `MinMaxScaler(feature_range=(-1, 1))` followed by multi-class classification using `LogisticRegression`, achieving high validation accuracy on question type classification.

#### Part 2: TextCNN for Sequence Modeling & Neural Embedding
* **Architectural Overview:** Implementation of the classic **TextCNN** architecture (*Kim, 2014*) using PyTorch layers for dynamic n-gram feature extraction.
* **Key Components:**
  * **Multi-Kernel 1D Convolutions:** Simultaneous processing of sequence channels using three distinct 1D convolutional filter sizes (kernel sizes $k \in \{3, 5, 7\}$) to capture unigram, bigram, and trigram context features.
  * **Multi-Feature Concatenation & Dense Output:** Channel-wise concatenation of pooled representations across all three convolution streams into a 2D tensor, routed through a fully connected layer for multi-class prediction.
  * **End-to-End Training:** Model optimization using the custom `BaseTrainer` framework with `Adam` optimizer and Cross-Entropy loss.

#### Part 3: Deep Multi-Layer Recurrent Architectures & Attention
* **Notebook:** `FIT3181_DeepLearning_Assignment2_Official[RNNs].ipynb`
* **Architectural Overview:** Modular design, comparative evaluation, and fine-tuning of multi-layer recurrent models across various cell types (`simple_rnn`, `lstm`, `gru`), sequence feature extraction pooling strategies (`mean`, `max`, `last_state`), pretrained embedding initialization modes, and an additive attention mechanism.
* **Key Components:**
  * **Modular `BaseRNN` Framework:** Dynamically constructs multi-layer recurrent networks by stacking cells (`nn.RNN`, `nn.LSTM`, `nn.GRU`) via `nn.ModuleList`. Supports three distinct sequence pooling strategies (`last_state`, `mean`, `max`) to extract fixed-size feature vectors for classification.
  * **Pretrained Embedding Strategy (`RNN`):** Extends `BaseRNN` with flexible embedding initialization using Gensim's `glove-wiki-gigaword-100`. Compares three training modes:
    * `scratch`: Randomly initialized embeddings trained from scratch.
    * `init-only`: Pretrained GloVe weights frozen during training (`freeze=True`).
    * `init-fine-tune`: Pretrained GloVe weights fine-tuned end-to-end (`freeze=False`).
  * **Additive Attention Mechanism (`MyAttention` & `AttentionRNN`):** Custom implementation of a Bahdanau-style attention layer placed on top of the last hidden layer. Computes alignment scores $s_t = V \tanh(U h_t)$, normalizes them via Softmax into attention weights $a_t$, and outputs a weighted sequence context vector $c = \sum_{t=1}^T a_t h_t$.
  * **Attention-Augmented Classification:** Concatenates the attention context vector with the aggregated sequence representation ($[c ; \text{rep}]$) before passing through the final linear classification head.
* **Techniques Used:**
  * Comparative grid-style training across cell types and pooling strategies using `BaseTrainer`.
  * Pretrained vector loading and disk caching (`embeddings/E.npy`) with out-of-vocabulary handling.

#### Part 4: Transformer-Based Models & Prefix Prompt-Tuning
* **Notebook:** `FIT3181_DeepLearning_Assignment2_Official[Transformers].ipynb`
* **Architectural Overview:** Full implementation of a custom Encoder-only Transformer classifier built from scratch, alongside a Parameter-Efficient Fine-Tuning (PEFT) framework implementing **Prefix Prompt-Tuning** on top of pre-trained BERT (`bert-base-uncased`).
* **Key Components:**
  * **Custom Transformer Classifier (`TransformerClassifier`):** Bare-metal implementation of stacked Transformer Encoder layers featuring multi-head self-attention (`MultiHeadAttention`), sinusoidal positional encoding (`PositionalEncoding`), and position-wise feed-forward networks (`PositionWiseFeedForward`). Aggregates hidden sequence representations via mean-pooling across timesteps before projecting to target class logits.
  * **HuggingFace Data Processing Pipeline:** Conversion of raw text data into structured PyTorch `Dataset` objects via `AutoTokenizer`, configured with sequence truncation and padding (`max_length=36`). Includes custom dataset splitting routines (`train_valid_test_split`) into train, validation, and test loaders.
  * **Prefix Prompt-Tuning Module (`PrefixTuningForClassification`):** Freezes all underlying parameters of `bert-base-uncased` while introducing a set of continuous, learnable prefix embeddings of shape `[prefix_length, hidden_size]`. Concatenates learnable prefix vectors onto input token embeddings along the sequence dimension.
  * **Dynamic Mask Extension & Pooling:** Extends the sequence attention mask to allow full self-attention over prefix tokens and performs temporal mean-pooling across the combined hidden state sequence to drive a linear classification head.
    
* **Techniques Used:**
  * Manual implementation of scaled dot-product multi-head attention and layer normalization.
  * Selective gradient freezing (`param.requires_grad = False`) for pre-trained Transformer weights.
  * Custom training harness (`FineTunedBaseTrainer`) configured with AdamW/Adam optimizers tailored specifically for prompt-tuning convergence over extended training schedules (100 epochs).
