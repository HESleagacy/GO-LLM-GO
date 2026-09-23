# 5. Training the parameters

This page follows source section 10. The architecture defines a differentiable function; training adjusts its parameters to predict observed tokens.

## 5.1 Maximum likelihood and negative log-likelihood

For a tokenized training sequence \((w_1,\ldots,w_n)\), the autoregressive log-likelihood is:

\[
\log P_\theta(w_{1:n})=\sum_{t=1}^{n}\log P_\theta(w_t\mid w_{1:t-1}).
\]

Across a corpus, maximum likelihood chooses parameters \(\theta\) that maximize this sum. Equivalently, training minimizes its negative, often averaged over tokens:

\[
\mathcal L(\theta)=-\frac{1}{N}\sum_{t=1}^{N}\log P_\theta(w_t\mid w_{<t}).
\]

For one observed target token, this is categorical cross-entropy against a one-hot target. It penalizes the model when it assigns low probability to the actual next token.

## 5.2 Gradient updates

Gradient descent updates parameters in the direction that reduces loss:

\[
\theta\leftarrow\theta-\eta\nabla_\theta\mathcal L(\theta),
\]

where \(\eta\) is the learning rate. Backpropagation applies the chain rule to compute gradients through the composed operations. In practice, modern training commonly uses minibatches and adaptive optimizers such as Adam variants; the paper's gradient-descent equation gives the basic idea.

## 5.3 What “stochastic” buys and what it does not promise

Sampling minibatches avoids computing each update over the entire corpus. Under suitable sampling assumptions, the minibatch gradient can be an unbiased estimate of the full-data gradient. But unbiased does not mean low variance, and stochastic optimization of a large non-convex network does not come with a blanket guarantee of convergence to a globally optimal solution. Learning-rate schedules, data quality, architecture, and numerical stability matter.

## 5.4 Training versus inference

During training, the model predicts many positions in parallel using a causal mask and compares each prediction with its known next token. During generation, it repeatedly predicts a distribution, chooses or samples a token, appends it to context, and runs again. Sampling policy (temperature, top-k, nucleus sampling) changes decoding behavior; it is not part of the core probability model equation.

The source also describes pretraining, supervised fine-tuning, and preference optimization/RLHF. These are common stages in many systems, not required steps in the mathematical definition of every language model.

## Go implementation boundary

The Go lab computes a forward pass only. It does not calculate gradients or update weights. That boundary is deliberate: implementing reliable autodiff and training would roughly multiply the code's scope and obscure the core equations. A later extension could add scalar reverse-mode autodiff, then a tiny trainable model; this package does not claim to have done that work.
