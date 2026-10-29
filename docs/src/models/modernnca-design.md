# ModernNCA: design

## What it is

`ModernNCAConfig` implements ModernNCA from *Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later* (Ye, Yin, Zhan, Chao, ICLR 2025).

ModernNCA predicts a row by looking at similar rows in the training set. A small neural network, the encoder, turns every row into a vector. To predict for a new row, the model finds the training rows whose vectors are close to it and averages their targets, giving closer rows more weight.

The model here is the one in the paper. What is different is how the work is carried out. The paper's code compares a batch against the whole training set in one step, so the training set has to fit in GPU memory. This implementation goes through the training set in blocks, so memory depends on the block size and not on the size of the data.

The block method is not our invention. It comes from FlashAttention, which solved the same memory problem for attention in transformers. We apply its two main ideas to ModernNCA. The section [Where the ideas come from](#where-the-ideas-come-from) gives the credit in detail.

This page gives the math, shows where the memory goes, explains the block method, and states plainly what is exact and what is not.

## The model

Let $f_\theta$ be the encoder. Take a row $x$ to predict, and $N$ candidate rows $(x_j, y_j)$ from the training set.

Encode everything:

$$z = f_\theta(x), \quad z_j = f_\theta(x_j)$$

Measure how far each candidate is from the row:

$$d_j = \lVert z - z_j \rVert_2$$

Turn distances into weights that sum to one. Close rows get large weights:

$$\alpha_j = \frac{\exp(-d_j / T)}{\sum_{k=1}^{N} \exp(-d_k / T)}$$

Predict with the weighted average of the candidates' targets:

$$\hat{p} = \sum_{j=1}^{N} \alpha_j y_j$$

Here $T$ is a temperature. A small $T$ makes the model listen only to the very closest rows.

- **Regression:** $y_j$ is a number and $\hat{p}$ is the prediction.
- **Multiclass:** $y_j$ is a one-hot vector, so $\hat{p}$ is a probability for each class, and the loss is $-\log \hat{p}_y$.
- **Binary:** $y_j$ is 0 or 1, so $\hat{p}$ is the probability of the positive class. It is converted to a logit before the loss.

**The encoder** is a linear layer into $\mathbb{R}^d$, then `n_blocks` blocks of (BatchNorm, Linear, ReLU, Dropout, Linear), then a final BatchNorm. There are no residual connections. If numerical feature embeddings are used, they come first and are applied to every row in the same way.

**Which rows are candidates:**

- **Training.** The candidates are the current batch plus a random part of the other training rows. A row is never allowed to look at itself, otherwise it would just copy its own target. The random part is redrawn at every step, and its size is `sample_rate` times the number of other rows. The paper calls this Stochastic Neighborhood Sampling and reports that it makes training faster and the model better.
- **Evaluation and inference.** The candidates are the whole training set. Nothing is sampled and nothing is masked.

## The problem: memory

Done directly, the math above needs a table of distances with one row per batch item and one column per candidate. That table has $B \times N$ entries. In training it has to be kept for the backward pass too.

| Batch $B$ | Candidates $N$ | Full table (`Float32`) | One block, `corpus_chunk_size = 2048` |
|---|---|---|---|
| 1024 | 100 000 | 0.4 GB | 8 MB |
| 1024 | 1 000 000 | 4 GB | 8 MB |
| 1024 | 10 000 000 | 40 GB | 8 MB |

Several tables of that size exist at once: distances, exponentials, weights, and their gradients. So the direct method fails long before the last line. ModernNCA has to look at the whole training set for every prediction, which makes this the one thing keeping it away from large datasets.

## The fix: go through the candidates in blocks

### The weighted average can be built up one block at a time

Split the candidates into blocks $\mathcal{C}_1, \dots, \mathcal{C}_K$ and write $s_j = -d_j / T$ for the score of candidate $j$. For each block compute three things:

$$m_k = \max_{j \in \mathcal{C}_k} s_j$$

$$\ell_k = \sum_{j \in \mathcal{C}_k} e^{s_j - m_k}$$

$$u_k = \sum_{j \in \mathcal{C}_k} e^{s_j - m_k} y_j$$

Keep three running values, starting from $M_0 = -\infty$, $L_0 = 0$, $U_0 = 0$:

$$M_k = \max(M_{k-1}, m_k)$$

$$L_k = L_{k-1} e^{M_{k-1} - M_k} + \ell_k e^{m_k - M_k}$$

$$U_k = U_{k-1} e^{M_{k-1} - M_k} + u_k e^{m_k - M_k}$$

After the last block the prediction is

$$\hat{p} = \frac{U_K}{L_K} = \frac{\sum_j e^{s_j} y_j}{\sum_j e^{s_j}}$$

which is the same $\hat{p}$ as in the model section. Subtracting the running maximum $M$ keeps every exponent at zero or below, so nothing overflows, and $M$ cancels in the final ratio so it does not change the answer. This way of updating a softmax one piece at a time is called online softmax (Milakov and Gimelshein, 2018), and it is what FlashAttention uses to process attention in blocks. It works for any way of splitting the candidates, including a single block, where it becomes the usual stable softmax.

In the code, `_softmax_acc` creates $(M, L, U)$, `_softmax_fold` adds one block, and `_softmax_result` returns $U/L$ along with $\log L + M$. That second value is the log of the sum of all $e^{s_j}$, and the backward pass needs it. `corpus_chunk_size` is the block width. Only one table of $B$ rows by `corpus_chunk_size` columns exists at any time.

### Evaluation and inference

Encode the batch once, then loop over the training set in blocks and fold each one in. The encoded training set is also produced in blocks, by `_encode_all`, and stored on the `Corpus`. It is rebuilt once per training round, because the parameters only change between rounds. Encoding once per round, and not once per batch, saves repeated work and gives the same numbers.

### Training

Training has an extra difficulty. The candidates are encoded with the current parameters, so the gradient has to flow back through the encoder for every candidate row. If automatic differentiation were left to handle that, it would keep the activations of every block, and memory would grow with $N$ again.

So `_attend_train` has a hand-written backward pass. The forward keeps only $\hat{p}$ and, for each batch row, the value $\mathrm{lse} = \log \sum_j e^{s_j}$. With $g = \partial \mathcal{L} / \partial \hat p$, the gradient with respect to a score is

$$\frac{\partial \mathcal{L}}{\partial s_j} = \alpha_j \left( y_j^{\top} g - \hat{p}^{\top} g \right)$$

where the weight is recovered from the saved value as

$$\alpha_j = e^{s_j - \mathrm{lse}}$$

The term $\hat{p}^{\top} g$ is one number per batch row and is computed once. Everything else belongs to a single candidate. So the backward can visit the blocks one by one: encode the block again, recompute its scores, get $\alpha_j$ from $\mathrm{lse}$, form the gradient above, pass it through the distance, and pass the result through the encoder for that block only. The parameter gradients from all blocks are added up. Recomputing each block in the backward, from one saved number per row, is the second idea taken from FlashAttention. The layer state from before each block is saved in the forward pass, so BatchNorm and dropout behave the same way when the block is recomputed.

The batch's own block is treated the same way, with its diagonal set to $-\infty$. That gives a weight of exactly zero in both the forward and the backward, which is how a row is kept from looking at itself.

### Candidate sampling

At every step the loader needs `floor(sample_rate * (N - B))` different rows, picked at random from the rows outside the current batch. The batch is a window of consecutive positions in a shuffled order. `ModernNCALoader` draws that many numbers from `1:(N - B)` and adds $B$ to any number at or after the start of the window. This skips the batch without building a mask of length $N$. With `sample_rate = 1` all other rows are used.

## What is exact and what is not

**Exact.** For a given encoder, the block method gives the same prediction and the same gradient as the direct method. It evaluates the same formulas in a different order. The only difference is `Float32` rounding from adding numbers in a different order.

**Not identical: BatchNorm during training.** BatchNorm in training mode normalises with the mean and variance of the rows it is given. The paper's code passes all candidates through the encoder together, so it uses the statistics of the full candidate set. Here each block is normalised with its own statistics, and the running averages are updated once per block. This is the same kind of difference as training with a batch of 2048 rows and not a larger one. It cannot be avoided without holding every candidate in memory, which is the problem this design removes. The paper's code already normalises the batch and the candidates in two separate calls, so it does not use one set of statistics either.

**When the model is identical to the direct method:**

- during evaluation and inference, where BatchNorm uses its running averages
- with `n_blocks = 0`, where the encoder has no BatchNorm
- with `corpus_chunk_size` at least as large as the candidate set, where there is only one block

## How it is tested

`test/modernnca.jl` contains a separate, direct implementation of the formulas with no blocks.

- **Forward, evaluation path.** The block version is compared with the direct version for regression, binary and multiclass targets, on 5000 candidates with a block size that does not divide 5000. They agree to a relative tolerance of `1e-4`.
- **End to end.** A small regression model trains, the logged evaluation metric equals the metric recomputed from the model's own predictions, and grouped inference returns rows in order.

## Settings

| Paper | `ModernNCAConfig` | Default here | Default in the paper's code | Role |
|---|---|---|---|---|
| embedding width | `d_embedding` | 128 | 128 | size of the encoder output |
| number of blocks | `n_blocks` | 2 | 0 | depth after the linear layer |
| block hidden width | `d_block` | 256 | 512 | width inside each block |
| dropout | `dropout` | 0.1 | 0.1 | inside each block |
| temperature $T$ | `temperature` | 1.0 | 1 | how sharply close rows are favoured |
| sampling rate | `sample_rate` | 0.8 | 0.5 | share of the other rows used as candidates per step |
| not in the paper | `corpus_chunk_size` | 2048 | | candidates per block. Changes memory, not results |
| not in the paper | `eps` | 1e-8 | | numerical safety only |

The first six are the paper's hyperparameters. Three of the defaults here differ from the paper's default configuration, as shown. `corpus_chunk_size` trades memory for speed: a smaller value lowers peak memory, a larger value runs fewer and bigger matrix products.

## Numerical safeguards

These do not appear in the model's formulas. They keep the arithmetic well behaved.

- `eps` is added under the square root in the distance, so the distance can be differentiated when two vectors are equal.
- The temperature is never allowed below `eps`.
- For multiclass, $\hat{p}$ is kept at or above `1e-7` before the log. The paper's code adds `1e-7` for the same reason.
- For binary, $\hat{p}$ is kept between `1e-6` and `1 - 1e-6` before it is turned into a logit.

## Where the ideas come from

The memory-saving method on this page is an application of FlashAttention to ModernNCA. Two ideas are taken from it:

1. **Process attention in blocks and combine the pieces exactly.** FlashAttention splits the attention computation into blocks and uses the online softmax update to combine them, so the full table of scores is never built. Our forward pass does the same over blocks of training rows.
2. **Recompute in the backward, do not store.** FlashAttention keeps only the softmax normaliser from the forward pass and rebuilds each block of the attention table during the backward. Our backward does the same, keeping one log-sum-exp value per batch row.

The online softmax update itself is from Milakov and Gimelshein. Rabe and Staats showed earlier that attention can be computed in blocks with memory that does not grow with the sequence length.

What is specific to this work:

- The scores are negative Euclidean distances between learned vectors, not dot products, so the backward through the score is different.
- The candidates are not fixed inputs. They are produced by the encoder, so the backward also has to pass through the encoder for every block, with the layer state saved so that BatchNorm and dropout repeat exactly.
- The self-exclusion mask and the per-step candidate sampling of ModernNCA are handled inside the block method.

One difference in scope. FlashAttention is also a fast GPU kernel, written to reduce traffic between fast and slow GPU memory. This implementation uses only the algorithm, written with ordinary array operations. It saves memory. It does not claim FlashAttention's speed gains.

## References

- H.-J. Ye, H.-H. Yin, D.-C. Zhan, W.-L. Chao. *Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later.* ICLR 2025. Code: github.com/LAMDA-Tabular/TALENT
- T. Dao, D. Y. Fu, S. Ermon, A. Rudra, C. Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135
- T. Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* 2023. arXiv:2307.08691
- M. Milakov, N. Gimelshein. *Online normalizer calculation for softmax.* 2018. arXiv:1805.02867
- M. N. Rabe, C. Staats. *Self-attention Does Not Need O(n²) Memory.* 2021. arXiv:2112.05682

## Summary

ModernNCA predicts a row from the training rows nearest to it in a learned space. This implementation computes the paper's prediction with the paper's sampling and self-exclusion, but it never holds more than one batch-by-block table in memory. Following FlashAttention, the weighted average is built up block by block, and the training backward recomputes each block in place of storing it. Memory drops from $O(B \cdot N)$ to $O(B \cdot c)$, where $c$ is `corpus_chunk_size`. The attention and its gradient are exact. The one visible effect of working in blocks is that BatchNorm, during training, sees block-sized batches.
