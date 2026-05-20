# Spectra of Generalized Attention, from Linear Attention to Mamba

(Potential) Authors: Mengda Li, Zhaorui Dong, Runze You, Jianfeng Yao*

## 1. Research Objective

We aim to study the spectra of structured attention mechanisms beyond the standard self-normalized softmax attention. Motivated by the "Going Beyond the SSD Layer 2" discussion in the Mamba-2 blog, we focus on how different structured attention masks lead to different limiting spectral behavior.

The central question is:

> How does the choice of structured mask \(L\) change the spectral domain of
> \[
> M = L \circ QK^\top,
> \]
> and can these spectral differences explain fundamental behavioral differences between Mamba-like models and Transformers?

In particular, we would like to compare linear attention, Mamba-2/SSD-type attention, Toeplitz-type attention, decay masks, and Fourier-type structured attention under a unified random matrix framework. Ideally, this spectral perspective can clarify what is gained and what is lost when moving from quadratic softmax attention to sub-quadratic or linear-time structured attention.

## 2. Related Works

- Mamba-2 blog and paper:
  - [State Space Duality (Mamba-2) Part II - The Theory](https://goombalab.github.io/blog/2024/mamba2-part2-theory/#going-beyond-the-ssd-layer-2)
  - [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- Hayase, Collins, and Karakida:
  - [Gaussian Equivalence for Self-Attention: Asymptotic Spectral Analysis of Attention Matrix](https://arxiv.org/abs/2510.06685)
  - [AISTATS/OpenReview version](https://openreview.net/attachment?id=ABbVGvqaj3&name=pdf)
  - The Gaussian-equivalence and initialization analysis tricks, especially concentration of the softmax normalization term and linearization of attention, may be useful for our setting.
- Hydra / matrix mixer framework:
  - [Hydra Part I - Matrix Mixer Framework](https://goombalab.github.io/blog/2024/hydra-part1-matrix-mixer/)
  - This gives a broader comparison of structured matrix mixers, including low-rank, Toeplitz, Fourier, and semiseparable matrix classes.
- Discussion reference:
  - [Difference between Mamba and Transformer](https://www.zhihu.com/question/651838446)
  - This motivates the "no free lunch" intuition: reducing attention from \(O(\ell^2)\) to \(O(\ell)\) should impose structural restrictions, and these restrictions may be visible in the spectrum.

## 3. Model

![image](https://goombalab.github.io/assets/img/2024-05-31-mamba-2/sma.png)

Following the structured masked attention viewpoint in the Mamba-2 paper/blog, we model a family of structured attention matrices by
\[
M = L \circ QK^\top \in \mathbb{R}^{\ell \times \ell},
\]
where \(\ell > 0\) is the sequence length, \(d > 0\) is the hidden dimension, \(Q,K \in \mathbb{R}^{\ell \times d}\), and \(L \in \mathbb{R}^{\ell \times \ell}\) is a structured mask.

Different choices of \(L\) recover different models or candidate models:

- If \(L\) is the causal mask, i.e. lower triangular with ones on and below the diagonal and zeros above the diagonal, then \(M\) corresponds to causal linear attention.
- If \(L\) is a **1-semiseparable matrix**, then \(M\) corresponds to the SSD/Mamba-2 structured masked attention form.
\[
    L = \begin{bmatrix} 1 & \\ a_1 & 1 & \\ a_2a_1 & a_2 & 1 \\ \vdots & \vdots & \ddots & \ddots \\ a_{\mathtt{T}-1}\dots a_1 & a_{\mathtt{T}-1}\dots a_2 & \dots & a_{\mathtt{T}-1} & 1 \\ \end{bmatrix} .
\]
- If \(L\) is a lower triangular Toeplitz matrix, then \(M\) resembles a causal convolutional or Toeplitz structured attention.
- If \(L\) is a lower triangular decay mask, then \(M\) resembles retention/decay-based attention.
- If \(L\) is a Fourier or Fourier-related structured matrix, then \(M\) may connect to FNet/Fourier-type sequence mixing and frequency-domain positional structure.

### 3.1 Assumptions

In Transformers, \(Q = XW_Q\) and \(K = XW_K\), where \(W_Q,W_K\) are projection matrices. Therefore, the left singular vectors of \(Q\) and \(K\) are generally dependent through the shared input \(X\). However, as a tractable first step, we may assume that:

- the right singular vectors of \(Q\) are independent of those of \(K\).

Under this hypothesis, the product \(QK^\top\) becomes easier to analyze. A scale such as \(1/\ell\) or \(1/d\), depending on the asymptotic regime and normalization convention, should appear to compress non-essential singular values while preserving the dominant information subspace.

We plan to study both the initialization stage and the trained stage.

- At initialization, we can suppose that \(Q\) and \(K\) are independent i.i.d. Gaussian matrices. In the Hayase--Collins--Karakida setting, if \(X\) is orthogonal and independent of an i.i.d. Gaussian \(W_Q\), then by the left Haar invariance of the Ginibre ensemble, \(XW_Q \stackrel{\mathrm{law}}{=} W_Q\). Hence, it is reasonable to begin with \(Q,K\) as Ginibre ensembles.
- At the trained stage, we can suppose that \(Q\) and \(K\) are effectively low rank, motivated by the observation that the effective rank of softmax attention remains almost constant as the sequence length increases (reference: my paper on softmax attention).

### 3.2 Spectral Quantities of Interest

We may consider the following spectral objects:

- the empirical singular value distribution of \(M\);
- the top singular values and possible outliers;
- the spectral norm and its scaling in \((\ell,d)\);
- the bulk support and whether it differs from Marchenko--Pastur-type behavior;
- the effect of masking \(L\) on the rank, effective rank, and possible spectral collapse of \(QK^\top\).

## 4. Desired Theoretical Results

The guiding intuition is that there is no free lunch when replacing \(O(\ell^2)\) softmax attention by \(O(\ell)\) Mamba-like structured attention. A linear-time structured attention mechanism should have some theoretical restriction compared with full attention, but it should also be better than simpler alternatives such as plain linear attention; otherwise its empirical advantage would be hard to explain.

Possible theoretical results include:

- Derive the limiting singular value distribution of
  \[
  M = L \circ QK^\top
  \]
  for several structured masks \(L\), beginning with the causal mask and 1-semiseparable mask.
- Compare the spectral domains of causal linear attention, Mamba-2/SSD, Toeplitz attention, decay masks, and Fourier masks.
- Identify which spectral features distinguish Mamba-like structured attention from linear attention and from full softmax attention.
- Understand whether the 1-semiseparable mask produces a richer or more stable spectral distribution than the causal all-one mask.
- Relate spectral restrictions to model behavior, for example copying, retrieval, long-context reasoning, effective memory, and rank collapse.
- Explore whether Toeplitz and Fourier masks are not only mathematically interesting, but also useful intermediate points between convolutional/frequency mixers and attention-like mixers.

## 5. Possible Proof Strategy

As a first stage, we can treat \(L\) as deterministic and structured, and take \(Q,K\) as independent Gaussian matrices. Then \(QK^\top\) is a random low-rank or full-rank bilinear matrix depending on the relation between \(\ell\) and \(d\). The main difficulty is to understand the Hadamard product
\[
L \circ QK^\top.
\]

Potential approaches:

- Start from moment calculations for
  \[
  \frac{1}{\ell}\mathrm{Tr}\big((MM^\top)^k\big),
  \]
  keeping track of how the entries and paths are weighted by \(L\).
- Use Gaussian concentration to show that the empirical spectrum concentrates around a deterministic limit.
- For Toeplitz and Fourier \(L\), exploit diagonalization or approximate diagonalization by Fourier bases.
- For semiseparable \(L\), use its low-rank submatrix structure and recurrence representation to simplify the weighted path expansion.
- Compare the result with Gaussian equivalence for softmax attention, especially after removing or controlling the row-normalization effect.

## 6. Codex Suggestions / Extra Thinking

<span style="color:blue">[Codex suggestion] A clean first theorem could fix \(d/\ell \to \gamma \in (0,\infty)\), take \(Q,K\) independent with entries \(N(0,1/d)\), and compare the limiting singular value laws for two masks only: the causal mask and a simple constant-decay 1-semiseparable mask \(L_{ij}=a^{i-j}\mathbf{1}_{i\ge j}\). This gives a focused "linear attention vs. Mamba-like decay" baseline before moving to general input-dependent SSD masks.</span>

<span style="color:blue">[Codex suggestion] It may be useful to separate three levels of difficulty: deterministic \(L\), random but input-independent \(L\), and input-dependent \(L(X)\). The first level is likely publishable mathematically if the spectral law is new; the third level is closer to real Mamba/Transformer behavior but much harder.</span>

<span style="color:blue">[Codex suggestion] For trained models, the low-rank assumption on \(Q,K\) suggests studying a spiked version of the initialization model. This could connect the spectral bulk at initialization with outliers that appear after training or early feature learning.</span>

<span style="color:blue">[Codex suggestion] A numerical section should be planned early. Simulating the empirical singular spectra for causal, decay, semiseparable, Toeplitz, and Fourier masks may reveal the right conjectures before the proof is fully clear.</span>

## 7. Venue(s)

This timeline should be suitable for ICLR 2027, although the call for papers and LaTeX format are not yet open.

If we complete the analysis for all of these models,  the project may be substantial enough for JMLR.
