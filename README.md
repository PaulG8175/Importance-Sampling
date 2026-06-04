# Importance Sampling for Exotic Option Pricing

> **Personal project**, done independently outside coursework.

Implementation of **Importance Sampling (IS)** — a generic variance reduction technique for Monte Carlo estimation — applied to vanilla and exotic option pricing in a multi-dimensional Black-Scholes model. Particularly effective for **deep out-of-the-money options** where standard MC wastes almost all samples in the zero-payoff region.

---

## Core idea

For any $g$ and $\theta \in \mathbb{R}^n$:

$$\mathbb{E}[g(X)] = \mathbb{E}\left[g(X+\theta)\*e^{-\theta \cdot X - \frac{1}{2}|\theta|^2}\right], \quad X \sim \mathcal{N}(0, I_n)$$

The estimator variance depends on $\theta$. The **optimal shift** $\theta^*$ minimises:

$$\theta^* = \underset{\theta \in \mathbb{R}^n}{\arg\min}\text{Var}\left(g(X+\theta)\*e^{-\theta \cdot X - \frac{1}{2}|\theta|^2}\right)$$

**Gradient of the variance:**
$$\nabla_\theta\,\text{Var}(f(\theta,X)) = \mathbb{E}\left[(\theta - X)\*g(X)^2\*e^{-\theta\cdot X + \frac{1}{2}|\theta|^2}\right]$$

$\theta^*$ is estimated via a **Newton algorithm** on the empirical gradient $u_N(\theta) = 0$.

---

## Newton algorithm

$$\begin{cases} \theta_0^N = 0 \\ u_N(\theta_j^N) + \nabla u_N(\theta_j^N)\cdot(\theta_{j+1}^N - \theta_j^N) = 0 \end{cases}$$

The same draws $(X_i)$ are reused across all iterations to avoid instability from sampling noise.

---

## Model

3-dimensional Black-Scholes with correlated Brownian motions:

$$dS_i(t) = S_i(t)\left(r\*dt + \sigma_i\*dW_i(t)\right)$$

$$
\Gamma =
\left(
\begin{array}{ccc}
1 & \rho_{12} & \rho_{13} \\
\rho_{12} & 1 & \rho_{23} \\
\rho_{13} & \rho_{23} & 1
\end{array}
\right)
$$

In the IS framework: $W(T) = \sqrt{T}\*L\*X$ with $\Gamma = LL^\top$ (Cholesky), $X \sim \mathcal{N}(0, I_3)$.

---

## Parameters

### Vanilla call (Q6–Q10)

| Parameter | Value |
|-----------|-------|
| `S₁,₀` | 1 |
| `σ` | 0.30 |
| `r` | 0.01 |
| `T` | 2 years |
| `K` | 1 → 2.5 |

### Exotic options (Q12–Q13)

| Parameter | Value |
|-----------|-------|
| `Sᵢ,₀` | 1 |
| `λᵢ` | 1/3 |
| `K` | 1.25 |
| `T` | 1 year |
| `σ` | (0.25, 0.28, 0.30) |
| `ρᵢⱼ` | 0.5 |

> All simulations use **Uniform random variables only** via Box-Muller transform.

---

## Key results

**Convergence of θ* vs K:**

- K small (ITM): θ* ≈ 0 — standard MC already efficient
- K large (deep OTM): θ* >> 0 — large shift needed to reach the payoff region; IS reduces variance by several orders of magnitude

**Options covered:**

### Basket Call
$$h(x_1, x_2, x_3) = \left(\lambda_1 x_1 + \lambda_2 x_2 + \lambda_3 x_3 - K\right)^+$$

### Symphony option
$$h(x_1,x_2,x_3) = \left[\frac{1}{2}\max(x_i-K) + \frac{1}{2}\min(x_i-K) - \text{median}(x_i-K)\right]^+$$

---

## Topics covered

| Q | Content |
|---|---------|
| Q2 | Proof of the IS change of measure |
| Q3 | Gradient of variance w.r.t. θ |
| Q5 | Black-Scholes analytical price |
| Q6 | Newton algorithm for θ* (1D) |
| Q7 | IS Monte Carlo estimator |
| Q8 | Convergence of (θⱼ) for K ∈ {0.35, …, 2.5} |
| Q9 | Standard deviation reduction across iterations |
| Q10 | IS vs standard MC for K=2.5 (deep OTM) |
| Q11 | Extension to dimension 3 |
| Q12 | Basket call: IS vs standard MC |
| Q13 | Symphony option: IS vs standard MC |
| Q14 | Control variate via put-call parity |

---

## Run

```bash
pip install numpy matplotlib
python importance_sampling_commented.py
```

## Dependencies

`numpy` · `matplotlib`
