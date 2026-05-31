# 🎯 PRB222 — Échantillonnage Préférentiel (Importance Sampling)

Projet numérique portant sur une méthode générique de réduction de variance par **échantillonnage préférentiel**, appliquée au pricing d'options vanilles et exotiques dans le modèle de Black-Scholes.

---

## 🎯 Objectif

Minimiser la variance de l'estimateur Monte Carlo en cherchant le shift optimal **θ\*** tel que :

```
E[g(X)] = E[g(X + θ) · exp(−θ·X − ½|θ|²)]
```

θ\* est estimé par un **algorithme de Newton** appliqué au gradient empirique de la variance.

---

## 📂 Structure

```
├── importance_sampling_commented.py   # Script principal commenté
```

---

## ⚙️ Modèle

Black-Scholes en dimension 3 :

```
dSi(t) = Si(t)(r dt + σi dWi(t))
```

avec W = (W1, W2, W3) browniens corrélés de matrice :

```
Γ = [[1,   ρ12, ρ13],
     [ρ12,  1,  ρ23],
     [ρ13, ρ23,  1 ]]
```

Simulation via décomposition de **Cholesky** : `W(T) = √T · L · X`, avec `X ~ N(0, I3)`.

---

## ⚙️ Paramètres

### Option vanille (Q6–Q10)

| Paramètre | Valeur |
|-----------|--------|
| `S1,0` | 1 |
| `σ` | 0.30 |
| `r` | 0.01 |
| `T` | 2 ans |
| `K` | 1 (puis 0.35 à 2.5) |

### Options exotiques (Q12–Q13)

| Paramètre | Valeur |
|-----------|--------|
| `Si,0` | 1 |
| `λi` | 1/3 |
| `K` | 1.25 |
| `T` | 1 an |
| `σ` | (0.25, 0.28, 0.30) |
| `ρij` | 0.5 |

> Les simulations utilisent uniquement des lois **Uniformes** via la méthode de Box-Muller.

---

## 📋 Questions traitées

| Q | Contenu |
|---|---------|
| Q1 | Solution de l'EDS Si(t) |
| Q2 | Preuve du changement de mesure (IS) |
| Q3 | Gradient de la variance par rapport à θ |
| Q4 | Identification du payoff (call européen) |
| Q5 | Prix analytique Black-Scholes |
| Q6 | Algorithme de Newton pour estimer θ\* |
| Q7 | Estimateur IS par Monte Carlo |
| Q8 | Convergence de la suite (θj) pour différents K |
| Q9 | Réduction de l'écart-type empirique au fil des itérations |
| Q10 | Comparaison MC standard vs IS pour K=2.5 (option OTM) |
| Q11 | Extension en dimension 3 pour options exotiques |
| Q12 | Call panier : IS vs MC standard |
| Q13 | Option Symphonie : IS vs MC standard |
| Q14 | Variable de contrôle via parité call-put panier |

---

## 🔑 Idée clé

Pour une option **OTM** (K grand), le MC standard gaspille presque tous ses tirages dans la zone de payoff nul. Le shift θ\* > 0 déplace les gaussiennes vers cette zone, réduisant drastiquement la variance.

---

## 🚀 Lancement

```bash
pip install numpy matplotlib
python importance_sampling_commented.py
```

---

## 📦 Dépendances

- `numpy`
- `matplotlib`
