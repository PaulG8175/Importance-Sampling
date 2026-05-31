import numpy as np
import matplotlib.pyplot as plt

# CDF de la loi normale centrée réduite (Abramowitz & Stegun)
def Abramowitz(x):
    if x<0:
        return 1-Abramowitz(-x)
    else:
        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        t = 1/(1+b0*x)
        return 1 - np.exp(-x**2 /2) * (b1*t+b2* t**2 +b3* t**3 +b4* t**4 +b5 * t**5)/np.sqrt(2*np.pi)


# Q1 : St = (Si,0 * exp((r-sig_i**2 /2)*t + sig_i * Wi(t))) 1<=i<=3

# Q2 : en partant de l'expression de droite et en faisant le changement
#  de variable u = x + teta, on retrouve l'expression de gauche

# Q3 : Var(f(teta,X)) = E(f(teta,X)**2) - E(f(teta,X))**2, on applique la dérivation sur les deux termes,
# on arrive facilement à la réponse

# Q4 : on a le payoff h = (ST - K)+, c'est un call européen

# Q5 : on utilise la preuve classique pour trouver un call européen, on a : 
# P = ST * phi(d1) - K*exp(-r*T)*phi(d2) avec phi focntion répartition N(0,1),
# d1 = (ln(S0/K) + (r + sig**2 /2)*T) / sig*sqrt(T) et d2 = d1 - sig*sqrt(T)

# ============================================================
# Q6) Algorithme de Newton à pas décroissant pour estimer theta*
# On cherche theta minimisant Var(f(theta,X))
# ============================================================

# Box-Muller 1D : génère N gaussiennes depuis des uniformes
def bx_mu(N):
    U,V = np.random.rand(N), np.random.rand(N)
    return np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V)

N=1000000

# u_N(theta) = gradient empirique de la variance (Q3) : vaut 0 à l'optimum
def Un(teta,g,X):
    return np.mean((teta-X)* g(X)**2 * np.exp(-teta*X + 0.5 * np.linalg.norm(teta)**2))

# Jacobien de u_N(theta) pour le schéma de Newton
def grad_Un(teta,g,X):
    return np.mean(g(X)**2 * np.exp(-teta*X + 0.5*teta**2)* (1 + (teta - X)**2))

# Itérations de Newton : theta_{j+1} = theta_j - u_N(theta_j) / grad_u_N(theta_j)
def teta_chap_N(g,X,niter):
    teta_suite = np.zeros(niter+1)
    for i in range(niter):
        teta_suite[i+1] = teta_suite[i] -Un(teta_suite[i],g,X)/grad_Un(teta_suite[i],g,X)
    return teta_suite

# ============================================================
# Q7) Estimateur MC avec échantillonnage préférentiel
# E[g(X)] = E[g(X+theta) * exp(-theta·X - 0.5*|theta|²)]
# ============================================================

# Même tirage X fixe pour toutes les itérations (évite l'effet tirage, Q6)
X = bx_mu(N)

sig=0.3
S1_0 = 1
r=0.01
T=2
K=1

# g : payoff actualisé du call européen exprimé en fonction de X ~ N(0,1)
def g(x):
    ST = S1_0 * np.exp((r-0.5* sig**2)*T + sig*np.sqrt(T)*x)
    return np.exp(-r*T) * np.maximum(ST - K, 0)

# Estimateur IS : moyenne de f(theta, Xi) = g(Xi+theta) * poids de Radon-Nikodym
def P_MC(teta,X):
    return np.sum(g(X+teta)*np.exp(-teta*X - 0.5*teta**2))/np.size(X)

# ============================================================
# Q8) Convergence de la suite (theta_j) pour plusieurs valeurs de K
# theta* > 0 pour les options OTM (K grand) : le shift déplace les tirages
# vers la zone de payoff non nul, réduisant la variance
# ============================================================

n=10  # nb d'itérations de Newton (convergence rapide en pratique)
K_list = np.array([0.35,0.54,0.7,1.24,1.6,2.5])
teta_chap_s = teta_chap_N(g,X,n)

plt.figure()
plt.plot(np.arange(n+1),teta_chap_s)
plt.title("suite teta_n en fonction de l'itération")
plt.xlabel("nombre itérations")
plt.ylabel("suite teta_n")
plt.show()


plt.figure()
for K in K_list:
    teta_chap_ = teta_chap_N(g,X,n)
    plt.plot(np.arange(n+1),teta_chap_, '*',label=f'K={K}')
plt.title("suite teta_n en fonction de l'itération")
plt.xlabel("nombre itérations")
plt.ylabel("suite teta_n")
plt.legend()
plt.show()


# ============================================================
# Q9) Evolution de l'écart-type empirique sqrt(Var/N) au fil des itérations
# L'IS réduit fortement la variance, surtout pour K grand (options OTM)
# ============================================================

# f(theta, x) : intégrande de l'estimateur IS
def f(teta,x):
    return g(x+teta)*np.exp(-teta*x - 0.5*teta**2)

# Variance empirique sans biais
def var_empi(Y):
    return np.sum((Y - np.mean(Y))**2) / (np.size(Y)-1)

plt.figure()
for K in K_list:
    teta_chap_suite = teta_chap_N(g,X,n)
    var_liste = np.zeros(n+1)
    for i,teta_chap in enumerate(teta_chap_suite):
        var_liste[i] = var_empi(f(teta_chap,X))
    plt.plot(np.arange(n+1),np.sqrt(var_liste/np.size(X)),label=f'K={K}')
plt.title("suite teta_n en fonction de l'itération")
plt.xlabel("nombre itérations")
plt.ylabel("suite teta_n")
plt.yscale("log")
plt.legend()
plt.show()


# ============================================================
# Q10) Comparaison MC standard vs IS pour K=2.5 (option très OTM)
# IS converge bien plus vite : variance réduite par le shift theta*
# ============================================================
K=2.5
n=10

N_list = [1000,1500,2000,3000,5000,8000,10000,15000,20000,30000,
50000,80000,100000,150000,200000,300000,500000,800000,1000000,
1500000,2000000,3000000,4000000,5000000]

P_MC_zero = np.zeros(np.size(N_list))
P_MC_chap = np.zeros(np.size(N_list))

for i,N in enumerate(N_list):
    X = bx_mu(N)
    teta_chap = teta_chap_N(g,X,n)[-1]  # dernière itération de Newton
    P_MC_zero[i] = P_MC(0,X)
    P_MC_chap[i] = P_MC(teta_chap,X)

# Prix théorique Black-Scholes (Q5)
d1 = (np.log(S1_0/K) + (r+ 0.5*sig**2)*T)/ (sig*np.sqrt(T))
d2 = d1 - sig*np.sqrt(T)
P_theo = S1_0*Abramowitz(d1) - K*np.exp(-r*T)*Abramowitz(d2)
print(f"prix théorique P = {P_theo}")


plt.figure()
plt.title("Convergence des deux méthodes")
plt.plot(N_list,P_MC_zero, label = "MC teta=0")
plt.plot(N_list,P_MC_chap, label="MC teta= teta_chapeau")
plt.axhline(y=P_theo, color = "black", label="valeur théorique")
plt.xscale("log")
plt.yscale("log")
plt.show()


plt.figure()
plt.plot(N_list, np.abs(P_MC_zero-P_theo),label="MC teta=0")
plt.plot(N_list, np.abs(P_MC_chap - P_theo), label="MC teta=teta_chapeau")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()


# ============================================================
# Q11) Extension en dimension 3 (n=3) pour options exotiques
# g(X) = exp(-rT)*h(S1(T), S2(T), S3(T)) avec X ~ N(0,I3)
# et W(T) = sqrt(T)*L*X où Gamma = L*L^T (Cholesky)
# ============================================================

# Q11 : on a St = (S1(t),S2(t),S3(t)), on a donc : g(X) = exp(-r*T)*h(S1(T,X),S2(T,X),S3(T,X))
# avec X normale(0,I3) et tel que W(T) = np.sqrt(T)*L*X et gamma = L* L^t

# Paramètres communs Q12-Q13
lamb = np.array([1/3,1/3,1/3])
Si_0 = 1
K=1.25
T=1
sig=np.array([0.25,0.28,0.3])
rho = 0.5
r=0.01
n=10

# Décomposition de Cholesky : L tel que Gamma = L*L^T
gamma = np.array([[1,rho,rho],[rho,1,rho],[rho,rho,1]])
L = np.linalg.cholesky(gamma)

# Payoff du call panier : (lambda·S - K)+
def h(x):
    return np.maximum(lamb@x - K,0)

# Jacobien de u_N en dimension 3 (matrice 3x3)
def grad_Un_Rn(teta,g,X):
    N = X.shape[1]
    gX2 = g_Rn(X)**2
    log_w = np.log(gX2 + 1e-300) - teta@X + 0.5*np.linalg.norm(teta)**2
    log_w = np.clip(log_w,-50,50)  # stabilité numérique
    w = np.exp(log_w)                        
    diff = teta[:,None] - X                    
    outer = np.einsum('iN,jN,N->ij', diff, diff, w)/N
    return np.eye(3)*np.mean(w) + outer

# Gradient u_N en dimension 3 (vecteur 3D)
def Un_Rn(teta,g,X):
    N = X.shape[1]
    gX2 = g_Rn(X)**2                          
    log_w = np.log(gX2 + 1e-300) - teta@X + 0.5*np.linalg.norm(teta)**2
    log_w = np.clip(log_w,-50,50)
    w = np.exp(log_w)                         
    return np.mean((teta[:,None]-X)*w, axis=1) 

# Newton en dimension 3 : résolution du système linéaire à chaque itération
def teta_chap_N_Rn(g,X,niter):
    teta_suite = np.zeros((3,niter+1))
    for i in range(niter):
        teta_suite[:,i+1] = teta_suite[:,i] - np.linalg.solve(grad_Un_Rn(teta_suite[:,i],g,X),Un_Rn(teta_suite[:,i],g,X))
    return teta_suite

# g en dimension 3 : payoff actualisé exprimé via X ~ N(0,I3)
# W(T) = sqrt(T)*L*X pour obtenir la corrélation souhaitée
def g_Rn(x):
    y = L@x  # y ~ N(0, Gamma) via Cholesky
    S = np.zeros((3,x.shape[1]))
    for i in range(3):
        S[i,:] = Si_0 * np.exp((r-0.5 * sig[i]**2)*T + sig[i]*np.sqrt(T)*y[i])
    return np.exp(-r*T)*h(S)

def P_MC_Rn(teta,X):
    return np.mean(f_Rn(teta,X))

# Simulation de X ~ N(0,I3) : 3 gaussiennes indépendantes via Box-Muller
def bx_mu_R3(N):
    X = np.zeros((3,N))
    for i in range(3):
        X[i,:] = bx_mu(N)
    return X

# Estimateur IS en dimension 3 : poids exp(-theta·x - 0.5*|theta|²)
def f_Rn(teta,x):
    return g_Rn(x+teta[:,None])* np.exp(-teta@x -0.5 * np.linalg.norm(teta)**2)


# ============================================================
# Q12) Call panier : comparaison MC standard vs IS
# theta* converge vers un vecteur non nul qui shift les tirages
# vers la zone où le panier dépasse K
# ============================================================

N_list  = np.array([500,1000,2000,5000,10000,20000,50000,100000,200000,500000])
P_MC_zero = np.zeros(np.size(N_list))
P_MC_chap = np.zeros(np.size(N_list))


for i,N in enumerate(N_list):
    X = bx_mu_R3(N)
    teta_chap = teta_chap_N_Rn(g_Rn,X,n)[:,-1]
    P_MC_zero[i]=P_MC_Rn(np.zeros(3),X)
    P_MC_chap[i]=P_MC_Rn(teta_chap,X)


plt.figure()
plt.title("Convergence des deux méthodes")
plt.plot(N_list,P_MC_zero, label = "MC teta=0")
plt.plot(N_list,P_MC_chap, label="MC teta= teta_chapeau")
plt.xscale("log")
plt.yscale("log")
plt.show()
