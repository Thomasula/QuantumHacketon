import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

#
# mat = sio.loadmat('BartLisa.mat')
# X = mat['D']  # shape (542300, 6)
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# # Parameters
# r = 2
# num_iter = 100
# n, d = X.shape
# p = r
#
# # Initialization
# S = X.T @ X
# U, Lambda_diag, _ = np.linalg.svd(S)
# Lambda = np.diag(Lambda_diag)
#
# sigma = (1 / (d - r)) * np.sum(Lambda_diag[r:])
#
# A = np.random.randn(d, r)
# A, _ = np.linalg.qr(A)
# Z = np.random.randn(n, r)
# Z = np.where(Z > 0, 0, Z)
# A = np.where(A > 0, 0, A)
#
# epsilon = 1e-6
# print("Init loss:", np.linalg.norm(X - Z @ A.T) ** 2)
#
# for i in range(num_iter):
#     A = X.T @ Z @ np.linalg.inv(Z.T @ Z + sigma * np.eye(r))
#     Sigma_A = np.linalg.inv((1 / sigma) * Z.T @ Z + np.eye(r))
#     Z = X @ A @ np.linalg.inv(A.T @ A + p * Sigma_A + epsilon*np.eye(r))
#     Z = np.where(Z > 0, 0, Z)
#     A = np.where(A > 0, 0, A)
#
#     # loss = np.linalg.norm(X - Z @ A.T) ** 2
#     # print(f"Iter {i + 1}, loss: {loss:.4f}, ||Z||: {np.linalg.norm(Z):.4f}")
#
# height, width = 850, 638
# print(f"norm X: {np.linalg.norm(X)}")
# print(f"norm: {np.linalg.norm(X - Z @ A.T) ** 2}")
# components = [Z[:, i].reshape((width, height)) for i in range(r)]


from matplotlib import cm, colors
#
# fig, axes = plt.subplots(1, r, figsize=(15, 5))
# for i, ax in enumerate(axes):
#     ax.imshow(components[i], cmap='gray')
#     ax.set_title(f'Component {i + 1}')
#     ax.axis('off')
# plt.tight_layout()
# plt.show()


# np.random.seed(42)
# n, d = 100, 3
# X = np.random.randn(n, d)
# true_beta = np.array([1.0, -2.0, 3.5])
# sigma = 2.0
# y = X @ true_beta + np.random.normal(0, sigma, size=n)
#
#
# def U(beta, X, y, sigma2, tau2):
#     resid = y - X @ beta
#     return 0.5 / sigma2 * np.dot(resid, resid) + 0.5 / tau2 * np.dot(beta, beta)
#
#
# def dU(beta, X, y, sigma2, tau2):
#     return -X.T @ (y - X @ beta) / sigma2 + beta / tau2
#
#
# def hamiltonian(beta, p, M_inv, X, y, sigma2, tau2):
#     return U(beta, X, y, sigma2, tau2) + 0.5 * np.dot(p.T, M_inv @ p)
#
#
# def leapfrog(p, dt, beta, M_inv, X, y, sigma2, tau2, L):
#     p_n = p - 0.5 * dt * dU(beta, X, y, sigma2, tau2)
#     beta_n = beta
#     for i in range(L):
#         beta_n = beta_n + dt * M_inv @ p_n
#         if i < L - 1:
#             p_n = p_n - dt * dU(beta_n, X, y, sigma2, tau2)
#     p_n = p_n - 0.5 * dt * dU(beta_n, X, y, sigma2, tau2)
#     return beta_n, p_n
#
#
# N = 1000
# h = 0.01
# L = 100
# tau = 10.0
# sigma2 = sigma**2
# M_inv = np.eye(d)
#
# beta_t = np.zeros(d)
# samples = np.zeros((N + 1, d))
# samples[0] = beta_t
#
# # same as old HMC code
# for i in range(N):
#     p_t = np.random.normal(0, 1, size=beta_t.shape)
#     beta_new, p_new = leapfrog(p_t, h, beta_t, M_inv, X, y, sigma2, tau, L)
#
#     E_new = hamiltonian(beta_new, p_new, M_inv, X, y, sigma2, tau)
#     E_old = hamiltonian(beta_t, p_t, M_inv, X, y, sigma2, tau)
#     alpha = min(1, np.exp(E_old - E_new))
#
#     if np.random.rand() < alpha:
#         beta_t = beta_new
#     samples[i + 1] = beta_t
#
#
# for j in range(d):
#     sns.histplot(samples[:, j], bins=50, kde=True, label=f'beta_{j}')
#     plt.axvline(true_beta[j], color='k', linestyle='--', label='true' if j == 0 else "")
# plt.xlabel("Coefficient value")
# plt.title("Ridge Regression")
# plt.legend()
# plt.show()


import numpy as np

def householder_unitary(psi: np.ndarray) -> np.ndarray:
    """
    Construct a real unitary matrix U such that U @ |0> = psi.

    Parameters:
        psi (np.ndarray): Real normalized state vector.

    Returns:
        U (np.ndarray): Real unitary matrix.
    """
    psi = psi.astype(np.float64)
    psi = psi / np.linalg.norm(psi)

    dim = len(psi)
    e1 = np.zeros(dim)
    e1[0] = 1.0

    v = psi - e1
    v = v / np.linalg.norm(v)

    U = np.eye(dim) - 2.0 * np.outer(v, v)
    return U

def print_matrix(U: np.ndarray, precision: int = 4):
    """
    Pretty-print a matrix with fixed precision.

    Parameters:
        U (np.ndarray): Matrix to print.
        precision (int): Decimal digits.
    """
    np.set_printoptions(precision=precision, suppress=True)
    print(U)

# p = np.ones(8)/8
# p /= np.sum(p)
#
#     # Target quantum state
# psi = np.sqrt(p)
#
#     # Generate unitary matrix
# U = householder_unitary(psi)
#
#     # Print the result
# print_matrix(U)
# print(sum(np.power(U[:, 0], 2)))

# import numpy as np
# from scipy.stats import entropy
# from qiskit import QuantumCircuit
# from qiskit.quantum_info import Statevector, Operator
# from qiskit.transpiler.passes import UnitarySynthesis
# from qiskit.transpiler import PassManager
# from qiskit.circuit.library import UnitaryGate
#
#
# # Set parameters
# num_qubits = 3
# num_states = 2 ** num_qubits
# lambda_depth = 0.1  # Penalty weight for depth
#
# # Target distribution (e.g., soft Gaussian-like on 3 qubits)
# x = np.arange(num_states)
# p_target = np.exp(-0.5 * ((x - 3.5) / 1.0)**2)
# p_target /= np.sum(p_target)  # Normalize
#
# psi = np.sqrt(p_target)
#
#     # Generate unitary matrix
# U = householder_unitary(psi)
#
#     # Print the result
# print_matrix(U)
# print(sum(np.power(U[:, 0], 2)))
#
#
#
# # Define a simple parameterized circuit
# def build_test_circuit(params):
#     qc = QuantumCircuit(num_qubits)
#     for i in range(num_qubits):
#         qc.ry(params[i], i)
#     for i in range(num_qubits - 1):
#         qc.cx(i, i + 1)
#     return qc
#
# # Example parameters (random initialization)
# params = np.random.uniform(0, 2 * np.pi, num_qubits)
# qc = build_test_circuit(params)
# # unitary_gate = UnitaryGate(U)
# #
# # qc_basic = QuantumCircuit(num_qubits)
# # qc_basic.append(unitary_gate, range(num_qubits))
# #
# # # Decompose using UnitarySynthesis
# # pm = PassManager([UnitarySynthesis()])
# # qc = pm.run(qc_basic)
#
# # Get the output probabilities from the circuit
# state = Statevector.from_instruction(qc)
# probs = np.abs(state.data) ** 2
#
# # Compute KL divergence
# def kl_divergence(p, q, eps=1e-10):
#     p = np.clip(p, eps, 1)
#     q = np.clip(q, eps, 1)
#     return np.sum(p * np.log(p / q))
#
# kl = kl_divergence(p_target, probs)
# depth_penalty = lambda_depth * qc.depth()
# loss = kl + depth_penalty
#
# print(qc.draw(output='text'))
#
# print(f"target prob.: {p_target}")
# print((f"prob.from circuit: {probs}"))
# print(f"KLD: {kl}, depth: {qc.depth()}, loss: {loss}")
#
# labels = [f'{i:03b}' for i in range(len(p_target))]
# x = np.arange(len(p_target))
#
# plt.figure(figsize=(10, 5))
# plt.bar(x - 0.15, p_target, width=0.3, label='Target', align='center')
# plt.bar(x + 0.15, probs, width=0.3, label='Circuit', align='center')
# plt.xticks(x, labels)
# plt.xlabel('Basis State')
# plt.ylabel('Probability')
# plt.title('Target vs Circuit Output')
# plt.legend()
# plt.tight_layout()
# plt.show()

import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# Configuration
NUM_QUBITS = 3
POP_SIZE = 10
N_GEN = 100
MUTATION_RATE = 0.4
ELITE_SIZE = 2
LAMBDA_DEPTH = 0.1

# Target distribution
x = np.arange(2**NUM_QUBITS)
p_target = np.exp(-0.5 * ((x - 3.5) / 1.0)**2)
p_target /= np.sum(p_target)
psi_target = np.sqrt(p_target)

# Helper functions
def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return np.sum(p * np.log(p / q))

GATES = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx', 'sx', 'cz']


def random_gate():
    g = random.choice(GATES)
    if g in ['rx', 'ry', 'rz']:
        return (g, random.randint(0, NUM_QUBITS - 1), random.uniform(0, 2*np.pi))
    elif g in ['cx', 'cz']:
        control = random.randint(0, NUM_QUBITS - 1)
        target = (control + 1) % NUM_QUBITS
        return (g, control, target)
    else:
        return (g, random.randint(0, NUM_QUBITS - 1))


def build_circuit(gates):
    qc = QuantumCircuit(NUM_QUBITS)
    for gate in gates:
        if gate[0] in ['rx', 'ry', 'rz']:
            getattr(qc, gate[0])(gate[2], gate[1])
        elif gate[0] == 'cx':
            qc.cx(gate[1], gate[2])
        elif gate[0] == 'cz':
            qc.cz(gate[1], gate[2])
        else:
            getattr(qc, gate[0])(gate[1])
    return qc


def fitness(ind):
    qc = build_circuit(ind)
    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data)**2
    kl = kl_divergence(p_target, probs)
    penalty = LAMBDA_DEPTH * qc.depth()
    return -(kl + penalty)  # maximize negative loss

# GA functions
def mutate(ind):
    ind = ind.copy()
    if random.random() < 0.5 and len(ind) > 1:
        ind[random.randint(0, len(ind)-1)] = random_gate()
    else:
        ind.append(random_gate())
    return ind

def crossover(p1, p2):
    if len(p1) < 2 or len(p2) < 2:
        return p1
    point = random.randint(1, min(len(p1), len(p2)) - 1)
    return p1[:point] + p2[point:]

def softmax(x, temperature=0.5):
    x = np.array(x)
    x = x - np.max(x)  # For numerical stability
    exps = np.exp(x / temperature)
    return exps / np.sum(exps)

def select(pop, fitnesses, temperature=0.5):
    probs = softmax(fitnesses, temperature)
    elites = random.choices(pop, weights=probs, k=ELITE_SIZE)
    # Keep elites deterministically
    # elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
    # elites = [pop[i] for i in elite_indices]
    return elites

def random_individual(length=10):
    return [random_gate() for _ in range(length)]

# Main GA loop
population = [random_individual() for _ in range(POP_SIZE)]
best_fitness = -np.inf
best_individual = None

for gen in range(N_GEN):
    fitnesses = [fitness(ind) for ind in population]
    max_fit = max(fitnesses)
    print(f"Gen {gen}, best fitness: {-max_fit:.4f}")
    if max_fit > best_fitness:
        best_fitness = max_fit
        best_individual = population[np.argmax(fitnesses)]
    selected = select(population, fitnesses)
    new_pop = selected[:ELITE_SIZE]
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, best_individual)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        new_pop.append(child)
    population = new_pop

# Evaluate best individual
qc_best = build_circuit(best_individual)
state_best = Statevector.from_instruction(qc_best)
probs_best = np.abs(state_best.data) ** 2
kl_best = kl_divergence(p_target, probs_best)

# Plot
labels = [f'{i:03b}' for i in range(len(p_target))]
x = np.arange(len(p_target))
print("Best circuit:\n", qc_best.draw())

plt.figure(figsize=(10, 5))
plt.bar(x - 0.15, p_target, width=0.3, label='Target', align='center')
plt.bar(x + 0.15, probs_best, width=0.3, label='Best Circuit', align='center')
plt.xticks(x, labels)
plt.xlabel('Basis State')
plt.ylabel('Probability')
plt.title(f'Final KL Divergence: {kl_best:.5f}')
plt.legend()
plt.tight_layout()
plt.show()


