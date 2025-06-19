import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import pandas as pd


# Read-in error table
df = pd.read_csv('ibm_aachen_calibrations_2025-06-19T09_34_49Z.csv')
# print(df.head())

mean_RX_error = df.loc[df['RX error '] != 1.0, 'RX error '].mean()
print(f"mean_RX_error: {mean_RX_error}")
mean_SX_error = df.loc[df['√x (sx) error '] != 1.0, '√x (sx) error '].mean()
print(f"mean_SX_error: {mean_SX_error}")
mean_X_error = df.loc[df['Pauli-X error '] != 1.0,'Pauli-X error ' ].mean()
print(f"mean_X_error: {mean_X_error}")

def extract_mean(s):
    values = [float(part.split(':')[1]) for part in s.split(';')]
    filtered = [v for v in values if v != 1.0]
    return sum(filtered) / len(filtered) if filtered else None  # or 0, np.nan, etc.

df['CZ error '] = df['CZ error '].apply(extract_mean)
mean_CZ_error = df['CZ error '].mean()
print(f"mean_CZ_error: {mean_CZ_error}")

df['RZZ error '] = df['RZZ error '].apply(extract_mean)
mean_RZZ_error = df['RZZ error '].mean()
print(f"mean_RZZ_error: {mean_RZZ_error}")

df['Gate time (ns)'] = df['Gate time (ns)'].apply(extract_mean)
mean_Gate_time = df['Gate time (ns)'].mean()
print(f"mean Gate time (ns) : {mean_Gate_time}")


# Configuration
NUM_QUBITS = 3
POP_SIZE = 10
N_GEN = 100
MUTATION_RATE = 0.4
alpha = 0.9
ELITE_SIZE = 2
LAMBDA_DEPTH = 0.025
LAMBDA_CNOT = 0.1
LAMBDA_GATE = 2*1e1

# Target distribution
# x = np.arange(2**NUM_QUBITS)
# p_target = np.exp(-0.5 * ((x - 3.5) / 1.0)**2)
# p_target /= np.sum(p_target)
# psi_target = np.sqrt(p_target)
x = np.linspace(0, 1, 2**NUM_QUBITS)
f_vals = np.sin(np.pi * x)
f_vals = np.maximum(f_vals, 0)  # Ensure positivity if using probability

# Target distribution: square to get amplitude proportions
p_target = f_vals**2
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
    penalty_depth = LAMBDA_DEPTH * qc.depth()
    cnot_count = 0

    gate_error_penalty = 0
    for gate in ind:
        if gate[0] == 'cx':
            cnot_count += 1
        elif gate[0] == 'rx':
            gate_error_penalty += mean_RX_error
        elif gate[0] == 'sx':
            gate_error_penalty += mean_SX_error
        elif gate[0] == 'x':
            gate_error_penalty += mean_X_error
        elif gate[0] == 'cz':
            gate_error_penalty += mean_CZ_error

    # Penalize based on number of CNOTs
    penalty_cnot = LAMBDA_CNOT * cnot_count
    gate_error_penalty *= LAMBDA_GATE
    print(f"penalty depth: {penalty_depth}, penalty cnot: {penalty_cnot}, kl: {kl}, gate error penalty: {gate_error_penalty}")
    return -(kl + penalty_depth + penalty_cnot + gate_error_penalty)  # maximize negative loss

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

# Run GA

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
    MUTATION_RATE *= alpha
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

from scipy.integrate import quad
true_integral, _ = quad(lambda x: np.sin(np.pi * x), 0, 1)
delta_x = 1 / (2**NUM_QUBITS)
# estimated_integral = np.sum(np.sqrt(probs_best)) * delta_x
estimated_integral = np.sum(probs_best * f_vals)
print(f"True integral: {true_integral}, Estimated: {estimated_integral}")

# cumulative_integral = []
# cum_sum = 0
# for i in range(len(f_vals)):
#     cum_sum += probs_best[i] * f_vals[i]
#     cumulative_integral.append(cum_sum)
#
# plt.plot(cumulative_integral)
# plt.show()
from scipy.integrate import cumtrapz

x_vals = (np.arange(len(f_vals)) + 0.5) / len(f_vals)
estimated_cumulative = np.cumsum(probs_best * f_vals)
true_cumulative = cumtrapz(f_vals, x_vals, initial=0)

# Plot cumulative integrals
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(x_vals, estimated_cumulative, label="Quantum Estimated Cumulative")
plt.plot(x_vals, true_cumulative, '--', label="True Cumulative Integral")
plt.xlabel("x")
plt.ylabel("Cumulative Integral")
plt.title("Cumulative Integral: Quantum vs True")
plt.legend()
plt.grid(True)

# Plot absolute error
plt.subplot(2, 1, 2)
abs_error = np.abs(estimated_cumulative - true_cumulative)
plt.plot(x_vals, abs_error, color='red', label="Absolute Error")
plt.xlabel("x")
plt.ylabel("Error")
plt.title("Cumulative Integral Absolute Error")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


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
