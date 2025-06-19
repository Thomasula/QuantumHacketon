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
x = np.arange(NUM_QUBITS)
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


