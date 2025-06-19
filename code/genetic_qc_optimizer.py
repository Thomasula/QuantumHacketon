import numpy as np
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# Configuration
NUM_QUBITS = 3
POP_SIZE = 50
N_GEN = 100
MUTATION_RATE = 0.5
CROSSOVER_RATE = 0.6
ELITE_SIZE = 15
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

GATES = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']

def random_gate():
    g = random.choice(GATES)
    if g in ['rx', 'ry', 'rz']:
        return (g, random.randint(0, NUM_QUBITS - 1), random.uniform(0, 2*np.pi))
    elif g == 'cx':
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


def fitness_tvd(ind):
    qc = build_circuit(ind)
    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data)**2
    return -0.5 * np.sum(abs(p_target - probs))

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

def softmax(x, temperature=0.5): # higher temperature makes the distribution more flat (more uniform selection)
    # recomended temperature [0.1, 0.8]
    x = np.array(x)
    x = x - np.max(x)  # For numerical stability
    exps = np.exp(x / temperature)
    return exps / np.sum(exps)

def select(pop, fitnesses, temperature=0.5):
    probs = softmax(fitnesses, temperature)
    selected = random.choices(pop, weights=probs, k=ELITE_SIZE)
    return selected

def random_individual(length=10):
    return [random_gate() for _ in range(length)]

# Main GA loop
population = [random_individual() for _ in range(POP_SIZE)]
best_fitness = -np.inf
best_individual = None

for gen in range(N_GEN):
    fitnesses = [fitness_tvd(ind) for ind in population]
    max_fit = max(fitnesses)
    print(f"Gen {gen}, best fitness: {-max_fit:.4f}")
    if max_fit > best_fitness:
        best_fitness = max_fit
        best_individual = population[np.argmax(fitnesses)]
    new_pop = select(population, fitnesses, temperature=0.6)
    while len(new_pop) < POP_SIZE:
        p1, p2 = best_individual, random.choice(new_pop)
        if random.random() < MUTATION_RATE:
            child = mutate(p1)
        if random.random() < CROSSOVER_RATE:
            child = crossover(p1, p2)
        new_pop.append(child)
    population = new_pop

# Evaluate best individual
qc_best = build_circuit(best_individual)
print(qc_best.draw(output='text'))
state_best = Statevector.from_instruction(qc_best)
probs_best = np.abs(state_best.data) ** 2
kl_best = kl_divergence(p_target, probs_best)

# Plot
labels = [f'{i:03b}' for i in range(len(p_target))]
x = np.arange(len(p_target))

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

