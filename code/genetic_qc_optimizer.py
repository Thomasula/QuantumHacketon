import random
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector

# ========================
# CONFIG
# ========================
NUM_QUBITS = 2
POPULATION_SIZE = 10
NUM_GENERATIONS = 20
MUTATION_RATE = 0.3
ELITE_SIZE = 2

TARGET_PROB_DIST = {
    '00': 0.1,
    '01': 0.2,
    '10': 0.3,
    '11': 0.4
}

def target_statevector(dist):
    vec = np.zeros(2**NUM_QUBITS, dtype=complex)
    for bitstring, prob in dist.items():
        idx = int(bitstring, 2)
        vec[idx] = np.sqrt(prob)
    norm = np.linalg.norm(vec)
    return vec / norm

TARGET_STATE = target_statevector(TARGET_PROB_DIST)

# ========================
# CIRCUIT ENCODING
# ========================
GATES = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx']


def random_gate():
    gate = random.choice(GATES)
    if gate in ['rx', 'ry', 'rz']:
        return (gate, random.randint(0, NUM_QUBITS - 1), random.uniform(0, 2 * np.pi))
    elif gate == 'cx':
        control = random.randint(0, NUM_QUBITS - 1)
        target = (control + 1) % NUM_QUBITS
        return (gate, control, target)
    else:
        return (gate, random.randint(0, NUM_QUBITS - 1))


def build_circuit(gate_list):
    qc = QuantumCircuit(NUM_QUBITS)
    for gate in gate_list:
        if gate[0] in ['rx', 'ry', 'rz']:
            getattr(qc, gate[0])(gate[2], gate[1])
        elif gate[0] == 'cx':
            qc.cx(gate[1], gate[2])
        else:
            getattr(qc, gate[0])(gate[1])
    return qc


def random_individual(length=10):
    return [random_gate() for _ in range(length)]


# ========================
# FITNESS FUNCTION
# ========================
def fitness(individual):
    qc = build_circuit(individual)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    state = result.get_statevector(qc)
    fidelity = np.abs(np.dot(np.conj(TARGET_STATE), state)) ** 2
    return fidelity


# ========================
# GA OPERATIONS
# ========================
def mutate(individual):
    new = individual.copy()
    if random.random() < 0.5 and len(new) > 1:
        idx = random.randint(0, len(new) - 1)
        new[idx] = random_gate()
    else:
        new.append(random_gate())
    return new


def crossover(parent1, parent2):
    split = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child = parent1[:split] + parent2[split:]
    return child


def select(population, fitnesses):
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
    return sorted_pop[:ELITE_SIZE] + random.choices(sorted_pop[:5], k=POPULATION_SIZE - ELITE_SIZE)


# ========================
# MAIN LOOP
# ========================
population = [random_individual() for _ in range(POPULATION_SIZE)]

for gen in range(NUM_GENERATIONS):
    fitnesses = [fitness(ind) for ind in population]
    print(f"Gen {gen}, max fitness: {max(fitnesses):.4f}")

    selected = select(population, fitnesses)
    next_population = selected[:ELITE_SIZE]

    while len(next_population) < POPULATION_SIZE:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        if random.random() < MUTATION_RATE:
            child = mutate(child)
        next_population.append(child)

    population = next_population
