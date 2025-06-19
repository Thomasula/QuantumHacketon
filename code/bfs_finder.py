from queue import SimpleQueue
import math
import cmath
from typing import List, Tuple

# apply_gate(state, gate_matrix, target_qubit)
def apply_gate(state: List[Tuple[float, float]], gate: List[List[complex]], target_qubit: int) -> List[Tuple[float, float]]:
    n = int(math.log2(len(state)))  # number of qubits
    assert 2 ** n == len(state), "State must be a power of 2 in length"

    complex_state = [complex(r, i) for (r, i) in state]
    new_state = complex_state.copy()

    for i in range(len(state)):
        if ((i >> (n - 1 - target_qubit)) & 1) == 0:
            # i and i_pair differ at the target_qubit bit
            i0 = i
            i1 = i | (1 << (n - 1 - target_qubit))

            a = complex_state[i0]
            b = complex_state[i1]

            new_state[i0] = gate[0][0] * a + gate[0][1] * b
            new_state[i1] = gate[1][0] * a + gate[1][1] * b

    return [(c.real, c.imag) for c in new_state]

def apply_cx(state: List[Tuple[float, float]], control_qubit: int, target_qubit: int) -> List[Tuple[float, float]]:
    n = int(math.log2(len(state)))
    assert 2 ** n == len(state), "State must have 2^n entries"

    complex_state = [complex(r, i) for (r, i) in state]
    new_state = complex_state.copy()

    for i in range(len(state)):
        # Check if control qubit is 1
        if ((i >> (n - 1 - control_qubit)) & 1) == 1:
            # Flip target qubit
            j = i ^ (1 << (n - 1 - target_qubit))  # flip the target bit
            if i < j:  # prevent double-swapping
                new_state[i], new_state[j] = complex_state[j], complex_state[i]

    return [(c.real, c.imag) for c in new_state]


X_GATE = [
    [0 + 0j, 1 + 0j],
    [1 + 0j, 0 + 0j]
]

CX_GATE = [
    [1+0j, 0+0j, 0+0j, 0+0j],  # |00⟩ → |00⟩
    [0+0j, 1+0j, 0+0j, 0+0j],  # |01⟩ → |01⟩
    [0+0j, 0+0j, 0+0j, 1+0j],  # |10⟩ → |11⟩
    [0+0j, 0+0j, 1+0j, 0+0j],  # |11⟩ → |10⟩
]

def RZ_GATE(theta: float):
    return [
        [1+0j, 0+0j],
        [0+0j, cmath.exp(1j * theta)]
    ]

SX_GATE = [
    [0.5 + 0.5j, 0.5 - 0.5j],
    [0.5 - 0.5j, 0.5 + 0.5j]
]

# !! NOT NATIVE - extra
H_GATE = [
    [1 / math.sqrt(2),  1 / math.sqrt(2)],
    [1 / math.sqrt(2), -1 / math.sqrt(2)]
]

# Initial state |000⟩
state = [(1.0, 0.0)] + [(0.0, 0.0)] * 7  # 3 qubits → 8 basis states

# Apply X to qubit 2 (rightmost qubit)
new_state = apply_gate(state, X_GATE, target_qubit=2)

print("New state:")
for i, amp in enumerate(new_state):
    if amp != (0.0, 0.0):
        print(f"|{i:03b}⟩: {amp[0]:.3f} + {amp[1]:.3f}i")


# number of qbits
N = 2
START = [(1.0, 0.0)] + [(0.0, 0.0)] * 3 # |00>
START = tuple(START)
GOAL  = [(1/math.sqrt(2), 0.0)] + [(0.0, 0.0)] + [(0.0, 0.0)] + [(1/math.sqrt(2), 0.0)] # |01>
GOAL  = tuple(GOAL)
visited = {START}

queue = SimpleQueue()
queue.put(START)
path = SimpleQueue()
path.put([START])

def print_state(state):
    for i, amp in enumerate(state):
        ### !!! change this when different number of qbits is used
        print(f"|{i:02b}⟩: {amp[0]:.3f} + {amp[1]:.3f}i")

print("START")
while True:
    # try all states:
    current_state = queue.get()
    current_path = path.get()
    print(current_path)
    print("current state:")
    print_state(current_state)
    simple_gates = [("SX_GATE", SX_GATE), ("X_GATE", X_GATE), ("H_GATE", H_GATE)]
    #controled_gates = [CX_GATE]
    new_states = []
    moves = []

    print("applying simple gates..")
    for name, g in simple_gates:
        for i in range(N):
            new_states.append(apply_gate(current_state, g, target_qubit=i))
            moves.append(name + " " + str(i))
            print(moves[-1])
            print_state(new_states[-1])

    print("applying controlled not gate..")
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            new_states.append(apply_cx(current_state, i, j))
            moves.append("CX_GATE" + " " + str(i) + " " + str(j))
            print(moves[-1])
            print_state(new_states[-1])

    for i in range(len(new_states)):
        ns = tuple(new_states[i])
        nm = moves[i]
        if ns == GOAL:
            print("FINISHED!")
            print("path to target state")
            final = current_path + [nm]
            for f in final:
                print(f)
            quit()

        if not ns in visited:
            visited.add(tuple(ns))
            queue.put(ns)
            path.put(current_path + [nm])

    input("press enter to continue")
    print()




