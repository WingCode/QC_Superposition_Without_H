from math import pi
from random import uniform as randUniform
from typing import List

import qiskit.providers.aer.noise as noise
from qiskit import QuantumCircuit, execute, Aer


def get_random_params():
    params = [randUniform(0, 2 * pi), randUniform(0, 2 * pi)]
    return params


def get_counts(params: List[float or int], shots: int = 1000) -> dict:
    """
    Here we run the circuit according to the given parameters for each gate and return the counts for each state.

    :param params: List of the parameters of the RY and RX gates of the circuit.
    :param shots: Total number of shots the circuit must execute
    """
    # Error probabilities
    prob_1 = 0.001  # 1-qubit gate
    prob_2 = 0.01  # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates

    # Make a circuit
    circ = QuantumCircuit(2, 2)

    # Set gates and measurement
    circ.ry(params[0], 0)
    circ.rx(params[1], 1)
    circ.cx(0, 1)
    circ.measure([0, 1], [0, 1])

    # Perform a noisy simulation and get the counts
    # noinspection PyTypeChecker
    result = execute(circ, Aer.get_backend('qasm_simulator'),
                     basis_gates=basis_gates,
                     noise_model=noise_model, shots=shots).result()
    counts = result.get_counts(0)

    return counts


def get_cost_vector(counts: dict) -> List[float]:
    """
    This function simply gives values that represent how far away from our desired goal we are. Said desired goal is that
    we get as close to 0 counts for the states |00> and |11>, and as close to 50% of the total counts for |01> and |10>
    each.

    :param counts: Dictionary containing the count of each state
    :return: List of ints that determine how far the count of each state is from the desired count for that state:
                -First element corresponds to |00>
                -Second element corresponds to |01>
                -Third element corresponds to |10>
                -Fourth element corresponds to |11>
    """
    # First we get the counts of each state. Try-except blocks are to avoid errors when the count is 0.
    try:
        a = counts['00']
    except KeyError:
        a = 0
    try:
        b = counts['01']
    except KeyError:
        b = 0
    try:
        c = counts['10']
    except KeyError:
        c = 0
    try:
        d = counts['11']
    except KeyError:
        d = 0

    # We then want the total number of shots to know what proportions we should expect
    totalShots = a + b + c + d

    # We return the absolute value of the difference of each state's observed and desired counts
    # Other systems to determine how far each state count is from the goal exist, but this one is simple and works well
    return [abs(a - 0), abs(b - totalShots / 2), abs(c - totalShots / 2), abs(d - 0)]
