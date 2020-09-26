from math import pi
from typing import List

from quantum.util import get_counts, get_cost_vector


def done_descending(costVector: List[float or int], shots: int, costTolerance: float) -> bool:
    """
    Determines whether we should stop the algorithm or not
    :param costVector: List of ints that determine how far the count of each state is from the desired count for that state
    :param shots: Number of shots taken by the circuit
    :param costTolerance: Float value that determines the error margin that will be allowed.
    :return: True if all costs are lower than costTolerance * shots, False otherwise
    """
    if not 1 >= costTolerance > 0:
        raise ValueError("costTolerance must be in the range (0, 1]")

    for cost in costVector:
        if cost > costTolerance * shots:
            return False
    return True


def get_gradient(counts: dict) -> List[int]:
    """
    The gradient will determine how strongly we will modify the values in params[] and in which direction.

    Let's remember again what we want each qubit to do:
        1) We want the first qubit to be measured as |0> 50% of the time, with the other 50% of the time being |1>, obviously.
        2) We want the second qubit to always be |1> before it reaches the CNOT.
    Remember that the counts give us information AFTER the CNOT.

    We can easily determine how far the first parameter (the one controlling the phase of the first qubit) is from its
    desired goal by seeing how often it is a |0> or a |1> once it has been measured. This can be inferred by adding the
    counts of |00> and |10>, we get the count of the first qubit being |0>. Of course, adding the counts of |01> and |11>
    gives us how often the first qubit is |1>.

    With that information, we can determine in which direction and how strongly we should modify the values in params[i]
    with a function such as (count |00> + count |10>) - (count |01> + count |11>). The more often |0> appears than |1>,
    the greater the output will be on the positive side, and the more often |1> appears than |0>, the greater the output
    will be on the negative side. This is what we will use as the gradient function for the parameter of the first qubit.

    The second qubit is a little trickier to think about, but the function is just as simple. We said we wanted the second
    qubit to always be |1> BEFORE it reaches the CNOT, but we don't have access to measurement before the CNOT. Then, how
    do we tell how we should modify the parameter for the RX gate on the second qubit?

    Going back to the other big comment in this file, we know that if there were no RX (or RY) gate on the second qubit,
    that the output will always be |00> or |11>, since the CNOT only flips the second qubit if the first one is flipped.
    This way, we know that if we are getting counts of |00> and |11> that are way too high above 0, the parameter of the
    gate is not where it should be. Therefore, the higher the total count of |00> and |11>, the stronger the change in
    phase we need to make, and a function that represents that well is simply the total count of |00> and |11> itself.

    There is a small problem with this function, and that is that it's only one-directional (we will never get a negative
    value for this function). This means that perhaps it would be optimal to decrease the value of the parameter at a given
    moment, rather than to increase it, but the function will always guide us to increase it. This isn't really a huge issue,
    since the phase is modulo 2*pi. What matters is that it tells us to increase the value at the right pace at every given
    moment: when we are really far from the goal, take big steps, and as we get closer to the goal, take smaller and smaller
    steps. The function proposed earlier does that very well.

    One extra thing to note: the function of the gradient of the second qubit highly depends on the effectiveness of the
    function of the first qubit. If the first one isn't working, the second one is flying blind.

    With these two functions dictating how strongly we should alter each parameter and tweaking some constants such as
    the learning rate that we will use later on, we will obtain the correct values for each parameter fairly quickly.

    :param counts: Dictionary containing the count of each state
    :return: List of ints that are a representation of how intensely we must change the values of params[]
                -params[0] corresponds to the phase of the RY gate of the first qubit
                -params[1] corresponds to the phase of the RX gate of the second qubit
    """
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

    """
    Let's quickly think about a way of improving the gradient than just the difference or addition of counts.
    
    First, let's note that without any modifications, the range of the gradients are [-0.5, 0.5] and [0, 1] for params[0]
    and params[1], respectively. If our values turned out to be at the extreme side of the undesired, then we know very 
    well by how much we should change the phase.
    
    For example, if for params[1] we got a gradient of 1, that means that params[1] is somewhere around |0> and so it needs
    a shift of pi to be |1>. To get to that point very quickly, we can just multiply the value of the gradient by pi. 
    That way, whenever the second parameter is on the complete opposite end of where it should be, we can move it very 
    quickly to where it should. When I only modify the second gradient by multiplying it by pi, this often reduces the 
    number of steps taken by the algorithm significantly (for example, from an average of 20 steps to an average of 6 steps).
    
    Now, you would think that the same goes for the first parameter: if the gradient is ±0.5, then params[0] is 
    probably around 0 or pi, and we want to move it to pi/2 or 3*pi/2, so we need the gradient to be ±pi/2. We take 
    advantage of the 0.5 and multiply it by pi in order to make the gradient equal to ±pi/2 whenever it is at 
    either extreme. But this is not the case. For some reason that I don't fully comprehend, it makes the algorithm take
    more steps in general, especially when combined with the modification of the second gradient value. 
    
    I suspect that it has to do with the fact that the second gradient value has a greater "tolerable margin of error" 
    than the first gradient value. That is, params[1] could be anywhere from 3*pi/4 to 5*pi/4 and still output often 
    enough it's intended goal, whilst that range of 2*pi/4 (or pi/2) means the opposite output of what is desired. Either
    way, I'd love to discuss it further with my mentor.
    """
    return [((b + d) - (a + c)) / totalShots, pi * (a + d) / totalShots]


def update_params(intial_params: List[float], gradient: List[int], learningRate: float or int) -> List[float]:
    """
    Here we update the parameters according to the gradient. A very simple update function is just subtracting the gradient,
    and modulating the intensity of said gradient with a learning rate value.

    :param params: List of the parameters of the RY and RX gates of the circuit.
    :param gradient: List of values which represent how intensely we should modify each parameter
    :param learningRate: Float or int value to modulate the gradient
    """
    updated_params = [None] * len(intial_params)
    for i in range(len(intial_params)):  # For every parameter value
        # params[i] -= learningRate * gradient[i]  # Subtract a modulated version of the value of the gradient
        updated_params[i] = intial_params[i] - learningRate * gradient[i]

        while updated_params[i] < 0:
            updated_params[i] = updated_params[i] + 2 * pi
        updated_params[i] = updated_params[i] % (2 * pi)

    return updated_params


def get_best_params(params, shots=1000, cost_tolerance=0.01, learning_rate=1):
    """
    Main method that does the entire gradient descent algorithm.
    :param params: Rotation angle parameters
    :param cost_tolerance: Float value that determines the error margin that will be allowed.
    :param learning_rate: Float or int value to modulate the gradient
    :param shots: Total number of shots the circuit must execute
    """

    # Initialize variables
    descending = True  # Symbolic, makes it easier for reader to follow
    counts = None
    cost_vector = None

    print("START PARAMS: " + str(params))  # Print to console the parameters we will begin with

    start_params = params
    steps = 0  # Keep track of how many steps were taken to descend
    while descending:
        # Get the initial counts that result from these parameters
        counts = get_counts(params, shots)

        # Find the cost of these parameters given the results they have produced
        cost_vector = get_cost_vector(counts)

        # Determine whether the cost is low enough to stop the algorithm
        if done_descending(cost_vector, shots, cost_tolerance):
            break

        # Calculate the gradient
        gradient = get_gradient(counts)

        # Update the params according to the gradient
        params = update_params(params, gradient, learning_rate)
        steps += 1  # Recording the number of steps taken

    # Show current situation
    # print("\tCOUNTS: " + str(counts)
    # 	  + "\n\tCOST VECTOR: " + str(cost_vector)
    # 	  + "\n\tGRADIENT: " + str(gradient)
    # 	  + "\n\tUPDATED PARAMS: " + str(params) + "\n")

    # Print the obtained results
    print("FINAL RESULTS:"
          + "\n\tCOUNTS: " + str(counts)
          + "\n\tCOST VECTOR" + str(cost_vector)
          + "\n\tSTART_PARAMS: " + str(start_params)
          + "\n\tPARAMS: " + str(params)
          + "\nSteps taken: " + str(steps))
    return start_params[0], start_params[1], params[0], params[1], steps
