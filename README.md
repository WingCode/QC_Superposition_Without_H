# QC_Superposition_Without_H
QOSF Task 2. Finds the optimal angle rotation using rx, ry bootstrapped by ML model.

## Task
```
Implement a circuit that returns |01> and |10> with equal probability.
Requirements :
The circuit should consist only of CNOTs, RXs and RYs. 
Start from all parameters in parametric gates being equal to 0 or randomly chosen. 
You should find the right set of parameters using gradient descent (you might use more advanced optimization methods if you like). 
Simulations must be done with sampling - i.e. a limited number of measurements per iteration and noise. 

Compare the results for different numbers of measurements: 1, 10, 100, 1000. 

Bonus question:
How to make sure you produce state |01> + |10> and not |01> - |10> ?
```

## Technique used
Traditional gradient descent method works in this way when qubit is random initialised with angle parameters:
1. Measure and get counts. Mostly the state |01> and |10> won't have 50% probability of occurring.
2. Check if state |01> and |10> are having 50% probability or very close to it depending on a threshold. For example: 48% |01> and 51% |10> state of occuring might be good enough according to threshold configured.
3. Calculate gradient to nudge it to 50% probability |01> & |10>
4. Update angle parameters with gradient.
5. Continue again in loop from Step 1 until Step 2 threshold is satisfied.

The problem with this approach is that when you have to get state of very close to equal say 49% |01> and 51% |10> probability of occuring, you will have to take around 230 steps from the random initialisation of angle parameters.

To improve upon this approach of traditional approach of Gradient descent, the following was implemented:
1. Get the random state intialisation.
2. Do multiple iterations for each random state intialisation and find the optimal rotation that is the optimal rotation to get from random state to optimal state ( very close to 50%  |01> and 50% |10>)
3. Note down for each iteration the starting random state and the optimal rotation.
4. Train a ML model (LGBM because it is all time favorite of any DS hacker ;) ) to predict the optimal rotation given a random state.
5. Save the model.

The advantage over here is that:
1. Given any random state it is able to directly find the optimal rotation. Most of the times, the optimal rotation given by model requires no more gradient iteration steps. Hence reducing the steps from 230 -> 1, 2 steps which is **~99% decrease in steps compared to gradient descent.**
2. We are able to build the model only once and use it subsquently. Quantum hardware is expensive currently and this prior step on classical computers makes things cheaper.

## Potential applications
1. If an ideal H gate is expensive to simulate ( running multiple iteration to get the exact state), we can use this method to cut short that.
2. This is some form of error correction. If we build a noisy H gate hardware and we are able to correct it by profiling using ML only once making the hardware useful.


Gradient descent logic taken from https://github.com/MIBbrandon/QOSF_tasks
