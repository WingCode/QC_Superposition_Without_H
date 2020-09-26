from core.main import get_equal_probability_params

results = get_equal_probability_params(shots=1000, cost_tolerance=0.01, learning_rate=1, bootstrap_model_available=True)
print(results)
