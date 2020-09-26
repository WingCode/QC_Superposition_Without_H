from tqdm import tqdm

from ml.bootstrap_model import BootstrapModel
from ml.gradient_descent import get_best_params
from quantum.util import get_random_params
from util import get_bootstrap_file_name, get_bootstrap_row

models_base_file_path = "./models"
bootstrap_data_base_file_path = "./bootstrap_data"


def get_equal_probability_params(shots=1000, cost_tolerance=0.01, learning_rate=1, bootstrap_model_available=True):
    # If model is available directly use it
    if bootstrap_model_available:
        params = get_random_params()
        bootstrap_model = BootstrapModel(models_base_file_path)
        predicted_params = bootstrap_model.predict_params(params)
        return get_best_params(predicted_params, shots, cost_tolerance, learning_rate)

    # If bootstrap_train_data is available, generate model and use it
    else:
        # If nothing is available generate bootstrap_train_data then generate model and then use predictions
        bootstrap_data_file_path = get_bootstrap_file_name(bootstrap_data_base_file_path, shots, cost_tolerance,
                                                           learning_rate)
        bootstrap_file = open(bootstrap_data_file_path, "w+")
        bootstrap_file.write("start_param_1,start_param_2,final_param_1,final_param_2\n")

        iter = 1000
        total_steps = 0
        for i in tqdm(range(iter)):
            # Random param init
            params = get_random_params()
            start_params_1, start_params_2, final_params_1, final_params_2, steps = get_best_params(params, shots,
                                                                                                    cost_tolerance,
                                                                                                    learning_rate)
            total_steps += steps

            row = get_bootstrap_row(start_params_1, start_params_2, final_params_1, final_params_2)
            bootstrap_file.write(row)
        bootstrap_file.close()
        print("Average steps taken per iteration: " + str(total_steps / iter))

        bootstrap_model = BootstrapModel()
        bootstrap_model.train_bootstrap_models(bootstrap_data_file_path, "./models")

        params = get_random_params()
        predicted_params = bootstrap_model.predict_params(params)
        return get_best_params(predicted_params, shots, cost_tolerance, learning_rate)
