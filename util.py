from os import path


def get_bootstrap_file_name(bootstrap_data_base_file_path, shots, cost_tolerance, learning_rate):
    bootstrap_data_file_name = bootstrap_data_base_file_path + "/data_shots_" + str(shots) + "_costTolerance_" + str(
            cost_tolerance) + "_learningRate_" + str(learning_rate) + ".csv"
    return bootstrap_data_file_name


def get_bootstrap_row(start_params_1, start_params_2, final_params_1, final_params_2):
    row = str(start_params_1) + "," + str(start_params_2) + "," + str(final_params_1) + "," + str(
        final_params_2) + "\n"
    return row


def check_file_exists(file_path):
    return path.exists(file_path)
