import itertools
import subprocess
import argparse
from multiprocessing import Pool


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--std_values",
        type=list,
        default=[0.75, 1, 1.5, 2],
        help="The values of std to sweep",
    )
    parser.add_argument(
        "--d1_values",
        type=list,
        default=[0.01, 0.1, 0.3, 0.5, 0.8, 1],
        help="The values of d1 to sweep",
    )
    parser.add_argument(
        "--d2_values",
        type=list,
        default=[0.01, 0.1, 0.3, 0.5, 0.8, 1],
        help="The values of d2 to sweep",
    )
    parser.add_argument(
        "--default_indexes",
        type=list,
        default=[0, 1, 2, 3],
        help="The values of d2 to sweep",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default='grid_search_results.txt',
        help="Output file name to save results of grid search.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=8,
        help="Number of threads for parallel computation",
    )

    return parser.parse_args()


# Function to run the adversarial_attacks.py script with given parameters
def run_script(params):
    std, d1, d2, index = params

    cmd = f"source ~/NeuralNets/MatrixStatistics/matrix/bin/activate &&" \
          f" python adversarial_attacks.py --std {std} --d1 {d1} --d2 {d2} " \
          f"--default_index {index}"
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
    )

    # Extract the metrics from the output
    output_lines = result.stdout.split('\n')
    good_defences = None
    wrong_rejection = None
    for line in output_lines:
        if "Percentage of good defences" in line:
            good_defences = float(line.split()[-1].strip(':'))
        if "Percentage of wrong rejections" in line:
            wrong_rejection = float(line.split()[-1].strip(':'))

    if good_defences is not None and wrong_rejection is not None:
        result_line = f"{std},{d1},{d2},default {index},{good_defences},{wrong_rejection}\n"
    else:
        result_line = f"{std},{d1},{d2},default {index},None\n"

    print(result_line.strip())

    return result_line


if __name__ == "__main__":
    args = parse_args()
    # Define the number of processes to use
    num_processes = args.nb_workers  # Adjust this number based on your system's capabilities

    # Define the parameter grid
    std_values = args.std_values
    d1_values = args.d1_values
    d2_values = args.d2_values
    indexes = args.default_indexes

    param_grid = list(itertools.product(std_values, d1_values, d2_values, indexes))

    # Use multiprocessing Pool to run the scripts in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_script, param_grid)

    # TODO: write each result when it is done computing
    with open(args.output_file, 'w') as f:
        f.write(f"std,d1,d2,default_index,good_defence,wrong_rejection\n")
        f.writelines(results)
