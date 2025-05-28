import sys
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser, Namespace
from typing import Union

from detect_adversarial_examples import main as detect_adversarial_examples


def parse_args(
        parser:Union[ArgumentParser, None] = None
    ) -> Namespace:
    """
        Args:
            parser: the parser to use.
        Returns:
            The parsed arguments.
    """
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "--output_file",
        type = str,
        default = 'grid_search_results.txt',
        help = "Output file name to read results of grid search.",
    )
    parser.add_argument(
        "--default_index",
        type = int,
        default = 0,
        help = "The index for default experiment",
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "The temporary directory to use.",
    )
    return parser.parse_args()


def check_default_index_exists(
        df: pd.DataFrame, 
        default_index: int
    ) -> bool:
    """
        Args:
            df: the dataframe to check.
            default_index: the default index to check.
        Returns:
            True if the default index exists, False otherwise.
    """
    return f'default {default_index}' in df['default_index'].values


def get_top_10_abs_difference(
        df: pd.DataFrame, 
        default_index: int,
        baseline: bool = False
    ) -> pd.DataFrame:
    """
        Args:
            df: the dataframe to get the top 10 absolute difference from.
            default_index: the default index to get the top 10 absolute difference from.
            baseline: whether to get the top 10 absolute difference from the baseline methods.
        Returns:
            The top 10 absolute difference.
    """
    if baseline:
        filtered_df = df[df['default_index'] == default_index].copy()
    else:
        filtered_df = df[df['default_index'] == f'default {default_index}'].copy()

    # Convert columns to float, coercing errors to NaN
    filtered_df['good_defence'] = pd.to_numeric(filtered_df['good_defence'], errors='coerce')
    filtered_df['wrong_rejection'] = pd.to_numeric(filtered_df['wrong_rejection'], errors='coerce')

    # Remove rows with NaN values
    filtered_df = filtered_df.dropna(subset=['good_defence', 'wrong_rejection'])

    filtered_df['abs_difference'] = filtered_df['good_defence'] - filtered_df['wrong_rejection']
    top_10 = filtered_df.nlargest(10, 'abs_difference')
    if baseline:
        return top_10[['method', 'parameter', 'good_defence', 'wrong_rejection']]
    else:
        return top_10[['t_epsilon', 'epsilon', 'epsilon_p', 'good_defence', 'wrong_rejection']]


def run_detect_adversarial_examples(
        t_epsilon: float, 
        epsilon: float, 
        epsilon_p: float, 
        default_index: int, 
        top: int, 
        temp_dir:Union[str, None] = None
    ) -> None:
    """
        Args:
            t_epsilon: the t_epsilon value.
            epsilon: the epsilon value.
            epsilon_p: the epsilon_p value.
            default_index: the default index of the experiment.
            top: the index of a value in the top 10 dataframe.
            temp_dir: the temporary directory.
    """

    # Run the command and capture the output
    result = detect_adversarial_examples(
        t_epsilon = t_epsilon, 
        epsilon = epsilon, 
        epsilon_p = epsilon_p, 
        default_index = default_index, 
        temp_dir = temp_dir
    )

    if result is not None:
        print(f"Successfully ran script with params --t_epsilon {t_epsilon} --epsilon {epsilon} --epsilon_p {epsilon_p}")

        with open(f'experiments/{default_index}/results/{top}_output_{t_epsilon}_{epsilon}_{epsilon_p}.txt', 'w') as f:
            f.write(result)


def main() -> None:
    """
        Main function to read the results of the grid search.
    """
    args = parse_args()

    if args.temp_dir is None:
        path_output = Path(f'experiments/{args.default_index}/grid_search/grid_search_{args.default_index}.txt')
    else:
        path_output = Path(f'{args.temp_dir}/experiments/{args.default_index}/grid_search/grid_search_{args.default_index}.txt')

    Path(f'experiments/{args.default_index}/results/').mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path_output)

    # Check if the default_index exists
    if not check_default_index_exists(df, args.default_index):
        print(f"Error: Default index {args.default_index} does not exist in {args.output_file}.")
        sys.exit(1)

    # Get the top 10 values for the highest absolute difference
    top_10_abs_diff = get_top_10_abs_difference(df, args.default_index)
    print("Top 10 values for highest absolute difference:")
    print(top_10_abs_diff)

    count = 0
    for top, rows in enumerate(top_10_abs_diff.iterrows()):
        _, row = rows
        run_detect_adversarial_examples(
            t_epsilon = row['t_epsilon'], 
            epsilon = row['epsilon'], 
            epsilon_p = row['epsilon_p'], 
            default_index = args.default_index, 
            top = top, 
            temp_dir = args.temp_dir
        )
        count += 1
        if count >= 3:
            break
    
    # Read results of baseline methods
    if args.temp_dir is None:
        path_output = Path(f'experiments/{args.default_index}/grid_search/grid_search_{args.default_index}_baseline.txt')
    else:
        path_output = Path(f'{args.temp_dir}/experiments/{args.default_index}/grid_search/grid_search_{args.default_index}_baseline.txt')

    df = pd.read_csv(path_output)

    top_10_abs_diff = get_top_10_abs_difference(df, args.default_index, baseline=True)
    print("Top 10 values for highest absolute difference in baseline methods:")
    print(top_10_abs_diff)


if __name__ == "__main__":
    main()