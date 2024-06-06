import argparse
import pandas as pd
import sys

def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default='grid_search_results.txt',
        help="Output file name to read results of grid search.",
    )
    parser.add_argument(
        "--default_index",
        type=int,
        required=True,
        default=0,
        help="The index for default experiment",
    )

    return parser.parse_args()


def check_default_index_exists(df, default_index):
    return f'default {default_index}' in df['default_index'].values


def get_top_10_abs_difference(df, default_index):
    filtered_df = df[df['default_index'] == f'default {default_index}'].copy()
    filtered_df['abs_difference'] = abs(filtered_df['good_defence'] - filtered_df['wrong_rejection'])
    top_10 = filtered_df.nlargest(10, 'abs_difference')
    return top_10[['std', 'd1', 'd2', 'good_defence', 'wrong_rejection']]


if __name__ == "__main__":
    args = parse_args()

    # Read the results file into a DataFrame
    df = pd.read_csv(args.output_file)

    # Check if the default_index exists
    if not check_default_index_exists(df, args.default_index):
        print(f"Error: Default index {args.default_index} does not exist in {args.output_file}.")
        sys.exit(1)

    # Get the top 10 values for the highest absolute difference
    top_10_abs_diff = get_top_10_abs_difference(df, args.default_index)
    print("Top 10 values for highest absolute difference:")
    print(top_10_abs_diff)