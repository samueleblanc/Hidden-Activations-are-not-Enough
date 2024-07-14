from multiprocessing import Pool
import argparse

from utils.utils import compute_train_statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_index", type=int, default=0, help="Default index experiment")
    parser.add_argument("--matrices_path", type=str, default=8, help="Number of threads for parallel computation")
    return parser.parse_args()


def main():
    args = parse_args()
    print('Computing matrix statistics', flush=True)
    compute_train_statistics(args.default_index) #, args.matrices_path)
    #arguments = (i for i in range(18))
    #with Pool(processes=args.nb_workers) as pool:
    #    pool.map(compute_train_statistics, arguments)


if __name__ == '__main__':
    main()
