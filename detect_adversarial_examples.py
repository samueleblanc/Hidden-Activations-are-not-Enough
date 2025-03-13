import os
import json
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from argparse import ArgumentParser, Namespace
from pathlib import Path

from utils.utils import get_ellipsoid_data, zero_std, get_model, get_dataset
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS


def parse_args(
        parser:ArgumentParser|None = None
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
        "--default_index",
        type = int,
        default = 0,
        help = "Index of default trained networks."
    )
    parser.add_argument(
        "--t_epsilon",
        type = float,
        default = 1,
        help = "This times the standard deviation gives a margin for rejection level."
    )
    parser.add_argument(
        "--epsilon",
        type = float,
        default = 0.1,
        help = "Determines how small should the standard deviation be per coordinate on matrix statistics."
    )
    parser.add_argument(
        "--epsilon_p",
        type = float,
        default = 0.1,
        help = "Determines how small should the standard deviation be per coordinate when detecting."
    )
    parser.add_argument(
        "--temp_dir",
        type = str,
        default = None,
        help = "Temporary directory to read data for computations from, such as weights, matrix statistics and adversarial matrices."
             "Useful on clusters but not on local experiments."
    )

    return parser.parse_args()


def reject_predicted_attacks(
        default_index: int,
        weights_path: str,
        architecture_index: int,
        residual: bool,
        input_shape: tuple[int,int,int],
        num_classes: int,
        dropout: bool,
        ellipsoids: dict,
        t_epsilon:float = 2,
        epsilon:float = 0.1,
        epsilon_p:float = 0.1,
        verbose:bool = True,
        temp_dir:str|None = None
    ) -> None:
    """
        Goes over the dataset and predicts if it is an adversarial example or not.

        Args:
            default_index: experiment index (See constants/constants.py).
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            num_classes: the number of classes.
            dropout: whether the model has dropout layers.
            ellipsoids: the ellipsoids.
            t_epsilon: t^epsilon from the paper.
            epsilon: the threshold for matrix statistics.
            epsilon_p: the threshold for detection.
            verbose: whether to print the results.
            temp_dir: the temporary directory.
    """
    if temp_dir is not None:
        reject_path = f'{temp_dir}/experiments/{default_index}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'{temp_dir}/experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    else:
        reject_path = f'experiments/{default_index}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'experiments/{default_index}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout
    )

    if os.path.exists(reject_path):
        print("Loading rejection level...", flush=True)
        file = open(reject_path)
        reject_at = json.load(file)[0]
    else:
        print(f"File does not exists: {reject_path}", flush=True)
        return

    if reject_at <= 0:
        print(f"Rejection level too small: {reject_at}", flush=True)
        return

    print(f"Will reject when 'zero dims' < {reject_at}.", flush=True)
    adv_succes = {attack: [] for attack in ["test"]+ATTACKS}  # Save adversarial examples that were not detected
    results = []  # (Rejected, Was attacked)
    # For test counts how many were trusted, and for attacks how many where detected
    counts = {
        key: {
                'not_rejected_and_attacked': 0,
                'not_rejected_and_not_attacked': 0,
                'rejected_and_attacked': 0,
                'rejected_and_not_attacked': 0
        } for key in ["test"] + ATTACKS
    }

    if temp_dir is not None:
        path_adv_matrices = f'{temp_dir}/experiments/{default_index}/adversarial_matrices/'
        test_labels = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/test/labels.pth')
    else:
        path_adv_matrices = f'experiments/{default_index}/adversarial_matrices/'
        test_labels = torch.load(f'experiments/{default_index}/adversarial_examples/test/labels.pth')

    test_acc = 0

    for a in ["test"]+ATTACKS:
        print(f"Trying for {a}", flush=True)
        try:
            if temp_dir is not None:
                attacked_dataset = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/{a}/adversarial_examples.pth')
            else:
                attacked_dataset = torch.load(f'experiments/{default_index}/adversarial_examples/{a}/adversarial_examples.pth')
        except:
            print(f"Attack {a} not found.", flush=True)
            continue
        not_rejected_and_attacked = 0
        not_rejected_and_not_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0

        for i in range(len(attacked_dataset)):
            current_matrix_path = f"{path_adv_matrices}{a}/{i}/matrix.pth"
            im = attacked_dataset[i]
            pred = torch.argmax(model.forward(im))
            mat = torch.load(current_matrix_path)

            b = get_ellipsoid_data(ellipsoids, pred, "std")
            c = zero_std(mat, b, epsilon_p).item()

            res = ((reject_at > c), (a != "test"))

            # if not rejected and it was an attack
            # so detected adversarial example
            if not res[0] and a != "test":
                not_rejected_and_attacked += 1
                counts[a]['not_rejected_and_attacked'] += 1
                if len(adv_succes[a]) < 10:
                    adv_succes[a].append(im)

            # if rejected and it was an attack
            if res[0] and a != 'test':
                rejected_and_attacked += 1
                counts[a]['rejected_and_attacked'] += 1

            # if rejected and it was test data
            # so wrong rejection of natural data
            if res[0] and a == "test":
                rejected_and_not_attacked += 1
                counts[a]['rejected_and_not_attacked'] += 1

            # if not rejected and it was test data
            if not res[0] and a == "test":
                not_rejected_and_not_attacked += 1
                counts[a]['not_rejected_and_not_attacked'] += 1
                if pred == test_labels[i]:
                    test_acc += 1

            results.append(res)

        if verbose:
            print("Attack method: ", a, flush=True)
            if a == 'test':
                print(f'Wrongly rejected test data : {rejected_and_not_attacked} out of {len(attacked_dataset)}', flush=True)
                print(f'Trusted test data : {not_rejected_and_not_attacked} out of {len(attacked_dataset)}', flush=True)

                if counts['test']['not_rejected_and_not_attacked'] == 0:
                    test_acc = 0
                else:
                    test_acc = test_acc / counts['test']['not_rejected_and_not_attacked']

                print("Accuracy on test data that was not rejected: ",
                      test_acc, flush=True)

            else:
                print(f'Detected adversarial examples : {rejected_and_attacked} out of {len(attacked_dataset)}', flush=True)
                print(f'Successful adversarial examples : {not_rejected_and_attacked} out of {len(attacked_dataset)}', flush=True)

    counts_file = f'experiments/{default_index}/counts_per_attack/counts_per_attack_{t_epsilon}_{epsilon}_{epsilon_p}.json'
    Path(f'experiments/{default_index}/counts_per_attack/').mkdir(parents=True, exist_ok=True)
    with open(counts_file, 'w') as json_file:
        json.dump(counts, json_file, indent=4)

    test_accuracy = f'experiments/{default_index}/counts_per_attack/test_accuracy_{t_epsilon}_{epsilon}_{epsilon_p}.json'
    with open(test_accuracy, 'w') as json_file:
        json.dump([test_acc], json_file, indent=4)

    good_defence = 0
    wrongly_rejected = 0
    num_att = 0
    for rej, att in results:
        if att:
            good_defence += int(rej)
            num_att += 1
        else:
            wrongly_rejected += int(rej)
    print(f"Percentage of good defences: {good_defence/num_att}", flush=True)
    print(f"Percentage of wrong rejections: {wrongly_rejected/(len(results)-num_att)}", flush=True)

    return f"Percentage of good defences: {good_defence/num_att}\nPercentage of wrong rejections: {wrongly_rejected/(len(results)-num_att)}"


def get_features(data: torch.Tensor, model) -> np.ndarray:
    """
        Extracts features from the data using the model's penultimate layer.

        Args:
            data: the data to extract features from.
            model: the model to extract features from.
        Returns:
            The features.
    """
    model.eval()
    with torch.no_grad():
        features = []
        for img in data:
            feat = model.forward(img.unsqueeze(0).float(), return_penultimate=True)
            features.append(feat.cpu().numpy().flatten())
        return np.array(features)

def reject_predicted_attacks_baseline(        
        default_index: int,
        weights_path: str,
        architecture_index: int,
        residual: bool,
        input_shape: tuple[int,int,int],
        num_classes: int,
        dropout: bool,
        verbose:bool = True,
        temp_dir:str|None = None
    ) -> None:
    """
        Goes over the dataset and predicts if it is an adversarial example or not using baseline methods.
        Uses K-nearest neighbors, Gaussian Mixture Model, and One-Class SVM for detection.
        Models are trained on training data and a subset of adversarial examples.

        Args:
            default_index: experiment index (See constants/constants.py).
            weights_path: the path to the weights.
            architecture_index: the index of the architecture (See constants/constants.py).
            residual: whether the model has residual connections.
            input_shape: the shape of the input.
            num_classes: the number of classes.
            dropout: whether the model has dropout layers.
            verbose: whether to print the results.
            temp_dir: the temporary directory.
    """
    output_file = f'experiments/{default_index}/grid_search/grid_search_{default_index}_baseline.txt'
    with open(output_file, 'w') as f:
        f.write("method,parameter,default_index,good_defence,wrong_rejection\n")

    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout
    )

    dataset = DEFAULT_EXPERIMENTS[f'experiment_{default_index}']['dataset']

    # Load train and test data
    if temp_dir is not None:
        train_data, _ = get_dataset(dataset, data_loader=False, data_path=temp_dir)
        test_labels = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/test/labels.pth')
    else:
        train_data, _ = get_dataset(dataset, data_loader=False)
        test_labels = torch.load(f'experiments/{default_index}/adversarial_examples/test/labels.pth')

    # Get features from training data
    print("Extracting features from training data...", flush=True)
    train_features = get_features(train_data.data, model)
    train_labels = np.zeros(len(train_features))  # 0 for clean data

    # Get some adversarial examples for training
    print("Collecting adversarial examples for training...", flush=True)
    adv_features = []
    for attack in ATTACKS:
        try:
            # TODO: The attacked data should be from the training set
            if temp_dir is not None:
                attack_data = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/{attack}/adversarial_examples.pth')
            else:
                attack_data = torch.load(f'experiments/{default_index}/adversarial_examples/{attack}/adversarial_examples.pth')
            
            # Take a small subset of each attack type for training
            subset_size = min(100, len(attack_data))
            attack_features = get_features(attack_data[:subset_size], model)
            adv_features.append(attack_features)
        except:
            print(f"Attack {attack} not found for training.", flush=True)
            continue

    if len(adv_features) > 0:
        adv_features = np.vstack(adv_features)
        adv_labels = np.ones(len(adv_features))  # 1 for adversarial data

        # Combine clean and adversarial data for training
        train_features = np.vstack([train_features, adv_features])
        train_labels = np.concatenate([train_labels, adv_labels])
        # Shuffle the data
        permutation = np.random.permutation(len(train_features))
        train_features = train_features[permutation]
        train_labels = train_labels[permutation]

    # TODO: Adjust parameters depending on the dataset
    knn_parameters = [3, 5, 7]
    kde_parameters = [0.1, 0.5, 1]
    gmm_parameters = [10, 20, 30]
    ocsvm_parameters = [0.01, 0.05, 0.1]
    iforest_parameters = [100, 200, 300]
    parameters = {
        'knn': knn_parameters,
        'kde': kde_parameters,
        'gmm': gmm_parameters,
        'ocsvm': ocsvm_parameters,
        'iforest': iforest_parameters
    }
    methods = list(parameters.keys())

    # Initialize results dictionaries for each method
    counts = {
        method: {
            parameters[method][parameter]: {
                attack: {
                    'not_rejected_and_attacked': 0,
                    'not_rejected_and_not_attacked': 0,
                    'rejected_and_attacked': 0,
                    'rejected_and_not_attacked': 0
                } for attack in ["test"] + ATTACKS
            } for parameter in range(len(parameters[method]))
        } for method in methods
    }

    test_acc = {
        method: {
            parameters[method][parameter]: 0
                for parameter in range(len(parameters[method]))
        } for method in methods
    }

    for param in range(len(knn_parameters)):
        print("Initializing detectors...", flush=True)
        knn = KNeighborsClassifier(n_neighbors=knn_parameters[param])
        kde = KernelDensity(bandwidth=kde_parameters[param])
        gmm = GaussianMixture(n_components=gmm_parameters[param], random_state=0)
        ocsvm = OneClassSVM(kernel='rbf', nu=ocsvm_parameters[param])
        iforest = IsolationForest(n_estimators=iforest_parameters[param])

        print("Training detection models...", flush=True)
        knn.fit(train_features, train_labels)
        kde.fit(train_features)
        gmm.fit(train_features)
        ocsvm.fit(train_features[train_labels == 0])  # Train only on clean data for One-Class SVM
        iforest.fit(train_features)

        for method in methods:
            # Test on both clean and adversarial data
            results = []
            for a in ["test"] + ATTACKS:
                print(f"Evaluating {a} examples", flush=True)
                try:
                    if temp_dir is not None:
                        attacked_dataset = torch.load(f'{temp_dir}/experiments/{default_index}/adversarial_examples/{a}/adversarial_examples.pth')
                    else:
                        attacked_dataset = torch.load(f'experiments/{default_index}/adversarial_examples/{a}/adversarial_examples.pth')
                except:
                    print(f"Attack {a} not found.", flush=True)
                    continue

                # Get features for current dataset
                current_features = get_features(attacked_dataset, model)

                if method == 'knn':
                    predictions = knn.predict(current_features)
                elif method == 'kde':
                    scores = kde.score_samples(current_features)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'gmm':
                    scores = gmm.score_samples(current_features)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'ocsvm':
                    predictions = ocsvm.predict(current_features) == -1
                elif method == 'iforest':
                    predictions = iforest.predict(current_features) == -1
                else:
                    raise ValueError(f"Method {method} not found.")

                for i, is_adversarial in enumerate(predictions):
                    results.append((is_adversarial, a != "test"))
                    if a == "test":
                        if not is_adversarial:
                            counts[method][parameters[method][param]][a]['not_rejected_and_not_attacked'] += 1
                            pred = torch.argmax(model.forward(attacked_dataset[i].unsqueeze(0)))
                            if pred == test_labels[i]:
                                test_acc[method][parameters[method][param]] += 1
                        else:
                            counts[method][parameters[method][param]][a]['rejected_and_not_attacked'] += 1
                    else:
                        if is_adversarial:
                            counts[method][parameters[method][param]][a]['rejected_and_attacked'] += 1
                        else:
                            counts[method][parameters[method][param]][a]['not_rejected_and_attacked'] += 1
                    
                # Print results
                if verbose:
                    print(f"\nResults for {method.upper()}:")
                    if a == 'test':
                        print(f'Wrongly rejected test data: {counts[method][parameters[method][param]][a]["rejected_and_not_attacked"]}')
                        print(f'Trusted test data: {counts[method][parameters[method][param]][a]["not_rejected_and_not_attacked"]}')
                        
                        if counts[method][parameters[method][param]][a]['not_rejected_and_not_attacked'] > 0:
                            test_acc[method][parameters[method][param]] = test_acc[method][parameters[method][param]] / counts[method][parameters[method][param]][a]['not_rejected_and_not_attacked']
                        else:
                            test_acc[method][parameters[method][param]] = 0
                        print(f"Accuracy on trusted test data: {test_acc[method][parameters[method][param]]}")
                    else:
                        print(f'Detected adversarial examples: {counts[method][parameters[method][param]][a]["rejected_and_attacked"]}')
                        print(f'Missed adversarial examples: {counts[method][parameters[method][param]][a]["not_rejected_and_attacked"]}')

            good_defence = 0
            wrongly_rejected = 0
            num_att = 0
            for rej, att in results:
                if att:
                    good_defence += int(rej)
                    num_att += 1
                else:
                    wrongly_rejected += int(rej)

            result_line = f"{method},{parameters[method][param]},{default_index},{good_defence/num_att},{wrongly_rejected/(len(results)-num_att)}\n"

            # Write the result to the file
            with open(output_file, 'a') as f:
                f.write(result_line)

    # Save results
    for method in methods:
        counts_file = f'experiments/{default_index}/counts_per_attack/baseline_{method}_counts.json'
        test_accuracy_file = f'experiments/{default_index}/counts_per_attack/baseline_{method}_accuracy.json'
        
        Path(f'experiments/{default_index}/counts_per_attack/').mkdir(parents=True, exist_ok=True)
        
        with open(counts_file, 'w') as f:
            json.dump(counts[method], f, indent=4)
        
        with open(test_accuracy_file, 'w') as f:
            json.dump([test_acc[method]], f, indent=4)


def main(
        default_index:int|None = None,
        t_epsilon:float|None = None,
        epsilon:float|None = None,
        epsilon_p:float|None = None,
        temp_dir:str|None = None,
        baseline:bool = False
    ) -> str:
    """
        Main function to detect adversarial examples.
        Args:
            default_index: the index of the default experiment.
            t_epsilon: the t^epsilon parameter.
            epsilon: the epsilon parameter.
            epsilon_p: the epsilon prime parameter.
            temp_dir: the temporary directory.
        Returns:
            The result of the detection.
    """
    if default_index is None:
        args = parse_args()
    else:
        args = Namespace(
            default_index = default_index, 
            t_epsilon = t_epsilon, 
            epsilon = epsilon, 
            epsilon_p = epsilon_p, 
            temp_dir = temp_dir
        )

    if args.default_index is not None:
        try:
            experiment = DEFAULT_EXPERIMENTS[f'experiment_{args.default_index}']

            architecture_index = experiment['architecture_index']
            residual = experiment['residual']
            dropout = experiment['dropout']
            dataset = experiment['dataset']
            epoch = experiment['epoch'] - 1

        except KeyError:
            print(f"Error: Default index {args.default_index} does not exist.")
            return -1
    else:
        raise ValueError("Default index not specified in constants/constants.py")

    print("Detecting adversarial examples for Experiment: ", args.default_index, flush=True)

    input_shape = (3, 32, 32) if dataset == 'cifar10' or dataset == 'cifar100' else (1, 28, 28)
    num_classes = 10  # TODO: This should be easily changed

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
        matrices_path = Path(f'{args.temp_dir}/experiments/{args.default_index}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"{args.temp_dir}/experiments/{args.default_index}/matrices/matrix_statistics.json")
    else:
        weights_path = Path(f'experiments/{args.default_index}/weights') / f'epoch_{epoch}.pth'
        matrices_path = Path(f'experiments/{args.default_index}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"experiments/{args.default_index}/matrices/matrix_statistics.json")

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed")

    ellipsoids = json.load(ellipsoids_file)

    if baseline:
        reject_predicted_attacks_baseline(
            default_index = args.default_index,
            weights_path = weights_path,
            architecture_index = architecture_index,
            residual = residual,
            input_shape = input_shape,
            num_classes = num_classes,
            dropout = dropout,
            verbose = True,
            temp_dir = args.temp_dir
        )

    result = reject_predicted_attacks(
        default_index = args.default_index,
        weights_path = weights_path,
        architecture_index = architecture_index,
        residual = residual,
        input_shape = input_shape,
        num_classes = num_classes,
        dropout = dropout,
        ellipsoids = ellipsoids,
        t_epsilon = args.t_epsilon,
        epsilon = args.epsilon,
        epsilon_p = args.epsilon_p,
        verbose = True,
        temp_dir = args.temp_dir
    )

    return result


if __name__ == '__main__':
    main()
