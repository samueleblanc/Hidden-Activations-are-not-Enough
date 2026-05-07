import os
import json
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis
from argparse import Namespace
from pathlib import Path
from typing import Union

from utils.utils import get_ellipsoid_data, zero_std, get_model, get_dataset, subset, get_num_classes, get_input_shape, get_device, get_parameters_baseline
from constants.constants import DEFAULT_EXPERIMENTS, ATTACKS


def reject_predicted_attacks(
        experiment_name: str,
        weights_path: str,
        architecture_index: int,
        input_shape,
        num_classes: int,
        ellipsoids: dict,
        t_epsilon:float = 2,
        epsilon:float = 0.1,
        epsilon_p:float = 0.1,
        verbose:bool = True,
        temp_dir:Union[str, None] = None
    ) -> str:
    """
        Goes over the dataset and predicts if it is an adversarial example or not.

        Args:
            experiment_name: experiment index (See constants/constants.py).
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
        reject_path = f'{temp_dir}/experiments/{experiment_name}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'{temp_dir}/experiments/{experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)
    else:
        reject_path = f'experiments/{experiment_name}/rejection_levels/reject_at_{t_epsilon}_{epsilon}.json'
        Path(f'experiments/{experiment_name}/rejection_levels/').mkdir(parents=True, exist_ok=True)

    device = get_device()

    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        device=device
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
        path_adv_matrices = f'{temp_dir}/experiments/{experiment_name}/adversarial_matrices/'
        test_labels = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(device)
    else:
        path_adv_matrices = f'experiments/{experiment_name}/adversarial_matrices/'
        test_labels = torch.load(f'experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(device)

    test_acc = 0

    counts_file = Path(f'experiments/{experiment_name}/counts_per_attack/counts_per_attack_{t_epsilon}_{epsilon}_{epsilon_p}.json')
    Path(f'experiments/{experiment_name}/counts_per_attack/').mkdir(parents=True, exist_ok=True)

    for a in ["test"]+ATTACKS:
        print(f"Trying for {a}", flush=True)

        if counts_file.exists():
            with open(counts_file, 'r') as file:
                counts_current = json.load(file)

            if all(value == 0 for value in counts_current[a].values()):
                counts = counts_current

            else:
                print(f"Attack {a} found and loaded.", flush=True)
                continue

        try:
            if temp_dir is not None:
                attacked_dataset = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth').to(device)
            else:
                attacked_dataset = torch.load(f'experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth').to(device)
        except:
            print(f"Attack {a} not found.", flush=True)
            continue
        not_rejected_and_attacked = 0
        not_rejected_and_not_attacked = 0
        rejected_and_attacked = 0
        rejected_and_not_attacked = 0

        for i in range(len(attacked_dataset)):
            current_matrix_path = f"{path_adv_matrices}{a}/{i}/matrix.pth"
            im = attacked_dataset[i].to(device)
            pred = torch.argmax(model.forward(im))
            mat = torch.load(current_matrix_path).to(device)

            b = get_ellipsoid_data(ellipsoids, pred.cpu().detach(), "std")
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

        with open(counts_file, 'w') as json_file:
            json.dump(counts, json_file, indent=4)

        test_accuracy = f'experiments/{experiment_name}/counts_per_attack/test_accuracy_{t_epsilon}_{epsilon}_{epsilon_p}.json'
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

    percentage_good = good_defence/num_att if num_att != 0 else 0
    print(f"Percentage of good defences: {percentage_good}", flush=True)

    percentage_bad = wrongly_rejected/(len(results)-num_att) if len(results) != num_att else 0
    print(f"Percentage of wrong rejections: {percentage_bad}", flush=True)

    return f"Percentage of good defences: {percentage_good}\nPercentage of wrong rejections: {percentage_bad}"


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
            img = img.to(get_device())
            feat = model.forward(img.unsqueeze(0).float(), return_penultimate=True)
            features.append(feat.cpu().numpy().flatten())
        return np.array(features)


def reject_predicted_attacks_baseline(
        experiment_name: str,
        weights_path: str,
        architecture_index: int,
        input_shape,
        num_classes: int,
        verbose:bool = True,
        temp_dir:Union[str, None] = None
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
    output_file = Path(f'experiments/{experiment_name}/grid_search/grid_search_{experiment_name}_baseline.txt')

    if not output_file.exists():
        with open(output_file, 'w') as f:
            f.write("method,parameter,attack,experiment_name,good_defence,wrong_rejection\n")

    model = get_model(
        path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
        device=get_device()
    )

    dataset = DEFAULT_EXPERIMENTS[f'{experiment_name}']['dataset']

    # Load train and test data
    if temp_dir is not None:
        train_data, _ = get_dataset(dataset, data_loader=False, data_path=temp_dir)
        test_labels = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(get_device())
    else:
        train_data, _ = get_dataset(dataset, data_loader=False)
        test_labels = torch.load(f'experiments/{experiment_name}/adversarial_examples/test/labels.pth').to(get_device())

    # Get features from training data
    print("Extracting features from training data...", flush=True)
    train_data, train_labels = subset(train_data, 10000, input_shape)  # The rejection level is computed on 10000 examples
    train_features = get_features(train_data, model)

    parameters = get_parameters_baseline(dataset)

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
            str(parameters[method][parameter]): 0
                for parameter in range(len(parameters[method]))
        } for method in methods
    }

    # For mahalanobis, we need class means and covariances
    class_means = {}
    class_covariances = {}
    for class_idx in range(num_classes):
        class_data = train_features[train_labels == class_idx]
        class_means[class_idx] = np.mean(class_data, axis=0)
        class_covariances[class_idx] = np.cov(class_data, rowvar=False)
    
    def get_min_mahalanobis_distances(features: torch.Tensor) -> np.ndarray:
        min_distances = np.array([])
        for feature in features:
            distances = [
                mahalanobis(feature, class_means[class_idx], class_covariances[class_idx])
                for class_idx in range(num_classes)
            ]
            min_distances = np.append(min_distances, min(distances))
        return min_distances

    # One file for all metods, parameters and attacks
    counts_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_counts.json')
    test_accuracy_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_accuracy.json')

    for param in range(len(parameters['knn'])):
        print("Initializing detectors...", flush=True)
        knn = NearestNeighbors(n_neighbors=parameters['knn'][param], metric="euclidean")
        kde = KernelDensity(bandwidth=parameters['kde'][param])
        gmm = GaussianMixture(n_components=parameters['gmm'][param])
        ocsvm = OneClassSVM(kernel='rbf', nu=parameters['ocsvm'][param])
        iforest = IsolationForest(n_estimators=parameters['iforest'][param])

        print("Training detection models...", flush=True)
        knn.fit(train_features)
        kde.fit(train_features)
        gmm.fit(train_features)
        ocsvm.fit(train_features)
        iforest.fit(train_features)
        mahalanobis_threshold = np.percentile(
            get_min_mahalanobis_distances(train_features), 
            parameters['mahalanobis'][param]
        )

        Path(f'experiments/{experiment_name}/counts_per_attack/').mkdir(parents=True, exist_ok=True)
        for method in methods:
            # Test on both clean and adversarial data
            results = []
            for a in ["test"] + ATTACKS:
                print(f"\n\nEvaluating {a} examples", flush=True)

                if counts_file.exists():
                    with open(counts_file, 'r') as file:
                        counts_current = json.load(file)

                    if all(value == 0 for value in counts_current[method][str(parameters[method][param])][a].values()):
                        counts = counts_current

                    else:
                        print(f'Found and load method: {method}, parameter: {param}, attack: {a}', flush=True)
                        continue

                if test_accuracy_file.exists():
                    with open(test_accuracy_file, 'r') as file:
                        current_acc = json.load(file)

                    #if all(value == 0 for value in current_acc[method][str(parameters[method][param])].values()):
                    #    test_acc = current_acc

                    #else:
                    #    continue

                try:
                    if temp_dir is not None:
                        attacked_dataset = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth')
                    else:
                        attacked_dataset = torch.load(f'experiments/{experiment_name}/adversarial_examples/{a}/adversarial_examples.pth')
                except:
                    print(f"Attack {a} not found.", flush=True)
                    continue

                # Get features for current dataset
                current_features = get_features(attacked_dataset, model)

                if method == 'knn':
                    distances, _ = knn.kneighbors(current_features)
                    average_distance = distances.mean(axis=1)
                    predictions = average_distance < np.percentile(average_distance, 10)
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
                elif method == 'softmax':
                    logits = [model.forward(att.unsqueeze(0)) for att in attacked_dataset]
                    probs = [torch.nn.functional.softmax(log, dim=1) for log in logits]
                    confidences = [p.max(dim=1).values for p in probs]
                    predictions = [conf < parameters['softmax'] for conf in confidences]
                elif method == 'mahalanobis':
                    predictions = get_min_mahalanobis_distances(current_features) < mahalanobis_threshold
                else:
                    raise ValueError(f"Method {method} not found.")

                for i, is_adversarial in enumerate(predictions):
                    results.append((is_adversarial, a != "test"))
                    if a == "test":
                        if not is_adversarial:
                            counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked'] += 1
                            pred = torch.argmax(model.forward(attacked_dataset[i].unsqueeze(0).to(get_device())))
                            if pred == test_labels[i]:
                                test_acc[method][str(parameters[method][param])] += 1
                        else:
                            counts[method][str(parameters[method][param])][a]['rejected_and_not_attacked'] += 1
                    else:
                        if is_adversarial:
                            counts[method][str(parameters[method][param])][a]['rejected_and_attacked'] += 1
                        else:
                            counts[method][str(parameters[method][param])][a]['not_rejected_and_attacked'] += 1
                    
                # Print results
                if verbose:
                    print(f"\nResults for {method.upper()}:")
                    if a == 'test':
                        print(f'Wrongly rejected test data: {counts[method][str(parameters[method][param])][a]["rejected_and_not_attacked"]}')
                        print(f'Trusted test data: {counts[method][str(parameters[method][param])][a]["not_rejected_and_not_attacked"]}')
                        
                        if counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked'] > 0:
                            test_acc[method][str(parameters[method][param])] = test_acc[method][str(parameters[method][param])] / counts[method][str(parameters[method][param])][a]['not_rejected_and_not_attacked']
                        else:
                            test_acc[method][str(parameters[method][param])] = 0
                        print(f"Accuracy on trusted test data: {test_acc[method][str(parameters[method][param])]}")
                    else:
                        print(f'Detected adversarial examples: {counts[method][str(parameters[method][param])][a]["rejected_and_attacked"]}')
                        print(f'Missed adversarial examples: {counts[method][str(parameters[method][param])][a]["not_rejected_and_attacked"]}')

                good_defence = 0
                wrongly_rejected = 0
                num_att = 0
                for rej, att in results:
                    if att:
                        good_defence += int(rej)
                        num_att += 1
                    else:
                        wrongly_rejected += int(rej)

                perc_good = good_defence/num_att if num_att != 0 else 0
                perc_bad = wrongly_rejected/(len(results)-num_att) if len(results) != num_att else 0
                result_line = f"{method},{parameters[method][param]},{a},{experiment_name},{perc_good},{perc_bad}\n"

                # Write the result to the file
                with open(output_file, 'a') as f:
                    f.write(result_line)

                with open(counts_file, 'w') as f:
                    json.dump(counts, f, indent=4)

                with open(test_accuracy_file, 'w') as f:
                    json.dump(test_acc, f, indent=4)


def reject_predicted_attacks_baseline_matrices(        
        experiment_name: str,
        num_classes: int,
        dataset: str,
        verbose:bool = True,
        temp_dir:Union[str, None] = None
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
    output_file = Path(f'experiments/{experiment_name}/grid_search/grid_search_{experiment_name}_baseline_matrices.txt')
    Path(f'experiments/{experiment_name}/grid_search/').mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("method,parameter,experiment_name,good_defence,wrong_rejection\n")

    counts_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_matrices_counts.json')
    test_accuracy_file = Path(f'experiments/{experiment_name}/counts_per_attack/baseline_matrices_accuracy.json')
    Path(f'experiments/{experiment_name}/counts_per_attack/').mkdir(parents=True, exist_ok=True)

    # TODO: The range in the loop should be easily adjustable.
    num_train_examples = 10000
    num_train_examples_per_class = num_train_examples // num_classes
    train_data = torch.Tensor()
    train_labels = torch.Tensor()
    for i in range(num_classes):
        for j in range(num_train_examples_per_class):
            if temp_dir is not None:
                train_data = torch.cat((train_data, torch.load(f'{temp_dir}/experiments/{experiment_name}/matrices/{i}/{j}/matrix.pt').unsqueeze(0)), dim=0)
                train_labels = torch.cat((train_labels, torch.Tensor([i])), dim=0)
            else:
                train_data = torch.cat((train_data, torch.load(f'experiments/{experiment_name}/matrices/{i}/{j}/matrix.pt').unsqueeze(0)), dim=0)
                train_labels = torch.cat((train_labels, torch.Tensor([i])), dim=0)

    train_data = train_data.reshape(train_data.shape[0], -1).detach().cpu().numpy()

    parameters = get_parameters_baseline(dataset)
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

    # For mahalanobis, we need class means and covariances
    class_means = {}
    class_covariances = {}
    for class_idx in range(num_classes):
        class_data = train_data[train_labels == class_idx]
        class_means[class_idx] = np.mean(class_data, axis=0)
        class_covariances[class_idx] = np.cov(class_data, rowvar=False)
    
    def get_min_mahalanobis_distances(features: torch.Tensor) -> np.ndarray:
        min_distances = np.array([])
        for feature in features:
            distances = [
                mahalanobis(feature, class_means[class_idx], class_covariances[class_idx])
                for class_idx in range(num_classes)
            ]
            min_distances = np.append(min_distances, min(distances))
        return min_distances

    if temp_dir is not None:
        test_labels = torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_examples/test/labels.pth')
    else:
        test_labels = torch.load(f'experiments/{experiment_name}/adversarial_examples/test/labels.pth')

    for param in range(len(parameters['knn'])):
        print("Initializing detectors...", flush=True)
        knn = NearestNeighbors(n_neighbors=parameters['knn'][param], metric="euclidean")
        kde = KernelDensity(bandwidth=parameters['kde'][param])
        gmm = GaussianMixture(n_components=parameters['gmm'][param])
        ocsvm = OneClassSVM(kernel='rbf', nu=parameters['ocsvm'][param])
        iforest = IsolationForest(n_estimators=parameters['iforest'][param])

        print("Training detection models...", flush=True)
        knn.fit(train_data)
        kde.fit(train_data)
        gmm.fit(train_data)
        ocsvm.fit(train_data)
        iforest.fit(train_data)
        mahalanobis_threshold = np.percentile(
            get_min_mahalanobis_distances(train_data), 
            parameters['mahalanobis'][param]
        )

        for method in methods:
            # Test on both clean and adversarial data
            results = []
            for a in ["test"] + ATTACKS:
                if counts_file.exists():
                    with open(counts_file, 'r') as file:
                        counts_current = json.load(file)

                    if all(value == 0 for value in counts_current[method][str(parameters[method][param])][a].values()):
                        counts = counts_current

                    else:
                        print(f'Found and load method: {method}, parameter: {param}, attack: {a}', flush=True)
                        continue

                if test_accuracy_file.exists():
                    with open(test_accuracy_file, 'r') as file:
                        current_acc = json.load(file)

                    if all(value == 0 for value in current_acc[method][str(parameters[method][param])].values()):
                        test_acc = current_acc

                    else:
                        continue

                attack_found = True
                print(f"\n\nEvaluating {a} examples", flush=True)
                attacked_dataset = torch.Tensor()
                for i in range(len(train_data)):
                    try:
                        if temp_dir is not None:
                            attacked_dataset = torch.cat((attacked_dataset, torch.load(f'{temp_dir}/experiments/{experiment_name}/adversarial_matrices/{a}/{i}/matrix.pth').unsqueeze(0)), dim=0)
                        else:
                            attacked_dataset = torch.cat((attacked_dataset, torch.load(f'experiments/{experiment_name}/adversarial_matrices/{a}/{i}/matrix.pth').unsqueeze(0)), dim=0)
                    except:
                        if i == 0:
                            print(f"Attack {a} not found.", flush=True)
                            attack_found = False
                            break
                
                if not attack_found:
                    continue
                
                attacked_dataset = attacked_dataset.reshape(attacked_dataset.shape[0], -1)

                if method == 'knn':
                    distances, _ = knn.kneighbors(attacked_dataset)
                    average_distance = distances.mean(axis=1)
                    predictions = average_distance < np.percentile(average_distance, 10)
                elif method == 'kde':
                    scores = kde.score_samples(attacked_dataset)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'gmm':
                    scores = gmm.score_samples(attacked_dataset)
                    predictions = scores < np.percentile(scores, 10)
                elif method == 'ocsvm':
                    predictions = ocsvm.predict(attacked_dataset) == -1
                elif method == 'iforest':
                    predictions = iforest.predict(attacked_dataset) == -1
                elif method == 'mahalanobis':
                    predictions = get_min_mahalanobis_distances(attacked_dataset) < mahalanobis_threshold
                else:
                    raise ValueError(f"Method {method} not found.")

                for i, is_adversarial in enumerate(predictions):
                    results.append((is_adversarial, a != "test"))
                    if a == "test":
                        if not is_adversarial:
                            counts[method][parameters[method][param]][a]['not_rejected_and_not_attacked'] += 1
                            pred = torch.argmax(attacked_dataset[i].sum(dim=0))
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

                result_line = f"{method},{parameters[method][param]},{a},{experiment_name},{good_defence/num_att},{wrongly_rejected/(len(results)-num_att)}\n"

                # Write the result to the file
                with open(output_file, 'a') as f:
                    f.write(result_line)

                with open(counts_file, 'w') as f:
                    json.dump(counts[method], f, indent=4)

                with open(test_accuracy_file, 'w') as f:
                    json.dump([test_acc[method]], f, indent=4)


def main(
        experiment_name:Union[str, None] = None,
        t_epsilon:Union[float, None] = None,
        epsilon:Union[float, None] = None,
        epsilon_p:Union[float, None] = None,
        temp_dir:Union[str, None] = None,
        baseline:bool = False
    ) -> str:
    """
        Main function to detect adversarial examples.
        Args:
            experiment_name: the index of the default experiment.
            t_epsilon: the t^epsilon parameter.
            epsilon: the epsilon parameter.
            epsilon_p: the epsilon prime parameter.
            temp_dir: the temporary directory.
        Returns:
            The result of the detection.
    """
    args = Namespace(
        experiment_name = experiment_name,
        t_epsilon = t_epsilon, 
        epsilon = epsilon, 
        epsilon_p = epsilon_p, 
        temp_dir = temp_dir
    )

    if args.experiment_name is not None:
        experiment = DEFAULT_EXPERIMENTS[f'{args.experiment_name}']
        architecture_index = experiment['architecture_index']
        dataset = experiment['dataset']
        epoch = experiment['epochs'] - 1
    else:
        raise ValueError("Experiment not specified in constants/constants.py")

    print("Detecting adversarial examples for Experiment: ", args.experiment_name, flush=True)

    input_shape = get_input_shape(dataset)
    num_classes = get_num_classes(dataset)

    if args.temp_dir is not None:
        weights_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/weights') / f'epoch_{epoch}.pth'
        matrices_path = Path(f'{args.temp_dir}/experiments/{args.experiment_name}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"{args.temp_dir}/experiments/{args.experiment_name}/matrices/matrix_statistics.json")
    else:
        weights_path = Path(f'experiments/{args.experiment_name}/weights') / f'epoch_{epoch}.pth'
        matrices_path = Path(f'experiments/{args.experiment_name}/matrices/matrix_statistics.json')
        ellipsoids_file = open(f"experiments/{args.experiment_name}/matrices/matrix_statistics.json")

    if not weights_path.exists():
        raise ValueError(f"Experiment needs to be trained")

    if not matrices_path.exists():
        raise ValueError(f"Matrix statistics have to be computed")

    ellipsoids = json.load(ellipsoids_file)

    if baseline:
        reject_predicted_attacks_baseline(
            experiment_name = args.experiment_name,
            weights_path = weights_path,
            architecture_index = architecture_index,
            input_shape = input_shape,
            num_classes = num_classes,
            verbose = True,
            temp_dir = args.temp_dir
        )
        reject_predicted_attacks_baseline_matrices(
            experiment_name = args.experiment_name,
            num_classes = num_classes,
            dataset = dataset,
            verbose = True,
            temp_dir = args.temp_dir
        )

    result = reject_predicted_attacks(
        experiment_name = args.experiment_name,
        weights_path = weights_path,
        architecture_index = architecture_index,
        input_shape = input_shape,
        num_classes = num_classes,
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
