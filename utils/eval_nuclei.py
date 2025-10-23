import os
import json
import numpy as np


def calculate_centroid(points):
    """
    Calculate the centroid of a polygon given its points.
    Points should be a list of [x, y] coordinates.
    """
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    return centroid


def extract_features_from_json(json_data, json_name):
    features_list = []
    # for polygon_data in json_data['features'].get('polygons', []):
    for polygon_data in json_data.get('polygons', []):
        category = polygon_data['name']
        score = polygon_data.get('score', 1)
        path_points = polygon_data['path_points']

        if len(path_points) < 3:
            continue  # A valid polygon needs at least 3 points

        exterior_coords = [coord[:2] for coord in path_points]
        centroid = calculate_centroid(exterior_coords)

        features_list.append({
            'filename': json_name,
            'category': category,
            'centroid': centroid.tolist(),
            'score': score
        })
    return features_list


def process_json_file(json_file_path):
    json_name = json_file_path#os.path.basename(json_file_path)
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
        features = extract_features_from_json(json_data, json_name)

    return features


def calculate_centroid_distance(gt_features, pred_features):
    results = []
    pred_structure = {}

    # Organize pred_features into a dictionary (pred_structure) for faster access
    for pred_feature in pred_features:
        match_key = pred_feature['category']
        if match_key not in pred_structure:
            pred_structure[match_key] = []
        pred_structure[match_key].append(pred_feature)

    for gt_feature in gt_features:
        match_key = gt_feature['category']
        eligible_predictions = []

        # Check if there are predictions matching the same filename and category
        if match_key in pred_structure:
            for pred_feature in pred_structure[match_key]:
                # Calculate the Euclidean distance between ground truth and prediction centroids
                distance = np.linalg.norm(np.array(gt_feature['centroid']) - np.array(pred_feature['centroid']))

                # Filter predictions based on a distance threshold (e.g., 15 units)
                if distance < 15:
                    eligible_predictions.append({
                        'pred_json': pred_feature['filename'],
                        'gt_category': gt_feature['category'],
                        'pred_category': pred_feature['category'],
                        'distance': distance,
                        'pred_score': pred_feature['score'],
                        'pred_feature': pred_feature,
                    })

        # Sort eligible predictions by descending prediction score and ascending distance
        eligible_predictions.sort(key=lambda x: (-x['pred_score'], x['distance']))

        # If we have any eligible prediction, take the best match
        if eligible_predictions:
            best_match = eligible_predictions[0]
            results.append(best_match)

            # Find and remove the used prediction from pred_structure
            for i, pred in enumerate(pred_structure[match_key]):
                if np.array_equal(pred['centroid'], best_match['pred_feature']['centroid']):
                    del pred_structure[match_key][i]
                    break

    return results


def calculate_classification_metrics(results, gt_features, pred_features):
    # Extract true positive categories (matched predictions)
    pred_tp = [match['pred_category'] for match in results]

    # Ground truth categories
    ground_truth = [feature['category'] for feature in gt_features]

    # All predicted categories
    pred_all = [feature['category'] for feature in pred_features]

    # Count occurrences of each category in ground truth, predictions, and true positives
    gt_dict = dict(zip(*np.unique(ground_truth, return_counts=True)))
    pred_dict = dict(zip(*np.unique(pred_all, return_counts=True)))
    tp_dict = dict(zip(*np.unique(pred_tp, return_counts=True)))

    micro_TP, micro_FP, micro_FN = 0, 0, 0
    results_metrics = {}

    # Calculate metrics for each category
    for category in np.unique(list(gt_dict.keys()) + list(pred_dict.keys())):
        TP = tp_dict.get(category, 0)
        FP = pred_dict.get(category, 0) - TP
        FN = gt_dict.get(category, 0) - TP

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        results_metrics[category] = {
            'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1_score': f1_score
        }

    # Calculate micro metrics (aggregated across categories)
    micro_precision = micro_TP / (micro_TP + micro_FP) if micro_TP + micro_FP > 0 else 0
    micro_recall = micro_TP / (micro_TP + micro_FN) if micro_TP + micro_FN > 0 else 0
    micro_f1_score = 2 * micro_precision * micro_recall / (
                micro_precision + micro_recall) if micro_precision + micro_recall > 0 else 0

    # Calculate macro F1 (average of F1 scores per category)
    macro_f1_score = np.mean([metrics['f1_score'] for metrics in results_metrics.values()])

    results_metrics['micro'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1_score': micro_f1_score
    }
    results_metrics['macro'] = {
        'f1_score': macro_f1_score
    }

    return results_metrics


def evaluate_files(ground_truth_file, pred_file):
    pred_features = process_json_file(pred_file)
    gt_features = process_json_file(ground_truth_file)

    results = calculate_centroid_distance(gt_features, pred_features)
    metrics = calculate_classification_metrics(results, gt_features, pred_features)

    return metrics