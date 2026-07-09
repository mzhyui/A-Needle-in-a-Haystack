from scipy.fft import fft2, ifft2, fftshift, ifftshift
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mutual_info_score


import umap
import joblib
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier

from tqdm import tqdm

from models.softmaskLayer import SoftMaskedLayer
from utils.dataUtils import NoiseDataset, FilteredDataset, ShrinkedDataset, VoidDataset, CorruptedDataset
from utils.subnetworkUtils import getTssImpt, getTssCommon, getTssCommon_soft, getTssCommon_soft_to_hard, computeNeuronImportance_v3
import utils.venn as venn


class Assessor():
    def __init__(self, dataset) -> None:
        self.datasets = _prepare_datasets(dataset)

    def updateTriggerDataset(self, dataset_test, mask, trigger, alpha, target_label, pos_choice, num_class, img_channel, img_size, attack_type):
        corrupt_ds = _prepare_datasets_trigger(
            dataset_test, mask, trigger, alpha, attack_type, target_label, pos_choice, num_class, img_channel, img_size)
        self.datasets['trigger'] = corrupt_ds

    def triImportance(self, model, w_local_list, last_glob_dict, trigger_model_idx, p, visual_path=None, assessor_path="./", assessing_layer='fc4', save=True, device: str | torch.device = 'cuda'):
        """
        Calculate trigger importance scores and classify them into TP/TN/FP categories.

        Returns:
            dict: Contains 'trigger', 'eval', and 'ne' analysis results
        """

        # Step 1: Extract layer names
        layer_names = [name for name, module in model.named_modules()
                       if isinstance(module, SoftMaskedLayer)]

        # Step 3: Prepare model state dictionaries
        model_state_dicts = {idx: state_dict for idx,
                             state_dict, _ in w_local_list}
        trigger_model_dict = model_state_dicts[trigger_model_idx]

        # Step 4: Calculate importance scores
        importance_scores = _calculate_importance_scores(
            model, trigger_model_dict, model_state_dicts, self.datasets,
            layer_names, p, device, trigger_model_idx
        )
        params_list = [model_i for idx, model_i, _ in w_local_list]
        importance_scores['grad'] = computeNeuronImportance_v3(
            model, last_glob_dict, params_list
        )
        importance_scores['grad_hard'] = computeNeuronImportance_v3(
            model, last_glob_dict, params_list, True
        )

        # Step 5: Classify importance scores
        # Assuming _classify_importance_scores is defined elsewhere
        classifications = _classify_importance_scores(
            importance_scores, layer_names)

        # Step 6: Plot importance scores (if visual_path is provided)
        if visual_path is not None:
            # _plot_importance_scores(importance_scores, layer_names, visual_path)
            # _plot_layerwise_umap([importance_scores['trigger_hard'], importance_scores['clean_hard'], importance_scores['noise_hard'], importance_scores['grad_hard']], importance_scores['noise'], importance_scores['clean'], importance_scores['grad'], model, trigger_model_dict, visual_path, 'gt')
            # _plot_layerwise_venn_intersections([importance_scores['trigger'], importance_scores['clean_hard'], importance_scores['noise_hard'], importance_scores['grad_hard']], model, visual_path, 'venn')
            _plot_comparative_trigger_heatmaps([importance_scores['trigger_hard'], importance_scores['clean_hard'],
                                               importance_scores['noise_hard'], importance_scores['grad_hard']], model, visual_path, 'gt')

        # # # Step 7: Train SVM with importance scores
        # svm_results = train_svm_classifier_improved(
        #     importance_scores, assessing_layer, assessor_path, save=save)

        # # Step 8: grad steo analysis
        # grad_step_results = analyze_neurons_by_gradient_percentile_aggregated(importance_scores['clean'], importance_scores['noise'],
        #                                                       importance_scores['trigger'],
        #                                                       importance_scores['grad'],
        #                                                       percentile=90)
        # grad_step_results = analyze_neurons_by_gradient_steps(importance_scores['clean'], importance_scores['noise'],
        #                                                       importance_scores['trigger'],
        #                                                       importance_scores['grad'],
        #                                                       )

        return classifications


def analyze_data_distribution(X, y, feature_names=['noise', 'clean', 'grad']):
    """Analyze and visualize data distribution"""
    print("\n" + "="*60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*60)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"\nClass Distribution:")
    for label, count in class_dist.items():
        percentage = (count / len(y)) * 100
        print(f"  Label {label}: {count} samples ({percentage:.2f}%)")

    imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1.0
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")

    # Feature statistics per class
    print(f"\nFeature Statistics by Class:")
    for label in unique:
        print(f"\n  Class {label}:")
        X_class = X[y == label]
        for i, fname in enumerate(feature_names):
            print(f"    {fname}: mean={X_class[:, i].mean():.4f}, "
                  f"std={X_class[:, i].std():.4f}, "
                  f"min={X_class[:, i].min():.4f}, "
                  f"max={X_class[:, i].max():.4f}")

    # Feature correlation
    print(f"\nFeature Correlations:")
    for i, fname in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {fname} <-> label: {corr:.4f}")

    return class_dist, imbalance_ratio


def train_svm_classifier_improved(importance_scores, layer_selection="", assessor_path="./", save=True):
    """
    Improved SVM classifier with better handling of imbalanced data.
    If save=False, loads existing model and evaluates on new data.
    """

    # ============================================
    # LOAD EXISTING MODEL IF save=False
    # ============================================
    if not save:
        return load_and_evaluate_model(importance_scores, layer_selection, assessor_path)

    # Extract features and labels
    labels_dict = importance_scores['trigger']
    feature_noise_dict = importance_scores['noise']
    feature_clean_dict = importance_scores['clean']
    feature_grad_dict = importance_scores['grad']

    all_labels = []
    all_features = []

    layer_names = list(labels_dict.keys())
    layer_names = list(filter(lambda ln: ln == layer_selection,
                       layer_names)) if layer_selection else layer_names

    for layer_name in layer_names:
        layer_labels = labels_dict[layer_name]
        layer_noise = feature_noise_dict[layer_name]
        layer_clean = feature_clean_dict[layer_name]
        layer_grad = feature_grad_dict[layer_name]

        # Convert to numpy
        for var_name, var in [('layer_labels', layer_labels),
                              ('layer_noise', layer_noise),
                              ('layer_clean', layer_clean),
                              ('layer_grad', layer_grad)]:
            if isinstance(var, torch.Tensor):
                var = var.cpu().numpy()
            locals()[var_name] = var
        if isinstance(layer_labels, torch.Tensor):
            layer_labels = layer_labels.cpu().numpy()
        if isinstance(layer_noise, torch.Tensor):
            layer_noise = layer_noise.cpu().numpy()
        if isinstance(layer_clean, torch.Tensor):
            layer_clean = layer_clean.cpu().numpy()
        if isinstance(layer_grad, torch.Tensor):
            layer_grad = layer_grad.cpu().numpy()

        # Flatten
        layer_labels_flat = layer_labels.flatten()
        layer_noise_flat = layer_noise.flatten()
        layer_clean_flat = layer_clean.flatten()
        layer_grad_flat = layer_grad.flatten()

        # Ensure same length
        min_len = min(len(layer_labels_flat), len(layer_noise_flat),
                      len(layer_clean_flat), len(layer_grad_flat))

        layer_labels_flat = layer_labels_flat[:min_len]
        layer_noise_flat = layer_noise_flat[:min_len]
        layer_clean_flat = layer_clean_flat[:min_len]
        layer_grad_flat = layer_grad_flat[:min_len]

        layer_features = np.column_stack(
            [layer_noise_flat, layer_clean_flat, layer_grad_flat])

        all_labels.extend(layer_labels_flat)
        all_features.extend(layer_features)

    X = np.array(all_features)
    y = np.array(all_labels)

    if len(X) == 0:
        print("No data available for SVM training")
        return None

    # ============================================
    # IMPROVED BINARIZATION STRATEGY
    # ============================================
    # Try multiple thresholding strategies
    print("\nTesting different thresholding strategies:")

    strategies = {
        'median': np.median(y),
        'mean': np.mean(y),
        'percentile_75': np.percentile(y, 75),
        'percentile_80': np.percentile(y, 80),
        'percentile_85': np.percentile(y, 85),
        'otsu': None  # Will compute using Otsu's method
    }

    # Otsu's method for automatic threshold
    hist, bin_edges = np.histogram(y, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute Otsu threshold
    total = len(y)
    sum_total = np.sum(y)
    sum_background = 0
    weight_background = 0
    max_variance = 0
    otsu_threshold = 0

    for i, val in enumerate(bin_centers):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += hist[i] * val
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance = weight_background * weight_foreground * \
            (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            otsu_threshold = val

    strategies['otsu'] = otsu_threshold

    # Evaluate each strategy
    best_strategy = None
    best_balance = 0

    for strategy_name, threshold in strategies.items():
        y_binary = (y > threshold).astype(int)
        unique, counts = np.unique(y_binary, return_counts=True)

        if len(unique) == 2:
            # Calculate balance score (closer to 1.0 is better)
            balance = min(counts) / max(counts)
            print(f"  {strategy_name}: threshold={threshold:.4f}, "
                  f"class_0={counts[0] if 0 in unique else 0}, "
                  f"class_1={counts[1] if 1 in unique else 0}, "
                  f"balance={balance:.4f}")

            # Prefer strategies with better balance (aim for 20-80% minority class)
            minority_ratio = min(counts) / sum(counts)
            if 0.2 <= minority_ratio <= 0.8 and balance > best_balance:
                best_balance = balance
                best_strategy = (strategy_name, threshold)

    # Use best strategy or fall back to percentile_80
    if best_strategy:
        strategy_name, threshold = best_strategy
    else:
        strategy_name = 'percentile_80'
        threshold = strategies['percentile_80']

    print(f"\nSelected strategy: {strategy_name} (threshold={threshold:.4f})")
    y_binary = (y > threshold).astype(int)

    # Analyze data distribution
    class_dist, imbalance_ratio = analyze_data_distribution(X, y_binary)

    # ============================================
    # HANDLE SEVERE IMBALANCE
    # ============================================

    # Apply SMOTE for severe imbalance (ratio > 3:1)
    if imbalance_ratio > 3.0:
        print(
            f"\nApplying SMOTE to handle severe imbalance (ratio: {imbalance_ratio:.2f}:1)")
        try:

            # Use SMOTETomek for better results (oversampling + cleaning)
            smote_tomek = SMOTETomek(random_state=42)
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y_binary)

            print(f"After SMOTE-Tomek: {len(X_resampled)} samples")
            new_class_dist, new_ratio = analyze_data_distribution(
                X_resampled, y_resampled)

            X = X_resampled
            y_binary = y_resampled
        except ImportError:
            print(
                "Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")
            print("Continuing with class weights only...")

    # ============================================
    # STRATIFIED SPLIT
    # ============================================
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ============================================
    # TRAIN MULTIPLE MODELS AND COMPARE
    # ============================================
    print("\n" + "="*60)
    print("TRAINING MULTIPLE CLASSIFIERS")
    print("="*60)

    models = {}
    results_comparison = {}

    # 3. Random Forest with balanced classes
    print("3. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42,
                                class_weight='balanced', max_depth=10)
    rf.fit(X_train_scaled, y_train)
    models['RandomForest'] = rf

    # Evaluate all models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_pred_train = model.predict(X_train_scaled)

        # Calculate metrics
        test_acc = accuracy_score(y_test, y_pred)
        train_acc = accuracy_score(y_train, y_pred_train)

        # Per-class metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        # F1 scores
        f1_macro = report['macro avg']['f1-score']
        f1_class_0 = report['0']['f1-score'] if '0' in report else 0
        f1_class_1 = report['1']['f1-score'] if '1' in report else 0

        # Precision and Recall for minority class (likely class 1)
        precision_1 = report['1']['precision'] if '1' in report else 0
        recall_1 = report['1']['recall'] if '1' in report else 0

        results_comparison[model_name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'f1_macro': f1_macro,
            'f1_class_0': f1_class_0,
            'f1_class_1': f1_class_1,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'report': classification_report(y_test, y_pred)
        }

        print(f"\n{model_name}:")
        print(f"  Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  Class 0 - F1: {f1_class_0:.4f}")
        print(
            f"  Class 1 - F1: {f1_class_1:.4f}, Precision: {precision_1:.4f}, Recall: {recall_1:.4f}")

    # Select best model based on F1 score for minority class
    best_model_name = max(results_comparison.items(),
                          key=lambda x: x[1]['f1_class_1'])[0]
    best_model = models[best_model_name]

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"{'='*60}")
    print(results_comparison[best_model_name]['report'])

    # Feature importance
    feature_importance = {}
    if hasattr(best_model, 'coef_'):  # Linear models
        feature_coeffs = np.abs(best_model.coef_[0])
        feature_importance = {
            'noise': feature_coeffs[0],
            'clean': feature_coeffs[1],
            'grad': feature_coeffs[2]
        }
    elif hasattr(best_model, 'feature_importances_'):  # Tree-based models
        feature_importance = {
            'noise': best_model.feature_importances_[0],
            'clean': best_model.feature_importances_[1],
            'grad': best_model.feature_importances_[2]
        }
    else:
        # For RBF SVM, train a linear SVM to get feature importance
        try:
            linear_svm = SVC(kernel='linear', random_state=42)
            linear_svm.fit(X_train_scaled, y_train)
            feature_coeffs = np.abs(linear_svm.coef_[0])
            feature_importance = {
                'noise': feature_coeffs[0],
                'clean': feature_coeffs[1],
                'grad': feature_coeffs[2]
            }
        except:
            feature_importance = {'noise': 1.0, 'clean': 1.0, 'grad': 1.0}

    print(f"\nFeature Importance:")
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")

    # Save model
    if save:
        model_filename = f'{assessor_path}/best_model_{best_model_name}.joblib'
        joblib.dump(best_model, model_filename)
        scaler_filename = f'{assessor_path}/scaler_{best_model_name}.joblib'
        joblib.dump(scaler, scaler_filename)

        # Save additional metadata
        metadata = {
            'best_model_name': best_model_name,
            'threshold_strategy': strategy_name,
            'threshold_value': threshold,
            'feature_importance': feature_importance,
            'layer_selection': layer_selection,
            'model_params': best_model.get_params() if hasattr(best_model, 'get_params') else {}
        }

        import json
        metadata_filename = f'{assessor_path}/model_metadata_{best_model_name}.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\nModel saved:")
        print(f"  Model: {model_filename}")
        print(f"  Scaler: {scaler_filename}")
        print(f"  Metadata: {metadata_filename}")

    # Return comprehensive results
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'all_models': models,
        'results_comparison': results_comparison,
        'scaler': scaler,
        'threshold_used': threshold,
        'threshold_strategy': strategy_name,
        'train_accuracy': results_comparison[best_model_name]['train_acc'],
        'test_accuracy': results_comparison[best_model_name]['test_acc'],
        'feature_importance': feature_importance,
        'class_distribution': class_dist,
        'imbalance_ratio': imbalance_ratio,
        'classification_report': results_comparison[best_model_name]['report']
    }


def load_and_evaluate_model(importance_scores, layer_selection="", assessor_path="./"):
    """
    Load existing model and evaluate on new data.
    """
    import joblib
    import json
    import os
    from glob import glob

    print("\n" + "="*60)
    print("LOADING EXISTING MODEL")
    print("="*60)

    # Find available models
    model_files = glob(f'{assessor_path}/best_model_*.joblib')
    if not model_files:
        raise FileNotFoundError(f"No saved models found in {assessor_path}/")

    # For simplicity, use the first model found (you can modify this logic)
    model_file = model_files[0]
    model_name = os.path.basename(model_file).replace(
        'best_model_', '').replace('.joblib', '')

    print(f"Loading model: {model_file}")

    # Load model and scaler
    try:
        model = joblib.load(model_file)
        scaler_file = f'{assessor_path}/scaler_{model_name}.joblib'
        scaler = joblib.load(scaler_file)
        print(f"Loaded scaler: {scaler_file}")
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        return None

    # Load metadata if available
    metadata_file = f'{assessor_path}/model_metadata_{model_name}.json'
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata_file}")
        print(
            f"Original threshold strategy: {metadata.get('threshold_strategy', 'unknown')}")
        print(
            f"Original threshold value: {metadata.get('threshold_value', 'unknown')}")

    # ============================================
    # PREPARE NEW DATA FOR EVALUATION
    # ============================================
    print("\nPreparing new data for evaluation...")

    # Extract features and labels (same as training)
    labels_dict = importance_scores['trigger']
    feature_noise_dict = importance_scores['noise']
    feature_clean_dict = importance_scores['clean']
    feature_grad_dict = importance_scores['grad']

    all_labels = []
    all_features = []

    layer_names = list(labels_dict.keys())
    layer_names = list(filter(lambda ln: ln == layer_selection,
                       layer_names)) if layer_selection else layer_names

    for layer_name in layer_names:
        layer_labels = labels_dict[layer_name]
        layer_noise = feature_noise_dict[layer_name]
        layer_clean = feature_clean_dict[layer_name]
        layer_grad = feature_grad_dict[layer_name]

        # Convert to numpy
        if isinstance(layer_labels, torch.Tensor):
            layer_labels = layer_labels.cpu().numpy()
        if isinstance(layer_noise, torch.Tensor):
            layer_noise = layer_noise.cpu().numpy()
        if isinstance(layer_clean, torch.Tensor):
            layer_clean = layer_clean.cpu().numpy()
        if isinstance(layer_grad, torch.Tensor):
            layer_grad = layer_grad.cpu().numpy()

        # Flatten
        layer_labels_flat = layer_labels.flatten()
        layer_noise_flat = layer_noise.flatten()
        layer_clean_flat = layer_clean.flatten()
        layer_grad_flat = layer_grad.flatten()

        # Ensure same length
        min_len = min(len(layer_labels_flat), len(layer_noise_flat),
                      len(layer_clean_flat), len(layer_grad_flat))

        layer_labels_flat = layer_labels_flat[:min_len]
        layer_noise_flat = layer_noise_flat[:min_len]
        layer_clean_flat = layer_clean_flat[:min_len]
        layer_grad_flat = layer_grad_flat[:min_len]

        layer_features = np.column_stack(
            [layer_noise_flat, layer_clean_flat, layer_grad_flat])

        all_labels.extend(layer_labels_flat)
        all_features.extend(layer_features)

    X = np.array(all_features)
    y = np.array(all_labels)

    if len(X) == 0:
        print("No data available for evaluation")
        return None

    # Apply same threshold strategy as training
    threshold = metadata.get('threshold_value')
    threshold_strategy = metadata.get('threshold_strategy', 'percentile_80')

    if threshold is None:
        print("No threshold found in metadata, computing percentile_80...")
        threshold = np.percentile(y, 80)

    print(f"Using threshold: {threshold:.4f} (strategy: {threshold_strategy})")
    y_binary = (y > threshold).astype(int)

    # Analyze data distribution
    unique, counts = np.unique(y_binary, return_counts=True)
    class_dist = dict(zip(unique, counts))
    imbalance_ratio = max(counts) / min(counts) if len(counts) > 1 else 1.0

    print(f"Data distribution: {class_dist}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    # ============================================
    # EVALUATE MODEL ON NEW DATA
    # ============================================
    print("\n" + "="*60)
    print("EVALUATING LOADED MODEL")
    print("="*60)

    # Scale features using loaded scaler
    X_scaled = scaler.transform(X)

    # Make predictions
    y_pred = model.predict(X_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_binary, y_pred)
    report_dict = classification_report(y_binary, y_pred, output_dict=True)
    report_str = classification_report(y_binary, y_pred)

    # Extract metrics
    f1_macro = report_dict['macro avg']['f1-score']
    f1_class_0 = report_dict['0']['f1-score'] if '0' in report_dict else 0
    f1_class_1 = report_dict['1']['f1-score'] if '1' in report_dict else 0
    precision_1 = report_dict['1']['precision'] if '1' in report_dict else 0
    recall_1 = report_dict['1']['recall'] if '1' in report_dict else 0

    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"Class 0 - F1: {f1_class_0:.4f}")
    print(
        f"Class 1 - F1: {f1_class_1:.4f}, Precision: {precision_1:.4f}, Recall: {recall_1:.4f}")
    print(f"\nDetailed Classification Report:")
    print(report_str)

    # Get feature importance from loaded metadata
    feature_importance = metadata.get(
        'feature_importance', {'noise': 1.0, 'clean': 1.0, 'grad': 1.0})
    print(f"\nFeature Importance (from training):")
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")

    # Get prediction probabilities if available
    prediction_probs = None
    if hasattr(model, 'predict_proba'):
        try:
            prediction_probs = model.predict_proba(X_scaled)
            print(f"\nPrediction confidence available")
        except:
            print(f"\nPrediction probabilities not available")

    # Return results in same format as training
    results_comparison = {
        model_name: {
            'train_acc': 'N/A (loaded model)',
            'test_acc': accuracy,
            'f1_macro': f1_macro,
            'f1_class_0': f1_class_0,
            'f1_class_1': f1_class_1,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'report': report_str
        }
    }

    return {
        'best_model': model,
        'best_model_name': model_name,
        'all_models': {model_name: model},
        'results_comparison': results_comparison,
        'scaler': scaler,
        'threshold_used': threshold,
        'threshold_strategy': threshold_strategy,
        'train_accuracy': 'N/A (loaded model)',
        'test_accuracy': accuracy,
        'feature_importance': feature_importance,
        'class_distribution': class_dist,
        'imbalance_ratio': imbalance_ratio,
        'classification_report': report_str,
        'predictions': y_pred,
        'prediction_probabilities': prediction_probs,
        'true_labels': y_binary,
        'metadata': metadata
    }


def find_and_select_model(assessor_path="./"):
    """
    Helper function to list and select from available models.
    """
    import os
    from glob import glob

    model_files = glob(f'{assessor_path}/best_model_*.joblib')
    if not model_files:
        print(f"No saved models found in {assessor_path}/")
        return None

    print("Available models:")
    for i, model_file in enumerate(model_files):
        model_name = os.path.basename(model_file).replace(
            'best_model_', '').replace('.joblib', '')
        metadata_file = f'{assessor_path}/model_metadata_{model_name}.json'

        if os.path.exists(metadata_file):
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(
                f"  {i}: {model_name} (threshold: {metadata.get('threshold_strategy', 'unknown')})")
        else:
            print(f"  {i}: {model_name} (no metadata)")

    return model_files


def analyze_neurons_by_gradient_steps(importance_scores_clean, importance_scores_noise,
                                      importance_scores_trigger, importance_scores_grad,
                                      layer_selection="", n_steps=5):
    """
    Analyze neurons by filtering in gradient-based steps and computing mutual information
    between noise/clean features (X) and trigger features (Y).

    Args:
        importance_scores_clean: Dict of clean activation scores per layer
        importance_scores_noise: Dict of noise activation scores per layer  
        importance_scores_trigger: Dict of trigger activation scores per layer (now discrete variable, not label)
        importance_scores_grad: Dict of gradient magnitudes per layer
        layer_selection: Specific layer to analyze (empty string for all layers)
        n_steps: Number of gradient threshold steps (default: 5)

    Returns:
        Dict containing analysis results for each layer and step
    """
    results = {}

    # Get layer names
    layer_names = list(importance_scores_trigger.keys())
    if layer_selection:
        layer_names = [ln for ln in layer_names if ln == layer_selection]

    print("=" * 80)
    print("GRADIENT-BASED NEURON FILTERING ANALYSIS")
    print("Calculating MI(X, Y) where X ∈ {clean, noise} and Y = trigger")
    print("=" * 80)

    for layer_name in layer_names:
        print(f"\n{'='*80}")
        print(f"Layer: {layer_name}")
        print(f"{'='*80}")

        # Extract data for this layer
        layer_trigger = importance_scores_trigger[layer_name]
        layer_noise = importance_scores_noise[layer_name]
        layer_clean = importance_scores_clean[layer_name]
        layer_grad = importance_scores_grad[layer_name]

        # Convert to numpy if needed
        if isinstance(layer_trigger, torch.Tensor):
            layer_trigger = layer_trigger.cpu().numpy()
        if isinstance(layer_noise, torch.Tensor):
            layer_noise = layer_noise.cpu().numpy()
        if isinstance(layer_clean, torch.Tensor):
            layer_clean = layer_clean.cpu().numpy()
        if isinstance(layer_grad, torch.Tensor):
            layer_grad = layer_grad.cpu().numpy()

        # Flatten all arrays
        layer_trigger_flat = layer_trigger.flatten()
        layer_noise_flat = layer_noise.flatten()
        layer_clean_flat = layer_clean.flatten()
        layer_grad_flat = layer_grad.flatten()

        # Ensure same length
        min_len = min(len(layer_trigger_flat), len(layer_noise_flat),
                      len(layer_clean_flat), len(layer_grad_flat))

        layer_trigger_flat = layer_trigger_flat[:min_len]
        layer_noise_flat = layer_noise_flat[:min_len]
        layer_clean_flat = layer_clean_flat[:min_len]
        layer_grad_flat = layer_grad_flat[:min_len]

        # Compute gradient percentile thresholds
        grad_percentiles = np.linspace(
            0, 100, n_steps + 1)[1:]  # [20, 40, 60, 80, 100]
        grad_thresholds = np.percentile(layer_grad_flat, grad_percentiles)

        print(f"\nTotal neurons: {len(layer_grad_flat)}")
        print(f"\nGradient thresholds (percentiles {grad_percentiles}):")
        for i, (percentile, threshold) in enumerate(zip(grad_percentiles, grad_thresholds)):
            print(
                f"  Step {i+1}: {percentile:.1f}th percentile = {threshold:.6f}")

        # Print trigger statistics
        print(f"\nTrigger statistics:")
        print(f"  Mean: {layer_trigger_flat.mean():.6f}")
        print(f"  Std: {layer_trigger_flat.std():.6f}")
        print(f"  Min: {layer_trigger_flat.min():.6f}")
        print(f"  Max: {layer_trigger_flat.max():.6f}")

        # Initialize results for this layer
        layer_results = {
            'grad_thresholds': grad_thresholds,
            'grad_percentiles': grad_percentiles,
            'steps': []
        }

        # Analyze each gradient threshold step
        print(f"\n{'Step':<6} {'Threshold':<12} {'Neurons':<10} {'%Keep':<8} "
              f"{'MI(Clean,Trig)':<15} {'MI(Noise,Trig)':<15} {'MI(Noise)-MI(Clean)':<20}")
        print("-" * 100)

        # Step 0: All neurons (baseline)
        mi_clean_baseline = compute_mutual_information_safe(
            layer_clean_flat, layer_trigger_flat
        )
        mi_noise_baseline = compute_mutual_information_safe(
            layer_noise_flat, layer_trigger_flat
        )
        mi_diff_baseline = mi_noise_baseline - mi_clean_baseline if (
            mi_noise_baseline is not None and mi_clean_baseline is not None) else None

        if mi_diff_baseline is not None:
            print(f"{'0':<6} {'All':<12} {len(layer_grad_flat):<10} {'100.0':<8} "
                  f"{mi_clean_baseline:<15.6f} {mi_noise_baseline:<15.6f} {mi_diff_baseline:<20.6f}")
        else:
            print(f"{'0':<6} {'All':<12} {len(layer_grad_flat):<10} {'100.0':<8} "
                  f"{'N/A':<15} {'N/A':<15} {'N/A':<20}")

        step_result = {
            'step': 0,
            'threshold': None,
            'percentile': 0,
            'n_neurons_kept': len(layer_grad_flat),
            'pct_kept': 100.0,
            'mi_clean_trigger': mi_clean_baseline,
            'mi_noise_trigger': mi_noise_baseline,
            'mi_diff': mi_diff_baseline,
            'mask': np.ones(len(layer_grad_flat), dtype=bool)
        }
        layer_results['steps'].append(step_result)

        # Analyze each threshold step
        for step_idx, (percentile, threshold) in enumerate(zip(grad_percentiles, grad_thresholds), 1):
            # Create mask for neurons above this gradient threshold
            mask = layer_grad_flat >= threshold
            n_kept = mask.sum()
            pct_kept = (n_kept / len(layer_grad_flat)) * 100

            if n_kept == 0:
                print(f"{step_idx:<6} {threshold:<12.6f} {n_kept:<10} {pct_kept:<8.2f} "
                      f"{'N/A':<15} {'N/A':<15} {'N/A':<20}")
                step_result = {
                    'step': step_idx,
                    'threshold': threshold,
                    'percentile': percentile,
                    'n_neurons_kept': n_kept,
                    'pct_kept': pct_kept,
                    'mi_clean_trigger': None,
                    'mi_noise_trigger': None,
                    'mi_diff': None,
                    'mask': mask
                }
                layer_results['steps'].append(step_result)
                continue

            # Filter neurons based on gradient threshold
            clean_filtered = layer_clean_flat[mask]
            noise_filtered = layer_noise_flat[mask]
            trigger_filtered = layer_trigger_flat[mask]

            # Compute mutual information: MI(X, Y) where X is clean/noise, Y is trigger
            mi_clean_trigger = compute_mutual_information_safe(
                clean_filtered, trigger_filtered
            )
            mi_noise_trigger = compute_mutual_information_safe(
                noise_filtered, trigger_filtered
            )

            if mi_clean_trigger is not None and mi_noise_trigger is not None:
                mi_diff = mi_noise_trigger - mi_clean_trigger
                print(f"{step_idx:<6} {threshold:<12.6f} {n_kept:<10} {pct_kept:<8.2f} "
                      f"{mi_clean_trigger:<15.6f} {mi_noise_trigger:<15.6f} {mi_diff:<20.6f}")
            else:
                mi_diff = None
                print(f"{step_idx:<6} {threshold:<12.6f} {n_kept:<10} {pct_kept:<8.2f} "
                      f"{'N/A':<15} {'N/A':<15} {'N/A':<20}")

            step_result = {
                'step': step_idx,
                'threshold': threshold,
                'percentile': percentile,
                'n_neurons_kept': n_kept,
                'pct_kept': pct_kept,
                'mi_clean_trigger': mi_clean_trigger,
                'mi_noise_trigger': mi_noise_trigger,
                'mi_diff': mi_diff,
                'mask': mask
            }
            layer_results['steps'].append(step_result)

        # Find best step (highest MI difference: MI(noise,trigger) - MI(clean,trigger))
        valid_steps = [s for s in layer_results['steps']
                       if s['mi_diff'] is not None]
        if valid_steps:
            best_step = max(valid_steps, key=lambda s: s['mi_diff'])
            print(f"\n{'*'*80}")
            print(f"BEST STEP for {layer_name}: Step {best_step['step']}")
            if best_step['threshold'] is not None:
                print(
                    f"  Threshold: {best_step['threshold']:.6f} ({best_step['percentile']:.1f}th percentile)")
            print(
                f"  Neurons kept: {best_step['n_neurons_kept']} ({best_step['pct_kept']:.2f}%)")
            print(f"  MI(Clean, Trigger): {best_step['mi_clean_trigger']:.6f}")
            print(f"  MI(Noise, Trigger): {best_step['mi_noise_trigger']:.6f}")
            print(
                f"  MI(Noise, Trigger) - MI(Clean, Trigger): {best_step['mi_diff']:.6f}")
            print(f"{'*'*80}")

            layer_results['best_step'] = best_step
        else:
            print(f"\n{'*'*80}")
            print(f"WARNING: No valid MI calculations for {layer_name}")
            print(f"{'*'*80}")

        results[layer_name] = layer_results

    # Summary across all layers
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL LAYERS")
    print(f"{'='*80}")

    print(f"\n{'Layer':<30} {'Best Step':<12} {'Neurons':<10} {'%Keep':<8} "
          f"{'MI(Clean,Trig)':<15} {'MI(Noise,Trig)':<15} {'MI Diff':<12}")
    print("-" * 110)

    for layer_name, layer_res in results.items():
        if 'best_step' in layer_res:
            best = layer_res['best_step']
            print(f"{layer_name:<30} {best['step']:<12} {best['n_neurons_kept']:<10} "
                  f"{best['pct_kept']:<8.2f} {best['mi_clean_trigger']:<15.6f} "
                  f"{best['mi_noise_trigger']:<15.6f} {best['mi_diff']:<12.6f}")
        else:
            print(f"{layer_name:<30} {'N/A':<12} {'N/A':<10} {'N/A':<8} "
                  f"{'N/A':<15} {'N/A':<15} {'N/A':<12}")

    return results


def analyze_neurons_by_gradient_percentile_aggregated(importance_scores_clean, importance_scores_noise,
                                                      importance_scores_trigger, importance_scores_grad,
                                                      percentile=90):
    """
    Aggregate all layer neurons and analyze by gradient-based filtering.
    Computes mutual information between all pairs: clean, noise, and trigger.

    Args:
        importance_scores_clean: Dict of clean activation scores per layer
        importance_scores_noise: Dict of noise activation scores per layer  
        importance_scores_trigger: Dict of trigger activation scores per layer
        importance_scores_grad: Dict of gradient magnitudes per layer
        percentile: Gradient percentile threshold (default: 90)

    Returns:
        Dict containing aggregated analysis results with all MI combinations
    """

    print("=" * 80)
    print("AGGREGATED GRADIENT-BASED NEURON FILTERING ANALYSIS")
    print(f"Calculating MI for all pairs: (Clean, Noise, Trigger)")
    print(f"Using {percentile}th percentile gradient threshold")
    print("=" * 80)

    # Aggregate all neurons across all layers
    all_trigger = []
    all_noise = []
    all_clean = []
    all_grad = []

    layer_names = list(importance_scores_trigger.keys())

    print(f"\nAggregating neurons from {len(layer_names)} layers...")

    for layer_name in layer_names:
        # Extract data for this layer
        layer_trigger = importance_scores_trigger[layer_name]
        layer_noise = importance_scores_noise[layer_name]
        layer_clean = importance_scores_clean[layer_name]
        layer_grad = importance_scores_grad[layer_name]

        # Convert to numpy if needed
        if isinstance(layer_trigger, torch.Tensor):
            layer_trigger = layer_trigger.cpu().numpy()
        if isinstance(layer_noise, torch.Tensor):
            layer_noise = layer_noise.cpu().numpy()
        if isinstance(layer_clean, torch.Tensor):
            layer_clean = layer_clean.cpu().numpy()
        if isinstance(layer_grad, torch.Tensor):
            layer_grad = layer_grad.cpu().numpy()

        # Flatten and append
        all_trigger.append(layer_trigger.flatten())
        all_noise.append(layer_noise.flatten())
        all_clean.append(layer_clean.flatten())
        all_grad.append(layer_grad.flatten())

        print(f"  {layer_name}: {layer_trigger.flatten().shape[0]} neurons")

    # Concatenate all layers
    all_trigger = np.concatenate(all_trigger)
    all_noise = np.concatenate(all_noise)
    all_clean = np.concatenate(all_clean)
    all_grad = np.concatenate(all_grad)

    total_neurons = len(all_grad)
    print(f"\nTotal aggregated neurons: {total_neurons}")

    # Compute gradient threshold
    grad_threshold = np.percentile(all_grad, percentile)

    print(f"\nGradient statistics:")
    print(f"  Mean: {all_grad.mean():.6f}")
    print(f"  Std: {all_grad.std():.6f}")
    print(f"  Min: {all_grad.min():.6f}")
    print(f"  Max: {all_grad.max():.6f}")
    print(f"  {percentile}th percentile threshold: {grad_threshold:.6f}")

    print(f"\nTrigger statistics:")
    print(f"  Mean: {all_trigger.mean():.6f}")
    print(f"  Std: {all_trigger.std():.6f}")
    print(f"  Min: {all_trigger.min():.6f}")
    print(f"  Max: {all_trigger.max():.6f}")

    # Helper function to compute all MI pairs
    def compute_all_mi_pairs(clean, noise, trigger, label=""):
        """Compute all pairwise mutual information"""
        print(f"\n{label}Mutual Information Matrix:")
        print(f"{'':>12} {'Clean':>12} {'Noise':>12} {'Trigger':>12}")
        print("-" * 50)

        mi_results = {}

        # MI(Clean, Noise)
        mi_clean_noise = compute_mutual_information_safe(clean, noise)
        mi_results['mi_clean_noise'] = mi_clean_noise

        # MI(Clean, Trigger)
        mi_clean_trigger = compute_mutual_information_safe(clean, trigger)
        mi_results['mi_clean_trigger'] = mi_clean_trigger

        # MI(Noise, Trigger)
        mi_noise_trigger = compute_mutual_information_safe(noise, trigger)
        mi_results['mi_noise_trigger'] = mi_noise_trigger

        # Print matrix
        print(f"{'Clean':>12} {'-':>12} {mi_clean_noise if mi_clean_noise else 'N/A':>12} {mi_clean_trigger if mi_clean_trigger else 'N/A':>12}")
        print(f"{'Noise':>12} {mi_clean_noise if mi_clean_noise else 'N/A':>12} {'-':>12} {mi_noise_trigger if mi_noise_trigger else 'N/A':>12}")
        print(f"{'Trigger':>12} {mi_clean_trigger if mi_clean_trigger else 'N/A':>12} {mi_noise_trigger if mi_noise_trigger else 'N/A':>12} {'-':>12}")

        # Compute differences
        if all(v is not None for v in [mi_clean_trigger, mi_noise_trigger, mi_clean_noise]):
            mi_diff_trigger = mi_noise_trigger - mi_clean_trigger
            mi_results['mi_diff_trigger'] = mi_diff_trigger

            print(f"\nKey Differences:")
            print(
                f"  MI(Noise, Trigger) - MI(Clean, Trigger): {mi_diff_trigger:+.6f}")
            print(f"  MI(Clean, Noise): {mi_clean_noise:.6f}")
        else:
            mi_results['mi_diff_trigger'] = None

        return mi_results

    # Step 0: Baseline (all neurons)
    print(f"\n{'='*80}")
    print("STEP 0: BASELINE (All Neurons)")
    print(f"{'='*80}")
    print(f"Neurons: {total_neurons} (100.0%)")

    baseline_mi = compute_all_mi_pairs(all_clean, all_noise, all_trigger, "")

    # Step 1: Filter by gradient threshold
    print(f"\n{'='*80}")
    print(f"STEP 1: FILTERED (>= {percentile}th percentile gradient)")
    print(f"{'='*80}")

    mask = all_grad >= grad_threshold
    n_kept = mask.sum()
    pct_kept = (n_kept / total_neurons) * 100

    print(f"Neurons kept: {n_kept} ({pct_kept:.2f}%)")
    print(f"Neurons removed: {total_neurons - n_kept} ({100 - pct_kept:.2f}%)")

    if n_kept == 0:
        print("WARNING: No neurons kept after filtering!")
        filtered_mi = {k: None for k in baseline_mi.keys()}
    else:
        # Filter neurons
        clean_filtered = all_clean[mask]
        noise_filtered = all_noise[mask]
        trigger_filtered = all_trigger[mask]

        # Compute MI on filtered neurons
        filtered_mi = compute_all_mi_pairs(
            clean_filtered, noise_filtered, trigger_filtered, "")

        # Compare with baseline
        print(f"\n{'='*80}")
        print("CHANGE FROM BASELINE:")
        print(f"{'='*80}")

        for key in ['mi_clean_noise', 'mi_clean_trigger', 'mi_noise_trigger', 'mi_diff_trigger']:
            if baseline_mi.get(key) is not None and filtered_mi.get(key) is not None:
                delta = filtered_mi[key] - baseline_mi[key]
                print(
                    f"  Δ{key}: {delta:+.6f} ({baseline_mi[key]:.6f} → {filtered_mi[key]:.6f})")

    # Compile results
    results = {
        'total_neurons': total_neurons,
        'percentile': percentile,
        'threshold': grad_threshold,
        'baseline': {
            'n_neurons': total_neurons,
            **baseline_mi
        },
        'filtered': {
            'n_neurons': n_kept,
            'pct_kept': pct_kept,
            **filtered_mi,
            'mask': mask if n_kept > 0 else None
        }
    }

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

    return results


def compute_mutual_information_safe(X, y, n_bins=20, min_samples=10):
    """
    Safely compute mutual information with fallback strategies.

    Args:
        X: Continuous feature values (1D array)
        y: Continuous target values (1D array)
        n_bins: Number of bins for discretization fallback
        min_samples: Minimum samples required for KNN-based MI

    Returns:
        Mutual information score or None if computation fails
    """
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.metrics import mutual_info_score

    # Check if we have enough samples
    n_samples = len(X)

    if n_samples < 2:
        return None

    # Try KNN-based MI estimation if we have enough samples
    if n_samples >= min_samples:
        try:
            # Adjust n_neighbors based on sample size
            n_neighbors = min(3, n_samples - 1)
            X_2d = X.reshape(-1, 1)
            mi = mutual_info_regression(
                X_2d, y, n_neighbors=n_neighbors, random_state=42)
            return mi[0]
        except Exception as e:
            print(
                f"    Warning: KNN-based MI failed ({str(e)}), falling back to binning method")

    # Fallback: Use binning method
    try:
        # Adjust number of bins based on sample size
        actual_bins = min(n_bins, max(2, n_samples // 5))

        # Check if X or y have zero variance
        if np.std(X) < 1e-10 or np.std(y) < 1e-10:
            return 0.0

        # Discretize both X and Y into bins
        X_binned = np.digitize(X, bins=np.linspace(
            X.min(), X.max(), actual_bins))
        y_binned = np.digitize(y, bins=np.linspace(
            y.min(), y.max(), actual_bins))

        # Compute mutual information between discrete variables
        mi = mutual_info_score(X_binned, y_binned)
        return mi
    except Exception as e:
        print(f"    Warning: Binning-based MI also failed ({str(e)})")
        return None


def compute_mutual_information_binned(X, y, n_bins=20):
    """
    Compute mutual information by binning both X and Y into discrete categories.

    Args:
        X: Continuous feature values (1D array)
        y: Continuous target values (1D array)
        n_bins: Number of bins for discretization

    Returns:
        Mutual information score
    """
    from sklearn.metrics import mutual_info_score

    # Adjust bins based on sample size
    n_samples = len(X)
    actual_bins = min(n_bins, max(2, n_samples // 5))

    # Discretize both X and Y into bins
    X_binned = np.digitize(X, bins=np.linspace(X.min(), X.max(), actual_bins))
    y_binned = np.digitize(y, bins=np.linspace(y.min(), y.max(), actual_bins))

    # Compute mutual information between discrete variables
    mi = mutual_info_score(X_binned, y_binned)

    return mi


def compute_contrastive_loss(trojan_features, clean_features, labels, target_label, temperature):
    """
    Compute contrastive loss to make trojan samples with target label similar
    and different from other classes
    """
    batch_size = trojan_features.size(0)

    # Create masks for positive and negative pairs
    target_mask = (labels == target_label).float()

    # Compute similarity matrix
    similarity_matrix = torch.matmul(
        trojan_features, trojan_features.T) / temperature

    # Create positive and negative masks
    # Positive: samples that should be similar (both should predict target_label)
    positive_mask = torch.eye(batch_size, device=trojan_features.device)

    # Negative: samples from different original classes
    negative_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    negative_mask.fill_diagonal_(0)

    # Compute contrastive loss
    exp_sim = torch.exp(similarity_matrix)

    # Positive pairs loss
    pos_sim = similarity_matrix * positive_mask

    # Negative pairs normalization
    neg_exp_sim = exp_sim * negative_mask
    neg_sum = neg_exp_sim.sum(dim=1, keepdim=True)

    # InfoNCE-style loss
    contrastive_loss = - \
        torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + neg_sum + 1e-8))
    contrastive_loss = contrastive_loss * positive_mask

    return contrastive_loss.sum() / (positive_mask.sum() + 1e-8)


def compute_contrastive_loss_v2(trojan_features, clean_features, labels, target_label, temperature):
    """
    Alternative contrastive loss: encourage trojan samples to be similar to each other
    when they should predict the same target label
    """
    batch_size = trojan_features.size(0)

    # All trojan samples should be similar since they all target the same label
    similarity_matrix = torch.matmul(
        trojan_features, trojan_features.T) / temperature

    # Create positive pairs (all pairs should be positive for target label)
    positive_mask = torch.ones(
        batch_size, batch_size, device=trojan_features.device)
    positive_mask.fill_diagonal_(0)

    # Compute loss to make all trojan features similar
    exp_sim = torch.exp(similarity_matrix)
    positive_sim = similarity_matrix * positive_mask

    loss = -torch.log(torch.exp(positive_sim) /
                      (exp_sim.sum(dim=1, keepdim=True) + 1e-8))
    loss = loss * positive_mask

    return loss.sum() / (positive_mask.sum() + 1e-8)


def inversion_train(model, target_label, train_loader, param, device: str | torch.device = "cuda", pbar: tqdm = None):
    if pbar is not None:
        desc = pbar.desc
        pbar.set_description(desc + f" | Inversion Label {target_label}")
    else:
        print("Processing label: {}".format(target_label))
    width, height = param["image_size"]

    trigger = torch.rand(
        (param["CWH"][0], param["CWH"][1], param["CWH"][2]), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]
    # Temperature for contrastive loss
    temperature = param.get("temperature", 0.1)
    contrastive_weight = param.get("contrastive_weight", 1.0)

    min_norm = np.inf
    min_norm_count = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [{"params": trigger}, {"params": mask}], lr=0.005)

    model.to(device)
    model.eval()

    # Extract feature extractor (assuming model has a feature extraction part)
    if hasattr(model, 'features'):
        feature_extractor = model.features
    elif hasattr(model, 'backbone'):
        feature_extractor = model.backbone
    elif hasattr(model, 'forward_feature_unflatten'):
        feature_extractor = model.forward_feature_unflatten
    else:
        # If no explicit feature extractor, use all layers except the last classifier
        feature_extractor = nn.Sequential(*list(model.children())[:-1])

    for epoch in range(Epochs):
        norm = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            # Create trojan images
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * \
                images + torch.unsqueeze(mask, dim=0) * trigger

            # Get predictions and features
            y_pred = model(trojan_images)
            clean_pred = model(images)

            # Extract features for contrastive learning
            with torch.no_grad():
                clean_features = feature_extractor(images)
                clean_features = F.adaptive_avg_pool2d(
                    clean_features, (1, 1)).flatten(1)
                clean_features = F.normalize(clean_features, dim=1)

            trojan_features = feature_extractor(trojan_images)
            trojan_features = F.adaptive_avg_pool2d(
                trojan_features, (1, 1)).flatten(1)
            trojan_features = F.normalize(trojan_features, dim=1)

            # Target labels for trojan images
            y_target = torch.full(
                (y_pred.size(0),), target_label, dtype=torch.long).to(device)
            correct += (y_pred.argmax(1) == y_target).sum().item()
            total += y_pred.size(0)

            # Classification loss for trojan images
            classification_loss = criterion(y_pred, y_target)

            # Contrastive loss components
            contrastive_loss = compute_contrastive_loss(
                trojan_features, clean_features, labels, target_label, temperature
            )

            # Feature consistency loss (optional: encourage trojan features to be different from clean)
            feature_distance_loss = - \
                F.cosine_similarity(trojan_features, clean_features).mean()

            # Total loss
            total_loss = (classification_loss +
                          contrastive_weight * contrastive_loss +
                          0.1 * feature_distance_loss +
                          lamda * torch.sum(torch.abs(mask)))

            total_loss.backward()
            optimizer.step()

            # Clip values
            with torch.no_grad():
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))

        # if epoch % 10 == 0:
        #     log.debug_(f"Label {target_label} Epoch {epoch}: Loss {total_loss.item():.4f}, Acc {correct/total*100:.2f}%, Mask Norm {norm.item():.4f}")

        # Early stopping
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1
        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()


def reverse_engineer(dataset, model, param, device: str | torch.device = "cuda", pbar: tqdm = None):

    norm_list = []
    trigger_list = []
    mask_list = []

    test_dataset_noise = NoiseDataset(dataset, list(
        range(param['num_classes'])), attackportion=1.0)
    test_loader_noise = DataLoader(
        test_dataset_noise, batch_size=128, shuffle=False, num_workers=4)
    # for label in range(param["num_classes"]):
    for label in param["target_label"]:
        trigger, mask = inversion_train(
            model, label, test_loader_noise, param, device=device, pbar=pbar)
        norm_list.append(mask.sum().item())

        trigger = trigger.cpu().detach().numpy()
        # trigger = np.transpose(trigger, (1,2,0))

        mask = mask.cpu().detach().numpy()

        trigger_list.append(trigger)
        mask_list.append(mask)

    return trigger_list, mask_list


def realTriggerImportance(model, model_state_dict_list, dataset_test, trigger_model_idx, mask, trigger, target_label, pos_choice, num_class, img_channel, img_size, device: str | torch.device = 'cuda'):
    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    dataset_test_list = [FilteredDataset(
        ShrinkedDataset(dataset_test, frac=1.0), [label]) for label in range(num_class)]

    dataset_test_list_corrupt = [CorruptedDataset(VoidDataset(FilteredDataset(
        ShrinkedDataset(dataset_test, frac=1.0), [label])),
        [target_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size, mask=mask) for label in range(num_class)]

    model_state_dicts = dict([(w_local_i[0], w_local_i[1])
                             for w_local_i in model_state_dict_list])
    w_dict = [w_local for idx, w_local,
              _ in model_state_dict_list if idx == trigger_model_idx][0]

    tss_mask_trigger = getTssImpt(
        model, w_dict, dataset_test_list_corrupt, p=0.3, device=device)[0]
    tss_impt_dict = getTssCommon(getTssImpt(
        model, w_dict, dataset_test_list, device=device), model, w_dict)
    tss_impt_dict_all = {
        idx: getTssCommon(getTssImpt(
            model, model_dict, dataset_test_list, p=0.3, device=device), model, model_dict, 1)
        for idx, model_dict in model_state_dicts.items()
    }
    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]
                            for idx in model_state_dicts.keys()]).mean(0)
        for layer in layer_names
    }

    # compare the two importance scores
    tss_impt_dict_tp = {}
    tss_impt_dict_tn = {}
    tss_impt_dict_fp = {}
    for layer in tss_mask_trigger:
        tss_impt_dict_tp[layer] = copy.deepcopy(tss_mask_trigger[layer])
        tss_impt_dict_tn[layer] = copy.deepcopy(tss_mask_trigger[layer])
        tss_impt_dict_fp[layer] = copy.deepcopy(tss_mask_trigger[layer])
        # Fixed: Use torch.logical_and for floating-point tensors
        mask_condition = torch.logical_and(
            tss_mask_trigger[layer] > 0,
            torch.logical_and(
                tss_impt_dict[layer] > 0, inter_model_tss[layer] > 0)
        )
        tss_impt_dict_tp[layer][mask_condition] = 1
        # Set other values to 0
        tss_impt_dict_tp[layer][~mask_condition] = 0
        mask_condition = torch.logical_and(
            tss_mask_trigger[layer] <= 0,
            torch.logical_and(
                tss_impt_dict[layer] <= 0, inter_model_tss[layer] <= 0)
        )
        tss_impt_dict_tn[layer][mask_condition] = 1
        # Set other values to 0
        tss_impt_dict_tn[layer][~mask_condition] = 0
        mask_condition = torch.logical_and(
            tss_mask_trigger[layer] > 0,
            torch.logical_or(tss_impt_dict[layer]
                             <= 0, inter_model_tss[layer] <= 0)
        )
        tss_impt_dict_fp[layer][mask_condition] = 1
        # Set other values to 0
        tss_impt_dict_fp[layer][~mask_condition] = 0

    return tss_impt_dict_tp, tss_impt_dict_tn, tss_impt_dict_fp


def noiseImportanceDiff(model, model_state_dict_list, dataset_test, trigger_model_idx, device='cuda'):
    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, (SoftMaskedLayer))]

    dataset_test_list_noise = [FilteredDataset(
        NoiseDataset(ShrinkedDataset(dataset_test, frac=10.0), victim_labels=list(range(10))), [label]) for label in range(10)]

    dataset_test_list = [FilteredDataset(
        ShrinkedDataset(dataset_test, frac=1.0), [label]) for label in range(10)]

    model_state_dicts = dict([(w_local_i[0], w_local_i[1])
                             for w_local_i in model_state_dict_list])
    w_dict = [w_local for idx, w_local,
              _ in model_state_dict_list if idx == trigger_model_idx][0]

    # tss_mask_trigger = getTssImpt_withTrigger(model, w_dict, [dataset_test], trigger, mask, p=0.3, device=device)[0]
    tss_impt_noise = getTssCommon(getTssImpt(
        model, w_dict, dataset_test_list_noise, device=device), model, w_dict)
    tss_impt_dict = getTssCommon(getTssImpt(
        model, w_dict, dataset_test_list, device=device), model, w_dict)
    tss_impt_dict_all = {
        idx: getTssCommon(getTssImpt(
            model, model_dict, dataset_test_list, p=0.3, device=device), model, model_dict, 1)
        for idx, model_dict in model_state_dicts.items()
    }
    inter_model_tss = {
        layer: torch.stack([tss_impt_dict_all[idx][layer]
                            for idx in model_state_dicts.keys()]).mean(0)
        for layer in layer_names
    }

    # compare the two importance scores
    tss_impt_dict_tp = {}
    tss_impt_dict_tn = {}
    tss_impt_dict_fp = {}
    for layer, _ in tss_impt_noise.items():
        tss_impt_dict_tp[layer] = copy.deepcopy(tss_impt_noise[layer])
        tss_impt_dict_tn[layer] = copy.deepcopy(tss_impt_noise[layer])
        tss_impt_dict_fp[layer] = copy.deepcopy(tss_impt_noise[layer])
        # Fixed: Use torch.logical_and for floating-point tensors
        mask_condition = torch.logical_and(
            tss_impt_noise[layer] > 0,
            torch.logical_and(
                tss_impt_dict[layer] > 0, inter_model_tss[layer] > 0)
        )
        tss_impt_dict_tp[layer][mask_condition] = 1
        # Set other values to 0
        tss_impt_dict_tp[layer][~mask_condition] = 0
        mask_condition = torch.logical_and(
            tss_impt_noise[layer] <= 0,
            torch.logical_and(
                tss_impt_dict[layer] <= 0, inter_model_tss[layer] <= 0)
        )
        tss_impt_dict_tn[layer][mask_condition] = 1
        # Set other values to 0
        tss_impt_dict_tn[layer][~mask_condition] = 0
        mask_condition = torch.logical_and(
            tss_impt_noise[layer] > 0,
            torch.logical_or(tss_impt_dict[layer]
                             <= 0, inter_model_tss[layer] <= 0)
        )
        tss_impt_dict_fp[layer][mask_condition] = 1
        # Set other values to 0
        tss_impt_dict_fp[layer][~mask_condition] = 0

    return tss_impt_dict_tp, tss_impt_dict_tn, tss_impt_dict_fp


def _prepare_datasets(dataset_test, frac=0.1):
    """Prepare filtered datasets for different labels and noise datasets."""
    all_labels = list(range(max(dataset_test.targets)))
    s_dataset = ShrinkedDataset(dataset_test, frac=frac)
    datasets = {
        'test_by_label': [
            FilteredDataset(s_dataset, [label])
            for label in all_labels
        ],
        'noise_by_label': [
            NoiseDataset(FilteredDataset(
                s_dataset,
                [label])
            ) for label in all_labels
        ],
        'void': [VoidDataset(s_dataset, victim_labels=all_labels)]
    }

    return datasets


def _prepare_datasets_trigger(dataset_test, mask, trigger, alpha, attack_type, target_label, pos_choice, num_class, img_channel, img_size):
    if attack_type == 'blend':
        corrupt_ds = [CorruptedDataset(VoidDataset(FilteredDataset(
            ShrinkedDataset(dataset_test, frac=1.0), [label])),
            [target_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size, mask=mask, alpha=alpha) for label in range(num_class)]
    else:
        corrupt_ds = [CorruptedDataset(VoidDataset(FilteredDataset(
            ShrinkedDataset(dataset_test, frac=1.0), [label])),
            [target_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size, mask=mask) for label in range(num_class)]

    return corrupt_ds


def _prepare_datasets_trigger_nonvoid(dataset_test, mask, trigger, alpha, attack_type, target_label, pos_choice, num_class, img_channel, img_size):
    if attack_type == 'blend':
        corrupt_ds = [CorruptedDataset(FilteredDataset(
            ShrinkedDataset(dataset_test, frac=1.0), [label]),
            [target_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size, mask=mask, alpha=alpha) for label in range(num_class)]
    else:
        corrupt_ds = [CorruptedDataset(FilteredDataset(
            ShrinkedDataset(dataset_test, frac=1.0), [label]),
            [target_label], pattern_tensor=trigger, pos_choice=pos_choice, victim_labels=range(num_class), attackportion=1, img_channel=img_channel, img_size=img_size, mask=mask) for label in range(num_class)]

    return corrupt_ds


def _prepare_datasets_noise(dataset_test, num_class):
    corrupt_ds = [NoiseDataset(FilteredDataset(
        ShrinkedDataset(dataset_test, frac=1.0), [label])) for label in range(num_class)]

    return corrupt_ds


def _calculate_importance_scores(model, trigger_model_dict, model_state_dicts, datasets,
                                 layer_names, p, device, trigger_model_idx):
    """Calculate various importance scores."""
    # Trigger importance
    # trigger_importance = getTssCommon(getTssImpt(
    #     model, trigger_model_dict, datasets['trigger'], p=p, device=device),
    #     model, trigger_model_dict
    # )

    trigger_importance_soft = getTssCommon_soft(getTssImpt(
        model, trigger_model_dict, datasets['trigger'], p=p, device=device),
        model, trigger_model_dict
    )
    trigger_importance = getTssCommon_soft_to_hard(trigger_importance_soft)

    # Noise importance
    # Using noise dataset importance as tss_impt_dict_noise, add to eval dataset importance calculation
    noise_importance = getTssCommon_soft(
        getTssImpt(model, trigger_model_dict,
                   datasets['noise_by_label'], p=p, device=device),
        model, trigger_model_dict
    )
    # noise_importance_hard = getTssCommon(
    #     getTssImpt(model, trigger_model_dict, datasets['noise_by_label'], p=p, device=device),
    #     model, trigger_model_dict, 1
    # )
    noise_importance_hard = getTssCommon_soft_to_hard(noise_importance)

    # Clean importance
    impt_mid_clean = getTssImpt(
        model, trigger_model_dict, datasets['test_by_label'], p=p, device=device)
    clean_importance = getTssCommon_soft(
        impt_mid_clean,
        model, trigger_model_dict
    )
    # clean_importance_hard = getTssCommon(
    #     impt_mid_clean,
    #     model, trigger_model_dict, 1
    # )
    clean_importance_hard = getTssCommon_soft_to_hard(clean_importance)

    # Clean importance for all models
    clean_importance_all = {
        idx: getTssCommon(
            getTssImpt(model, state_dict,
                       datasets['test_by_label'], p=p, device=device),
            model, state_dict, 1
        ) for idx, state_dict in model_state_dicts.items()
    }

    # Inter-model average importance
    inter_model_importance = {
        layer: torch.stack([clean_importance_all[idx][layer]
                           for idx in model_state_dicts.keys()]).mean(0)
        for layer in layer_names
    }

    return {
        'trigger_hard': trigger_importance,
        'trigger': trigger_importance_soft,
        'noise': noise_importance,
        'noise_hard': noise_importance_hard,
        'clean': clean_importance,  # Use the specific trigger model
        'clean_hard': clean_importance_hard,
        'inter_model': inter_model_importance,
    }


def _classify_importance_scores(importance_scores, layer_names):
    """Classify importance scores into TP/TN/FP categories for different scenarios."""
    trigger_imp = importance_scores['trigger']
    noise_imp = importance_scores['noise']
    clean_imp = importance_scores['clean']
    inter_model_imp = importance_scores['inter_model']

    # Initialize classification dictionaries
    classifications = {
        'noise': _classify_trigger_vs_noise(trigger_imp, noise_imp),
        'eval': _classify_eval_comparison(trigger_imp, clean_imp),
        'ne': _classify_noise_evaluation(trigger_imp, noise_imp, clean_imp, inter_model_imp),
        'intersection': _classify_trigger_vs_intersection(trigger_imp, clean_imp, noise_imp),
        'union': _classify_trigger_vs_union(trigger_imp, clean_imp, noise_imp),
    }

    return classifications


def _plot_importance_scores(importance_scores, layer_names, visual_path):
    """Plot importance scores for visualization."""

    for score_type, scores in importance_scores.items():
        for layer in layer_names:
            plt.figure(figsize=(10, 5))
            plt.title(f"{score_type.capitalize()} Importance - {layer}")

            # Get the scores and flatten them
            score_data = scores[layer].cpu().numpy().flatten()

            # Calculate approximate square dimensions
            total_elements = len(score_data)
            n = int(np.sqrt(total_elements))

            # Pad with zeros if necessary to make it square
            padded_size = n * n
            if total_elements < padded_size:
                score_data = np.pad(
                    score_data, (0, padded_size - total_elements), mode='constant')
            elif total_elements > padded_size:
                # Truncate if we have more elements than needed
                score_data = score_data[:padded_size]

            # Reshape into n x n
            score_matrix = score_data.reshape(n, n)

            plt.imshow(score_matrix, cmap='hot', interpolation=None)
            plt.colorbar()
            plt.savefig(f"{visual_path}/{score_type}_{layer}.png")
            plt.close()  # Added to prevent memory issues


def _create_classification_dict(base_dict):
    """Create TP/TN/FP dictionary copies."""
    return {
        'tp': copy.deepcopy(base_dict),
        'tn': copy.deepcopy(base_dict),
        'fp': copy.deepcopy(base_dict)
    }


def _apply_binary_mask(tensor_dict, layer, tp_mask, tn_mask, fp_mask):
    """Apply binary masks to classification tensors."""
    tensor_dict['tp'][layer] = torch.where(tp_mask, 1, 0)
    tensor_dict['tn'][layer] = torch.where(tn_mask, 1, 0)
    tensor_dict['fp'][layer] = torch.where(fp_mask, 1, 0)


def _classify_trigger_vs_noise(trigger_imp, noise_imp):
    """Classify trigger importance vs noise importance."""
    classification = _create_classification_dict(trigger_imp)

    for layer in trigger_imp.keys():
        trigger_important = trigger_imp[layer] > 0

        # TP: trigger important AND noise important
        tp_mask = torch.logical_and(noise_imp[layer] > 0, trigger_important)

        # FP: trigger not important AND noise important
        fp_mask = torch.logical_and(noise_imp[layer] > 0, ~trigger_important)

        # TN: trigger not important AND noise not important
        tn_mask = torch.logical_and(noise_imp[layer] <= 0, ~trigger_important)

        _apply_binary_mask(classification, layer, tp_mask, tn_mask, fp_mask)

    return trigger_imp, classification['tp'], classification['tn'], classification['fp']


def _classify_trigger_vs_intersection(trigger_imp, clean_imp, noise_imp):
    """Classify trigger importance vs intersection of clean and noise importance. (trigger as ground truth)"""
    classification = _create_classification_dict(trigger_imp)

    for layer in trigger_imp.keys():
        clean_and_noise_important = torch.logical_and(
            noise_imp[layer] > 0, clean_imp[layer] > 0
        )
        clean_and_noise_not_important = torch.logical_and(
            noise_imp[layer] <= 0, clean_imp[layer] <= 0
        )

        # TP: trigger important AND (clean important AND noise important)
        tp_mask = torch.logical_and(
            trigger_imp[layer] > 0, clean_and_noise_important)

        # FP: trigger important AND (clean not important OR noise not important)
        fp_mask = torch.logical_and(
            trigger_imp[layer] > 0, clean_and_noise_not_important)

        # TN: trigger not important AND (clean not important AND noise not important)
        tn_mask = torch.logical_and(
            trigger_imp[layer] <= 0, clean_and_noise_not_important)

        _apply_binary_mask(classification, layer, tp_mask, tn_mask, fp_mask)

    return trigger_imp, classification['tp'], classification['tn'], classification['fp']


def _classify_trigger_vs_union(trigger_imp, clean_imp, noise_imp):
    """Classify trigger importance vs union of clean and noise importance. (trigger as ground truth)"""
    classification = _create_classification_dict(trigger_imp)

    for layer in trigger_imp.keys():
        clean_or_noise_important = torch.logical_or(
            noise_imp[layer] > 0, clean_imp[layer] > 0
        )
        clean_and_noise_not_important = torch.logical_and(
            noise_imp[layer] <= 0, clean_imp[layer] <= 0
        )

        # TP: trigger important AND (clean important OR noise important)
        tp_mask = torch.logical_and(
            trigger_imp[layer] > 0, clean_or_noise_important)

        # FP: trigger important AND (clean not important AND noise not important)
        fp_mask = torch.logical_and(
            trigger_imp[layer] > 0, clean_and_noise_not_important)

        # TN: trigger not important AND (clean not important AND noise not important)
        tn_mask = torch.logical_and(
            trigger_imp[layer] <= 0, clean_and_noise_not_important)

        _apply_binary_mask(classification, layer, tp_mask, tn_mask, fp_mask)

    return trigger_imp, classification['tp'], classification['tn'], classification['fp']


def _classify_noise_evaluation(trigger_imp, noise_imp, clean_imp, inter_model_imp):
    """Classify noise evaluation scenario."""
    classification = _create_classification_dict(trigger_imp)

    for layer in trigger_imp.keys():
        clean_and_inter_important = torch.logical_and(
            inter_model_imp[layer] > 0, clean_imp[layer] > 0
        )
        clean_and_inter_not_important = torch.logical_and(
            inter_model_imp[layer] <= 0, clean_imp[layer] <= 0
        )

        # TP: noise important AND (clean important AND inter_model important)
        tp_mask = torch.logical_and(
            noise_imp[layer] > 0, clean_and_inter_important)

        # FP: noise important AND (clean not important OR inter_model not important)
        fp_mask = torch.logical_and(
            noise_imp[layer] > 0, clean_and_inter_not_important)

        # TN: noise not important AND (clean not important AND inter_model not important)
        tn_mask = torch.logical_and(
            noise_imp[layer] <= 0, clean_and_inter_not_important)

        _apply_binary_mask(classification, layer, tp_mask, tn_mask, fp_mask)

    return clean_imp, classification['tp'], classification['tn'], classification['fp']


def _classify_eval_comparison(trigger_imp, clean_imp):
    """Classify evaluation comparison of trigger importance and clean importance."""
    classification = _create_classification_dict(trigger_imp)

    for layer in trigger_imp.keys():
        # TP: trigger > 0 AND clean > 0
        tp_mask = torch.logical_and(
            trigger_imp[layer] > 0, clean_imp[layer] > 0)

        # TN: trigger <= 0 AND clean <= 0
        tn_mask = torch.logical_and(
            trigger_imp[layer] <= 0, clean_imp[layer] <= 0)

        # FP: trigger > 0 but clean <= 0
        fp_mask = torch.logical_and(
            trigger_imp[layer] > 0, clean_imp[layer] <= 0)

        _apply_binary_mask(classification, layer, tp_mask, tn_mask, fp_mask)

    return trigger_imp, classification['tp'], classification['tn'], classification['fp']


def _classify_full_comparison(trigger_imp, noise_imp, clean_imp, inter_model_imp):
    """Classify full comparison of all importance scores."""
    classification = _create_classification_dict(trigger_imp)

    for layer in trigger_imp.keys():
        # TP: trigger > 0, clean > 0, inter_model > 0, noise <= 0
        tp_mask = torch.logical_and(
            torch.logical_and(trigger_imp[layer] > 0, clean_imp[layer] > 0),
            torch.logical_and(
                inter_model_imp[layer] > 0, noise_imp[layer] <= 0)
        )

        # TN: trigger <= 0, clean <= 0, inter_model <= 0
        tn_mask = torch.logical_and(
            torch.logical_and(trigger_imp[layer] <= 0, clean_imp[layer] <= 0),
            inter_model_imp[layer] <= 0
        )

        # FP: trigger > 0 but (clean <= 0 OR inter_model <= 0 OR noise > 0)
        fp_mask = torch.logical_and(
            trigger_imp[layer] > 0,
            torch.logical_or(
                torch.logical_or(clean_imp[layer] <=
                                 0, inter_model_imp[layer] <= 0),
                noise_imp[layer] > 0
            )
        )

        _apply_binary_mask(classification, layer, tp_mask, tn_mask, fp_mask)

    return clean_imp, classification['tp'], classification['tn'], classification['fp']


def _plot_layerwise_tsne(trigger_importance, noise_importance, clean_importance, grad_importance,
                         model, model_dict, visual_path, title, figsize=(15, 10), n_cols=3,
                         perplexity=30, random_state=42):
    """
    Plot layer-wise t-SNE clustering of weights based on importance values.

    Args:
        trigger_importance: Ground truth trigger importance (0-1 values)
        noise_importance: Soft noise importance values
        clean_importance: Soft clean importance values
        grad_importance: Gradient-based importance values
        model: The neural network model
        model_dict: Model state dictionary
        figsize: Figure size for the plot
        n_cols: Number of columns in subplot grid
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
    """

    # Load model state
    model = copy.deepcopy(model)
    model.load_state_dict(model_dict)

    # Get all SoftMaskedLayer names
    layer_names = []
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            layer_names.append(n)

    if not layer_names:
        print("No SoftMaskedLayer found in the model")
        return

    # Calculate grid dimensions
    n_layers = len(layer_names)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, layer_name in enumerate(layer_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Extract importance values for this layer
        trigger_vals = trigger_importance[layer_name].flatten().cpu().numpy()
        noise_vals = noise_importance[layer_name].flatten().cpu().numpy()
        clean_vals = clean_importance[layer_name].flatten().cpu().numpy()
        grad_vals = grad_importance[layer_name].flatten().cpu().numpy()

        # Create feature matrix: [noise_importance, clean_importance]
        features = np.column_stack([noise_vals, clean_vals, grad_vals])

        # Create labels based on trigger importance (0-1 ground truth)
        # You can adjust the threshold as needed
        threshold = 0.5
        labels = (trigger_vals > threshold).astype(int)

        # Handle case where all labels are the same
        if len(np.unique(labels)) == 1:
            ax.scatter(features[:, 0], features[:, 1],
                       c=labels, alpha=0.6, s=20)
            ax.set_title(f'{layer_name}\n(All same label)')
            ax.set_xlabel('Noise Importance')
            ax.set_ylabel('Clean Importance')
            continue

        # Apply t-SNE only if we have enough points
        if len(features) < 4:  # t-SNE needs at least 4 points
            ax.scatter(features[:, 0], features[:, 1],
                       c=labels, cmap='viridis', alpha=0.6, s=20)
            ax.set_title(f'{layer_name}\n(Too few points for t-SNE)')
            ax.set_xlabel('Noise Importance')
            ax.set_ylabel('Clean Importance')
            continue

        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Apply t-SNE
            tsne_perplexity = min(perplexity, len(features) - 1)
            tsne = TSNE(n_components=2, perplexity=tsne_perplexity,
                        random_state=random_state, max_iter=1000)
            features_tsne = tsne.fit_transform(features_scaled)

            # Create scatter plot
            scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1],
                                 c=labels, cmap='viridis', alpha=0.7, s=30)

            # Add colorbar for this subplot
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Trigger (0=No, 1=Yes)')

            ax.set_title(f'{layer_name}\n({len(features)} weights)')
            ax.set_xlabel('t-SNE Component 1')
            ax.set_ylabel('t-SNE Component 2')

        except Exception as e:
            print(f"Error processing layer {layer_name}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}',
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{layer_name}\n(Error)')

    # Hide empty subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Layer-wise t-SNE Clustering: Weights by Noise/Clean Importance and Trigger Labels',
                 y=1.02, fontsize=14)
    plt.savefig(f"{visual_path}/{title}.png")
    plt.close()  # Added to prevent memory issues


def _plot_layerwise_umap(gt_label_list: list, noise_importance, clean_importance, grad_importance,
                         model, model_dict, visual_path, title, n_cols=4,
                         n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Plot layer-wise UMAP clustering of weights based on importance values.
    """

    # Get all SoftMaskedLayer names
    layer_names = []
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            layer_names.append(n)

    if not layer_names:
        print("No SoftMaskedLayer found in the model")
        return

    figsize = (n_cols * 3, len(layer_names))

    # Calculate total number of plots needed
    total_plots = len(layer_names) * len(gt_label_list)
    n_rows = (total_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    plot_idx = 0

    for layer_idx, layer_name in enumerate(layer_names):
        # Extract importance values for this layer
        noise_vals = noise_importance[layer_name].flatten().cpu().numpy()
        clean_vals = clean_importance[layer_name].flatten().cpu().numpy()
        grad_vals = grad_importance[layer_name].flatten().cpu().numpy()

        # Create feature matrix
        features = np.column_stack([noise_vals, clean_vals, grad_vals])

        # Standardize features once per layer
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Adjust UMAP parameters based on data size
        umap_n_neighbors = min(n_neighbors, len(features) - 1)
        umap_n_neighbors = max(2, umap_n_neighbors)

        # Apply UMAP once per layer
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=min_dist,
            # random_state=random_state,
            n_epochs=200
        )
        features_umap = reducer.fit_transform(features_scaled)

        # Create plots for each gt_label_list item
        for gt_idx, gt_label_dict in enumerate(gt_label_list):
            trigger_vals = gt_label_dict[layer_name].flatten().cpu().numpy()
            labels = (trigger_vals > 0.5).astype(int)

            # Calculate subplot position
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]

            # Create scatter plot
            scatter = ax.scatter(features_umap[:, 0], features_umap[:, 1],
                                 c=labels, cmap='viridis', alpha=0.7, s=30)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Trigger (0=No, 1=Yes)')
            ax.set_title(
                f'{layer_name} (GT_{gt_idx})\n({len(features)} weights)')
            ax.set_xlabel('UMAP Component 1')
            ax.set_ylabel('UMAP Component 2')

            plot_idx += 1

    # Hide empty subplots
    for idx in range(total_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Layer-wise UMAP Clustering: Weights by Noise/Clean/Grad Importance and Trigger Labels',
                 y=1.02, fontsize=14)
    plt.savefig(f"{visual_path}/{title}.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_comparative_trigger_heatmaps(gt_label_list: list, model, visual_path, title, n_cols=None):
    """
    Plot comparative heatmaps showing all GT labels side by side for each layer.
    Uses the first item in gt_label_list as ground truth reference.
    """
    # Get all SoftMaskedLayer names
    layer_names = []
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            layer_names.append(n)
    if not layer_names:
        print("No SoftMaskedLayer found in the model")
        return
        
    n_gt_labels = len(gt_label_list)
    if n_gt_labels == 0:
        print("No GT labels provided")
        return
        
    if n_cols is None:
        n_cols = min(n_gt_labels, 4)  # Default to 4 columns max
    
    # Use first item as ground truth reference
    gt_reference = gt_label_list[0]
    
    # Helper function to process mask shape consistently
    def process_mask_shape(mask_vals):
        """Process mask to 2D shape consistently"""
        original_shape = mask_vals.shape
        if len(original_shape) == 1:
            size = original_shape[0]
            height = int(np.sqrt(size))
            width = (size + height - 1) // height
            padded_size = height * width
            if padded_size > size:
                mask_padded = np.zeros(padded_size)
                mask_padded[:size] = mask_vals.flatten()
                mask_vals = mask_padded
            mask_vals = mask_vals.reshape(height, width)
        elif len(original_shape) == 2:
            pass  # Already 2D
        elif len(original_shape) == 3:
            mask_vals = mask_vals.reshape(original_shape[0], -1)
        elif len(original_shape) == 4:
            mask_vals = mask_vals.reshape(original_shape[0] * original_shape[1],
                                        original_shape[2] * original_shape[3])
        else:
            mask_vals = mask_vals.flatten()
            size = len(mask_vals)
            height = int(np.sqrt(size))
            width = (size + height - 1) // height
            padded_size = height * width
            if padded_size > size:
                mask_padded = np.zeros(padded_size)
                mask_padded[:size] = mask_vals
                mask_vals = mask_padded
            mask_vals = mask_vals.reshape(height, width)
        return mask_vals
    
    # Create one figure per layer
    for layer_idx, layer_name in enumerate(layer_names):
        n_rows = (n_gt_labels + n_cols - 1) // n_cols
        figsize = (n_cols * 4, n_rows * 3)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Normalize axes to always be 2D array for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Process GT reference mask for this layer
        gt_trigger_vals = gt_reference[layer_name].cpu().numpy()
        gt_processed = process_mask_shape(gt_trigger_vals)
        gt_binary = (gt_processed > 0.5).astype(float)
        
        plot_idx = 0
        for gt_idx, gt_label_dict in enumerate(gt_label_list):
            # Extract and process trigger values
            trigger_vals = gt_label_dict[layer_name].cpu().numpy()
            processed_vals = process_mask_shape(trigger_vals)
            binary_mask = (processed_vals > 0.5).astype(float)
            
            # Calculate subplot position
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col]
            
            # Calculate mask statistics compared to GT reference (first item)
            if gt_idx == 0:
                # This is the reference, show as blue
                im = ax.imshow(processed_vals, cmap='Blues', aspect='auto',
                             vmin=0, vmax=1, interpolation='nearest')
                mask_rate_text = "Reference"
            else:
                # For comparison, we need to ensure both masks have the same final shape
                # Pad the smaller one to match the larger one
                if gt_binary.shape != binary_mask.shape:
                    max_h = max(gt_binary.shape[0], binary_mask.shape[0])
                    max_w = max(gt_binary.shape[1], binary_mask.shape[1])
                    
                    # Pad gt_binary if needed
                    if gt_binary.shape != (max_h, max_w):
                        gt_padded = np.zeros((max_h, max_w))
                        gt_padded[:gt_binary.shape[0], :gt_binary.shape[1]] = gt_binary
                        gt_binary_comp = gt_padded
                    else:
                        gt_binary_comp = gt_binary
                    
                    # Pad binary_mask if needed
                    if binary_mask.shape != (max_h, max_w):
                        mask_padded = np.zeros((max_h, max_w))
                        mask_padded[:binary_mask.shape[0], :binary_mask.shape[1]] = binary_mask
                        binary_mask_comp = mask_padded
                        
                        # Also pad the display values
                        display_padded = np.zeros((max_h, max_w))
                        display_padded[:processed_vals.shape[0], :processed_vals.shape[1]] = processed_vals
                        processed_vals = display_padded
                    else:
                        binary_mask_comp = binary_mask
                else:
                    gt_binary_comp = gt_binary
                    binary_mask_comp = binary_mask
                
                # Calculate overlap statistics
                current_active = np.sum(binary_mask_comp > 0)
                gt_active = np.sum(gt_binary_comp > 0)
                intersection = np.sum((binary_mask_comp > 0) & (gt_binary_comp > 0))
                union = np.sum((binary_mask_comp > 0) | (gt_binary_comp > 0))
                
                # Calculate metrics
                jaccard = intersection / union if union > 0 else 0
                precision = intersection / current_active if current_active > 0 else 0
                sensitivity = intersection / gt_active if gt_active > 0 else 0  # Also known as recall
                
                # Use red colormap for comparison items
                im = ax.imshow(processed_vals, cmap='Reds', aspect='auto',
                             vmin=0, vmax=1, interpolation='nearest')
                mask_rate_text = f"IoU: {jaccard:.3f}\nPrec: {precision:.3f}\nSens: {sensitivity:.3f}"
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Trigger Prob.')
            
            # Calculate basic statistics
            trigger_count = np.sum(processed_vals > 0.5)
            total_weights = processed_vals.size
            
            # Set title with statistics
            ax.set_title(
                f'GT Label {gt_idx}\n{mask_rate_text}\nTriggers: {trigger_count}/{total_weights}', 
                fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plot_idx += 1
        
        # Hide empty subplots
        for idx in range(n_gt_labels, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        fig.suptitle(f'Trigger Comparison for Layer: {layer_name}',
                     y=0.98, fontsize=14, fontweight='bold')
        
        # Save figure
        save_filename = f"{visual_path}/{title}_{layer_name}_comparison.png"
        plt.savefig(save_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison for layer {layer_name}: {save_filename}")

def _plot_layerwise_venn_intersections(gt_label_list: list, model, visual_path, title,
                                       figsize=(15, 20), n_cols=3, threshold=0.5):
    """
    Plot layer-wise Venn diagrams showing intersections between different ground truth label lists.
    """

    # Get all SoftMaskedLayer names
    layer_names = []
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            layer_names.append(n)

    if not layer_names:
        print("No SoftMaskedLayer found in the model")
        return

    if len(gt_label_list) < 2:
        print("Need at least 2 label lists for intersection analysis")
        return

    # Calculate number of rows needed
    n_rows = (len(layer_names) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for layer_idx, layer_name in enumerate(layer_names):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        ax = axes[row, col]

        # Convert trigger values to binary labels for each gt_label_list
        label_sets = []
        for gt_idx, gt_label_dict in enumerate(gt_label_list):
            trigger_vals = gt_label_dict[layer_name].flatten().cpu().numpy()
            trigger_indices = set(np.where(trigger_vals > threshold)[0])
            label_sets.append(trigger_indices)

        # Create Venn diagram based on number of label lists
        if len(gt_label_list) == 2:
            venn = venn2(label_sets, set_labels=[
                         f'GT_{i}' for i in range(len(gt_label_list))], ax=ax)
            venn2_circles(label_sets, ax=ax)

        elif len(gt_label_list) == 3:
            venn = venn3(label_sets, set_labels=[
                         f'GT_{i}' for i in range(len(gt_label_list))], ax=ax)
            venn3_circles(label_sets, ax=ax)

        elif len(gt_label_list) == 4:
            # Use custom 4-circle Venn diagram
            set_labels = [f'GT_{i}' for i in range(4)]
            _plot_detailed_4_circle_venn(label_sets, fig, ax, set_labels)

        else:
            # For more than 4 sets, show pairwise intersections as a heatmap
            n_sets = len(gt_label_list)
            intersection_matrix = np.zeros((n_sets, n_sets))

            for i in range(n_sets):
                for j in range(n_sets):
                    if i == j:
                        intersection_matrix[i, j] = len(label_sets[i])
                    else:
                        intersection_matrix[i, j] = len(
                            label_sets[i].intersection(label_sets[j]))

            im = ax.imshow(intersection_matrix, cmap='Blues', aspect='auto')
            ax.set_xticks(range(n_sets))
            ax.set_yticks(range(n_sets))
            ax.set_xticklabels([f'GT_{i}' for i in range(n_sets)], fontsize=8)
            ax.set_yticklabels([f'GT_{i}' for i in range(n_sets)], fontsize=8)

            # Add text annotations
            for i in range(n_sets):
                for j in range(n_sets):
                    text = ax.text(j, i, int(intersection_matrix[i, j]),
                                   ha="center", va="center", color="black", fontsize=8)

            plt.colorbar(im, ax=ax, shrink=0.6)

        ax.set_title(f'{layer_name}\nTrigger Intersections', fontsize=10)

    # Hide empty subplots
    for idx in range(len(layer_names), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.suptitle(
        f'{title} - Layer-wise Trigger Label Intersections', y=1.02, fontsize=14)
    plt.savefig(f"{visual_path}/{title}_venn_intersections.png",
                dpi=300, bbox_inches='tight')
    plt.close()


def _plot_detailed_4_circle_venn(label_sets, fig, ax, set_labels=None):
    labels = venn.get_labels(label_sets, fill=['number', 'logic'])
    fig, ax = venn.venn4(labels, names=set_labels, fig=fig, ax=ax)


def analyze_frequency_features(original_img, backdoored_img):
    """
    Analyze frequency domain features of original and backdoored images
    """
    # Convert to grayscale for frequency analysis (or analyze each channel separately)
    if len(original_img.shape) == 3:
        original_gray = np.mean(original_img, axis=2)
        backdoored_gray = np.mean(backdoored_img, axis=2)
    else:
        original_gray = original_img
        backdoored_gray = backdoored_img

    # Compute 2D FFT
    fft_original = fft2(original_gray)
    fft_backdoored = fft2(backdoored_gray)

    # Shift zero frequency to center
    fft_original_shifted = fftshift(fft_original)
    fft_backdoored_shifted = fftshift(fft_backdoored)

    # Compute magnitude spectra
    magnitude_original = np.abs(fft_original_shifted)
    magnitude_backdoored = np.abs(fft_backdoored_shifted)

    # Compute phase spectra
    phase_original = np.angle(fft_original_shifted)
    phase_backdoored = np.angle(fft_backdoored_shifted)

    # Compute log magnitude for better visualization
    log_mag_original = np.log(magnitude_original + 1)
    log_mag_backdoored = np.log(magnitude_backdoored + 1)

    return {
        'fft_original': fft_original_shifted,
        'fft_backdoored': fft_backdoored_shifted,
        'magnitude_original': magnitude_original,
        'magnitude_backdoored': magnitude_backdoored,
        'phase_original': phase_original,
        'phase_backdoored': phase_backdoored,
        'log_mag_original': log_mag_original,
        'log_mag_backdoored': log_mag_backdoored
    }


def plot_frequency_comparison(original_img, backdoored_img, img_path='./'):
    """
    Plot comprehensive frequency domain comparison
    """
    freq_data = analyze_frequency_features(original_img, backdoored_img)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Original images (spatial domain)
    axes[0, 0].imshow(original_img.astype(np.uint8))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(backdoored_img.astype(np.uint8))
    axes[0, 1].set_title('Backdoored Image')
    axes[0, 1].axis('off')

    # Difference in spatial domain
    spatial_diff = np.abs(backdoored_img - original_img)
    axes[0, 2].imshow(spatial_diff.astype(np.uint8))
    axes[0, 2].set_title('Spatial Difference')
    axes[0, 2].axis('off')

    # Magnitude difference
    mag_diff = np.abs(
        freq_data['magnitude_backdoored'] - freq_data['magnitude_original'])
    axes[0, 3].imshow(np.log(mag_diff + 1), cmap='hot')
    axes[0, 3].set_title('Magnitude Spectrum Difference')
    axes[0, 3].axis('off')

    # Log magnitude spectra
    axes[1, 0].imshow(freq_data['log_mag_original'], cmap='gray')
    axes[1, 0].set_title('Original Log Magnitude')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(freq_data['log_mag_backdoored'], cmap='gray')
    axes[1, 1].set_title('Backdoored Log Magnitude')
    axes[1, 1].axis('off')

    # Phase spectra
    axes[1, 2].imshow(freq_data['phase_original'], cmap='hsv')
    axes[1, 2].set_title('Original Phase')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(freq_data['phase_backdoored'], cmap='hsv')
    axes[1, 3].set_title('Backdoored Phase')
    axes[1, 3].axis('off')

    # Frequency analysis plots
    # Power spectral density
    center = freq_data['magnitude_original'].shape[0] // 2

    # Radial average for power spectrum
    y, x = np.ogrid[:freq_data['magnitude_original'].shape[0],
                    :freq_data['magnitude_original'].shape[1]]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype(int)

    # Calculate radial averages
    tbin = np.bincount(r.ravel(), freq_data['magnitude_original'].ravel())
    nr = np.bincount(r.ravel())
    radial_prof_orig = tbin / nr

    tbin_back = np.bincount(
        r.ravel(), freq_data['magnitude_backdoored'].ravel())
    radial_prof_back = tbin_back / nr

    axes[2, 0].plot(radial_prof_orig[:len(
        radial_prof_orig)//2], label='Original')
    axes[2, 0].plot(radial_prof_back[:len(
        radial_prof_back)//2], label='Backdoored')
    axes[2, 0].set_title('Radial Power Spectrum')
    axes[2, 0].set_xlabel('Frequency')
    axes[2, 0].set_ylabel('Power')
    axes[2, 0].legend()
    axes[2, 0].set_yscale('log')

    # High frequency components comparison
    high_freq_mask = r > center // 2
    high_freq_orig = freq_data['magnitude_original'][high_freq_mask]
    high_freq_back = freq_data['magnitude_backdoored'][high_freq_mask]

    axes[2, 1].hist(high_freq_orig, bins=50, alpha=0.7,
                    label='Original', density=True)
    axes[2, 1].hist(high_freq_back, bins=50, alpha=0.7,
                    label='Backdoored', density=True)
    axes[2, 1].set_title('High Frequency Distribution')
    axes[2, 1].set_xlabel('Magnitude')
    axes[2, 1].set_ylabel('Density')
    axes[2, 1].legend()
    axes[2, 1].set_yscale('log')

    # Statistical comparison
    stats_text = f"""
    Original Stats:
    Mean Magnitude: {np.mean(freq_data['magnitude_original']):.2f}
    Std Magnitude: {np.std(freq_data['magnitude_original']):.2f}
    Max Magnitude: {np.max(freq_data['magnitude_original']):.2f}
    
    Backdoored Stats:
    Mean Magnitude: {np.mean(freq_data['magnitude_backdoored']):.2f}
    Std Magnitude: {np.std(freq_data['magnitude_backdoored']):.2f}
    Max Magnitude: {np.max(freq_data['magnitude_backdoored']):.2f}
    
    Difference:
    Mean Diff: {np.mean(mag_diff):.2f}
    Max Diff: {np.max(mag_diff):.2f}
    """

    axes[2, 2].text(0.1, 0.5, stats_text, transform=axes[2, 2].transAxes,
                    verticalalignment='center', fontsize=8)
    axes[2, 2].axis('off')
    axes[2, 2].set_title('Statistical Comparison')

    # Frequency band energy comparison
    bands = [(0, center//4), (center//4, center//2),
             (center//2, 3*center//4), (3*center//4, center)]
    band_names = ['Low', 'Mid-Low', 'Mid-High', 'High']

    energies_orig = []
    energies_back = []

    for low, high in bands:
        mask = (r >= low) & (r < high)
        energies_orig.append(np.sum(freq_data['magnitude_original'][mask]**2))
        energies_back.append(
            np.sum(freq_data['magnitude_backdoored'][mask]**2))

    x_pos = np.arange(len(band_names))
    width = 0.35

    axes[2, 3].bar(x_pos - width/2, energies_orig,
                   width, label='Original', alpha=0.8)
    axes[2, 3].bar(x_pos + width/2, energies_back,
                   width, label='Backdoored', alpha=0.8)
    axes[2, 3].set_xlabel('Frequency Bands')
    axes[2, 3].set_ylabel('Energy')
    axes[2, 3].set_title('Energy per Frequency Band')
    axes[2, 3].set_xticks(x_pos)
    axes[2, 3].set_xticklabels(band_names)
    axes[2, 3].legend()
    axes[2, 3].set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{img_path}/frequency_comparison.png', dpi=300)
    plt.close()

    return freq_data


def detailed_frequency_analysis(original_img, backdoored_img):
    """
    Perform detailed frequency analysis with metrics
    """
    freq_data = analyze_frequency_features(original_img, backdoored_img)

    # Calculate various frequency domain metrics
    metrics = {}

    # Spectral distortion
    spectral_diff = np.abs(
        freq_data['fft_backdoored'] - freq_data['fft_original'])
    metrics['spectral_distortion'] = np.mean(spectral_diff)

    # High frequency energy ratio
    center = freq_data['magnitude_original'].shape[0] // 2
    y, x = np.ogrid[:freq_data['magnitude_original'].shape[0],
                    :freq_data['magnitude_original'].shape[1]]
    r = np.sqrt((x - center)**2 + (y - center)**2)

    high_freq_mask = r > center // 2
    low_freq_mask = r <= center // 2

    high_freq_energy_orig = np.sum(
        freq_data['magnitude_original'][high_freq_mask]**2)
    low_freq_energy_orig = np.sum(
        freq_data['magnitude_original'][low_freq_mask]**2)
    high_freq_energy_back = np.sum(
        freq_data['magnitude_backdoored'][high_freq_mask]**2)
    low_freq_energy_back = np.sum(
        freq_data['magnitude_backdoored'][low_freq_mask]**2)

    metrics['hf_lf_ratio_original'] = high_freq_energy_orig / \
        low_freq_energy_orig
    metrics['hf_lf_ratio_backdoored'] = high_freq_energy_back / \
        low_freq_energy_back
    metrics['hf_lf_ratio_change'] = metrics['hf_lf_ratio_backdoored'] / \
        metrics['hf_lf_ratio_original']

    # Phase coherence
    phase_diff = np.abs(
        freq_data['phase_backdoored'] - freq_data['phase_original'])
    # Handle phase wrapping
    phase_diff = np.minimum(phase_diff, 2*np.pi - phase_diff)
    metrics['phase_coherence'] = 1 - np.mean(phase_diff) / np.pi

    # Frequency domain SNR
    signal_power = np.mean(freq_data['magnitude_original']**2)
    noise_power = np.mean(
        (freq_data['magnitude_backdoored'] - freq_data['magnitude_original'])**2)
    metrics['frequency_snr_db'] = 10 * \
        np.log10(signal_power / (noise_power + 1e-10))

    print("\n=== Detailed Frequency Analysis ===")
    print(f"Spectral Distortion: {metrics['spectral_distortion']:.4f}")
    print(
        f"High/Low Freq Ratio (Original): {metrics['hf_lf_ratio_original']:.4f}")
    print(
        f"High/Low Freq Ratio (Backdoored): {metrics['hf_lf_ratio_backdoored']:.4f}")
    print(f"High/Low Freq Ratio Change: {metrics['hf_lf_ratio_change']:.4f}")
    print(f"Phase Coherence: {metrics['phase_coherence']:.4f}")
    print(f"Frequency Domain SNR: {metrics['frequency_snr_db']:.2f} dB")

    return metrics
