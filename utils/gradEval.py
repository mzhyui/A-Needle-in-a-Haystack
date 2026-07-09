
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np


from models.smFunction import impt_norm




def calculate_model_correlations_from_neuron_analysis(model:nn.Module, model_state_dicts, clean_samples, corrupt_samples, layer_names=None, 
                                                    correlation_metrics=['importance_diff', 'activation_diff'],
                                                    aggregation_method='mean',
                                                    p=0.5,
                                                    device:str|torch.device='cuda'):
    """
    Calculate correlations between multiple models based on their neuron-wise responses
    to void/corrupt samples
    
    Args:
        clean_samples
        corrupt_samples
        layer_names: List of specific layers to analyze (if None, analyzes all common layers)
        correlation_metrics: Which metrics to use for correlation calculation
        aggregation_method: How to aggregate across layers ('mean', 'weighted', 'concatenate')
        device: Device to run computation on
    
    Returns:
        correlation_results: Dictionary containing correlation matrices and detailed analysis
    """
    
    model_signatures = {}
    
    # print(f"Analyzing {len(model_state_dicts)} models...")
    
    # Step 1: Extract neuron signatures for each model
    for idx, model_dict in model_state_dicts.items():
        model.load_state_dict(model_dict)
        # print(f"\nAnalyzing dict {idx}...")
        
        # Get activation paths for clean and corrupt inputs
        activation_clean = get_activation_path_batch(model, clean_samples[0], 
                                                   clean_samples[1], 
                                                   p=p,
                                                   device=device)
        activation_corrupt = get_activation_path_batch(model, corrupt_samples[0], 
                                                     corrupt_samples[1], 
                                                     p=p,
                                                     device=device)
        
        # Get neuron-wise analysis
        neuron_analysis = get_neuron_wise_activation_analysis(activation_clean, activation_corrupt, p=p)
        
        # Extract signature vectors for each layer
        model_signatures[idx] = extract_model_signature(neuron_analysis, 
                                                        correlation_metrics, 
                                                        layer_names)
    
    # Step 2: Calculate correlations between models
    correlation_results = compute_model_correlations(model_signatures, aggregation_method)
    
    return correlation_results

def get_neuron_wise_activation_analysis(activation_clean:dict, activation_corrupt:dict, layer_name=None, metric='importance', p=0.5):
    """
    Analyze neuron-wise activation differences between clean and corrupted inputs
    
    Args:
        activation_clean: Dictionary containing activation info for clean input
        activation_corrupt: Dictionary containing activation info for corrupted input
        layer_name: Specific layer to analyze (if None, analyzes all layers)
        threshold: Threshold for considering a neuron as "active"
    
    Returns:
        neuron_analysis_dict: Dictionary containing detailed neuron-wise analysis
    """
    
    neuron_analysis_dict = {}
    
    # Select layers to analyze
    layers_to_analyze = [layer_name] if layer_name else list(activation_clean.keys())
    
    for layer in layers_to_analyze:
        if layer not in activation_clean or layer not in activation_corrupt:
            print(f"Warning: Layer {layer} not found in both activation dictionaries")
            continue
            
        # Get importance scores (normalized gradients)
        clean_importance = activation_clean[layer][metric]
        corrupt_importance = activation_corrupt[layer][metric]
        
        # Convert to binary activation maps based on threshold
        clean_active = (clean_importance >= p)
        corrupt_active = (corrupt_importance >= p)
        # clean_active = activation_clean[layer]['active_neurons']
        # corrupt_active = activation_corrupt[layer]['active_neurons']
        
        # Calculate differences
        importance_diff = corrupt_importance - clean_importance
        activation_diff = corrupt_active.float() - clean_active.float()
        
        # Categorize neurons based on activation changes
        newly_activated = (corrupt_active & ~clean_active)  # Activated in corrupt but not clean
        newly_deactivated = (~corrupt_active & clean_active)  # Deactivated in corrupt
        consistently_active = (corrupt_active & clean_active)  # Active in both
        consistently_inactive = (~corrupt_active & ~clean_active)  # Inactive in both
        
        # Store analysis results
        neuron_analysis_dict[layer] = {
            'clean_importance': clean_importance,
            'corrupt_importance': corrupt_importance,
            'importance_diff': importance_diff,
            'clean_active': clean_active,
            'corrupt_active': corrupt_active,
            'activation_diff': activation_diff,
            'newly_activated': newly_activated,
            'newly_deactivated': newly_deactivated,
            'consistently_active': consistently_active,
            'consistently_inactive': consistently_inactive,
            'stats': {
                'total_neurons': clean_importance.numel(),
                'newly_activated_count': newly_activated.sum().item(),
                'newly_deactivated_count': newly_deactivated.sum().item(),
                'consistently_active_count': consistently_active.sum().item(),
                'consistently_inactive_count': consistently_inactive.sum().item(),
                'clean_active_count': clean_active.sum().item(),
                'corrupt_active_count': corrupt_active.sum().item(),
                'max_importance_increase': importance_diff.max().item(),
                'max_importance_decrease': importance_diff.min().item(),
                'mean_importance_change': importance_diff.mean().item(),
                'std_importance_change': importance_diff.std().item()
            }
        }
    
    return neuron_analysis_dict

def extract_model_signature(neuron_analysis, correlation_metrics, layer_names=None):
    """
    Extract characteristic signature vectors from neuron analysis
    """
    signature = {}
    
    layers_to_use = layer_names if layer_names else list(neuron_analysis.keys())
    
    for layer in layers_to_use:
        if layer not in neuron_analysis:
            continue
            
        layer_data = neuron_analysis[layer]
        layer_signature = {}
        
        for metric in correlation_metrics:
            if metric == 'importance_diff':
                # Flatten and normalize importance differences
                diff = layer_data['importance_diff'].flatten().cpu().numpy()
                layer_signature[metric] = diff
                
            elif metric == 'activation_diff':
                # Flatten activation differences
                diff = layer_data['activation_diff'].flatten().cpu().numpy()
                layer_signature[metric] = diff
                
            elif metric == 'combined_stats':
                # Use statistical summaries as features
                stats = layer_data['stats']
                combined = np.array([
                    stats['newly_activated_count'] / stats['total_neurons'],
                    stats['newly_deactivated_count'] / stats['total_neurons'],
                    stats['max_importance_increase'],
                    stats['max_importance_decrease'],
                    stats['mean_importance_change'],
                    stats['std_importance_change']
                ])
                layer_signature[metric] = combined
                
        signature[layer] = layer_signature
    
    return signature

def compute_model_correlations(model_signatures, aggregation_method='mean'):
    """
    Compute correlation matrices between models with comprehensive NaN prevention
    """
    model_names = list(model_signatures.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        raise ValueError("Need at least 2 models for correlation computation")
    
    # Get all layers and metrics
    sample_model = model_signatures[model_names[0]]
    layers = list(sample_model.keys())
    metrics = list(sample_model[layers[0]].keys())
    
    correlation_results = {
        'model_names': model_names,
        'layers': layers,
        'metrics': metrics,
        'layer_wise_correlations': {},
        'metric_wise_correlations': {},
        'aggregated_correlation': np.eye(n_models),
        'similarity_scores': {}
    }
    
    # Layer-wise correlations
    for layer in layers:
        correlation_results['layer_wise_correlations'][layer] = {}
        
        for metric in metrics:
            vectors = []
            for model_name in model_names:
                if layer in model_signatures[model_name] and metric in model_signatures[model_name][layer]:
                    vector = model_signatures[model_name][layer][metric]
                else:
                    vector = np.array([0.0])  # Default empty vector
                
                vectors.append(vector)
            
            # Compute safe correlation matrix
            corr_matrix = safe_correlation_matrix(vectors)
            correlation_results['layer_wise_correlations'][layer][metric] = corr_matrix
    
    # Metric-wise aggregation across layers
    for metric in metrics:
        metric_correlations = []
        valid_layers = []
        
        for layer in layers:
            if (layer in correlation_results['layer_wise_correlations'] and 
                metric in correlation_results['layer_wise_correlations'][layer]):
                metric_correlations.append(correlation_results['layer_wise_correlations'][layer][metric])
                valid_layers.append(layer)
        
        if metric_correlations:
            avg_corr = safe_aggregate_correlations(metric_correlations, method=aggregation_method)
            
            correlation_results['metric_wise_correlations'][metric] = {
                'correlation_matrix': avg_corr,
                'valid_layers': valid_layers,
                'layer_count': len(valid_layers)
            }
    
    # Overall aggregated correlation
    all_metric_correlations = [
        correlation_results['metric_wise_correlations'][metric]['correlation_matrix'] 
        for metric in metrics 
        if metric in correlation_results['metric_wise_correlations']
    ]
    
    if all_metric_correlations:
        correlation_results['aggregated_correlation'] = safe_aggregate_correlations(
            all_metric_correlations, method=aggregation_method
        )
    
    # Calculate additional similarity metrics
    try:
        correlation_results['similarity_scores'] = calculate_similarity_scores(
            correlation_results['aggregated_correlation']
        )
    except Exception as e:
        print(f"Warning: Could not calculate similarity scores ({e})")
        correlation_results['similarity_scores'] = {}
    
    return correlation_results

def safe_aggregate_correlations(correlation_matrices, method='mean', weights=None):
    """
    Safely aggregate correlation matrices
    """
    if not correlation_matrices:
        return np.eye(2)  # Default case
    
    # Clean each correlation matrix
    clean_matrices = []
    for matrix in correlation_matrices:
        clean_matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        clean_matrix = np.clip(clean_matrix, -1, 1)
        np.fill_diagonal(clean_matrix, 1.0)
        clean_matrices.append(clean_matrix)
    
    try:
        if method == 'mean':
            result = np.mean(clean_matrices, axis=0)
        elif method == 'weighted' and weights is not None:
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize weights
            result = np.average(clean_matrices, axis=0, weights=weights)
        else:
            result = np.mean(clean_matrices, axis=0)
        
        # Final validation
        result = np.nan_to_num(result, nan=0.0)
        result = np.clip(result, -1, 1)
        np.fill_diagonal(result, 1.0)
        
        return result
        
    except Exception:
        # Fallback to identity matrix
        shape = clean_matrices[0].shape if clean_matrices else (2, 2)
        return np.eye(shape[0])

def safe_correlation_matrix(vectors):
    """
    Compute correlation matrix with comprehensive NaN prevention
    """
    n_models = len(vectors)
    
    if n_models < 2:
        return np.eye(max(1, n_models))
    
    # Clean all vectors
    cleaned_vectors = []
    for vector in vectors:
        cleaned_vector = clean_vector(vector)
        
        # Ensure minimum length
        if len(cleaned_vector) == 0:
            cleaned_vector = np.array([0.0])
        elif len(cleaned_vector) == 1:
            # Extend single values to avoid correlation issues
            cleaned_vector = np.array([cleaned_vector[0], cleaned_vector[0] + 1e-10])
        
        cleaned_vectors.append(cleaned_vector)
    
    # Ensure all vectors have the same length
    max_length = max(len(v) for v in cleaned_vectors)
    padded_vectors = []
    
    for vector in cleaned_vectors:
        if len(vector) < max_length:
            # Pad with the last value or zeros
            padding = np.full(max_length - len(vector), vector[-1] if len(vector) > 0 else 0.0)
            vector = np.concatenate([vector, padding])
        padded_vectors.append(vector)
    
    try:
        # Stack vectors for correlation computation
        data_matrix = np.stack(padded_vectors)
        
        # Check for sufficient variance
        variances = np.var(data_matrix, axis=1)
        
        # Handle zero variance cases
        if np.any(variances == 0):
            # Add small noise to zero-variance vectors
            for i, var in enumerate(variances):
                if var == 0:
                    data_matrix[i] += np.random.normal(0, 1e-10, size=data_matrix.shape[1])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data_matrix)
        
        # Final NaN check and replacement
        if np.any(np.isnan(corr_matrix)) or np.any(np.isinf(corr_matrix)):
            # Replace problematic values
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            # Ensure diagonal is 1
            np.fill_diagonal(corr_matrix, 1.0)
        
        # Ensure matrix is symmetric and within bounds
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Ensure symmetry
        corr_matrix = np.clip(corr_matrix, -1, 1)  # Ensure valid correlation bounds
        
        return corr_matrix
        
    except Exception as e:
        print(f"Warning: Correlation computation failed ({e}), returning identity matrix")
        return np.eye(n_models)

def clean_vector(vector):
    """
    Clean vector by handling NaN, inf, and other problematic values
    """
    vector = np.asarray(vector, dtype=np.float64)
    
    # Replace NaN with 0
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for constant vectors (zero variance)
    if len(vector) > 1 and np.var(vector) == 0:
        # Add small random noise to break ties
        vector = vector + np.random.normal(0, 1e-10, size=vector.shape)
    
    return vector

def calculate_similarity_scores(correlation_matrix):
    """
    Calculate various similarity scores from correlation matrix
    """
    n_models = correlation_matrix.shape[0]
    
    similarity_scores = {
        'pairwise_correlations': {},
        'average_correlation_per_model': {},
        'model_clustering_scores': {},
        'outlier_scores': {}
    }
    
    # Pairwise correlations (excluding diagonal)
    for i in range(n_models):
        for j in range(i+1, n_models):
            pair_name = f'Model_{i}_vs_Model_{j}'
            similarity_scores['pairwise_correlations'][pair_name] = correlation_matrix[i, j]
    
    # Average correlation per model
    for i in range(n_models):
        # Average correlation with all other models
        others = np.concatenate([correlation_matrix[i, :i], correlation_matrix[i, i+1:]])
        similarity_scores['average_correlation_per_model'][f'Model_{i}'] = np.mean(others)
    
    # Simple outlier detection (models with low average correlation)
    avg_corrs = list(similarity_scores['average_correlation_per_model'].values())
    mean_avg_corr = np.mean(avg_corrs)
    std_avg_corr = np.std(avg_corrs)
    
    for i, avg_corr in enumerate(avg_corrs):
        z_score = (avg_corr - mean_avg_corr) / (std_avg_corr + 1e-8)
        similarity_scores['outlier_scores'][f'Model_{i}'] = z_score
    
    return similarity_scores

def get_activation_path_batch(model:nn.Module, batch_input:torch.Tensor, batch_labels:torch.Tensor, device:str|torch.device='cuda', aggregation='mean', p=0.9) -> dict:
    """
    Get the activation path (back propagation) for a batch of input images
    
    Args:
        model: The neural network model with SoftMaskedLayer modules
        batch_input: Batch of input images tensor from dataloader
        batch_labels: Batch of target labels from dataloader
        ft_task: Fine-tuning task index (default: 0)
        device: Device to run computation on
        aggregation: How to aggregate gradients across batch ('mean', 'sum', 'individual')
    
    Returns:
        activation_path_dict: Dictionary containing gradient information for each SoftMaskedLayer
    """
    criterion = nn.CrossEntropyLoss()
    
    # Set model to appropriate mode for getting activation path
    model.eval()  # or model.train() depending on your needs

    # Ensure inputs are on correct device
    batch_input = batch_input.to(device)
    batch_labels = batch_labels.to(device)
    
    # Handle different label formats
    if batch_labels.dim() > 1:
        batch_labels = batch_labels.squeeze()
    
    batch_size = batch_input.size(0)
    
    # Clear any existing gradients
    model.zero_grad()
    
    if aggregation == 'individual':
        # Store individual activation paths for each sample
        batch_activation_paths = []
        
        for i in range(batch_size):
            model.zero_grad()
            single_input = batch_input[i:i+1]  # Keep batch dimension
            single_label = batch_labels[i:i+1]
            
            with torch.enable_grad():
                single_input.requires_grad_(True)
                output = model(single_input)
                loss = criterion(output, single_label)
                loss.backward()
            
            # Collect activation path for this sample
            sample_activation_path = {}
            for name, module in model.named_modules():
                # Get gradient information (activation path)
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    grad = module.weight.grad.clone().detach()
                    importance = impt_norm(grad)
                    thresh_mask = (importance >= p)
                    sample_activation_path[name] = {
                        'gradient': grad,
                        'importance': importance,
                        'importance_thresholded': importance * thresh_mask.float(),
                        'active_neurons': int(thresh_mask.sum().item()),
                        'total_neurons': grad.numel()
                    }
                    
                    # Calculate activation percentage
                    activation_percentage = sample_activation_path[name]['active_neurons'] / sample_activation_path[name]['total_neurons']
                    sample_activation_path[name]['activation_percentage'] = activation_percentage
                    
                    print(f'Layer: {name}')
                    print(f'  Active neurons: {sample_activation_path[name]["active_neurons"]}/{sample_activation_path[name]["total_neurons"]} ({activation_percentage:.2%})')
            
            batch_activation_paths.append(sample_activation_path)
        
        return batch_activation_paths
    
    else:
        # Aggregate gradients across the entire batch
        with torch.enable_grad():
            batch_input.requires_grad_(True)
            output = model(batch_input)
            
            # Compute loss for the entire batch
            loss = criterion(output, batch_labels)
            
            # Backward pass to compute gradients
            loss.backward()
        
        # Collect activation path information
        activation_path_dict = {}
        
        for name, module in model.named_modules():
        # Get gradient information (activation path)
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad = module.weight.grad.clone().detach()
                        
                # Apply aggregation if specified
                if aggregation == 'sum':
                    # Gradients are already summed across batch by default
                    aggregated_grad = grad
                elif aggregation == 'mean':
                    # Average the gradients across batch
                    aggregated_grad = grad / batch_size
                else:
                    aggregated_grad = grad
                
                importance = impt_norm(aggregated_grad)
                thresh_mask = (importance >= p)
                active_neurons = thresh_mask.sum().item()
                total_neurons = aggregated_grad.numel()
                
                activation_path_dict[name] = {
                    'gradient': aggregated_grad,
                    'importance': importance,
                    'importance_thresholded': importance * thresh_mask.float(),
                    'active_neurons': active_neurons,
                    'total_neurons': total_neurons,
                    'activation_percentage': active_neurons / total_neurons,
                    'batch_size': batch_size
                }
                
                # print(f'Layer: {name}')
                # print(f'  Batch size: {batch_size}')
                # print(f'  Active neurons: {active_neurons}/{total_neurons} ({activation_path_dict[name]["activation_percentage"]:.2%})')
    
        # Clear gradients after collection
        model.zero_grad()
        
        return activation_path_dict

def visualize_model_correlations(correlation_results, save_path=None):
    """
    Create comprehensive visualization of model correlations
    """
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main correlation heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    corr_matrix = correlation_results['aggregated_correlation']
    model_names = correlation_results['model_names']
    
    im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_yticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.set_yticklabels(model_names)
    ax1.set_title('Model Correlation Matrix (Aggregated)', fontsize=14, fontweight='bold')
    
    # Add correlation values to heatmap
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text = ax1.text(j, i, f'{corr_matrix[i, j]:.3f}', ha="center", va="center", 
                           color="white" if abs(corr_matrix[i, j]) > 0.5 else "black")
    
    plt.colorbar(im, ax=ax1, label='Correlation')
    
    # Average correlation per model
    ax2 = fig.add_subplot(gs[0, 2])
    avg_corrs = list(correlation_results['similarity_scores']['average_correlation_per_model'].values())
    ax2.bar(range(len(avg_corrs)), avg_corrs, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Model Index')
    ax2.set_ylabel('Average Correlation')
    ax2.set_title('Average Correlation per Model')
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels([f'M{i}' for i in range(len(model_names))])
    ax2.grid(True, alpha=0.3)
    
    # Metric-wise correlation comparison
    ax3 = fig.add_subplot(gs[1, :])
    metrics = list(correlation_results['metric_wise_correlations'].keys())
    
    if metrics:
        # Plot correlation strength for each metric
        metric_strengths = []
        for metric in metrics:
            corr_mat = correlation_results['metric_wise_correlations'][metric]['correlation_matrix']
            # Calculate average off-diagonal correlation
            mask = ~np.eye(corr_mat.shape[0], dtype=bool)
            avg_strength = np.mean(np.abs(corr_mat[mask]))
            metric_strengths.append(avg_strength)
        
        bars = ax3.bar(metrics, metric_strengths, color=['coral', 'lightgreen', 'lightblue'][:len(metrics)])
        ax3.set_ylabel('Average Correlation Strength')
        ax3.set_title('Correlation Strength by Metric Type')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, strength in zip(bars, metric_strengths):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{strength:.3f}', ha='center', va='bottom')
    
    # Hierarchical clustering dendrogram
    ax4 = fig.add_subplot(gs[2, :])
    # Convert correlation to distance
    distance_matrix = 1 - np.abs(corr_matrix)
    condensed_distances = squareform(distance_matrix, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distances, method='average')
    
    # Create dendrogram
    dendrogram(linkage_matrix, labels=model_names, ax=ax4, leaf_rotation=45)
    ax4.set_title('Hierarchical Clustering of Models')
    ax4.set_ylabel('Distance (1 - |correlation|)')
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def find_top_k_outliers(correlation_results, k=3, outlier_methods=['z_score', 'isolation']):
    """
    Find top-K outlier models based on multiple criteria
    
    Parameters:
    -----------
    correlation_results : dict
        Results from correlation analysis
    k : int
        Number of top outliers to return
    outlier_methods : list
        Methods to use for outlier detection
        
    Returns:
    --------
    dict : Comprehensive outlier analysis results
    """
    import numpy as np
    from scipy import stats
    
    model_names = correlation_results['model_names']
    corr_matrix = correlation_results['aggregated_correlation']
    n_models = len(model_names)
    
    outlier_scores = {}
    outlier_rankings = {}
    
    # Method 1: Z-score based on average correlation
    if 'z_score' in outlier_methods:
        avg_correlations = []
        for i in range(n_models):
            # Calculate average correlation with other models (excluding self)
            mask = np.ones(n_models, dtype=bool)
            mask[i] = False
            avg_corr = np.mean(np.abs(corr_matrix[i, mask]))
            avg_correlations.append(avg_corr)
        
        z_scores = np.abs(stats.zscore(avg_correlations))
        outlier_scores['z_score'] = dict(zip(model_names, z_scores))
        outlier_rankings['z_score'] = sorted(
            zip(model_names, z_scores), 
            key=lambda x: x[1], reverse=True
        )[:k]
    
    # Method 2: Correlation variance (inconsistency)
    if 'correlation_variance' in outlier_methods:
        corr_variances = []
        for i in range(n_models):
            mask = np.ones(n_models, dtype=bool)
            mask[i] = False
            corr_var = np.var(corr_matrix[i, mask])
            corr_variances.append(corr_var)
        
        outlier_scores['correlation_variance'] = dict(zip(model_names, corr_variances))
        outlier_rankings['correlation_variance'] = sorted(
            zip(model_names, corr_variances), 
            key=lambda x: x[1], reverse=True
        )[:k]
    
    # Method 3: Distance from correlation centroid
    if 'centroid_distance' in outlier_methods:
        # Calculate centroid of correlation patterns
        centroid = np.mean(corr_matrix, axis=0)
        distances = []
        
        for i in range(n_models):
            distance = np.linalg.norm(corr_matrix[i] - centroid)
            distances.append(distance)
        
        outlier_scores['centroid_distance'] = dict(zip(model_names, distances))
        outlier_rankings['centroid_distance'] = sorted(
            zip(model_names, distances), 
            key=lambda x: x[1], reverse=True
        )[:k]
    
    # Method 4: Isolation Forest (if sklearn available)
    if 'isolation' in outlier_methods:
        try:
            from sklearn.ensemble import IsolationForest
            
            # Use correlation patterns as features
            iso_forest = IsolationForest(contamination=0.3, random_state=42)
            outlier_predictions = iso_forest.fit_predict(corr_matrix)
            outlier_scores_iso = -iso_forest.score_samples(corr_matrix)
            
            outlier_scores['isolation'] = dict(zip(model_names, outlier_scores_iso))
            outlier_rankings['isolation'] = sorted(
                zip(model_names, outlier_scores_iso), 
                key=lambda x: x[1], reverse=True
            )[:k]
            
        except ImportError:
            print("sklearn not available for Isolation Forest method")
    
    # Method 5: Mahalanobis distance
    if 'mahalanobis' in outlier_methods:
        try:
            mean = np.mean(corr_matrix, axis=0)
            cov = np.cov(corr_matrix.T)
            
            # Add small regularization to handle potential singularity
            cov += np.eye(cov.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov)
            
            mahal_distances = []
            for i in range(n_models):
                diff = corr_matrix[i] - mean
                distance = np.sqrt(diff.T @ inv_cov @ diff)
                mahal_distances.append(distance)
            
            outlier_scores['mahalanobis'] = dict(zip(model_names, mahal_distances))
            outlier_rankings['mahalanobis'] = sorted(
                zip(model_names, mahal_distances), 
                key=lambda x: x[1], reverse=True
            )[:k]
            
        except np.linalg.LinAlgError:
            print("Mahalanobis distance calculation failed (singular matrix)")
    
    # Ensemble ranking - combine multiple methods
    ensemble_scores = {name: 0 for name in model_names}
    
    for method, rankings in outlier_rankings.items():
        for rank, (model_name, score) in enumerate(rankings):
            # Higher rank = higher outlier score (reverse rank for scoring)
            ensemble_scores[model_name] += (k - rank)
    
    ensemble_ranking = sorted(
        ensemble_scores.items(), 
        key=lambda x: x[1], reverse=True
    )[:k]
    
    # Calculate detailed statistics for top outliers
    top_outliers = [model for model, _ in ensemble_ranking]
    outlier_analysis = {}
    
    for outlier in top_outliers:
        idx = model_names.index(outlier)
        
        # Correlation statistics
        correlations_with_others = corr_matrix[idx, :]
        mask = np.ones(n_models, dtype=bool)
        mask[idx] = False
        other_correlations = correlations_with_others[mask]
        
        analysis = {
            'model_name': outlier,
            'avg_correlation': np.mean(np.abs(other_correlations)),
            'min_correlation': np.min(other_correlations),
            'max_correlation': np.max(other_correlations),
            'correlation_std': np.std(other_correlations),
            'ensemble_score': ensemble_scores[outlier],
            'rankings_by_method': {}
        }
        
        # Find ranking in each method
        for method, rankings in outlier_rankings.items():
            for rank, (model_name, score) in enumerate(rankings):
                if model_name == outlier:
                    analysis['rankings_by_method'][method] = {
                        'rank': rank + 1,
                        'score': score
                    }
                    break
        
        outlier_analysis[outlier] = analysis
    
    return {
        'top_k_outliers': top_outliers,
        'ensemble_ranking': ensemble_ranking,
        'outlier_scores_by_method': outlier_scores,
        'outlier_rankings_by_method': outlier_rankings,
        'detailed_analysis': outlier_analysis,
        'summary': {
            'total_models': n_models,
            'k': k,
            'methods_used': list(outlier_rankings.keys())
        }
    }

def analyze_model_outliers(correlation_results, k=3):
    """
    Complete pipeline for outlier analysis
    """
    # Find outliers
    outlier_results = find_top_k_outliers(
        correlation_results, 
        k=k, 
        outlier_methods=['z_score', 'correlation_variance', 'centroid_distance', 'isolation']
    )
    
    # Print summary
    print(f"\nTop-{k} Outlier Models:")
    print("=" * 50)
    
    for i, (model, score) in enumerate(outlier_results['ensemble_ranking'], 1):
        analysis = outlier_results['detailed_analysis'][model]
        print(f"{i}. {model}")
        print(f"   Ensemble Score: {score}")
        print(f"   Avg Correlation: {analysis['avg_correlation']:.3f}")
        print(f"   Correlation Std: {analysis['correlation_std']:.3f}")
        print(f"   Method Rankings: {analysis['rankings_by_method']}")
        print()
    
    # # Visualize
    # visualize_outliers(correlation_results, outlier_results)
    
    return outlier_results

def find_top_k_clustered_outliers(correlation_results, k=3, methods=['density_separation', 'clique_isolation', 'modularity']):
    """
    Find top-K points that are closely connected to each other but separated from other points
    
    Parameters:
    -----------
    correlation_results : dict
        Results from correlation analysis
    k : int
        Number of top clustered outlier groups to return
    methods : list
        Methods to use for finding clustered outliers
        
    Returns:
    --------
    dict : Comprehensive clustered outlier analysis results
    """
    import numpy as np
    from scipy import stats
    from itertools import combinations
    
    model_names = correlation_results['model_names']
    corr_matrix = correlation_results['aggregated_correlation']
    n_models = len(model_names)
    
    # Convert correlation to similarity/distance matrix
    similarity_matrix = np.abs(corr_matrix)
    distance_matrix = 1 - similarity_matrix
    
    clustered_groups = {}
    group_scores = {}
    
    # Method 1: Density-based separation analysis
    if 'density_separation' in methods:
        def calculate_group_cohesion_separation(indices):
            """Calculate internal cohesion vs external separation for a group"""
            if len(indices) < 2:
                return 0
            
            # Internal cohesion (higher is better)
            internal_similarities = []
            for i, j in combinations(indices, 2):
                internal_similarities.append(similarity_matrix[i, j])
            avg_internal_similarity = np.mean(internal_similarities)
            
            # External separation (lower external similarity is better)
            external_similarities = []
            other_indices = [i for i in range(n_models) if i not in indices]
            
            for group_idx in indices:
                for other_idx in other_indices:
                    external_similarities.append(similarity_matrix[group_idx, other_idx])
            
            avg_external_similarity = np.mean(external_similarities) if external_similarities else 0
            
            # Score: high internal cohesion, low external similarity
            separation_score = avg_internal_similarity - avg_external_similarity
            return separation_score, avg_internal_similarity, avg_external_similarity
        
        # Find all possible groups of size 2 to k+2
        density_groups = []
        for group_size in range(2, min(k+3, n_models+1)):
            for indices in combinations(range(n_models), group_size):
                score, internal, external = calculate_group_cohesion_separation(indices)
                group_names = [model_names[i] for i in indices]
                
                density_groups.append({
                    'indices': indices,
                    'models': group_names,
                    'separation_score': score,
                    'internal_cohesion': internal,
                    'external_separation': 1 - external,  # Convert to separation metric
                    'size': len(indices)
                })
        
        # Sort by separation score and take top k
        density_groups.sort(key=lambda x: x['separation_score'], reverse=True)
        clustered_groups['density_separation'] = density_groups[:k]
    
    # Method 2: Clique-based isolation
    if 'clique_isolation' in methods:
        def find_tight_cliques(threshold=0.7):
            """Find groups where all members are highly correlated with each other"""
            cliques = []
            
            # Start with pairs and expand
            for group_size in range(2, min(k+3, n_models+1)):
                for indices in combinations(range(n_models), group_size):
                    # Check if all pairs in group exceed threshold
                    min_similarity = float('inf')
                    for i, j in combinations(indices, 2):
                        min_similarity = min(min_similarity, similarity_matrix[i, j])
                    
                    if min_similarity >= threshold:
                        # Calculate isolation from rest
                        group_names = [model_names[i] for i in indices]
                        other_indices = [i for i in range(n_models) if i not in indices]
                        
                        max_external_similarity = 0
                        for group_idx in indices:
                            for other_idx in other_indices:
                                max_external_similarity = max(
                                    max_external_similarity, 
                                    similarity_matrix[group_idx, other_idx]
                                )
                        
                        isolation_score = min_similarity - max_external_similarity
                        
                        cliques.append({
                            'indices': indices,
                            'models': group_names,
                            'min_internal_similarity': min_similarity,
                            'max_external_similarity': max_external_similarity,
                            'isolation_score': isolation_score,
                            'size': len(indices)
                        })
            
            return sorted(cliques, key=lambda x: x['isolation_score'], reverse=True)
        
        clique_groups = find_tight_cliques()
        clustered_groups['clique_isolation'] = clique_groups[:k]
    
    # Method 3: Modularity-based community detection
    if 'modularity' in methods:
        try:
            # Convert similarity to adjacency matrix (threshold-based)
            threshold = np.percentile(similarity_matrix, 75)  # Top 25% connections
            adjacency = (similarity_matrix >= threshold).astype(int)
            np.fill_diagonal(adjacency, 0)
            
            def calculate_modularity(partition, adjacency):
                """Calculate modularity of a partition"""
                m = np.sum(adjacency) / 2  # Total number of edges
                if m == 0:
                    return 0
                
                modularity = 0
                for community in partition:
                    for i in community:
                        for j in community:
                            if i != j:
                                k_i = np.sum(adjacency[i])
                                k_j = np.sum(adjacency[j])
                                expected = (k_i * k_j) / (2 * m)
                                modularity += (adjacency[i, j] - expected)
                
                return modularity / (2 * m)
            
            # Simple community detection using edge betweenness (simplified version)
            modularity_groups = []
            
            # Try different small group combinations
            for group_size in range(2, min(k+3, n_models+1)):
                for indices in combinations(range(n_models), group_size):
                    # Create partition: this group vs rest
                    rest_indices = [i for i in range(n_models) if i not in indices]
                    partition = [list(indices)]
                    if rest_indices:
                        partition.append(rest_indices)
                    
                    mod_score = calculate_modularity(partition, adjacency)
                    
                    # Calculate additional metrics
                    group_names = [model_names[i] for i in indices]
                    internal_edges = sum(adjacency[i, j] for i, j in combinations(indices, 2))
                    possible_internal = len(list(combinations(indices, 2)))
                    internal_density = internal_edges / possible_internal if possible_internal > 0 else 0
                    
                    # External connections
                    external_edges = sum(
                        adjacency[i, j] for i in indices for j in rest_indices
                    )
                    possible_external = len(indices) * len(rest_indices)
                    external_density = external_edges / possible_external if possible_external > 0 else 0
                    
                    modularity_groups.append({
                        'indices': indices,
                        'models': group_names,
                        'modularity_score': mod_score,
                        'internal_density': internal_density,
                        'external_density': external_density,
                        'separation_ratio': internal_density / (external_density + 1e-8),
                        'size': len(indices)
                    })
            
            modularity_groups.sort(key=lambda x: x['modularity_score'], reverse=True)
            clustered_groups['modularity'] = modularity_groups[:k]
            
        except Exception as e:
            print(f"Modularity analysis failed: {e}")
    
    # Method 4: Silhouette-based cluster quality
    if 'silhouette' in methods:
        def calculate_group_silhouette(indices):
            """Calculate silhouette-like score for a group"""
            if len(indices) < 2:
                return 0
            
            silhouette_scores = []
            other_indices = [i for i in range(n_models) if i not in indices]
            
            for i in indices:
                # Average distance to other points in same cluster
                intra_distances = [distance_matrix[i, j] for j in indices if j != i]
                avg_intra = np.mean(intra_distances) if intra_distances else 0
                
                # Average distance to points in other clusters
                inter_distances = [distance_matrix[i, j] for j in other_indices]
                avg_inter = np.mean(inter_distances) if inter_distances else 1
                
                # Silhouette score
                silhouette = (avg_inter - avg_intra) / max(avg_inter, avg_intra) if max(avg_inter, avg_intra) > 0 else 0
                silhouette_scores.append(silhouette)
            
            return np.mean(silhouette_scores)
        
        silhouette_groups = []
        for group_size in range(2, min(k+3, n_models+1)):
            for indices in combinations(range(n_models), group_size):
                silhouette_score = calculate_group_silhouette(indices)
                group_names = [model_names[i] for i in indices]
                
                silhouette_groups.append({
                    'indices': indices,
                    'models': group_names,
                    'silhouette_score': silhouette_score,
                    'size': len(indices)
                })
        
        silhouette_groups.sort(key=lambda x: x['silhouette_score'], reverse=True)
        clustered_groups['silhouette'] = silhouette_groups[:k]
    
    # Ensemble scoring - combine results from different methods
    all_groups = []
    for method, groups in clustered_groups.items():
        for i, group in enumerate(groups):
            group_key = tuple(sorted(group['models']))
            all_groups.append((group_key, group, method, k - i))  # Higher rank = higher score
    
    # Aggregate scores for same groups found by different methods
    group_ensemble_scores = {}
    group_details = {}
    
    for group_key, group_data, method, score in all_groups:
        if group_key not in group_ensemble_scores:
            group_ensemble_scores[group_key] = 0
            group_details[group_key] = {
                'models': list(group_key),
                'found_by_methods': [],
                'method_scores': {}
            }
        
        group_ensemble_scores[group_key] += score
        group_details[group_key]['found_by_methods'].append(method)
        group_details[group_key]['method_scores'][method] = group_data
    
    # Sort by ensemble score
    final_ranking = sorted(
        group_ensemble_scores.items(), 
        key=lambda x: x[1], reverse=True
    )[:k]
    
    # Detailed analysis of top groups
    top_groups_analysis = {}
    for i, (group_key, ensemble_score) in enumerate(final_ranking):
        group_models = list(group_key)
        group_indices = [model_names.index(model) for model in group_models]
        
        # Calculate comprehensive statistics
        internal_similarities = []
        for i, j in combinations(group_indices, 2):
            internal_similarities.append(similarity_matrix[i, j])
        
        external_similarities = []
        other_indices = [i for i in range(n_models) if i not in group_indices]
        for group_idx in group_indices:
            for other_idx in other_indices:
                external_similarities.append(similarity_matrix[group_idx, other_idx])
        
        analysis = {
            'models': group_models,
            'size': len(group_models),
            'ensemble_score': ensemble_score,
            'avg_internal_similarity': np.mean(internal_similarities),
            'min_internal_similarity': np.min(internal_similarities),
            'max_internal_similarity': np.max(internal_similarities),
            'avg_external_similarity': np.mean(external_similarities) if external_similarities else 0,
            'max_external_similarity': np.max(external_similarities) if external_similarities else 0,
            'separation_score': np.mean(internal_similarities) - np.mean(external_similarities) if external_similarities else np.mean(internal_similarities),
            'found_by_methods': group_details[group_key]['found_by_methods'],
            'method_details': group_details[group_key]['method_scores']
        }
        
        top_groups_analysis[f'group_{i+1}'] = analysis
    
    return {
        'top_k_clustered_groups': [list(group_key) for group_key, _ in final_ranking],
        'ensemble_ranking': final_ranking,
        'groups_by_method': clustered_groups,
        'detailed_analysis': top_groups_analysis,
        'correlation_matrix': corr_matrix,
        'similarity_matrix': similarity_matrix,
        'summary': {
            'total_models': n_models,
            'k': k,
            'methods_used': list(clustered_groups.keys()),
            'total_groups_found': len(group_ensemble_scores)
        }
    }

# Additional utility function for visualization
def visualize_clustered_outliers(results, save_path=None):
    """
    Visualize the clustered outlier groups
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation matrix heatmap with group highlights
    corr_matrix = results['correlation_matrix']
    model_names = []
    for group_name, analysis in results['detailed_analysis'].items():
        model_names.extend(analysis['models'])
    
    # Add remaining models
    all_models = set()
    for i in range(corr_matrix.shape[0]):
        all_models.add(f'model_{i}')  # Assuming generic naming
    remaining_models = list(all_models - set(model_names))
    model_names.extend(remaining_models)
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                xticklabels=model_names, yticklabels=model_names, ax=axes[0,0])
    axes[0,0].set_title('Correlation Matrix with Clustered Groups')
    
    # 2. t-SNE visualization of groups
    if corr_matrix.shape[0] > 3:
        tsne = TSNE(n_components=2, random_state=42, perplexity=15)
        embedded = tsne.fit_transform(corr_matrix)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(results['top_k_clustered_groups'])))
        
        # Plot all points in gray first
        axes[0,1].scatter(embedded[:, 0], embedded[:, 1], c='lightgray', alpha=0.6)
        
        # Highlight clustered groups
        for i, group in enumerate(results['top_k_clustered_groups']):
            group_indices = [j for j, model in enumerate(model_names) if model in group]
            if group_indices:
                axes[0,1].scatter(embedded[group_indices, 0], embedded[group_indices, 1], 
                                    c=[colors[i]], label=f'Group {i+1}', s=100, alpha=0.8)
        
        axes[0,1].set_title('t-SNE: Clustered Outlier Groups')
        axes[0,1].legend()
    
    # 3. Group separation scores
    group_names = list(results['detailed_analysis'].keys())
    separation_scores = [analysis['separation_score'] for analysis in results['detailed_analysis'].values()]
    
    bars = axes[1,0].bar(group_names, separation_scores)
    axes[1,0].set_title('Group Separation Scores')
    axes[1,0].set_ylabel('Separation Score')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Color bars by score
    for bar, score in zip(bars, separation_scores):
        bar.set_color(plt.cm.RdYlBu_r(score / max(separation_scores)))
    
    # 4. Internal vs External similarity comparison
    internal_sims = [analysis['avg_internal_similarity'] for analysis in results['detailed_analysis'].values()]
    external_sims = [analysis['avg_external_similarity'] for analysis in results['detailed_analysis'].values()]
    
    x_pos = np.arange(len(group_names))
    width = 0.35
    
    axes[1,1].bar(x_pos - width/2, internal_sims, width, label='Internal Similarity', alpha=0.8)
    axes[1,1].bar(x_pos + width/2, external_sims, width, label='External Similarity', alpha=0.8)
    
    axes[1,1].set_title('Internal vs External Similarity')
    axes[1,1].set_ylabel('Average Similarity')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(group_names, rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

