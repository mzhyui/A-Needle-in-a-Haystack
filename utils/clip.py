import copy

import torch


def clipping(local_weights: dict, local_model: torch.nn.Module):
    reference_weights = local_model.state_dict()
    weight_deltas = copy.deepcopy(local_weights)
    normalized_deltas = copy.deepcopy(local_weights)

    for layer_name in local_weights:
        weight_deltas[layer_name] = local_weights[layer_name] - reference_weights[layer_name]
        normalized_deltas[layer_name] = torch.nn.functional.normalize(
            weight_deltas[layer_name].float(), dim=0
        )

    for layer_name in local_weights:
        local_weights[layer_name] = (
            local_weights[layer_name]
            - torch.nn.functional.normalize(
                normalized_deltas[layer_name].float(), dim=0
            ).long()
        )
    return local_weights
