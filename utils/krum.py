from collections import OrderedDict
from typing import Any, List, Tuple

import torch

"""
from https://github.com/FedML-AI/FedML

defense @ server, added by Xiaoyang, Chulin, 07/09/2022
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
https://arxiv.org/pdf/1703.02757.pdf
"Distributed momentum for byzantine-resilient stochastic gradient descent"
https://infoscience.epfl.ch/record/287261
"""


def compute_euclidean_distance(first_vector, second_vector, device='cuda'):
    first_vector = first_vector.to(device)
    second_vector = second_vector.to(device)
    return (first_vector - second_vector).norm()


def vectorize_weight(state_dict):
    weights = [
        tensor.flatten()
        for parameter_name, tensor in state_dict.items()
        if is_weight_param(parameter_name)
    ]
    return torch.cat(weights)


def is_weight_param(parameter_name):
    return (
        "running_mean" not in parameter_name
        and "running_var" not in parameter_name
        and "num_batches_tracked" not in parameter_name
    )


class KrumDefense:
    def __init__(self, byzantine_client_num, krum_param_m=1):
        self.byzantine_client_num = byzantine_client_num

        # krum_param_m = 1: krum; krum_param_m > 1: multi-krum
        self.krum_param_m = krum_param_m  # krum

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        client_count = len(raw_client_grad_list)
        # in the Krum paper, it says 2 * byzantine_client_num + 2 < client #
        if not 2 * self.byzantine_client_num + 2 <= client_count - self.krum_param_m:
            raise ValueError(
                "byzantine_client_num conflicts with requirements in Krum: 2 * byzantine_client_num + 2 < client number - krum_param_m"
            )

        client_vectors = [
            vectorize_weight(weights)
            for _, weights in raw_client_grad_list
        ]
        scores = self._compute_krum_score(client_vectors)
        selected_indices = torch.argsort(torch.tensor(scores)).tolist()
        selected_indices = selected_indices[:self.krum_param_m]
        return [raw_client_grad_list[index] for index in selected_indices]

    def _compute_krum_score(self, client_vectors):
        scores = []
        client_count = len(client_vectors)
        for client_index, client_vector in enumerate(client_vectors):
            distances = []
            for other_index, other_vector in enumerate(client_vectors):
                if client_index != other_index:
                    distances.append(
                        compute_euclidean_distance(
                            client_vector, other_vector
                        ).item() ** 2
                    )
            distances.sort()
            neighbor_count = client_count - self.byzantine_client_num - 2
            scores.append(sum(distances[:neighbor_count]))
        return scores
