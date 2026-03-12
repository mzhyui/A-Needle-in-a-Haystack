import copy
import math
import random
import numpy as np
from dataclasses import dataclass

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, functional

from utils.logger import myLogger
from utils.attackUtils import compute_all_losses_and_grads, compute_noise_loss, get_fl_update



def dual_ascent(lagrange_step, noise_lists, random_neurons,
                lagrange_mul, layer_name, device='cuda'):
    size = 0
    for name, data in noise_lists[0].state_dict().items():
        if layer_name in name:
            size += data.view(-1).shape[0]
    sum_var = torch.zeros(size)
    sum_var = sum_var.to(device)
    for i, noise_list in enumerate(noise_lists):
        size = 0
        for name, data in noise_lists[i].state_dict().items():
            if layer_name in name:
                for j in range(data.shape[0]):
                    if j in random_neurons:
                        sum_var[size:size +
                                data[j].view(-1).shape[0]] += data[j].view(-1)
                    size += data[j].view(-1).shape[0]
    loss = torch.norm(sum_var, p=2)
    lagrange_mul += lagrange_step * loss.item()
    return loss.item(), lagrange_mul


# def compute_blind_loss(model, criterion, batch, attack, fixed_model=None):
#     """

#     :param model:
#     :param criterion:
#     :param batch:
#     :param attack: Do not attack at all. Ignore all the parameters
#     :return:
#     """
#     # batch = batch.clip(self.params.clip_batch)
#     # loss_tasks = self.loss_tasks.copy() if attack else ['normal']
#     loss_tasks = ['backdoor']
#     batch_back = make_backdoor_batch(batch, attack=attack)
#     scale = dict()

#     if len(loss_tasks) == 1:
#         loss_values = compute_all_losses_and_grads(
#             loss_tasks,
#             model, criterion, batch, batch_back
#         )
#     else:
#         loss_values = compute_all_losses_and_grads(
#             loss_tasks,
#             model, criterion, batch, batch_back,
#             fixed_model=fixed_model)

#         for t in loss_tasks:
#             scale[t] = fixed_scales[t]

#     if len(loss_tasks) == 1:
#         scale = {loss_tasks[0]: 1.0}
#     blind_loss = scale_losses(loss_tasks, loss_values, scale)

#     return blind_loss


def read_indicator(fl_number_of_adversaries, dataset_str, global_update, indicators,
                   ind_layer, weakDP):
    accept = []
    feedbacks = []
    if weakDP:
        return accept, weakDP

    for adv_id in range(fl_number_of_adversaries):
        [I, ind_val] = indicators[adv_id]
        feedbacks.append(global_update[ind_layer]
                         [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
    for [I, ind_val] in indicators[fl_number_of_adversaries:]:
        feedbacks.append(global_update[ind_layer]
                         [I[0]][I[1]][I[2]][I[3]].item() / ind_val)
    # logger.info(f'3DFed: feedbacks {feedbacks}')
    # Simple Net is more unstable in parameters, we thus relax the threshold for MNIST
    threshold = 1e-5 if 'MNIST' not in dataset_str else 1e-4
    # logger.warning(f"Avg indicator feedback: \
    #     {np.mean(feedbacks)}")
    for feedback in feedbacks:
        if feedback > 1 or feedback < - threshold:
            weakDP = True
            break
        if feedback <= threshold:
            accept.append('r')  # r = rejected
        elif feedback > threshold and \
                feedback <= max(feedbacks) * 0.8:  # 0.5
            accept.append('c')  # c = clipped
        elif feedback > threshold:
            accept.append('a')  # a = accepted
    return accept, weakDP

def getAnalogUpdate(backdoor_update_dict, benign_update_dict, model_name='lenetsm'):
    # print([key for key in backdoor_update_dict.keys()])
    if model_name == 'lenetsm':
        backdoor_update_slice = abs(
            backdoor_update_dict['conv1.layer.weight'].cpu().numpy()) .flatten()
        benign_update_slice = abs(
            benign_update_dict['conv1.layer.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update_slice + benign_update_slice
        
        no_layer = 0 # TODO 2025-03-04 git.V.32546: fix the layer
        num_candidate = 10
        gradient = np.zeros(shape=(6,1,5,5)) # TODO 2025-03-04 git.V.32546: fix the shape
        curvature = np.zeros(shape=(6,1,5,5))
    elif model_name == 'vggsm':
        backdoor_update_slice = abs(
            backdoor_update_dict['conv1.layer.weight'].cpu().numpy()) .flatten()
        benign_update_slice = abs(
            benign_update_dict['conv1.layer.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update_slice + benign_update_slice
        
        no_layer = 0
        num_candidate = 10
        gradient = np.zeros(shape=(64,3,3,3)) # TODO 2025-03-04 git.V.32546: fix the shape
        curvature = np.zeros(shape=(64,3,3,3))
    elif model_name == 'resnet20sm':
        backdoor_update_slice = abs(
            backdoor_update_dict['conv1.layer.weight'].cpu().numpy()) .flatten()
        benign_update_slice = abs(
            benign_update_dict['conv1.layer.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update_slice + benign_update_slice
        
        no_layer = 0
        num_candidate = 10
        gradient = np.zeros(shape=(16,3,3,3)) # TODO 2025-03-04 git.V.32546: fix the shape
        curvature = np.zeros(shape=(16,3,3,3))

    elif model_name == 'vitsm':
        backdoor_update_slice = abs(
            backdoor_update_dict['embedding.patch_embeddings.projection.layer.weight'].cpu().numpy()) .flatten()
        benign_update_slice = abs(
            benign_update_dict['embedding.patch_embeddings.projection.layer.weight'].cpu().numpy()).flatten()
        analog_update = backdoor_update_slice + benign_update_slice
        
        no_layer = 0
        num_candidate = 10
        gradient = np.zeros(shape=(1,1,192,192)) # TODO 2025-03-04 git.V.32546: fix the shape
        curvature = np.zeros(shape=(1,1,192,192))

    else:
        raise ValueError('Unknown model', model_name)


    return analog_update, gradient, curvature, no_layer, num_candidate

def getIndex(curvature, index, model_name='lenetsm'):
    temp = np.arange(len(curvature))
    if model_name == 'lenetsm':
        temp = np.reshape(temp, (6,1,5,5)) # TODO 2025-03-04 git.V.32546: fix the shape
    elif model_name == 'vggsm':
        temp = np.reshape(temp, (64,3,3,3))
    elif model_name == 'resnet20sm':
        temp = np.reshape(temp, (16,3,3,3))
    elif model_name == 'vitsm':
        temp = np.reshape(temp, (192,12,4,4))
    else:
        raise ValueError('Unknown model', model_name)
    for i in range(len(index)):
        index[i] = np.where(temp == index[i])
        index[i] = [index[i][0][0], index[i][1][0],
                    index[i][2][0], index[i][3][0]]
        
    return index
    

# def design_indicator(fl_number_of_adversaries, dataset_str, k, model, backdoor_update, benign_update,
#                      criterion, train_loader, poisoning_proportion, backdoor_dynamic_position, resize_scale, pattern_tensor, input_shape, mask_value, backdoor_label, device='cuda', logger=myLogger()):
#     total_devices = fl_number_of_adversaries + k
#     num_candidate = 512  # 512
#     if 'cifar' in dataset_str:
#         num_candidate = 10
#         backdoor_update = abs(
#             backdoor_update['layer4.1.conv2.weight'].cpu().numpy()) .flatten()
#         benign_update = abs(
#             benign_update['layer4.1.conv2.weight'].cpu().numpy()).flatten()
#         analog_update = backdoor_update + benign_update
#         no_layer = 57  # layer4.1.conv2.weight
#         gradient = np.zeros(shape=(512, 512, 3, 3))
#         curvature = np.zeros(shape=(512, 512, 3, 3))
#     elif 'Imagenet' in dataset_str:
#         backdoor_update = abs(
#             backdoor_update['layer4.1.conv1.weight'].cpu().numpy()) .flatten()
#         benign_update = abs(
#             benign_update['layer4.1.conv1.weight'].cpu().numpy()).flatten()
#         analog_update = backdoor_update + benign_update
#         # no_layer = 48 # layer4.0.conv2.weight
#         no_layer = 54  # layer4.1.conv1.weight
#         gradient = np.zeros(shape=(512, 512, 3, 3))
#         curvature = np.zeros(shape=(512, 512, 3, 3))
#     elif 'mnist' in dataset_str:
#         num_candidate = 10
#         backdoor_update = abs(
#             backdoor_update['conv2.weight'].cpu().numpy()) .flatten()
#         benign_update = abs(
#             benign_update['conv2.weight'].cpu().numpy()).flatten()
#         analog_update = backdoor_update + benign_update
#         no_layer = 2  # conv2.weight
#         gradient = np.zeros(shape=(50, 20, 5, 5))
#         curvature = np.zeros(shape=(50, 20, 5, 5))
#     else:
#         raise ValueError('Unknown task')

#     # Get gradient and curvature
#     for i, data in enumerate(train_loader):
#         batch = get_batch(i, data)
#         batch_back = make_backdoor_batch(poisoning_proportion, backdoor_dynamic_position,
#                                          resize_scale, pattern_tensor, input_shape, mask_value, device, backdoor_label, batch)
#         # Compute gradient and curvature for normal loss
#         outputs = model(batch.inputs)
#         loss = criterion(outputs, batch.labels)
#         grad = torch.autograd.grad(loss.mean(),
#                                    [x for x in model.parameters() if
#                                     x.requires_grad],
#                                    retain_graph=True,
#                                    create_graph=True
#                                    )[no_layer]
#         grad.requires_grad_()
#         grad_sum = torch.sum(grad)
#         curv = torch.autograd.grad(grad_sum,
#                                    [x for x in model.parameters() if
#                                     x.requires_grad],
#                                    retain_graph=True
#                                    )[no_layer]
#         gradient += grad.detach().cpu().numpy()
#         curvature += curv.detach().cpu().numpy()

#         # Compute gradient and curvature for backdoor loss
#         outputs = model(batch_back.inputs)
#         loss = criterion(outputs, batch_back.labels)
#         grad = torch.autograd.grad(loss.mean(),
#                                    [x for x in model.parameters() if
#                                     x.requires_grad],
#                                    create_graph=True,
#                                    retain_graph=True
#                                    )[no_layer]
#         grad.requires_grad_()
#         grad_sum = torch.sum(grad)
#         curv = torch.autograd.grad(grad_sum,
#                                    [x for x in model.parameters() if
#                                     x.requires_grad],
#                                    retain_graph=True
#                                    )[no_layer]
#         gradient += grad.detach().cpu().numpy()
#         curvature += curv.detach().cpu().numpy()

#     update_val = []
#     idx_candidate = []
#     for i, grad in enumerate(analog_update):
#         if len(idx_candidate) < num_candidate * total_devices:
#             update_val.append(grad)
#             idx_candidate.append(i)
#         elif grad < max(update_val):
#             temp = update_val.index(max(update_val))
#             update_val[temp] = grad
#             idx_candidate[temp] = i

#     # gradient = np.abs(gradient.flatten()).tolist()
#     index = []
#     curv_val = []
#     curvature = np.abs(curvature.flatten()).tolist()
#     for idx in idx_candidate:
#         if len(index) < total_devices:
#             curv_val.append(curvature[idx])
#             index.append(idx)
#         elif curvature[idx] == 0:
#             # The index having max curvature
#             temp = curv_val.index(max(curv_val))
#             if analog_update[idx] < analog_update[index[temp]]:
#                 curv_val[temp] = curvature[idx]
#                 index[temp] = idx
#         elif curvature[idx] < max(curv_val):
#             temp = curv_val.index(max(curv_val))
#             curv_val[temp] = curvature[idx]
#             index[temp] = idx
#     logger.info(f'Curvature value: {curv_val}')

#     if 'cifar' in dataset_str:
#         temp = []
#         for i in range(len(curvature)):
#             temp.append(i)
#         temp = np.reshape(temp, (512, 512, 3, 3))
#         for i in range(len(index)):
#             index[i] = np.where(temp == index[i])
#             index[i] = [index[i][0][0], index[i][1][0],
#                         index[i][2][0], index[i][3][0]]
#     elif 'Imagenet' in dataset_str:
#         temp = []
#         for i in range(len(curvature)):
#             temp.append(i)
#         temp = np.reshape(temp, (512, 512, 3, 3))
#         for i in range(len(index)):
#             index[i] = np.where(temp == index[i])
#             index[i] = [index[i][0][0], index[i][1][0],
#                         index[i][2][0], index[i][3][0]]
#     else:
#         temp = []
#         for i in range(len(curvature)):
#             temp.append(i)
#         temp = np.reshape(temp, (50, 20, 5, 5))
#         for i in range(len(index)):
#             index[i] = np.where(temp == index[i])
#             index[i] = [index[i][0][0], index[i][1][0],
#                         index[i][2][0], index[i][3][0]]
#     return index


# def decoy_model_design(params: Params, k, backdoor_update, benign_update,
#                        benign_model, global_model, local_dataset, indicators, ind_layer):
#     if k <= 0:
#         return indicators
#     decoy_lists = []
#     optimizer_lists = []
#     decoy_loss_idx = find_decoy_params(
#         params, backdoor_update, benign_update, k)
#     for i in range(k):
#         decoy_lists.append(deepcopy(global_model))
#         optimizer_lists.append(optim.SGD(
#             decoy_lists[i].parameters(),
#             lr=params.lr,
#             weight_decay=params.decay,
#             momentum=params.momentum))

#     # Decoy model training
#     for _ in tqdm(range(params.fl_local_epochs)):
#         for i, data in enumerate(local_dataset):
#             batch = get_batch(i, data, params)
#             for j in range(k):
#                 decoy_lists[j].zero_grad()
#             losses = compute_decoy_loss(params, decoy_lists, benign_model,
#                                         decoy_loss_idx, batch, k)
#             for j in range(k):
#                 losses[j].backward(retain_graph=True)
#                 optimizer_lists[j].step()

#     # Save decoy model updates and implant indicators
#     for i in range(k):
#         dec_params = get_fl_update(decoy_lists[i], global_model)
#         if 'MNIST' in params.task:
#             logger.info(dec_params['fc1.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]] -
#                         benign_update['fc1.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]])
#         else:
#             logger.info(dec_params['fc.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]] -
#                         benign_update['fc.weight'][decoy_loss_idx[i][0]][decoy_loss_idx[i][1]])

#         # Implant the indicator
#         j = params.fl_number_of_adversaries + i
#         I = indicators[j]
#         dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].mul_(1e5)
#         # avoid zero value
#         if dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]] == 0:
#             if 'MNIST' in params.task:
#                 dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-2)
#             else:
#                 dec_params[ind_layer][I[0]][I[1]][I[2]][I[3]].add_(1e-3)
#         indicators[j] = [I, dec_params[ind_layer]
#                          [I[0]][I[1]][I[2]][I[3]].item()]

#         save_name = '{0}/saved_updates/update_{1}.pth'.format(params.folder_path,
#                                                               params.fl_total_participants-1-i)
#         torch.save(dec_params, save_name)
#     return indicators


def get_batch(batch_id, data, device='cuda'):
    """Process data into a batch.

    Specific for different datasets and data loaders this method unifies
    the output by returning the object of class Batch.
    :param batch_id: id of the batch
    :param data: object returned by the Loader.
    :return:
    """
    inputs, labels = data
    batch = Batch(batch_id, inputs, labels)
    return batch.to(device)


@dataclass
class Batch:
    batch_id: int
    inputs: torch.Tensor
    labels: torch.Tensor

    # For PIPA experiment we use this field to store identity label.
    aux: torch.Tensor = None

    def __post_init__(self):
        self.batch_size = self.inputs.shape[0]

    def to(self, device):
        inputs = self.inputs.to(device)
        labels = self.labels.to(device)
        if self.aux is not None:
            aux = self.aux.to(device)
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)

    def clone(self):
        inputs = self.inputs.clone()
        labels = self.labels.clone()
        if self.aux is not None:
            aux = self.aux.clone()
        else:
            aux = None
        return Batch(self.batch_id, inputs, labels, aux)

    def clip(self, batch_size):
        if batch_size is None:
            return self

        inputs = self.inputs[:batch_size]
        labels = self.labels[:batch_size]

        if self.aux is None:
            aux = None
        else:
            aux = self.aux[:batch_size]

        return Batch(self.batch_id, inputs, labels, aux)


def make_backdoor_batch(poisoning_proportion, backdoor_dynamic_position, resize_scale, pattern_tensor, input_shape, mask_value, device, backdoor_label, batch: Batch, test=False, attack=True) -> Batch:

    # Don't attack if only normal loss task.
    if not attack:
        return batch

    if test:
        attack_portion = batch.batch_size
    else:
        attack_portion = round(
            batch.batch_size * poisoning_proportion)

    backdoored_batch = batch.clone()
    synthesize_inputs(batch, backdoor_dynamic_position, resize_scale, pattern_tensor,
                      input_shape, mask_value, device, attack_portion=attack_portion)
    synthesize_labels(backdoor_label, batch=batch,
                      attack_portion=attack_portion)

    return backdoored_batch


def synthesize_inputs(batch, backdoor_dynamic_position, resize_scale, pattern_tensor, input_shape, mask_value, device, attack_portion=None):
    pattern, mask = get_pattern(
        backdoor_dynamic_position, resize_scale, pattern_tensor, input_shape, mask_value, device)
    batch.inputs[:attack_portion] = (1 - mask) * \
        batch.inputs[:attack_portion] + \
        mask * pattern

    return


def synthesize_labels(backdoor_label, batch, attack_portion=None):
    batch.labels[:attack_portion].fill_(backdoor_label)

    return


def get_pattern(backdoor_dynamic_position, resize_scale, pattern_tensor, input_shape, mask_value, device):
    if backdoor_dynamic_position:
        resize = random.randint(resize_scale[0], resize_scale[1])
        pattern = pattern_tensor
        if random.random() > 0.5:
            pattern = functional.hflip(pattern)
        image = transforms.ToPILImage()(pattern)
        pattern = transforms.ToTensor()(
            functional.resize(image, resize, interpolation=0)).squeeze()

        x = random.randint(0, input_shape[1] - pattern.shape[0] - 1)
        y = random.randint(0, input_shape[2] - pattern.shape[1] - 1)
        pattern, mask = make_pattern(
            pattern, x, y, input_shape, mask_value, device)
    else:
        pattern, mask = pattern_tensor, torch.zeros(input_shape).to(device)

    return pattern, mask


def make_pattern(pattern_tensor, x_top, y_top, input_shape, mask_value, device):
    full_image = torch.zeros(input_shape)
    full_image.fill_(mask_value)

    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    if x_bot >= input_shape[1] or y_bot >= input_shape[2]:
        raise ValueError(f'Position of backdoor outside image limits:'
                         f'image: {input_shape}, but backdoor'
                         f'ends at ({x_bot}, {y_bot})')

    full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor

    mask = 1 * (full_image != mask_value).to(device)
    pattern = full_image.to(device)

    return pattern, mask




def get_update_norm(local_update_dict):
    squared_sum = 0
    for name, value in local_update_dict.items():
        if 'tracked' in name or 'running' in name:
            continue
        squared_sum += torch.sum(torch.pow(value, 2)).item()
    update_norm = math.sqrt(squared_sum)
    return update_norm


def scale_update(local_update_dict, gamma):
    for name, value in local_update_dict.items():
        if value.dtype == torch.long or value.dtype == torch.int64:
            # For integer tensors, we need to convert, scale, then convert back
            float_tensor = value.float()
            float_tensor.mul_(gamma)
            local_update_dict[name] = float_tensor.long()
        else:
            # For floating point tensors, direct multiplication works
            value.mul_(gamma)

# def benign_training(global_model: nn.Module):
#     benign_model = copy.deepcopy(global_model, local_dataset)
#     if params.optimizer == 'SGD':
#         benign_optimizer = optim.SGD(benign_model.parameters(),
#                                 lr=params.lr,
#                                 weight_decay=params.decay,
#                                 momentum=params.momentum)
#     elif params.optimizer == 'Adam':
#         benign_optimizer = optim.Adam(benign_model.parameters(),
#                                 lr=params.lr,
#                                 weight_decay=params.decay)

#     benign_model.train()
#     for _ in range(params.fl_local_epochs):
#         for i, data in enumerate(local_dataset):
#             batch = get_batch(i, data)
#             benign_model.zero_grad()
#             loss = compute_blind_loss(benign_model, 
#                     nn.CrossEntropyLoss(reduction='none'), 
#                     batch, attack=False, fixed_model=None)
#             loss.backward()
#             benign_optimizer.step()
#             if i == params.max_batch_id:
#                 break
#     return benign_model