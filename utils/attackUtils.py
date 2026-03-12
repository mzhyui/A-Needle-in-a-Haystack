from copy import deepcopy
import copy
import random
import numpy as np
from PIL import Image

from sklearn import preprocessing
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms

from utils.dataUtils import ImageLabelDataset, TensorDatasetImg


def DataSet_distill_clean_data(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)
    unloader = transforms.ToPILImage()
    list_clean_data_knowledge_distill = []
    for i, (inputs, targets) in enumerate(dataloader):
        # print('target:', target[0])
        # sys.exit()
        # if distill_data_name=="cifar100":
        #     if target[0] in [13, 58, 81, 89]:
        #         # print(target[0])
        #         continue
        inputs, targets = inputs.to(device), targets.to(device)
        # compute output
        with torch.no_grad():
            output = model(inputs)
        # print('Output size:', output.size())
        # print(output)
        for j in range(inputs.size(0)):  # 遍历批次中的每个样本

            input_i = inputs[j]  # 获取第j个样本的输入

            output_i = output[j]  # 获取第j个样本的输出
            targets_i = targets[j]  # 获取第j个样本的标签

            # 转换成 PIL 图像
            input_i = unloader(input_i)

            list_clean_data_knowledge_distill.append(
                (input_i, output_i, targets_i))
    # torch.save(list_clean_data_knowledge_distill,
    #            './dataset/distill_' + distill_data_name)

    return list_clean_data_knowledge_distill


def select_img(dataset, images_batch, outputs_batch, batch_n, batch_size, com_ratio):
    data_compression = []
    data_num = images_batch.shape[0]
    max_num = int(data_num * com_ratio)
    if max_num * data_num == 0:
        return []
    n_selected = 0
    images_sim = np.dot(images_batch, images_batch.transpose())
    outputs_sim = np.dot(outputs_batch, outputs_batch.transpose())
    co_sim = np.multiply(images_sim, outputs_sim)

    index = random.randint(0, data_num - 1)

    while n_selected < max_num:
        index = np.argmin(co_sim[index])  # Select the least similar image
        data_compression.append(dataset[batch_n * batch_size + index])
        n_selected += 1
        co_sim[:, index] = 1  # Mark as selected
    return data_compression


def preprocessDistilledData(model, dataloader, batch_size, dataset_str, device='cuda'):
    dataset = DataSet_distill_clean_data(model, dataloader, device=device)
    random.shuffle(dataset)
    data_num = len(dataset)

    images = []
    outputs = []
    targets = []
    for i in range(data_num):
        img = np.array(dataset[i][0]).flatten()
        output = np.array(dataset[i][1].cpu())
        target = dataset[i][2]
        img = img.reshape(1, -1)
        images.append(preprocessing.normalize(img, norm='l2').squeeze())
        output = output.reshape(1, -1)
        outputs.append(preprocessing.normalize(output, norm='l2').squeeze())
        targets.append(target)
    images = np.array(images)
    outputs = np.array(outputs)

    batch_num = int(data_num / batch_size) + (data_num % batch_size != 0)
    data_compression = []
    com_ratio = 0.5

    def inner_select_img(images_batch, outputs_batch, batch_n):
        data_num = images_batch.shape[0]
        max_num = int(data_num * com_ratio)
        if max_num * data_num == 0:
            return []
        n_selected = 0
        images_sim = np.dot(images_batch, images_batch.transpose())
        # print(images_sim)
        # sys.exit()
        outputs_sim = np.dot(outputs_batch, outputs_batch.transpose())
        co_sim = np.multiply(images_sim, outputs_sim)
        # print(co_sim)
        # sys.exit()

        index = random.randint(0, data_num - 1)
        # print(index)

        while n_selected < max_num:
            index = np.argmin(co_sim[index])
            data_compression.append(dataset[batch_n * batch_size + index])
            n_selected += 1
            co_sim[:, index] = 1

        return []

    for i in range(batch_num):
        images_batch = images[i *
                              batch_size:min((i + 1) * batch_size, data_num)]
        outputs_batch = outputs[i *
                                batch_size:min((i + 1) * batch_size, data_num)]
        # data_compression += select_img(dataset, images_batch,
        #                                outputs_batch, i, batch_size, com_ratio)
        inner_select_img(images_batch, outputs_batch, i)

    images = []
    labels = []
    soft_labeels = []
    for i, data_item in enumerate(data_compression):
        img = data_item[0]
        out = data_item[1]
        label = data_item[2]
        images.append(img)
        # labels.append(torch.argmax(out))
        labels.append(label)
        soft_labeels.append(out)


    train_set_soft = TensorDatasetImg(images, soft_labeels, dataset_str)
    # TODO 2025-10-20 git.V.a1f8f: update with soft labels
    train_set = ImageLabelDataset(images, labels, dataset_str)
    return train_set

# TODO 2025-02-27 git.V.894ba: why use cos_sim between malicious?


def compute_cos_sim_loss_1(local_model, dataset_str, shared_models, fake_model, device='cuda'):
    loss = 0

    global_model = copy.deepcopy(local_model)
    global_vec = get_one_vec_variable(dataset_str, global_model, False, device=device)
    local_vec = get_one_vec_variable(dataset_str, local_model, True, device=device)
    update_vec = local_vec - global_vec
    # for i, shared_model in enumerate(shared_models):
    for _, other_model in enumerate(shared_models):

        # loaded_params = torch.load(shared_model)
        # other_model = deepcopy(local_model)
        # other_model.load_state_dict(loaded_params)
        other_vec = get_one_vec_variable(dataset_str, other_model, False, device=device)
        cs_sim = F.cosine_similarity(update_vec, other_vec, dim=0)
        cs_sim = (cs_sim) ** 2
        loss += cs_sim

    fake_vec = get_one_vec_variable(dataset_str, fake_model, False, device=device)
    fake_norm_update_vec = fake_vec-global_vec
    cs_sim = F.cosine_similarity(update_vec, fake_norm_update_vec, dim=0)
    cs_sim = (cs_sim)**2
    loss += cs_sim

    return loss


def compute_all_losses_and_grads(loss_tasks, attack, model, criterion,
                                 batch, batch_back,
                                 fixed_model=None):
    loss_values = {}
    for t in loss_tasks:
        if t == 'normal':
            loss_values[t] = compute_normal_loss(model,
                                                 criterion,
                                                 batch.inputs,
                                                 batch.labels)
        elif t == 'backdoor':
            loss_values[t] = compute_backdoor_loss(model,
                                                   criterion,
                                                   batch_back.inputs,
                                                   batch_back.labels)
        elif t == 'eu_constraint':
            loss_values[t] = compute_euclidean_loss(attack.params,
                                                    model,
                                                    fixed_model)
        elif t == 'cs_constraint':
            loss_values[t] = compute_cos_sim_loss(attack.params,
                                                  model,
                                                  fixed_model)

    return loss_values


def compute_cos_sim_loss(fl_weight_scale,
                         model,
                         fixed_model):
    model_vec = get_one_vec(model)
    target_var = get_one_vec(fixed_model)
    cs_sim = F.cosine_similarity(fl_weight_scale*(model_vec-target_var)
                                 + target_var, target_var, dim=0)
    loss = 1e3 * (1 - cs_sim)
    return loss


def compute_backdoor_loss(model, criterion, inputs_back, labels_back):
    outputs = model(inputs_back)
    loss = criterion(outputs, labels_back)
    loss = loss.mean()

    return loss


def compute_normal_loss(model, criterion, inputs, labels):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss = loss.mean()

    return loss


def compute_noise_ups_loss(backdoor_update, noise_masks, random_neurons, model_name):
    losses = []
    backdoor_update = backdoor_update.state_dict()
    for i, noise_mask in enumerate(noise_masks):
        UPs = []
        for j in random_neurons:
            if 'lenetsm' in model_name:  # TODO 2025-03-04 git.V.32546: fix
                UPs.append(torch.abs(backdoor_update['fc3.layer.weight'][j] +
                                     noise_mask.fc3.layer.weight[j]).sum()
                           + torch.abs(backdoor_update['fc3.layer.bias'][j] +
                                       noise_mask.fc3.layer.bias[j]))
            elif 'vggsm' in model_name:
                UPs.append(torch.abs(backdoor_update['fc14.layer.weight'][j] +
                                     noise_mask.fc14.layer.weight[j]).sum()
                           + torch.abs(backdoor_update['fc14.layer.bias'][j] +
                                       noise_mask.fc14.layer.bias[j]))
            elif 'resnet20sm' in model_name:
                UPs.append(torch.abs(backdoor_update['linear.layer.weight'][j] +
                                     noise_mask.linear.layer.weight[j]).sum()
                           + torch.abs(backdoor_update['linear.layer.bias'][j] +
                                       noise_mask.linear.layer.bias[j]))
            elif 'vitsm' in model_name:
                UPs.append(torch.abs(backdoor_update['classifier.layer.weight'][j] +
                                     noise_mask.classifier.layer.weight[j]).sum()
                           + torch.abs(backdoor_update['classifier.layer.bias'][j] +
                                       noise_mask.classifier.layer.bias[j]))
            else:
                raise ValueError('Unknown model', model_name)
        UPs_loss = 0
        for j, UP in enumerate(UPs):
            if 'vggsm' in model_name:
                UPs_loss += 5e-4 / UP
            else:
                UPs_loss += 1e-1 / UP  # (UPs[j] * params.fl_num_neurons)
        noise_masks[i].requires_grad_(True)
        UPs_loss.requires_grad_(True)
        losses.append(UPs_loss)
    return losses


def compute_noise_norm_loss(noise_masks, random_neurons, model_name, device='cuda'):
    size = 0
    # layer_name = 'fc2' if 'lenetsm' in model_name else 'fc'
    if 'lenetsm' in model_name:
        layer_name = 'fc3.layer'
    elif 'vggsm' in model_name:
        layer_name = 'fc14.layer'
    elif 'resnet20sm' in model_name:
        layer_name = 'linear.layer'
    elif 'vitsm' in model_name:
        layer_name = 'classifier.layer'
    else:
        raise ValueError('Unknown model', model_name)
    # print(noise_masks[0].named_parameters())
    for name, layer in noise_masks[0].named_parameters():
        if layer_name in name:
            size += layer.view(-1).shape[0]
    losses = []
    for i, noise_mask in enumerate(noise_masks):
        sum_var = torch.zeros(size, device=device)
        noise_size = 0
        for name, layer in noise_mask.named_parameters():
            if layer_name in name:
                for j in range(layer.shape[0]):
                    if j in random_neurons:
                        sum_var[noise_size:noise_size + layer[j].view(-1).shape[0]] = \
                            layer[j].view(-1)
                    noise_size += layer[j].view(-1).shape[0]
        if 'lenetsm' in model_name:
            loss = 8e-2 * torch.norm(sum_var, p=2)
        else:
            loss = 3e-2 * torch.norm(sum_var, p=2)
        losses.append(loss.item())
    return losses


def compute_lagrange_loss(noise_masks, random_neurons, model_name, device='cuda'):
    losses = []
    size = 0
    if 'lenetsm' in model_name:
        layer_name = 'fc3.layer'
    elif 'vggsm' in model_name:
        layer_name = 'fc14.layer'
    elif 'resnet20sm' in model_name:
        layer_name = 'linear.layer'
    elif 'vitsm' in model_name:
        layer_name = 'classifier.layer'
    else:
        raise ValueError('Unknown model', model_name)
    
    for name, layer in noise_masks[0].named_parameters():
        if layer_name in name:
            size += layer.view(-1).shape[0]
    sum_var = torch.zeros(size, device=device)
    for i, noise_mask in enumerate(noise_masks):
        size = 0
        for name, layer in noise_mask.named_parameters():
            if layer_name in name:
                for j in range(layer.shape[0]):
                    if j in random_neurons:
                        sum_var[size:size + layer[j].view(-1).shape[0]] += \
                            layer[j].view(-1)
                    size += layer[j].view(-1).shape[0]

    if 'lenetsm' in model_name:
        loss = 1e-1 * torch.norm(sum_var, p=2)
    else:
        loss = 1e-2 * torch.norm(sum_var, p=2)
    for i in range(len(noise_masks)):
        losses.append(loss.item())
    return losses


def compute_euclidean_loss(model, fixed_model, device='cuda'):
    size = 0
    for name, layer in model.named_parameters():
        size += layer.view(-1).shape[0]
    sum_var = torch.zeros(size, device=device)
    size = 0
    for name, layer in model.named_parameters():
        sum_var[size:size + layer.view(-1).shape[0]] = (layer -
                                                        fixed_model.state_dict()[name]).view(-1)
        size += layer.view(-1).shape[0]
    loss = torch.norm(sum_var, p=2)
    return loss


def compute_noise_loss(backdoor_update, noise_masks, alpha, random_neurons, lagrange_mul, model_name, device='cuda'):
    loss = []
    # Compute UPs loss
    ups_loss = compute_noise_ups_loss(
        backdoor_update, noise_masks, random_neurons, model_name)
    # Compute norm constrain
    norm_loss = compute_noise_norm_loss(
        noise_masks, random_neurons, model_name, device=device)
    for i in range(len(ups_loss)):
        loss.append(ups_loss[i] * alpha + norm_loss[i] * (1 - alpha))
    # Compute lagrange constrain
    lagrange_loss = compute_lagrange_loss(
        noise_masks, random_neurons, model_name, device=device)
    for i in range(len(ups_loss)):
        loss[i] += lagrange_mul * lagrange_loss[i]
        loss[i] /= (1 + lagrange_mul)
    return loss


def get_one_vec(model_or_state_dict, device='cuda'):
    # Check if the input is a model or a state_dict
    if isinstance(model_or_state_dict, torch.nn.Module):
        state_dict = model_or_state_dict.state_dict()
    elif isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        raise ValueError("Input must be a PyTorch model or state_dict.")

    size = sum(p.numel() for p in state_dict.values())
    sum_var = torch.zeros(size, device=device)
    index = 0
    for name, param in state_dict.items():
        # if 'fc' in name:
        numel = param.numel()
        sum_var[index:index + numel] = param.view(-1)
        index += numel

    return sum_var


def get_one_vec_variable(dataset_str, model, variable=False, device='cuda'):
    size = 0
    if dataset_str == 'GTSRB':
        s = 'fc2'
    else:
        s = 'fc'
    for name, layer in model.named_parameters():
        if s in name:
            size += layer.view(-1).shape[0]
    if variable:
        sum_var = Variable(torch.zeros(size, device=device))
    else:
        sum_var = torch.zeros(size, device=device)
    size = 0
    for name, layer in model.named_parameters():
        if s in name:
            if variable:
                sum_var[size:size + layer.view(-1).shape[0]] = (layer).view(-1)
            else:
                sum_var[size:size +
                        layer.view(-1).shape[0]] = (layer.data).view(-1)
            size += layer.view(-1).shape[0]

    return sum_var


# def train_with_grad_control(model, epoch, trainloader, criterion, optimizer, lambda1, task, former_model, user_id, fake_normal_model):

#     if task.params.task == 'Cifar10':
#         len1 = 10
#     elif task.params.task == 'CIFAR100':
#         len1 = 100
#     elif task.params.task == 'GTSRB':
#         len1 = 43
#     target_one_hot = torch.ones(len1).to(device)
#     ave_val = -10.0 / (len(target_one_hot))
#     target_one_hot = torch.mul(target_one_hot, ave_val)

#     model.eval()  # set as eval() to evade batchnorm

#     for i, (input, target, poisoned_flags) in enumerate(
#             trainloader):

#         img = input[0]
#         (channels, height, width) = img.shape
#         scale = 0.25
#         trigger_height = int(height * scale)
#         if trigger_height % 2 == 1:
#             trigger_height -= 1
#         trigger_width = int(width * scale)
#         if trigger_width % 2 == 1:
#             trigger_width -= 1
#         start_h = height - 2 - trigger_height
#         start_w = width - 2 - trigger_width

#         trans = transforms.ToTensor()
#         f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
#         trigger1 = Image.open(f).convert('RGB')
#         trigger1 = trans(trigger1)

#         index_clean = [index for (index, flag) in enumerate(
#             poisoned_flags) if flag == 0]
#         index_poison = [index for index, flag in enumerate(
#             poisoned_flags) if flag == 1]

#         input = input.to(device)
#         input[index_poison, :, start_h:start_h + trigger_height,
#               start_w:start_w + trigger_width] = trigger1.to(device)

#         target_one_hot1 = deepcopy(target_one_hot)
#         target_one_hot1[task.params.backdoor_label] = 10
#         target[index_poison] = target_one_hot1
#         target = target.to(device)
#         output = model(input)

#         output_clean = output[index_clean]
#         target_clean = target[index_clean]

#         output_poison = output[index_poison]
#         target_poison = target[index_poison]

#         loss_clean = criterion(output_clean, target_clean)
#         loss_poison = criterion(output_poison, target_poison)
#         eu_loss = 0
#         eu_loss = compute_euclidean_loss(model, task.model)
#         cos_loss = 0
#         if epoch > task.params.start_epoch:
#             cos_loss = compute_cos_sim_loss_1(
#                 model, task, user_id, fake_normal_model)
#         else:
#             cos_loss = compute_cos_sim_loss(model, task, user_id)
#         if len(output_poison) > 0:

#             loss = loss_clean + loss_poison+0.5*eu_loss+0.5*cos_loss
#         else:
#             loss = loss_clean+0.5*eu_loss+0.5*cos_loss
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

def predict_the_global_model(state_dict1, state_dict2, alpha):
    # s1:state_dict()
    sum_state_dict = {key: ((2-alpha)/1-alpha) * state_dict1[key] - (
        1/(1-alpha)) * state_dict2[key] for key in state_dict1.keys()}

    return sum_state_dict


def update_the_Ss(state_dict1, state_dict2, global_model_state_dict, alpha):
    s1_new = {key: alpha * global_model_state_dict[key] + (
        1-alpha) * state_dict1[key] for key in state_dict1.keys()}
    s2_new = {key: alpha * s1_new[key] + (1-alpha) * state_dict2[key]
              for key in state_dict2.keys()}
    return s1_new, s2_new


def get_fl_update(local_model_dict, global_model_dict):
    local_update = dict()
    for name, data in local_model_dict.items():
        # if self.check_ignored_weights(name):
        #     continue
        if 'num_batches_tracked' in name:
            continue
        local_update[name] = (data - global_model_dict[name])
    return local_update

# def ref_f(model, dataloader, num_class, batch_size, dim_f=10, device='cuda'):
#     model.eval()
#     F = {}
#     F_out = []
#     for class_ in range(num_class):
#         F[class_] = []
#     for batch_idx, (fs, labels) in enumerate(dataloader):
#         fs = fs.to(dtype=torch.float).to(device)
#         b, c, w, h = fs.shape
#         labels = torch.as_tensor(
#             labels, dtype=torch.long, device=device).view(-1, 1).squeeze().squeeze()
        
#         # Apply the same "divide by 8 and take mean" process to input
#         reduced_batch_size = b // 8
#         if reduced_batch_size > 0:
#             fs_reshaped = fs[:reduced_batch_size * 8].view(reduced_batch_size, 8, c, w, h)
#             fs_reduced = fs_reshaped.mean(dim=1)  # Take mean across the 8 samples
            
#             # Also need to handle labels accordingly
#             labels_reshaped = labels[:reduced_batch_size * 8].view(reduced_batch_size, 8)
#             labels_reduced = labels_reshaped.mode(dim=1)[0]  # Take mode (most frequent) label
            
#             out, features = model.forward_feature(fs_reduced)
            
#             # Expand features back to original batch size
#             features = features.repeat_interleave(8, dim=0)
#             labels = labels.repeat_interleave(8, dim=0) if labels.dim() == 1 else labels_reduced.repeat_interleave(8, dim=0)
            
#             # Handle remainder
#             if b % 8 != 0:
#                 remainder = b % 8
#                 fs_remainder = fs[reduced_batch_size * 8:]
#                 labels_remainder = labels[reduced_batch_size * 8:]
#                 out_remainder, features_remainder = model.forward_feature(fs_remainder)
                
#                 features = torch.cat([features, features_remainder], dim=0)
#                 labels = torch.cat([labels, labels_remainder], dim=0)
#         else:
#             # If batch size < 8, process normally
#             out, features = model.forward_feature(fs)
        
#         # Rest of the processing remains the same
#         for b_ in (range(batch_size)):
#             if b_ < len(labels):
#                 label = labels[b_].item()
#                 tmp = features[b_,:].detach().cpu()
#                 F[label].append(tmp)
    
#     for class_ in range(num_class):
#         if F[class_] == []:
#             F[class_].append(torch.rand_like(features[0, :]).detach().cpu())
    
#     for class_ in range(num_class):
#         F[class_] = torch.stack(F[class_]).mean(dim=0).unsqueeze(0)
        
#         # Apply the same process to F_out shape - divide by 8 and take mean
#         reduced_batch_size = batch_size // 8
#         if reduced_batch_size > 0:
#             F[class_] = F[class_].expand(reduced_batch_size, dim_f)
#             F[class_] = F[class_].repeat_interleave(8, dim=0)
            
#             # Handle remainder
#             if batch_size % 8 != 0:
#                 remainder = batch_size % 8
#                 last_feature = F[class_][-1:].repeat(remainder, 1)
#                 F[class_] = torch.cat([F[class_], last_feature], dim=0)
#         else:
#             F[class_] = F[class_].expand(batch_size, dim_f)
            
#         F_out.append(F[class_])
    
#     F_out = torch.stack(F_out)
#     F_out = F_out.permute(1, 0, 2).reshape(num_class*batch_size, dim_f).view(batch_size, num_class, dim_f).mean(1)
#     return F_out.to(device)

def ref_f(model, dataloader, num_class, batch_size, dim_f=10, device='cuda'):
    model.eval()
    F = {}
    F_out = []
    for class_ in range(num_class):
        F[class_] = []
    for batch_idx, (fs, labels) in enumerate(dataloader):
        fs = fs.to(dtype=torch.float).to(device)
        b, c, w, h = fs.shape
        labels = torch.as_tensor(
            labels, dtype=torch.long, device=device).view(-1, 1).squeeze().squeeze()
        out, features = model.forward_feature(fs)
        # print('batch_idx', batch_idx, 'features shape', features.shape)
        for b_ in (range(batch_size)):
            label = labels[b_].item()
            tmp = features[b_,:].detach().cpu()
            F[label].append(tmp)

    for class_ in range(num_class):
        if F[class_] == []:
            # F[class_].append(features[class_, :].detach().cpu()) # TODO 2025-03-06 git.V.33573: solve empty label
            # replace with rand vector like features
            F[class_].append(torch.rand_like(features[0, :]).detach().cpu())

    for class_ in range(num_class):
        F[class_] = torch.stack(F[class_]).mean(dim=0).unsqueeze(0)
        # print('F[class_]', F[class_].shape)
        # dim_f = num_class
        F[class_] = F[class_].expand(batch_size, dim_f)
        F_out.append(F[class_])
    
    F_out = torch.stack(F_out)
    F_out = F_out.permute(1, 0, 2).reshape(num_class*batch_size, dim_f)
    return F_out.to(device)
