import torch

from models.softmaskLayer import SoftMaskedLayer
from models.resnet import BasicBlockSM
from models.ViT import EncoderSM, BlockSM, MLPSM, AttentionHeadSM, EmbeddingsSM, PatchEmbeddingsSM


def set_compute_mask_impt(model, compute_impt):
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            m.layer.compute_mask_impt = compute_impt
        elif isinstance(m, BasicBlockSM):
            m.conv1.layer.compute_mask_impt = compute_impt
            m.conv2.layer.compute_mask_impt = compute_impt
            for k, q in m.shortcut.named_modules():
                if isinstance(q, SoftMaskedLayer):
                    q.layer.compute_mask_impt = compute_impt
        elif isinstance(m, AttentionHeadSM):
            m.query.layer.compute_mask_impt = compute_impt
            m.key.layer.compute_mask_impt = compute_impt
            m.value.layer.compute_mask_impt = compute_impt
        elif isinstance(m, MLPSM):
            m.dense_1.layer.compute_mask_impt = compute_impt
            m.dense_2.layer.compute_mask_impt = compute_impt
        elif isinstance(m, PatchEmbeddingsSM):
            m.projection.layer.compute_mask_impt = compute_impt
        elif isinstance(m, EncoderSM):
            for block in m.blocks:
                if isinstance(block, BlockSM):
                    for subname, subblock in block.attention.heads.named_modules():
                        if isinstance(subblock, AttentionHeadSM):
                            subblock.query.layer.compute_mask_impt = compute_impt
                            subblock.key.layer.compute_mask_impt = compute_impt
                            subblock.value.layer.compute_mask_impt = compute_impt
        elif isinstance(m, EmbeddingsSM):
            m.patch_embeddings.projection.layer.compute_mask_impt = compute_impt


# def set_module_impt(module, compute_impt):



def set_ft_task(model, ft_task):
    for n, m in model.named_modules():
        if isinstance(m, SoftMaskedLayer):
            m.layer.ft_task = ft_task
        elif isinstance(m, BasicBlockSM):
            m.conv1.layer.ft_task = ft_task
            m.conv2.layer.ft_task = ft_task
            for k, q in m.shortcut.named_modules():
                if isinstance(q, SoftMaskedLayer):
                    q.layer.ft_task = ft_task
        elif isinstance(m, AttentionHeadSM):
            m.query.layer.ft_task = ft_task
            m.key.layer.ft_task = ft_task
            m.value.layer.ft_task = ft_task
        elif isinstance(m, MLPSM):
            m.dense_1.layer.ft_task = ft_task
            m.dense_2.layer.ft_task = ft_task
        elif isinstance(m, PatchEmbeddingsSM):
            m.projection.layer.ft_task = ft_task
        elif isinstance(m, EncoderSM):
            for block in m.blocks:
                if isinstance(block, BlockSM):
                    for subname, subblock in block.attention.heads.named_modules():
                        if isinstance(subblock, AttentionHeadSM):
                            subblock.query.layer.ft_task = ft_task
                            subblock.key.layer.ft_task = ft_task
                            subblock.value.layer.ft_task = ft_task
        elif isinstance(m, EmbeddingsSM):
            m.patch_embeddings.projection.layer.ft_task = ft_task


# def set_ft_task(model, ft_task):
#     count_smlayers=0
#     for n, m in model.named_modules():
#         if isinstance(m, SoftMaskedLayer):
#             print(0)
#             count_smlayers+=1
#             m.layer.ft_task = ft_task
#         elif isinstance(m, BasicBlockSM):
#             print(1)
#             count_smlayers+=1
#             m.conv1.layer.ft_task = ft_task
#             m.conv2.layer.ft_task = ft_task
#             for k, q in m.shortcut.named_modules():
#                 if isinstance(q, SoftMaskedLayer):
#                     q.layer.ft_task = ft_task
#         elif isinstance(m, AttentionHeadSM):
#             print(2)
#             count_smlayers+=1
#             m.query.layer.ft_task = ft_task
#             m.key.layer.ft_task = ft_task
#             m.value.layer.ft_task = ft_task
#         elif isinstance(m, MLPSM):
#             print(3)
#             count_smlayers+=1
#             m.dense_1.layer.ft_task = ft_task
#             m.dense_2.layer.ft_task = ft_task
#         elif isinstance(m, PatchEmbeddingsSM):
#             print(4)
#             count_smlayers+=1
#             m.projection.layer.ft_task = ft_task
#         elif isinstance(m, EncoderSM):
#             print(5)
#             for block in m.blocks:
#                 if isinstance(block, BlockSM):
#                     for subname, subblock in block.attention.heads.named_modules():
#                         if isinstance(subblock, AttentionHeadSM):
#                             print(6)
#                             count_smlayers+=1
#                             subblock.query.layer.ft_task = ft_task
#                             subblock.key.layer.ft_task = ft_task
#                             subblock.value.layer.ft_task = ft_task
#         elif isinstance(m, EmbeddingsSM):
#             print(7)
#             count_smlayers+=1
#             m.patch_embeddings.projection.layer.ft_task = ft_task

#     layer_names = [n for n, m in model.named_modules()
#                    if isinstance(m, SoftMaskedLayer)]
#     print(len(layer_names))
#     print(count_smlayers)
#     count_totallayers = 0
#     for n, m in model.named_modules():
#         count_totallayers += 1
#     print(count_totallayers)



def impt_norm(impt:torch.Tensor) -> torch.Tensor:
    impt = impt.clone()
    tanh = torch.nn.Tanh()
    for layer in range(impt.size(0)):
        impt[layer] = (impt[layer] - impt[layer].mean()) / (impt[
            layer].std() + 1e-6)  # 2D, we need to deal with this for each layer
    impt = tanh(impt).abs()

    return impt
