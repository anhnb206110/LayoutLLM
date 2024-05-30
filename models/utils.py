import torch

def pad_input_ids(question_ids, target_length, pad_token_id=0):
    current_length = question_ids.size(1)
    if current_length >= target_length:
        return question_ids
    padding_length = target_length - current_length
    padding_tensor = torch.full((question_ids.size(0), padding_length), pad_token_id, dtype=question_ids.dtype)
    padded_input_ids = torch.cat([question_ids.cpu(), padding_tensor.cpu()], dim=1).cpu()

    return padded_input_ids

def pad_bbox(bbox, target_length, pad_value=None):
    current_length = len(bbox[0])
    if current_length >= target_length:
        return bbox
    padding_length = target_length - current_length
    if pad_value is None:
        pad_value = torch.tensor([0,0,0,0])
    padding_tensor = torch.zeros((bbox.size(0), padding_length, 4), dtype=bbox.dtype, device=bbox.device) + pad_value
    padded_bbox = torch.cat([bbox.cpu(), padding_tensor.cpu()], dim=1).cpu()
    return padded_bbox

    
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
