import torch
from torch.nn import BCELoss

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_torch_device


def with_ordinals(y, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Helper function for ordinal loss, transforming input labels to ordinal values.
    :param y: labels, shape [batch_size, slate_length]
    :param n: number of ordinals
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: ordinals, shape [batch_size, slate_length, n]
    """

    dev = get_torch_device()
    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=dev)
    unsqueezed = y.unsqueeze(2).repeat(1, 1, n)
    mask = unsqueezed == padded_value_indicator
    ordinals = (unsqueezed >= one_to_n).type(torch.float)
    ordinals[mask] = padded_value_indicator
    return ordinals

def with_smoothed_ordinals(y, n, padded_value_indicator=PADDED_Y_VALUE):
    dev = get_torch_device()
    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=dev)
    unsqueezed = y.unsqueeze(2).repeat(1, 1, n)
    mask = unsqueezed == padded_value_indicator

    upper_mask = (unsqueezed < one_to_n)
    new_ordinals = (unsqueezed >= one_to_n).type(torch.int) * one_to_n
    new_ordinals = new_ordinals / (torch.max(new_ordinals, -1))[0].unsqueeze(-1)
    new_ordinals[upper_mask] = 0.0
    new_ordinals[mask] = padded_value_indicator
    return new_ordinals


def ordinal(y_pred, y_true, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Ordinal loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length, n]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param n: number of ordinal values, int
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = get_torch_device()

    y_pred = y_pred.clone()
    y_true = with_ordinals(y_true.clone(), n)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output


def weighted_ordinal(y_pred, y_true, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Ordinal loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length, n]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param n: number of ordinal values, int
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = get_torch_device()

    y_pred = y_pred.clone()
    y_true = with_ordinals(y_true.clone(), n)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0
    weights = y_true + 1

    document_loss = torch.sum(ls*weights, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output


def weighted_ordinal_2(y_pred, y_true, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Ordinal loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length, n]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param n: number of ordinal values, int
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = get_torch_device()

    y_pred = y_pred.clone()
    weights = y_true + 1

    y_true = with_ordinals(y_true.clone(), n)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator


    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss*weights) / torch.sum(sum_valid)

    return loss_output



def weighted_ordinal_3(y_pred, y_true, n, padded_value_indicator=PADDED_Y_VALUE):
    """
    Ordinal loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length, n]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param n: number of ordinal values, int
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = get_torch_device()

    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=device)

    y_pred = y_pred.clone()
    y_true = with_ordinals(y_true.clone(), n)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator


    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    weights = y_true * one_to_n

    document_loss = torch.sum(ls * weights, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output