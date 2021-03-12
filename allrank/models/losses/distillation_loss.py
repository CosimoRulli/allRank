import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.losses import DEFAULT_EPS
from allrank.models.losses.ordinal import with_ordinals
import torch.nn.functional as F
from torch.nn import BCELoss
from allrank.models.model_utils import get_torch_device
import torch.nn.functional


def caruana_distillation_ndcg_loss(y_stud, y_teach, y_true, gt_loss_func, alpha, padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):
    """
    Caruana distillation loss for ordinal loss on the predictions of the single labels,
    :param y_stud: [batch_size, slate_length], student's output
    :param y_teach: [batch_size, slate_lengt], teacher's output
    :param y_true: [batch_size, slate_length], ground truth labels
    :param gt_loss_func: loss function between y_stud and y_true
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss [batch_size, slate_length]
    """

    y_stud = y_stud.clone()
    y_teach = y_teach.clone()

    if alpha != 1:
        gt_loss = gt_loss_func(y_stud, y_true.clone())
    else:
        gt_loss = 0.
    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    distill_loss = F.mse_loss(y_stud, y_teach, reduction="none")
    distill_loss[mask] = 0.0
    d_loss = torch.sum(distill_loss) / torch.sum(valid_mask)

    total_loss = alpha * d_loss + (1 - alpha) * gt_loss

    return total_loss



def caruana_distillation_loss_2_weighted(y_stud, y_teach, y_true, gt_loss_func, alpha, padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):
    y_stud = y_stud.clone()
    y_teach = y_teach.clone()

    if alpha != 1:
        #todo richiede l'applicazione della sigmoide  prima di fuznionare
        gt_loss = gt_loss_func(y_stud, y_true.clone())
    else:
        gt_loss = 0.
    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator
    expanded_y_true = with_ordinals(y_true, y_stud.size(-1));

    masked_y_stud = y_stud * expanded_y_true
    distill_loss = F.mse_loss(masked_y_stud, y_teach, reduction='none')
    distill_loss = torch.sum(distill_loss, dim=-1)
    distill_loss[mask] = 0.0
    d_loss = torch.sum(distill_loss) / torch.sum(valid_mask)

    total_loss = alpha * d_loss + (1 - alpha) * gt_loss
    return total_loss




def caruana_distillation_loss_2(y_stud, y_teach, y_true, gt_loss_func, alpha, padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):
    """
    Caruana distillation loss for ordinal loss on the predictions of the single labels,
    :param y_stud: [batch_size, slate_length, n], student's output before the sigmoid activation
    :param y_teach: [batch_size, slate_length,n], teacher's output before the sigmoid activation
    :param y_true: [batch_size, slate_length], ground truth labels
    :param gt_loss_func: loss function between y_stud and y_true
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss [batch_size, slate_length]
    """

    y_stud = y_stud.clone()
    y_teach = y_teach.clone()

    if alpha != 1:
        gt_loss = gt_loss_func(y_stud, y_true.clone())
    else:
        gt_loss = 0.
    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    distill_loss = F.mse_loss(y_stud, y_teach, reduction='none')
    distill_loss = torch.sum(distill_loss, dim=-1)
    distill_loss[mask] = 0.0
    d_loss = torch.sum(distill_loss) / torch.sum(valid_mask)

    total_loss = alpha * d_loss + (1 - alpha) * gt_loss
    return total_loss


def caruana_distillation_loss(y_stud, y_teach, y_true, gt_loss_func, alpha, padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):
    """
    Caruana distillation loss for ordinal loss,
    :param y_stud: [batch_size, slate_length, n], student's output before the sigmoid activation
    :param y_teach: [batch_size, slate_length,n], teacher's output before the sigmoid activation
    :param y_true: [batch_size, slate_length], ground truth labels
    :param gt_loss_func: loss function between y_stud and y_true
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss [batch_size, slate_length]
    """

    y_stud = y_stud.clone()
    y_teach = y_teach.clone()

    if alpha != 1:
        gt_loss = gt_loss_func(y_stud, y_true.clone())
    else:
        gt_loss = 0.

    y_teach = torch.sum(y_teach, -1)
    y_stud = torch.sum(y_stud, -1)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    distill_loss = F.mse_loss(y_stud, y_teach, reduction='none')
    distill_loss[mask] = 0.0
    d_loss = torch.sum(distill_loss) / torch.sum(valid_mask)

    #d_loss = torch.sum(distill_loss, dim =-1)
    #sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)
    total_loss = alpha * d_loss + (1 - alpha) * gt_loss
    return total_loss


def hinton_distillation_loss(y_stud, y_teach, y_true, gt_loss_func, alpha,  padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):

    device = get_torch_device()
    y_stud = y_stud.clone()
    y_teach = y_teach.clone()
    y_true = y_true.clone()
    gt_loss = gt_loss_func(y_stud, y_true.clone())
    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator
    #y_true = with_ordinals(y_true.clone(), y_stud.size(-1))


    stud_log = torch.log(y_stud)
    distill_loss = -torch.sum(y_teach * stud_log, dim=-1)
    distill_loss[mask] = 0.0

    d_loss = torch.sum(distill_loss) / torch.sum(valid_mask)
    #d_loss = torch.mean(distill_loss)
    total_loss = alpha * d_loss + (1 - alpha) * gt_loss
    return total_loss



    #mask = y_true == padded_value_indicator
    #y_stud[mask] = float('-inf')
    #y_teach[mask] = float('-inf')
    #y_stud = y_stud / temperature
    #y_teach = y_teach / temperature
    #pred_stud = F.softmax(y_stud, dim=1) + eps
    #pred_teach = F.softmax(y_teach, dim=1) + eps

    #stud_log = torch.log(pred_stud)

    #distill_loss = torch.mean(-torch.sum(pred_teach * stud_log, dim=1))



def caruana_distillation_loss_full(y_stud, hidden_stud, attn_std, y_teach, hidden_teach, attn_teach, y_true, gt_loss_func, alpha, padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):

    last_layer_loss = caruana_distillation_loss(y_stud, y_teach, y_true, gt_loss_func, alpha=0.0)

    intermediate_loss = compute_intermediate_distillation(hidden_stud, attn_std, hidden_teach, attn_teach, y_true == padded_value_indicator)

    return last_layer_loss + alpha * intermediate_loss




def hinton_distillation_loss_bce(y_stud, y_teach, y_true, gt_loss_func, alpha, padded_value_indicator=PADDED_Y_VALUE,
                      eps=DEFAULT_EPS):

    device = get_torch_device()
    y_stud = y_stud.clone()
    y_teach = y_teach.clone()

    if alpha != 1.0:
        gt_loss = gt_loss_func(y_stud, y_true.clone())
    else:
        gt_loss = 0
    y_true = with_ordinals(y_true.clone(), y_stud.size(-1))

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_stud, y_teach)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    distill_loss = torch.sum(document_loss) / torch.sum(sum_valid)

    total_loss = alpha * distill_loss + (1 - alpha) * gt_loss

    return total_loss



def hinton_distillation_loss_full(y_stud, hidden_stud, attn_std, y_teach, hidden_teach, attn_teach, y_true,
                                  gt_loss_func, alpha,padded_value_indicator=PADDED_Y_VALUE,
                                  eps=DEFAULT_EPS):

    final_loss = hinton_distillation_loss_bce(y_stud, y_teach, y_true, gt_loss_func, alpha = 1.0 )

    intermediate_distillation_loss = compute_intermediate_distillation(hidden_stud, attn_std, hidden_teach, attn_teach,
                                                          y_true == padded_value_indicator)
    loss = final_loss + alpha * intermediate_distillation_loss

    return loss

def compute_intermediate_distillation(hidden_stud, attn_std, hidden_teach, attn_teach , mask):

    hidden_stud = hidden_stud.clone()
    attn_std = attn_std.clone()
    hidden_teach = hidden_teach.clone()
    attn_teach = attn_teach.clone()
    inf_mask = attn_std == float("-inf")
    attn_teach[inf_mask] = 0.0
    attn_std[inf_mask] = 0.0
    attn_loss = F.mse_loss(attn_std, attn_teach)

    hidden_stud[mask] = 0.
    hidden_teach[mask] = 0.

    hidden_state_loss = F.mse_loss(hidden_stud, hidden_teach)
    return hidden_state_loss + attn_loss