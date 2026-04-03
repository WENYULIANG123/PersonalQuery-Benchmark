import torch
import torch.nn.functional as F

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    :param lengths: 1d tensor [batch_size]
    :param max_len: int
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1))
            .long())



def mono_annealing(step, period=40000):
    return min(step / period, 1)


def cyclic_annealing(step, period1=2000, period2=3000):
    residual = step % period2
    return min(residual / period1, 1)

def my_undecorate(func, name):
    for cell in func.__closure__:
        if hasattr(cell.cell_contents, '__name__') and cell.cell_contents.__name__  == name:
            return cell.cell_contents
    return None
 
def dpo_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps,
        beta=0.1, label_smoothing=0):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        if label_smoothing > 0:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            losses = -F.logsigmoid(beta * logits)

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

def concatenate_rejected(batch):
    batch['input_ids'] = torch.cat((batch['input_ids'], batch['rejected_input_ids']), dim=0)
    batch['attention_mask'] = torch.cat((batch['attention_mask'], batch['rejected_attention_mask']), dim=0)
    batch['ling'] = torch.cat((batch['ling'], batch['ling']), dim=0)
    if 'ling_attention_mask' in batch:
        batch['ling_attention_mask'] = torch.cat((batch['ling_attention_mask'], batch['ling_attention_mask']), dim=0)
    return batch
