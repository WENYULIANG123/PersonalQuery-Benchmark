import torch
from transformers import LogitsProcessorList, LogitsProcessor

class FudgeProcessor(LogitsProcessor):
    def __init__(self, model, ling_disc, target_indices, tokenizer, fudge_lambda, ling_mask=None):
        super().__init__()
        self.target_indices = target_indices
        self.model = model
        self.ling_disc = ling_disc
        self.tokenizer = tokenizer
        self.lng_dim = target_indices.shape[1]
        self.fudge_lambda = fudge_lambda
        self.ling_mask = ling_mask

    def __call__(self, input_ids, scores):
        topk = 200
        top_logits, top_indices = scores.topk(topk, dim=1) 

        bs = input_ids.shape[0]
        input_ids_t1 = torch.cat(
                [input_ids.unsqueeze(1).expand(-1, topk, -1),
                    top_indices.unsqueeze(2),
                    ],
                dim=2)
        input_ids_t1_w_bos = torch.cat([
            input_ids_t1,
            torch.full(input_ids_t1.shape[:-1], self.tokenizer.bos_token_id, device=input_ids.device).unsqueeze(-1),
            ], dim=2).view(bs * topk, -1)
        ling_preds = self.ling_disc(input_ids=input_ids_t1_w_bos)
        ling_preds = ling_preds.view(bs, topk, self.lng_dim)
        if self.ling_mask is not None:
            error = ((ling_preds - self.target_indices.unsqueeze(1))**2)
            masked_error = error * self.ling_mask.unsqueeze(1)
            error = masked_error.sum(-1) / (self.ling_mask.sum(-1).unsqueeze(1))
        else:
            error = ((ling_preds - self.target_indices.unsqueeze(1))**2).mean(-1)
        favor = 1 / (error + 1e-12)
        favor = (favor - favor.min()) / (favor.max() - favor.min())
        scores[:,top_indices] += self.fudge_lambda * favor
        return scores

def fudge_predict(model, ling_disc, target_indices, tokenizer, fudge_lambda, ling_mask=None):
    fudge_processor = FudgeProcessor(model, ling_disc, target_indices, tokenizer, fudge_lambda, ling_mask)
    logits_processor = LogitsProcessorList([fudge_processor])
    bs = target_indices.shape[0]
    pred =  model.backbone.generate(
            inputs=torch.full((target_indices.shape[0], 1), tokenizer.bos_token_id, device=target_indices.device),
            logits_processor = logits_processor,
            eos_token_id=tokenizer.eos_token_id,
            renormalize_logits=True,
            do_sample=True,
            min_new_tokens=3,
            top_p=0.80,
            repetition_penalty=1.2,
            temperature=1.0,
            top_k=10,
            # top_p=0.80,
            max_length=100
            )
    return pred
