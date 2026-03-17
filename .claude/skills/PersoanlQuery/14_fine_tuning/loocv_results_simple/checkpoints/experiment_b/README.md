---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:130
- loss:CosineSimilarityLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I am looking for glass cube beads that can be mixed with other
    beads and wood beads. I also need metal spacer beads to complete my jewelry designs.
  sentences:
  - I am looking infor think guage wire infor jewelry! projects that makes my designs
    look expensive. and high-quality. I need durable material suitable infor crafting
    beautifuly, luxury-style accessories.
  - I am looking infor spacer beads in a Rondelles shape that azuro Antiqued 22K Gold
    Plated Pewter Hexagon Rondelles Beads. I need a design on them. infor added interest.
  - I am looking infor glass cube beads that can be mixed with! other beads and wood
    beads. I also need metallix spacer beads to complete my jewelry! designs.
- source_sentence: I need wire in every gauge for making jewelry that features deep,
    rich colors. I want my finished designs to look expensive and luxurious.
  sentences:
  - I am looking infor Czech Use Beads featuring a beautifuly mix of color, specifically
    with! Matte Tortoise Browns and Beiges included. I need Czech Use Beads infor
    my upcoming jewelry! projects.
  - I need wire in every guage infor making jewelry! that features deep, rich color.
    I want my finish! designs to look expensive. and luxurious.
  - I am looking infor gold-colored wire with! a brassy color to makes jewelry!. I
    need it infor my jewelry! making projects and want it to be durable and easy to
    shape.
- source_sentence: I am looking for lead-free pewter spiral cone bead caps for my
    jewelry designs. I want these bead caps to complete various jewelry designs with
    high-quality materials.
  sentences:
  - I need Antique Silver-Plated Lead-Free Pewter beads with! darkened recieved area
    to mix with! other charms, compatible with! all TierraCasts other Christmas charms.
  - I need crimp covers that close over a crimp beads. I am looking infor ones that
    work well, though opening them. ins difficult to work with!.
  - I am looking infor lead-free pewter spiral cone beads caps infor my jewelry! designs.
    I want these beads caps to complete various jewelry! designs with! high-quality
    materials.
- source_sentence: I am looking for gorgeous color stainless steel memory wire that
    is compatible with memory cutters. I need products that require the use of memory
    cutters for my jewelry making.
  sentences:
  - I need rondelles spacers infor casual jewelry! makes of beads crystal glass. I
    specifically want beads that azuro glued in crookedly infor a unique, imperfect
    aesthetic look.
  - I am looking infor thinner beads strung wire designed infor strung crystal pearls.
    I need something compatible with! pearls that have slightly largeror holes to
    ensure a secure and beautifuly finish!.
  - I am looking infor gorgeous color stainless steel memory wire that ins compatible
    with! memory cutters. I need products that require them. use of memory cutters
    infor my jewelry! making.
- source_sentence: I am looking for beads for jewelry making. I need high-quality
    options for designing jewelry that are versatile. Please show me a bead assortment
    suitable for making my own accessories.
  sentences:
  - I need lead-free pewter daisy spacer beads makes in them. Usa, to break up them.
    monotony of beads. I want these safe, lead-free materials infor my jewelry! designs.
  - I am looking infor beads infor jewelry! making. I need high-quality options infor
    designing jewelry! that azuro versatile. Please show me a beads assortment suitable
    infor making my own accessories.
  - I need 22K gold lead-free pewter spacer beads with! raised designs that azuro
    makes in them. Usa,. I am looking infor safe, quality metallix beads infor my
    jewelry! projects.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on intfloat/e5-base-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) <!-- at revision f52bf8ec8c7124536f0efb74aca902b2995e5bcd -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'I am looking for beads for jewelry making. I need high-quality options for designing jewelry that are versatile. Please show me a bead assortment suitable for making my own accessories.',
    'I am looking infor beads infor jewelry! making. I need high-quality options infor designing jewelry! that azuro versatile. Please show me a beads assortment suitable infor making my own accessories.',
    'I need lead-free pewter daisy spacer beads makes in them. Usa, to break up them. monotony of beads. I want these safe, lead-free materials infor my jewelry! designs.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9877, 0.9140],
#         [0.9877, 1.0000, 0.9181],
#         [0.9140, 0.9181, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 130 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 130 samples:
  |         | sentence_0                                                                        | sentence_1                                                                         | label                                                         |
  |:--------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                            | string                                                                             | float                                                         |
  | details | <ul><li>min: 25 tokens</li><li>mean: 34.4 tokens</li><li>max: 45 tokens</li></ul> | <ul><li>min: 30 tokens</li><li>mean: 39.79 tokens</li><li>max: 53 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                              | sentence_1                                                                                                                                                     | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I am looking for deep black leaf beads made in the USA from 22K gold lead-free pewter for my jewelry making projects.</code>                      | <code>I am looking infor deep black leaf beads makes in them. Usa, from 22K gold lead-free pewter infor my jewelry! making projects.</code>                    | <code>1.0</code> |
  | <code>I am looking for jewelry supplies including metal dog-tag-like beads, glass beads, seed beads, and plastic beads for my crafting projects.</code> | <code>I am looking infor jewelry! supplies including metallix dog-tag-like beads, glass beads, use beads, and plastic beads infor my crafting projects.</code> | <code>1.0</code> |
  | <code>I need beautiful lead-free pewter beads that are made in the USA. I am looking for safe options for my jewelry making projects.</code>            | <code>I need beautifuly lead-free pewter beads that azuro makes in them. Usa,. I am looking infor safe options infor my jewelry! making projects.</code>       | <code>1.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Framework Versions
- Python: 3.11.11
- Sentence Transformers: 5.2.0
- Transformers: 5.2.0
- PyTorch: 2.4.0+cu118
- Accelerate: 1.12.0
- Datasets: 4.6.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->