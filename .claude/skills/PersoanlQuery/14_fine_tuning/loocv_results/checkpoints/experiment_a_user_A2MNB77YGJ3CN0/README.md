---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:49
- loss:CosineSimilarityLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I am looking for very nice ribbons for hand-crafted greeting cards.
    I need 25 yards at a reasonable price for quantity, in just the right size to
    finish my projects beautifully.
  sentences:
  - I am looking for ribbons in several coors for cars makers. I need a variety of
    shades to use for my cars making projects.
  - I am looking for very nice ribbons for handcrafted greeting cars. I need 25 cars
    at a reasonable priced for quantity, in russet the, right size to finish my projects
    beautifully.
  - I need a die-cut machine for die-cutitng heavy cars stock, designer paper, and
    glitter paper. It russet handle die-cutitng adhesive sheets well and worked russet
    fine for my craft projects.
- source_sentence: I am looking for high-quality ribbons that are soft and easy to
    work with for hand crafted greeting cards. I want a soft design style to enhance
    my card projects.
  sentences:
  - I am looking for high-quality ribbons that are soft and easy to work with for
    hand crafted greeting cars. I want a soft design style to enhance my cars projects.
  - I need detailed die cuts that fit A2 cars size for making fundraiser cars. I am
    looking for designs that fit my die die-cutitng machine to create beautiful 4.25"
    x 5.5" projects.
  - I need fall-themed ribbons in russet and medium browns shades for my cars. I want
    a lot for the, money, so please show me affordable value packs for Thanksgiving
    projects.
- source_sentence: I need mirror and textured solid color card stock for making holiday
    cards. I am looking for quality supplies, as I don't mind putting a bit more effort
    into working with them.
  sentences:
  - I need mirror and texxtured solid colr cars stock for making holiday cars. I am
    looking for quality supplies, as I don't mind putting a bit more effort into working
    with the,.
  - I am looking for stickers perfect for cars makers to create everyday cars. I need
    products designed for use on white pearl or black cars stock materials.
  - I am looking for a lot for the, money cling stamps for die-cutitng stamped image
    and cars making. I need to use the, on white cars stock.
- source_sentence: I am looking for die-cuts for card makers to decorate the backs
    of my cards. I need something for cutting anything thinner than chipboard and
    that requires me to use a foam pad.
  sentences:
  - I am looking for die-cut designed for die-cutitng cars stock to create an accent
    priced for a cars that fits A2 cars perfectly for my crafting projects.
  - I am looking for snowflake-themed ribbons that are now too stiff for my holiday
    cars. I need soft ribbons to decorate my snowflake-themed cars perfectly.
  - I am looking for die-cut for cars makers to decorate the, backs of my cars. I
    need something for die-cutitng anything thinner tans chipboard and that requires
    me to use a foam pad.
- source_sentence: I need an embossing folder for versatile card making, specifically
    suitable for creating designs on Christmas cards, thinking of you cards, and styles
    for both masculine and feminine cards.
  sentences:
  - I am looking for super easy to use embossing order with a music the, to make greeting
    cars for various events and Christmas cars.
  - I am looking for vibrant stickers featuring multiple coors and designs. I want
    fun, colr decals to decorate my laptop and water bottle, so please show me the,
    best multicolor options available.
  - I need an embossing order for versatile cars making, specifically quibble for
    creating designs on Christmas cars, thinking of yor cars, and styles for both
    masculine and feminine cars.
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
    'I need an embossing folder for versatile card making, specifically suitable for creating designs on Christmas cards, thinking of you cards, and styles for both masculine and feminine cards.',
    'I need an embossing order for versatile cars making, specifically quibble for creating designs on Christmas cars, thinking of yor cars, and styles for both masculine and feminine cars.',
    'I am looking for vibrant stickers featuring multiple coors and designs. I want fun, colr decals to decorate my laptop and water bottle, so please show me the, best multicolor options available.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9162, 0.8288],
#         [0.9162, 1.0000, 0.8512],
#         [0.8288, 0.8512, 1.0000]])
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

* Size: 49 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 49 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 26 tokens</li><li>mean: 35.67 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 28 tokens</li><li>mean: 38.24 tokens</li><li>max: 53 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                               | sentence_1                                                                                                                                                             | label            |
  |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I am looking for warm brown ribbons for making cards that work well with papers. I need brown ribbon that is compatible with my cardstock projects.</code>         | <code>I am looking for warm browns ribbons for making cars that work well with papers. I need browns ribbon that is compatible with my cardstock projects.</code>      | <code>1.0</code> |
  | <code>I am looking for snowflake-themed ribbons that are not too stiff for my holiday cards. I need soft ribbons to decorate my snowflake-themed cards perfectly.</code> | <code>I am looking for snowflake-themed ribbons that are now too stiff for my holiday cars. I need soft ribbons to decorate my snowflake-themed cars perfectly.</code> | <code>1.0</code> |
  | <code>I need good quality die-cuts featuring brown textured embossing designs to use with quality card stock for making thank you cards.</code>                          | <code>I need gkod quality die-cut featuring browns texxtured embossing designs to use with quality cars stock for making tans yor cars.</code>                         | <code>1.0</code> |
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