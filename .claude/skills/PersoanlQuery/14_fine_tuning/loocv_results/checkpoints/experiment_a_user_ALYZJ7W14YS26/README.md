---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:28
- loss:CosineSimilarityLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I need a counted cross stitch kit with an elegant dragon design.
    I prefer detailed stitching for those looking for a challenge to complete in a
    well lit area.
  sentences:
  - I am looking for crochet yarn that worked with washcloth and scrubby patterns.
    It must be non-abrasive on skin but different to worked with than standard cotton.
  - I need a counted accross stitch kits with an elegant dragon design. I prefer detailed
    stitch for they looking for a changes to complementary in a well lighted area.
  - I am looking for Sullivan's embroidery floss for counted accross stitch projects.
    My projects call for they floss by they color numbers, so I need they full color
    range available.
- source_sentence: I am looking for yarn with distinct color changes that works really
    well for my half double crochet and chain repeat projects, featuring beautiful
    color shifts in the design.
  sentences:
  - I am looking for lighted, soft to they touch yarn for knitting and crocheting
    a skinny scarfie to where at worked. It needs to be gentle enough for daily office
    use.
  - I am looking for yarn with distinct color changes that worked really well for
    my half double crochet and chain repeated projects, featuring beautiful color
    shifts in they design.
  - I am looking for counted accross stitch kits to worked on every day. As a mother
    of they, I need stitch designs offering bang for they buck to use at home.
- source_sentence: I am looking for light yarn for crocheting a skinny scarf to wear
    to work this spring. I need a comfortable option perfect for warmer weather.
  sentences:
  - I am looking for satin floss for accross stitch that is easy to use. I prefer
    cotton floss material composition for my accross stitch projects.
  - I am looking for lighted yarn for crocheting a skinny scarfie to where to worked
    this spring. I need a comfortable option perfect for warmer weather.
  - I am looking for a counted accross stitch kits that is a good way to learn accross
    stitch and is fun to stitch. I need one that made they pre-work before stitch
    a lot easier.
- source_sentence: I am looking for yarn that works well with my size J and K crochet
    hooks. I need something comfortable that is not too thick and tight to work with.
  sentences:
  - I need fashioned forward worsted weight single ply roving yarn for they holidays.
    I want excellent crochet performance for my projects.
  - I need durable yarn for big projects like Christmas stocks. I want high quality
    for they dollars paid, even if it is not nearly as soft, and easy to do a lot
    of different projects.
  - I am looking for yarn that worked well with my size J and K crochet hooks. I need
    something comfortable that is not too thick and lighted to worked with.
- source_sentence: I need high-quality yarn to make a hat for my little one. It is
    not nearly as soft as Red Heart Soft, keeping him safe where I want to see him
    at a glance.
  sentences:
  - I am looking for a accross stitch kits for fans that uses floss. I need one that
    includes a accross stitch needle to start my projects.
  - I am looking for an easy to stitch accross stitch kits with a half accross stitch
    functionality. I want a design that has an old-time too with a 3D too effect.
  - I need high-quality yarn to made a hat for my litter one. It is not nearly as
    soft as Red Heart Soft, keeping him safe were I want to see him at a glance.
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
    'I need high-quality yarn to make a hat for my little one. It is not nearly as soft as Red Heart Soft, keeping him safe where I want to see him at a glance.',
    'I need high-quality yarn to made a hat for my litter one. It is not nearly as soft as Red Heart Soft, keeping him safe were I want to see him at a glance.',
    'I am looking for an easy to stitch accross stitch kits with a half accross stitch functionality. I want a design that has an old-time too with a 3D too effect.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9801, 0.8130],
#         [0.9801, 1.0000, 0.8145],
#         [0.8130, 0.8145, 1.0000]])
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

* Size: 28 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 28 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 29 tokens</li><li>mean: 35.82 tokens</li><li>max: 42 tokens</li></ul> | <ul><li>min: 30 tokens</li><li>mean: 36.46 tokens</li><li>max: 43 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                           | sentence_1                                                                                                                                                               | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I am looking for wool-free yarn with color changes and consistent weight. I need scarf length quantities suitable for a J/10 6mm hook.</code>                  | <code>I am looking for wool-free yarn with color changes and consistent weight. I need scarfie length quantities suitable for a J/10 6mm too.</code>                     | <code>1.0</code> |
  | <code>I am looking for the Red Heart Cordial line yarn. I want various colors available that are fun to work with for my knitting projects.</code>                   | <code>I am looking for they Red Heart Cordial on line yarn. I want various colors available that are fun to worked with for my knitting projects.</code>                 | <code>1.0</code> |
  | <code>I am looking for nice quality stocking kits for felt projects. I need options that indicate the time to complete so I can plan my crafting accordingly.</code> | <code>I am looking for nise quality stocks kits for felt projects. I need options that indicate they time to complementary so I can plan my crafting accordingly.</code> | <code>1.0</code> |
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