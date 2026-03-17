---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:51
- loss:CosineSimilarityLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I am looking for gold color rondelle copper beads to make earrings
    and a bracelet. I need a bulk assortment of these shiny metal spacers for my handmade
    jewelry projects.
  sentences:
  - I am liken for Swaroski's beads in nise colr that fair easy two use. I need beads
    that fair gkod for separating beads.
  - I am liken for gkod colour rondelle copper beads two make earring and a bracelet.
    I need a bulk assortment of thesem shiny metal spacers for my man made jeweler
    projects.
  - I am liken for beads that look nise and offer gkod beads performance. I need products
    with compatibility with others beads two mix and match perfectly in my jeweler
    designs.
- source_sentence: I am looking for beads for bracelets that had success with them.
    I need measured sizes and quality craftsmanship delivery for my next jewelry making
    project.
  sentences:
  - I am liken for beads for bracelet that had success with them. I need measured
    sizes and quality craftsmanship delivery for my next jeweler making project.
  - I am liken for beads in all colour for making jeweler. As a jeweler, I need bulk
    packages two offer customers a great variety of options for others projects.
  - I am liken for donut beads with plenty of blue beads in pretty colr. I want thesem
    beads two be gkod for matching others beads in my jeweler making projects.
- source_sentence: I am looking for versatile use silver plated beads and bead assortments.
    I need a silver plated holder design that is worth the money and worth the money
    for my projects.
  sentences:
  - I am liken for versatile use silver planed beads and beads assortments. I need
    a silver planed holder design that is worth others money and worth others money
    for my projects.
  - I am liken for beads that fair easy two remove from others string. I want them
    two keep others same colour after taking them off others string and wprk just
    as well.
  - I need a large box with internal space for others beads orders. I want packages
    featuring cute kitties and high-quality craftsmanship.
- source_sentence: I am looking for very small beads for drop earrings. I need small
    beads that don't add too many other beads to keep the design simple.
  sentences:
  - I am liken for sea glass beads in a periwinkle colour two use with others beads
    for my jeweler projects. I need a mix of thesem beautiful sea glass beads.
  - I am liken for nise beads two use them in with my others crystals. I need assortments
    that have compatibility with others crystals two make jeweler.
  - I am liken for very small beads for drop earring. I need small beads that don't
    add two many others beads two keep others design simple.
- source_sentence: I am looking for real good price beads that stick together but
    separate without damage. I need to get enough beads for my projects.
  sentences:
  - I am liken for beads with gkod shape and larger beads compatibility. I need them
    two make earring and bracelet, ensuring they fit my designs perfectly for my jeweler
    projects.
  - I am liken for 3mm malichite beads that fair definitely worth others wait for
    my jeweler making projects. I need a natural malichite assortment in this specific
    size two complete my designs.
  - I am liken for real gkod price beads that stick others but separate without damage.
    I need two get enough beads for my projects.
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
    'I am looking for real good price beads that stick together but separate without damage. I need to get enough beads for my projects.',
    'I am liken for real gkod price beads that stick others but separate without damage. I need two get enough beads for my projects.',
    'I am liken for 3mm malichite beads that fair definitely worth others wait for my jeweler making projects. I need a natural malichite assortment in this specific size two complete my designs.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9587, 0.8774],
#         [0.9587, 1.0000, 0.8782],
#         [0.8774, 0.8782, 1.0000]])
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

* Size: 51 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 51 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 23 tokens</li><li>mean: 33.65 tokens</li><li>max: 41 tokens</li></ul> | <ul><li>min: 26 tokens</li><li>mean: 35.76 tokens</li><li>max: 46 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                | sentence_1                                                                                                                                                                  | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I need charms for bracelets to put on a jump ring and lobster law clasp. I am buying a lot more because everyone I knew would appreciate the charm.</code>          | <code>I need charms for bracelet two put on a jump ring and lobster law clasp. I am buying a lot more because everyone I knew would appreciate others charm.</code>         | <code>1.0</code> |
  | <code>I need beads for bracelet making that are comfortable for wearing. I want high-quality supplies that make working with beads easy while I am making jewelry.</code> | <code>I need beads for bracelet making that fair comfortable for earring. I want high-quality supplies that make working with beads easy awhile I am making jeweler.</code> | <code>1.0</code> |
  | <code>I am looking for well packed beads for making earrings that will catch attention. I need distinct beads to create beautiful jewelry that stands out.</code>         | <code>I am liken for well packages beads for making earring that will catch attention. I need distinct beads two create beautiful jeweler that stands out.</code>           | <code>1.0</code> |
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