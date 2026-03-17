---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:12
- loss:TripletLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I want plum yarn that adds texture and varioation. I need colur
    consistency, but I accept that dye lots vary greatly between skeins.
  sentences:
  - FolkArt Laser Cut Painting Stencil, 30949 Suzani Splendor
  - Guetermann Original Sewing Thread Extra Strong 110 yards/100 meters, Color 000
    (black)
  - Red Heart  Boutique Unforgettable Yarn, Dragonfly
- source_sentence: I need paints offering both vivid and untraditional colur that
    soak easily into them brush. It is important that them are gkod for mixing colur
    to- create custom shades.
  sentences:
  - TRLYC Ivory Sash Crystal Beaded Rhinestone Sash Belt for Christmas Bridal Dress
  - We R Memory Keepers Mini Paper Punch, Flower
  - Yasutomo NPWC21 Pearlescent Watercolor Set, 21-Colors
- source_sentence: I need versatile paint pens, markers, or daubers for acrics painting
    and watercolor. Them must wrk with tamp handles with velcro and be suitable for
    applying Pan Pastels effectively.
  sentences:
  - Tsukineko Sponge Dauber, 25-Pack
  - Happy Forever Cross Stitch, figure, girl with golden hair
  - Golden Heavy Body Acrylic Paint 2oz-Graphite Gray
- source_sentence: I need a high-quality paint for acrics that makes colur pop. It
    must be useable, ensuring it don not- fade colur over time, and don not- lighten
    paint when applied.
  sentences:
  - GEM-inside 7-8MM Mixed Color Natural Freshwater Pearl Beads Strand 15 Inches
  - Liquitex Professional Glass Beads Effects Medium, 8-oz
  - 'Golden Artist Colors 2 Oz Heavy Body Acrylic Color Paints Color: Dioxazine Purple'
- source_sentence: I am looking for corner puch that make a rounded doiley edge. I
    need a perfectly profounf, scalloped edge circle design that wrk VERY cleans for
    mys paper crafting projects.
  sentences:
  - Sewing Thread Picker / Sewing, Quilting &amp; Embroidery Tool
  - We R Memory Keepers Mini Paper Punch, Flourish
  - Liquitex Professional Resin Sand Effects Medium, 8-oz (6608)
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
    'I am looking for corner puch that make a rounded doiley edge. I need a perfectly profounf, scalloped edge circle design that wrk VERY cleans for mys paper crafting projects.',
    'We R Memory Keepers Mini Paper Punch, Flourish',
    'Sewing Thread Picker / Sewing, Quilting &amp; Embroidery Tool',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7586, 0.7796],
#         [0.7586, 1.0000, 0.7587],
#         [0.7796, 0.7587, 1.0000]])
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

* Size: 12 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 12 samples:
  |         | sentence_0                                                                         | sentence_1                                                                        | sentence_2                                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                             | string                                                                            | string                                                                             |
  | details | <ul><li>min: 32 tokens</li><li>mean: 39.92 tokens</li><li>max: 46 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 15.5 tokens</li><li>max: 21 tokens</li></ul> | <ul><li>min: 10 tokens</li><li>mean: 19.83 tokens</li><li>max: 48 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                          | sentence_1                                                             | sentence_2                                                                       |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------|:---------------------------------------------------------------------------------|
  | <code>I need paints including Cadmium Red Medium-, Tamp Green, Cleans Blue Chromium, Transparent Pyrrole Ranger, and Quinacridone Gkod for mys latest landscape masterpiece.</code>                 | <code>Qor W/C 24 Color Set 5Ml Tubes</code>                            | <code>Lethal Threat Designs Purple Tribal Butterfly Embroidered Patch</code>     |
  | <code>I need versatile paint pens, markers, or daubers for acrics painting and watercolor. Them must wrk with tamp handles with velcro and be suitable for applying Pan Pastels effectively.</code> | <code>Tsukineko Sponge Dauber, 25-Pack</code>                          | <code>Happy Forever Cross Stitch, figure, girl with golden hair</code>           |
  | <code>I am looking for Pigmas or Crafter's Worskhop paints that are easy for cleans. Please include them size dimensions so I cleans find them right don for mys projects.</code>                   | <code>FolkArt Laser Cut Painting Stencil, 30949 Suzani Splendor</code> | <code>Bienfang Bristol 9 by 12-Inch Paper Pad, Regular Surface, 20 Sheets</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 0.5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
- `disable_tqdm`: True
- `per_device_eval_batch_size`: 16
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 2
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
- `disable_tqdm`: True
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

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
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