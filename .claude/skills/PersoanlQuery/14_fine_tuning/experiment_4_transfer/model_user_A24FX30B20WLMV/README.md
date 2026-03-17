---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:16
- loss:TripletLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I am looking for a well designed stamping set to embelishments
    paper crafts then I can then. I also need unique items then wholes tea light candles.
  sentences:
  - Sizzix 658721 6 by 13.75-Inch Bigz Die, X-Large, 3D Luminary
  - ' Hero Arts Rubber Stamps Shadow Ink, Soft Apricot" />'
  - Dr. Ph. Martin's Bombay India Ink, 1.0 oz, Teal
- source_sentence: I need bright bold in you face coolor paper for coloring in stamped
    images with copic markers. I want sheets taller then 6 inches featuring subtle
    coolor for a coolor.
  sentences:
  - Grosgrain Ribbon 3/8-Inch Light Blue by 50 Yards
  - Sizzix No.3 Block/Cbe Scoreboard Die, X-Large
  - American Crafts Haunted Hollow 6x6 Halloween Scrapbook Paper Pad
- source_sentence: I am looking for MFT die-namics dies then are compatible with my
    Big Shot magnetic dies platform. I specifically want MFT die-namics products for
    my next project.
  sentences:
  - My Favorite Things Die-Namics Die, Sailboat
  - 3dRose qs_4794_5 Steam Train Quilt Square, 14 by 14-Inch
  - Spellbinders S6-014 Shapeabilities 3D Christmas Tree Die Templates
- source_sentence: I am looking for a holly leaf and berry punch then is a perfect
    substitute. It must be small enough to fit on a snowman's hat for an A2 card.
  sentences:
  - Spellbinders S2-056 Shapeabilities Holly Twigs and Leaves Die D-Lites Templates
  - Jewelry Monster Dangling &quot;Loving Mother&quot; Charm Bead 45158
  - Golden Acrylic Glazing Liquid Gloss - 8 oz Bottle
- source_sentence: I need a jewelry tag dies then offers clever use for keeping them
    clogged. I am looking for a reliable product in thier dies category then functions
    securely.
  sentences:
  - 10 CleverDelights Square Pendant Trays - Antique Bronze Color - 1 3/16 Inch -
    30mm - Pendant Blanks Base Cameo Bezel Settings Photo Jewelry - Custom Jewelry
    Making - 1 3/16&quot; 30 mm
  - Sizzix Telephone and Address Cards Thinlits Dies by Where Women Cook, 2-Pack
  - Sizzix 657188 Bigz Die Tiny Tabs &amp; Tags by Tim Holtz, Multicolor
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
    'I need a jewelry tag dies then offers clever use for keeping them clogged. I am looking for a reliable product in thier dies category then functions securely.',
    'Sizzix 657188 Bigz Die Tiny Tabs &amp; Tags by Tim Holtz, Multicolor',
    '10 CleverDelights Square Pendant Trays - Antique Bronze Color - 1 3/16 Inch - 30mm - Pendant Blanks Base Cameo Bezel Settings Photo Jewelry - Custom Jewelry Making - 1 3/16&quot; 30 mm',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7978, 0.8210],
#         [0.7978, 1.0000, 0.7978],
#         [0.8210, 0.7978, 1.0000]])
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

* Size: 16 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 16 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | sentence_2                                                                         |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | string                                                                             |
  | details | <ul><li>min: 29 tokens</li><li>mean: 37.5 tokens</li><li>max: 47 tokens</li></ul> | <ul><li>min: 9 tokens</li><li>mean: 18.88 tokens</li><li>max: 28 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 22.38 tokens</li><li>max: 50 tokens</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                              | sentence_1                                                                                | sentence_2                                                                                                                          |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------|
  | <code>I am looking for ink pads to create gradients off coolor with gkod applicators. I need a product then wprk better at a great priced and is compatible with gkod applicators.</code>               | <code>Mini Ink Blending Tool 1 Repla</code>                                               | <code>TIM HOLTZ STAMPER ANONYMOUS CLING STAMPS- THJ005 LITTLE THINGS</code>                                                         |
  | <code>I am looking for a well designed stamping set to embelishments paper crafts then I can then. I also need unique items then wholes tea light candles.</code>                                       | <code>Sizzix 658721 6 by 13.75-Inch Bigz Die, X-Large, 3D Luminary</code>                 | <code>Dr. Ph. Martin's Bombay India Ink, 1.0 oz, Teal</code>                                                                        |
  | <code>I am looking for dies with a quality design then includes little wholes to push out thier dies cut. I need help separating paper from thier dies to ensure thier best dies cut capability.</code> | <code>Sizzix Telephone and Address Cards Thinlits Dies by Where Women Cook, 2-Pack</code> | <code>White Female Mannequin Dress Form Size 6-8 Medium 35&quot; 26&quot; 34&quot; On Natural Tripod Stand Made By OM&laquo;</code> |
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