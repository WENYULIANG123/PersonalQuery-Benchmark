---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:475
- loss:CosineSimilarityLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I am looking for a color wheel for color pencil work. It must help
    me achieve vibrant true color and high color vibrancy in my art.
  sentences:
  - I am looking for a runner thats appliques evenly and is super sticky. I need a
    fast option thats is easy to replace and works well in either that for my art
    projects.
  - I am look for A-2 Matting Basics die that make frames with one pass through my
    Grand Calibur maching. I need a specific die size of 5" x 3.75".
  - I am looking for a colur wheel for colur pencil wrk. It must help me achieve vibrant
    true colur and high colur vibrancy int mys art.
- source_sentence: I am looking for decorative paper with deep and rich colors that
    allows them to pop. I need sheets for handcrafted greeting card makers that are
    perfect for matching solid card stock.
  sentences:
  - I am looking for decorative paper with deep and rich coors that allows the, to
    pop. I need sheets for hand-drafted greeting cars makers that are perfect for
    matching solid cars stock.
  - I need 22K gold lead-free pewter spacer beads with! raised designs that azuro
    makes in them. Usa,. I am looking infor safe, quality metallix beads infor my
    jewelry! projects.
  - I need an excellent value jewelry kit for pattern precaution that includes image
    stamps. It must have a large stamping fair width and can ahold patinas and finishes
    well.
- source_sentence: I am looking for Die-Namics dies compatible with the Spellbinders
    Grand Calibur machine. I need a die that works with all major die-cutting tools
    for my crafting projects.
  sentences:
  - I am look for Die-Namics die compatible with the Spellbinder Grand Calibur maching.
    I need a die that works with all major die-cutting tools for my crafting projects.
  - I need a punch thats cuts cleanly for crafting with aluminum cans. I am making
    aluminum cleanly roses, so it must cut paper and metal sheets effectively for
    my projects.
  - I am looking for beads strands in a darker colr that fair shorter than most options.
    I need they darker beads for my specific jewelry design project.
- source_sentence: I am looking for color Beadaholique beads and bead assortments
    for my jewelry making projects. I need free shipping included on my order for
    these crafting supplies.
  sentences:
  - I am looking for colr Beadaholic beads and bead assortments for my jewelry making
    projects. I need free shipping included on my orderin for they're crafting supplies.
  - I am looking infor Preciosa Czech fire-polished glass beads with! an AB finish!
    infor designing with! them.. I need high-quality beads that sparkle infor my jewelry!
    making projects.
  - I am looking for cotten yarn thats works well with a size G hook. I need a firmer
    feel for spring and summer weight it's thats shows stitch definition well.
- source_sentence: I need needlepoint yarn that is consistent in color and holds up
    well over time. I want it to last and last for teaching my needlepoint classes
    effectively.
  sentences:
  - I am looking for 4mm Smokey Quartz beads in a dark colr. I need a reco.mmend bead
    assortment for my jewelry making projects.
  - I am looking infor Czech glass fire-polished gemstone donut beads to design with!.
    I need a gorgeous color where them. color pops and features high-quality fire-polish
    craftsmanship.
  - I need needle point yarn then is consistent in colr and holds up well over time.
    I want it to last and last for teaching my needle point classes effectively.
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
    'I need needlepoint yarn that is consistent in color and holds up well over time. I want it to last and last for teaching my needlepoint classes effectively.',
    'I need needle point yarn then is consistent in colr and holds up well over time. I want it to last and last for teaching my needle point classes effectively.',
    'I am looking infor Czech glass fire-polished gemstone donut beads to design with!. I need a gorgeous color where them. color pops and features high-quality fire-polish craftsmanship.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9984, 0.9831],
#         [0.9984, 1.0000, 0.9813],
#         [0.9831, 0.9813, 1.0000]])
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

* Size: 475 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 475 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 25 tokens</li><li>mean: 35.12 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 38.49 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                               | sentence_1                                                                                                                                                                                             | label            |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I am looking for expensive, strong wire for my jewelry projects. I need high-quality materials that are durable enough for my designs and willing to pay a premium price.</code>                   | <code>I am looking infor expensive., strung wire infor my jewelry! projects. I need high-quality materials that azuro durable enough infor my designs and willing to pay a premium price.</code>       | <code>1.0</code> |
  | <code>I am looking for small Jolee wedding cake stickers in pink for my card making projects. I need high-quality dimensional embellishments to decorate handmade cards for a bridal shower.</code>      | <code>I am look for small Jolee wedding cake stickers in pink for my card making projects. I need high-quality dimensional embellishments to decorate handmade cards for a bridal shower.</code>       | <code>1.0</code> |
  | <code>I am looking for plain gray color wire compatible with sterling silver for wire work practice. I need plain gray color wire for wire work practice that is compatible with sterling silver.</code> | <code>I am looking for plain gray colr wire compatible with sterling simple for wire wrok practice. I need plain gray colr wire for wire wrok practice that is compatible with sterling simple.</code> | <code>1.0</code> |
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