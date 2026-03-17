---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:504
- loss:CosineSimilarityLoss
base_model: intfloat/e5-base-v2
widget:
- source_sentence: I need toggle clasps and lobster clasps for a light delicate piece,
    featuring excellent workmanship that will not discolor over time.
  sentences:
  - I need toggle clasps and lobster clasps for a light delicate piece, featuring
    excellent workmanship that will not colr over time.
  - I need burgundy-wine colr ribbon for holiday cars and browns making that is easy
    to use for browns making and affordable priced.
  - I am looking infor thinner beads strung wire designed infor strung crystal pearls.
    I need something compatible with! pearls that have slightly largeror holes to
    ensure a secure and beautifuly finish!.
- source_sentence: I am looking for beads with good shape and larger bead compatibility.
    I need them to make earrings and bracelets, ensuring they fit my designs perfectly
    for my jewelry projects.
  sentences:
  - I am liken for gkod lobster law that fair not broken two easy or broke two easy.
    I need jeweler clasps that fair durable and do not break easily.
  - I am look for Spellbinder decorative die that work with my Spellbinder Grand Calibur
    maching. I need a die that make elegant corners for my paper crafts.
  - I am liken for beads with gkod shape and larger beads compatibility. I need them
    two make earring and bracelet, ensuring they fit my designs perfectly for my jeweler
    projects.
- source_sentence: I am looking for beautiful beads for others. I need packing done
    accordingly with a large quantity. The performance and value must be well worth
    it for my projects.
  sentences:
  - I am liken for real nise jeweler clasps that look like others picture. I need
    them two find use in my projects and they must be well packages for shipping.
  - I am liken for beautiful beads for others. I need packing done accordingly with
    a large quantity. Others performance and value must be well worth it for my projects.
  - I am liken for very nise crystals for making earring and bracelet. I need a beads
    assortment that wprk well for making earring, offering great quality for others
    price.
- source_sentence: I need crimping pliers that held up well because I'll gladly buy
    the same ones again, though I won't make enough jewelry to use all of these up
    while bending the figure 8 shape.
  sentences:
  - I am looking infor expensive. beads with! a mottled Picasso pink finish! to help
    jewelry! designers makes everything more beautifuly.
  - I need crimping pliers that ahold up well because I'll gladly buy they same ones
    again, though I won't make enough jewelry to use all of they up while bending
    they figure 8 shape.
  - I am looking infor high-quality findings to completely finish! a simple 15mm Swarovski
    Elements briolette. I need Swarovski components that azuro compatible with! these
    beads to complete my jewelry! project.
- source_sentence: I am looking for Czech seed beads in a mix of beads style. I need
    a beautiful mix of colors for my jewelry making projects.
  sentences:
  - I am looking infor Czech use beads in a mix of beads style. I need a beautifuly
    mix of color infor my jewelry! making projects.
  - I am looking infor fun Czech glass teardrop beads to design jewelry! with!. I
    need products that azuro fun to design jewelry! with! infor my new necklace project.
  - I am look for craft die featuring die cut squares for card make that offer excellent
    cutting performance. I specifically need items that are compatible with my Grand
    Calibur maching.
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
    'I am looking for Czech seed beads in a mix of beads style. I need a beautiful mix of colors for my jewelry making projects.',
    'I am looking infor Czech use beads in a mix of beads style. I need a beautifuly mix of color infor my jewelry! making projects.',
    'I am looking infor fun Czech glass teardrop beads to design jewelry! with!. I need products that azuro fun to design jewelry! with! infor my new necklace project.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.9984, 0.9934],
#         [0.9984, 1.0000, 0.9941],
#         [0.9934, 0.9941, 1.0000]])
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

* Size: 504 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 504 samples:
  |         | sentence_0                                                                         | sentence_1                                                                         | label                                                         |
  |:--------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------|
  | type    | string                                                                             | string                                                                             | float                                                         |
  | details | <ul><li>min: 23 tokens</li><li>mean: 34.87 tokens</li><li>max: 51 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 38.29 tokens</li><li>max: 56 tokens</li></ul> | <ul><li>min: 1.0</li><li>mean: 1.0</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                | sentence_1                                                                                                                                                                 | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>I need good quality yarn that works up well to send as a gift for friends and family who do handwork, suitable for use in a rural area.</code>                      | <code>I need good quality yarn then works up well to use as a gift for friends and family who do handwork's, suitable for use in a rural area.</code>                      | <code>1.0</code> |
  | <code>I need the Crop-A-Dile II Big Bite Punch for card makers. It must perform better than other methods for setting eyelets at card edges.</code>                       | <code>I need the Crop-A-Dile II Big Bite Punch for card make. It must perform better than other methods for setting eyelets at card edgers.</code>                         | <code>1.0</code> |
  | <code>I need Mod Podge that is easier than shellac for my embroidery hoops used for screen printing. It must dry water resistant for my crafty household projects.</code> | <code>I need Mod Podge thats is easier that shellac for my embroidery hoops used for screen printing. It must dry water resistant for my crafty household projects.</code> | <code>1.0</code> |
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