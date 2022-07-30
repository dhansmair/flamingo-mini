# Flamingo mini
*work in progress*
Implementation of the flamingo vision-language model by deepmind, based on Lucidrains [insert link]
implementation of the perceiver resampler and the gated cross-attention layers. 

- Aims to be compatible with the huggingface transformers library.
- There are two versions available at the moment, based on GPT-2 and OPT.

## Install
requires python3.8

```bash
$ git clone https://github.com/dhansmair/flamingo-mini.git
$ cd flamingo-mini
$ pip install .
```

## Usage
```python
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor

# create a model for training
# for configuration options, see FlamingoConfig
device = ...
config = FlamingoConfig()
model = FlamingoModel(config)
model.to(device)
processor = FlamingoProcessor(model.config, device=device)
```

### load a pretrained image-captioning model
```python
# use a pretrained model
model = FlamingoModel.from_pretrained('dhansmair/flamingo-mini-test')
processor = FlamingoProcessor(model.config)

# ...
```

## Citations

```bibtex
@article{Alayrac2022Flamingo,
    title   = {Flamingo: a Visual Language Model for Few-Shot Learning},
    author  = {Jean-Baptiste Alayrac et al},
    year    = {2022}
}
```
