# Flamingo mini
Implementation of the <a href="https://www.deepmind.com/blog/tackling-multiple-tasks-with-a-single-visual-language-model">Flamingo</a> vision-language model by deepmind, based on Lucidrains https://github.com/lucidrains/flamingo-pytorch implementation of the perceiver resampler and the gated cross-attention layers. 

At the moment there are two versions available, based on GPT-2 and OPT. They have been tested with openai CLIP vision encoders `openai/clip-vit-base-patch32` and `openai/clip-vit-large-patch14`.

The implementation aims to be compatible with the huggingface transformers library. So you can use `save_pretrained()`, `from_pretrained()`, `push_to_hub()`, and so on.
Powered by hf transformers, the model is enabled with different text generation strategies such as beam search and sampling strategies such as top-k sampling.

A pretrained model is available at https://huggingface.co/dhansmair/flamingo-mini-test.
Be aware that this model was trained for image captioning on the conceptual captions dataset.
You can find a demo of the model in this hf space: https://huggingface.co/spaces/dhansmair/flamingo-cap

## Install
requires python3.8

```bash
git clone https://github.com/dhansmair/flamingo-mini.git
cd flamingo-mini
pip install .
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

### load the pretrained image-captioning model
```python
model = FlamingoModel.from_pretrained('dhansmair/flamingo-mini-test')
processor = FlamingoProcessor(model.config)
```
A complete example is provided in `examples/image_captioning.py`.

## Citations

```bibtex
@article{Alayrac2022Flamingo,
    title   = {Flamingo: a Visual Language Model for Few-Shot Learning},
    author  = {Jean-Baptiste Alayrac et al},
    year    = {2022}
}
```
