import torch
from flamingo_mini import FlamingoConfig, FlamingoModel, FlamingoProcessor


config = FlamingoConfig()
print(config)

# model = FlamingoModel(config)
# processor = FlamingoProcessor(config)
model = FlamingoModel.from_pretrained('dhansmair/flamingo-mini-test', use_auth_token=True)

print(model)
