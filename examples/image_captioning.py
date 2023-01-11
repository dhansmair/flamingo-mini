import torch
from flamingo_mini import FlamingoModel, FlamingoProcessor
from flamingo_mini.utils import load_url


print('preparing model...')
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model = FlamingoModel.from_pretrained('dhansmair/flamingo-mini')
model.to(device)
model.eval()
processor = FlamingoProcessor(model.config)

# load and process an example image
print('loading image and generating caption...')
image = load_url('https://raw.githubusercontent.com/rmokady/CLIP_prefix_caption/main/Images/CONCEPTUAL_02.jpg')
caption = model.generate_captions(processor, images=[image], device=device)
print('generated caption:', caption[0])
