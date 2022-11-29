"""
this script was used to generate the cc3m_coco_val.json file.
It contains the caption annotations of the cc3m validation set in coco format.
"""
import json
import os
from tqdm import tqdm
from PIL import Image

from ...datasets import ConceptualCaptions
from ... import paths


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    
    save_as = CURRENT_DIR + '/cc3m_coco_val.json'
    
    dataset = ConceptualCaptions(
        paths.cc3m_root,
        split='val',
        transform=None,
    )
    dataset.prefixes = ['']
    
    result = {
        "info": {
            "description": "annotation file for Conceptual Captions Validation set in COCO format",
            "url": "",
            "version": "1.0",
            "year": 2022,
            "contributor": "",
            "date_created": "2022/10/20"
            },
        "licenses": [],
        "images": [],
        "annotations": [],
    }
    
    
    for image_id, image, caption in tqdm(dataset):
        
        if not isinstance(image, Image.Image):
            result['images'].append({
                "filename": str(image_id) + ".jpg",
                "height": 0,
                "width": 0,
                "id": image_id
            })

            result['annotations'].append({
                "image_id": image_id,
                "id": image_id,
                "caption": caption
            })
        else:
            result['images'].append({
                "filename": str(image_id) + ".jpg",
                "height": image.height,
                "width": image.width,
                "id": image_id
            })

            result['annotations'].append({
                "image_id": image_id,
                "id": image_id,
                "caption": caption
            })
        
        
    print('result data generated.')
    
    
    # store the json file
    with open(save_as, 'w') as f:
        json.dump(result, f)
        
    print('result file stored.')