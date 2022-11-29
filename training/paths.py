"""
configuration file containing all the paths to the data resources
(coco and cc3m images and annotation files)
"""

###########################################################
# image directories and annotation files
###########################################################

coco_img_dir = '/nfs/data3/zhangya/coco2017/images'
coco_ann_train = '/home/stud/hansmair/datasets/coco2017/captions_train2017.json'
coco_ann_val = '/home/stud/hansmair/datasets/coco2017/captions_val2017.json'

cc3m_root = '/nfs/data3/zhangya/cc3m/data'
cc3m_ann_val = '/home/stud/hansmair/flamingo-mini/training/evaluation/data/cc3m_coco_val.json'


###########################################################
# paths where to store results, checkpoints etc.
###########################################################

checkpoint_dir = '/home/stud/hansmair/flamingo-mini/training/checkpoints'
config_dir = '/home/stud/hansmair/flamingo-mini/training/configs'

evaluation_results_dir = '/home/stud/hansmair/flamingo-mini/training/evaluation/results'
evaluation_summary_dir = '/home/stud/hansmair/flamingo-mini/training/evaluation/summary'
evaluation_log_dir = '/home/stud/hansmair/flamingo-mini/training/evaluation/logs'



# sanity check
# import os
# for path in [v for k, v in locals().items() if '__' not in k]:
#     if not isinstance(path, str): continue
#     assert os.path.isdir(path) or os.path.isfile(path)