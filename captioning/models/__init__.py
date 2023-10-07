# 新增模型，其余完全相同
# 新增模型包括Vision_TransformerModel，Swin系列，Video_Swin系列

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch


from .Video_Swin_TranCAP_L_concept_align import Video_Swin_TranCAP_L_concept_align # cz add



def setup(opt):
        
    if opt.caption_model == 'Video_Swin_TranCAP_L_concept_align':
        # cz add
        model = Video_Swin_TranCAP_L_concept_align(opt)

    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
