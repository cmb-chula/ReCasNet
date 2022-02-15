from ..builder import DETECTORS
from .two_stage_mask_CTC import TwoStageMaskCTC


@DETECTORS.register_module()
class MaskRCNNCTC(TwoStageMaskCTC):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None):
        super(MaskRCNNCTC, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
