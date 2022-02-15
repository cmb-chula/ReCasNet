_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_CTC.py',
    '../_base_/datasets/CTC_b.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
# model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
