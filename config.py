import torch
import os

class Config:
    MODEL_ROOT = './backbone_resume.pth'
    LOG_ROOT = './head_resume.pth'
    BACKBONE_RESUME_ROOT = ''
    HEAD_RESUME_ROOT = ''
    TRAIN_FILES = './dataset/face_train_ms1mv2.txt'

    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    EMBEDDING_SIZE = 512
    BATCH_SIZE = 500
    DROP_LAST = True
    BACKBONE_LR = 0.05
    QUALITY_LR = 0.01
    NUM_EPOCH = 90
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9

    HEAD_GPUS = [0]
    BACKBONE_GPUS = [0 , 1]

    PRETRAINED_BACKBONE = ''
    PRETRAINED_QUALITY = ''

    NUM_EPOCH_WARM_UP = 1
    FIXED_BACKBONE_FEATURE = False

config = Config()
