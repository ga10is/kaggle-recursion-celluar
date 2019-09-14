import os

IMG_SIZE = (512, 512)
INPUT_SIZE = (256, 256)
TEST_INPUT_SIZE = (256, 256)  # (1024, 1024)

INPUT = './data/'
TRAIN_PATH = os.path.join(INPUT, 'train.csv')
TEST_PATH = os.path.join(INPUT, 'test.csv')
TRAIN_IMG_PATH = os.path.join(INPUT, 'data_raw', 'train')
TEST_IMG_PATH = os.path.join(INPUT, 'data_raw', 'test')

N_SAMPLES = 5
RUN_TTA = False

N_CLASSES = 1108

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 128
NUM_WORKERS = 8
PRINT_FREQ = 10
ITER_PER_CYCLE = 20
EPOCHS = 5 + ITER_PER_CYCLE * 4

ADAM_LR = 5e-4
SGD_LR = 1e-3
MIN_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

USE_PRETRAINED = False
PRETRAIN_PATH = 'analysis/recursion/models/res34_unet_1/best_model.pth'

DROPOUT_RATE = 0.2
LATENT_DIM = 512
TEMPERATURE = 60
MARGIN = 0.5
