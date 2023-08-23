import sys
sys.path.append('/home/hazen/Documents/PROJECTS/2.ObjDetSD/KITTI-Detection') # Added to train on custom data
from utils.config import Config
from utils.misc import init_env  
import warnings
warnings.filterwarnings("ignore")
import wandb
import os


cfg = Config().parse()
init_env(cfg)




######################################################################################
##################################    WANDB   ########################################
######################################################################################
# two important variables `JITTER_DATA` and `PRESERVE_ASPECT` are defined in the `src
# .common` script

EXPERIMENT_NAME = cfg.exp_id    # name of the experiment
SEED = cfg.seed                                       # seed for seeding all libraries and samplers
BATCH_SIZE = cfg.batch_size                                 # mini-batch size for training
# SHUFFLE = True                                  # should dataloaders shuffle the data
LR = cfg.lr                                       # starting learning rate
# LR_GAMMA = 0.68                                 # the amount by which learning rate get adjusted
# LR_STEP_SIZE = 4                                # adjust learning rate after how many steps
WEIGHT_DECAY = cfg.weight_decay                             # the amount of weight decay after each scheduler step
NUM_EPOCHS = cfg.num_epochs                                 # total epochs
# EVAL_EPOCHS = 1                                 # after how many epoch to perform evaluation
# WARMUP_EPOCHS = 0                               # for training visibility head alone in the beginning
NUM_WORKERS = cfg.num_workers                                 # number of cpu workers in dataloader
LOG_WANDB = not(cfg.no_log_wandb)                      # log data to wandb.ai or not
WANDB_ID = wandb.util.generate_id()             # ID for current wandb run (could use older id to resume)
# KEYPOINTS_COUNT = len(KEYPOINT_MAPPING)         # number of keypoints for training
# device to use, works for single and multiple gpus as well
DEVICE = cfg.gpus
# loading state dict containing model weights, optimizer state
# LOAD_STATE_DICT = ""
# PRELOAD_WEIGHTS = LOAD_STATE_DICT
# base directories containing multiple folders of data in them

# PROJECT_BASE_DIR = str(pathlib.Path(__file__).parent.resolve())
# TRAIN_BASE_DIR = "/mnt_drive/synthetic frames/parsed/KPs"
# TEST_BASE_DIR = "/mnt_drive/finalized_datasets/keypoints/real_test"
# EXERIMENT_DIR_PATH = join(PROJECT_BASE_DIR, "experiments", EXPERIMENT_NAME)

WANDB_CONFIG = {
    "experiment_name": EXPERIMENT_NAME,
    "learning_rate": LR,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED,
    "num_workers": NUM_WORKERS,
}

if LOG_WANDB:
    KEY = "6109333195f38e41b3869cff8d20dbee0dc05ff3"
    # store this id to use it later when resuming
    ID = WANDB_ID
    # or via environment variables
    os.environ["WANDB_API_KEY"] = KEY
    os.environ["WANDB_NOTEBOOK_NAME"] = "train.ipynb"
    # os.environ["WANDB_RESUME"] = "allow"
    # os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
    wandb.init(
        project="LicensePlateDetector",
        entity="hazenai",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=EXPERIMENT_NAME,
        config=WANDB_CONFIG,
        id=ID,
        resume="allow",
    )
    wandb.config = WANDB_CONFIG
    print("wandb run id: ", ID)


######################################################################################
##################################    MAIN    ########################################
######################################################################################




if cfg.mode == 'train':
    from train import train  
    train(cfg)
elif cfg.mode == 'eval':
    from eval import eval
    eval(cfg)
elif cfg.mode == 'tfliteeval':
    from tfliteeval import TfliteEval
    TfliteEval(cfg)
elif cfg.mode == 'onnx':
    from toonnx import ToOnnx
    ToOnnx(cfg)

elif cfg.mode == 'demo':
    from demo import demo
    demo(cfg)
else:
    raise ValueError('Mode {} is invalid.'.format(cfg.mode))



if LOG_WANDB:
    wandb.finish()
