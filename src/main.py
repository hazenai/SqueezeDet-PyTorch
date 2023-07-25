import sys
sys.path.append('/home/hazen/Documents/PROJECTS/2.ObjDetSD/KITTI-Detection') # Added to train on custom data
from utils.config import Config
from utils.misc import init_env  
import warnings
warnings.filterwarnings("ignore")


cfg = Config().parse()
init_env(cfg)

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