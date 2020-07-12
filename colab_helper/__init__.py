from .ColabHelper import ColabHelper
from .GetData import GetData, _list_folders, _prevent_duplicates, img_generator
from .HardwareAssistant import Rig, info, gpu_info, tpu_info, default_strategy
from .Plots import plot_minibatch
from .Preproccessing import Preproccessing, decode_img, get_label

__version__ = '0.1.2-dev20200712'
__author__ = 'Guillermo Alcántara Gonzälez (Echo9k)'
