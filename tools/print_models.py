import argparse
import logging
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torchvision
import torch.onnx as torch_onnx

from ssd.config import cfg
from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.engine.inference import do_evaluation
from ssd.engine.trainer import do_train
from ssd.modeling.data_preprocessing import TrainAugmentation
from ssd.modeling.multibox_loss import MultiBoxLoss
from ssd.modeling.ssd import MatchPrior
from ssd.modeling.vgg_ssd import build_ssd_model
from ssd.module.prior_box import PriorBox
from ssd.utils import distributed_util
from ssd.utils.logger import setup_logger
from ssd.utils.lr_scheduler import WarmupMultiStepLR
from ssd.utils.misc import str2bool

import tensorboardX


def train(cfg, args):
    #logging.basicConfig(filename='./output/LOG/'+__name__+'.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d%I:%M:%S %p')
    logging.basicConfig(filename='./output/LOG/'+__name__+'.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', 
        level = logging.INFO)
    logger = logging.getLogger('SSD.trainer')
    # -----------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------
    ssd_model = build_ssd_model(cfg)
    ssd_model.init_from_base_net(args.vgg)
    ssd_model = torch.nn.DataParallel(ssd_model, device_ids=range(torch.cuda.device_count()))
    device = torch.device(cfg.MODEL.DEVICE)
    print(ssd_model)
    logger.info(ssd_model)
    model = torchvision.models.AlexNet(num_classes=10)
    logger.info(model)
    writer = tensorboardX.SummaryWriter(log_dir="./output/model_graph/",comment="myresnet")
    #dummy_input = torch.autograd.Variable(torch.rand(1, 3, 227, 227))
    dummy_input = torch.autograd.Variable(torch.rand(1, 3, 300, 300))
    #writer.add_graph(model=ssd_model, input_to_model=(dummy_input, ))
    model_onnx_path = 'torch_model.onnx'
    output = torch_onnx.export(ssd_model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)

    #ssd_model.to(device)
    print('----------------')

def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--vgg', help='Pre-trained vgg model path, download from https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth')
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--log_step', default=50, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=5000, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    #if args.distributed:
    #    torch.cuda.set_device(args.local_rank)
    #    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    #logger = setup_logger("SSD", distributed_util.get_rank())
    #logger.info("Using {} GPUs".format(num_gpus))
    #logger.info(args)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        #logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))

    train(cfg, args)

if __name__ == '__main__':
    main()
