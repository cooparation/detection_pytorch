import argparse
import logging
import os

import torch
import torchvision

from ssd.config import cfg
from ssd.modeling.vgg_ssd import build_ssd_model

def save(net, input, save_path):
    net.eval()
    traced_script_module = torch.jit.trace(net, input)
    traced_script_module.save(save_path)

def load(model_path): # 'scriptsmodel.pt'
    return torch.jit.load(model_path)

# convert pytorch model to torch scripts
def convert2scriptmodule(cfg, args):

    ssd_model = build_ssd_model(cfg)

    print(ssd_model)

    input = torch.Tensor(1, 3, cfg.INPUT.IMAGE_SIZE, cfg.INPUT.IMAGE_SIZE)
    model_path = args.model_path # 'ssd300_vgg_final.pth'
    ssd_model.load_state_dict(torch.load(model_path))
    save(ssd_model, input, args.model_out) # 'script.pt'

    #net = torch.jit.load(args.model_out)
    #print('--torch scripts--\n', net)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert pytorch model to torch script')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument('--model_path', help='path to torchmodel.pth')
    parser.add_argument('--model_out', help='path to torchscripts.pt')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        #logger.info(config_str)
    #logger.info("Running with config:\n{}".format(cfg))

    convert2scriptmodule(cfg, args)

    print('------ convert done ------')
