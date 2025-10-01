import hydra
import torch
from argparse import Namespace
from solver import Solver
import sys


def run_formant_tracking(cfg):
    cfg = Namespace(**dict(cfg))
    print(torch.cuda.is_available())
    print(sys.executable)
    cfg.device = torch.device("cuda" if (torch.cuda.is_available() and cfg.is_cuda) else "cpu")
    print(f"Using device: {cfg.device}")
    solver = Solver(cfg)
    encoder_output, pred_formants, fnames = solver.test()
    print("!~Done Shrem~!")
    return encoder_output, pred_formants, fnames

@hydra.main(version_base=None,config_path='conf', config_name='config')
def main(cfg):
    run_formant_tracking(cfg)

if __name__ == "__main__":
    main()
