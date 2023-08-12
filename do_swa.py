import torch
import argparse
from pathlib import Path


def do_swa(checkpoint):
    
    skip = []
    
    K = len(checkpoint)
    swa = None
    
    for k in range(K):
        state_dict = torch.load(checkpoint[k], map_location=lambda storage, loc: storage)['state_dict']
        if K==1:
            return state_dict
        if swa is None:
            swa = state_dict
        else:
            for k, v in state_dict.items():
                if any(s in k for s in skip): continue
                swa[k] += v
    
    for k, v in swa.items():
        if any(s in k for s in skip): continue
        try:
            swa[k] /= K
        except:
            swa[k] //= K

    return swa

def main(workdir):
    workdir = Path(workdir)
    checkpoint_paths = list(workdir.glob("epoch_*.pth"))
    checkpoint_paths = list(map(lambda x: str(x), checkpoint_paths))

    swa_model = do_swa(checkpoint_paths)

    state_dict = torch.load(checkpoint_paths[-1], map_location=lambda storage, loc: storage)

    state_dict['state_dict'] = swa_model

    save_path = workdir.joinpath("swa_last.pth")

    torch.save(state_dict, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do SWA')
    parser.add_argument('--workdir', help='work directory',)
    args = parser.parse_args()
    main(args.workdir)
