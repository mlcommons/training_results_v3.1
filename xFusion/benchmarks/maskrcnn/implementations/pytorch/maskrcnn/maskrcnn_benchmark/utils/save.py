import torch

def save(blob, folder, name, iteration):
    fname = "%s/%s_%d_%d.pt" % (folder, name, torch.distributed.get_rank(), iteration)
    with open(fname, "wb") as f:
        torch.save(blob, f)
        print("%d :: Wrote %s to %s" % (torch.distributed.get_rank(), name, fname))
