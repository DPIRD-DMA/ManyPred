import torch

# The workstation and CI runners are shared, and pytest-xdist runs several workers, so keep
# each worker to a few threads
torch.set_num_threads(min(4, torch.get_num_threads()))
