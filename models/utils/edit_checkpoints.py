import torch

"""
Adds new metadata to a checkpoint
INPUT:
    expects a dictionary
    each key value pair in the dictionary will be added to the checkpoint metadata
"""
def add_checkpoint_metadata(checkpoint_path, new_metadata):

    if type(new_metadata) != dict: raise Exception("add_checkpoint_metadata expects a dictionary input")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    for key, value in new_metadata.items():
        checkpoint[key] = value

    torch.save(checkpoint, checkpoint_path)


