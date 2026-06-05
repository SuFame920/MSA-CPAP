import torch
import os
import io


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    ds = getattr(args, 'dataset', '').lower()

    # 按数据集简单分流；其它数据集就落到 checkpoints_<dataset>
    if 'mosi' in ds:
        ckpt_dir = 'pre_trained_best_models_mosi'
    elif 'mosei' in ds:
        ckpt_dir = 'pre_trained_best_models_mosei'
    else:
        ckpt_dir = f'checkpoints_{ds or "default"}'

    os.makedirs(ckpt_dir, exist_ok=True)

    path = os.path.join(ckpt_dir, f'{name}.pt')

    torch.save(model.state_dict(), path)


def load_model(args, name=''):
    # name = save_load_name(args, name)
    name = name
    with open(f'pre_trained_models/{name}.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())
    model = torch.load(buffer)
    return model


def random_shuffle(tensor, dim=0):
    if dim != 0:
        perm = (i for i in range(len(tensor.size())))
        perm[0] = dim
        perm[dim] = 0
        tensor = tensor.permute(perm)

    idx = torch.randperm(t.size(0))
    t = tensor[idx]

    if dim != 0:
        t = t.permute(perm)

    return t

