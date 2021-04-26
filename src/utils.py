import gc
import h5py
import math
import matplotlib.pyplot as plt
import os
import psutil
import sys
import torch
import numpy as np
import torch.nn.init as init
import uuid


def nice_print(**kwargs):
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            print(f'Tensor "{k}" has shape {v.shape}')
        else:
            print(f'Variable "{k}" has value `{v}`')


def print_shape(o):
    dims = []
    while True:
        try:
            dims.append(len(o))
            o = next(iter(o))
        except:
            print(dims)
            return


def mem_report():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0 ** 30  # memory use in GB...I think
    print("memory GB:", memoryUse)


def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks
       E.g., grouper('ABCDEFG', 3) --> ABC DEF
    """
    args = [iter(iterable)] * n
    return zip(*args)


def window(seq, size=2, stride=1):
    """Returns a sliding window (of width n) over data from the iterable
       E.g., s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...  
    """
    it = iter(seq)
    result = []
    for elem in it:
        result.append(elem)
        if len(result) == size:
            yield result
            result = result[stride:]



def draw(pre_seqs, targets, results):
    plt.ion()
    targets = targets.cpu().detach().numpy()
    results = results.cpu().detach().numpy()
    pre_seqs = pre_seqs.cpu().detach().numpy()
    pre_seqs = np.concatenate(pre_seqs,axis=0)

    fig, axs = plt.subplots(2, 15, figsize=(5, 5), constrained_layout=True)

    for i, img in enumerate(pre_seqs):
        axs[0][i].imshow(img[0][0].squeeze())
        axs[1][i].imshow(img[0][0].squeeze())

    for i, target, result in enumerate(zip((targets, results)), start=10):
        axs[0][i].imshow(target[0][0].squeeze())
        axs[1][i].imshow(result[0][0].squeeze())

    plt.pause(2)
    plt.close('all')
    return 0

def weights_init(init_type="gaussian"):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find("Conv") == 0 or classname.find("Linear") == 0) and hasattr(
            m, "weight"
        ):
            # print m.__class__.__name__
            if init_type == "gaussian":
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == "default":
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def h5_virtual_file(filenames, name="data"):
    """
    Assembles a virtual h5 file from multiples
    """
    vsources = []
    total_t = 0
    for path in filenames:
        data = h5py.File(path, "r").get(name)
        t, *features_shape = data.shape
        total_t += t
        vsources.append(h5py.VirtualSource(path, name, shape=(t, *features_shape)))

    # Assemble virtual dataset
    layout = h5py.VirtualLayout(shape=(total_t, *features_shape), dtype=data.dtype)
    cursor = 0
    for vsource in vsources:
        # we generate slices like layour[0:10,:,:,:]
        indices = (slice(cursor, cursor + vsource.shape[0]),) + (slice(None),) * (
            len(vsource.shape) - 1
        )
        layout[indices] = vsource
        cursor += vsource.shape[0]
    # Add virtual dataset to output file
    f = h5py.File(f"{uuid.uuid4()}.h5", "w", libver="latest")
    f.create_virtual_dataset(name, layout)
    return f

def MNISTdataLoader(path):
    # loadmovingmnistdata,datashape=[timesteps,batchsize,width,height]=[20,batch_size,64,64]
    # BSHW->SBHW
    data=np.load(path)
    data_trans=data.transpose(1,0,2,3)
    return data_trans