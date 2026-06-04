import torch.utils.data
import numpy as np, h5py
import random


def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
    data_fs_s1=MyLoadDataSet(target_file)

    target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
    data_fs_s2=MyLoadDataSet(target_file)


    # 模拟输入数据集
    # data_fs_s1 = np.random.rand(16, 1, 256,256).astype(np.float32)
    # data_fs_s2 = np.random.rand(16, 1, 256,256).astype(np.float32)
    #
    # # distribution --> [0,1]
    # data_fs_s1 = (data_fs_s1 - data_fs_s1.max()) / (data_fs_s1.max() - data_fs_s1.min() + 1e-6)
    # data_fs_s2 = (data_fs_s2 - data_fs_s2.max()) / (data_fs_s2.max() - data_fs_s2.min() + 1e-6)
    #
    # # [0,1] --> [-1,1]
    # data_fs_s1 = data_fs_s1 * 2 - 1
    # data_fs_s2 = data_fs_s2 * 2 - 1


    print("dataf1=", data_fs_s1.shape , "  dataf2=", data_fs_s2.shape)

    dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))  
    return dataset 



#Dataset loading from load_dir and converintg to 256x256 
def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True):
    f = h5py.File(load_dir,'r') 
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
    else:
        data=np.transpose(np.array(f[variable]),(1,0,3,2))
    data=data.astype(np.float32) 
    if padding:
        pad_x=int((256-data.shape[2])/2)
        pad_y=int((256-data.shape[3])/2)
        print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
        data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))   
    if Norm:    
        data=(data-0.5)/0.5      
    return data

def MyLoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True):
    f = h5py.File(load_dir,'r')
    if np.array(f[variable]).ndim==3:
        data=np.expand_dims(np.array(f[variable]),axis=1)

    data=data.astype(np.float32)
    if Norm:
        data=(data-0.5)/0.5  # [0,1] --> [-1,1]
    return data
