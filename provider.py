import os
import sys
import numpy as np
#import h5py
import pandas as pd
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
#DATA_DIR = os.path.join(BASE_DIR, 'data')
#if not os.path.exists(DATA_DIR):
#    os.mkdir(DATA_DIR)
#if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
 #   www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
  #  zipfile = os.path.basename(www)
   # os.system('wget %s; unzip %s' % (www, zipfile))
    #os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
   # os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx,idx,idx], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def filling_data(filename,batch_size):

    label=np.zeros(batch_size)#only one category for each csv
    # label=np.zeros((60000))
    data=pd.read_csv(filename, encoding='cp936',header=None)

    ctgname=filename.replace('pts','category')
    ctg=pd.read_csv(ctgname, encoding='cp936',header=None)


    length_each_batch=60000//batch_size
    data_total=np.zeros((batch_size,length_each_batch,3))
    ctg_total=np.zeros((batch_size,length_each_batch))

    var=0.001
    #get noise
    length=len(data)
    is_same=(length==len(ctg))
    if is_same==False:
        return data_total,label,ctg_total,is_same

    if length > 60000:
        acc_list = random.sample(range(length), 60000)
        for i in range(batch_size):
            begid=i*length_each_batch
            endid=(i+1)*length_each_batch
            data_total[i,:,:]=data.loc[acc_list[begid:endid]]
            tmp = np.zeros((length_each_batch, 1))
            tmp = ctg.loc[acc_list[begid:endid]]
            tmp = np.array(tmp)
            tmp = tmp.reshape(1, -1)
            ctg_total[i, :] = tmp
            # ctg_total[i,:]


        # total[i, :, :] = data.loc[acc_list]
        # tmp = np.zeros((60000, 1))
        # tmp = ctg.loc[acc_list]
        # tmp = np.array(tmp)
        # tmp = tmp.reshape(1, -1)
        # ctg_total[i, :] = tmp

    else:
        noise=np.random.normal(0,var,[60000-length,3])

        # random select node to filling with noise
        insert_data_list=random.sample(range(length),(60000-length))
        insert_data_list.sort()
        acc_list=range(length_each_batch)
        #insert_data
        insert_data=np.array(data.loc[insert_data_list])
        insert_data=insert_data+noise

        #insert_data and output it
        data=data.append(pd.DataFrame(insert_data))  ####!!!!!data=!!!!!!!!!!!
        # data.to_csv(insert_file_path+'p/'+filename, mode='w', encoding='cp936', index=False, header=None)
        # data=np.array(data)
        # data2=np.zeros((1,60000,3))
        # data2[0:1,:,:]=data


        insert_data = np.array(ctg.loc[insert_data_list])
        ctg=ctg.append(pd.DataFrame(insert_data))
        data=np.array(data)
        ctg=np.array(ctg)

        for i in range(batch_size):
            begid=i*length_each_batch
            endid=(i+1)*length_each_batch
            data_total[i,:,:]=data[begid:endid,:]
            tmp = np.zeros((length_each_batch, 1))
            tmp = ctg[begid:endid]
            tmp = np.array(tmp)
            tmp = tmp.reshape(1, -1)
            ctg_total[i, :] = tmp

    return (data_total, label, ctg_total,is_same)



def loadDataFile_with_seg(filename,batch_size):
    return filling_data(filename,batch_size)

def loadDataFile_with_seg_batch(train_list,file_batch_size,batch_size):
    new_batch_size=batch_size*file_batch_size
    length_each_batch = 60000 // batch_size

    data_total=np.zeros((new_batch_size,length_each_batch,3))
    labels=np.zeros(new_batch_size)
    ctg_total=np.zeros((new_batch_size,length_each_batch))
    is_same=True
    for i in range(file_batch_size):
        filename=train_list[i]

    # label=np.zeros((60000))
        data = pd.read_csv(filename, encoding='cp936', header=None)

        ctgname = filename.replace('pts', 'category')
        ctg = pd.read_csv(ctgname, encoding='cp936', header=None)

        #length_each_batch = 60000 // batch_size
        # data_total = np.zeros((batch_size, length_each_batch, 3))
        # ctg_total = np.zeros((batch_size, length_each_batch))

        var = 0.001
        # get noise
        length = len(data)
        is_same = (length == len(ctg))
        if is_same == False:
            return data_total, labels, ctg_total, is_same

        if length > 60000:
            acc_list = random.sample(range(length), 60000)
            for j in range(batch_size):
                cur_batch_id=i*batch_size+j
                begid = j * length_each_batch
                endid = (j + 1) * length_each_batch
                data_total[cur_batch_id, :, :] = data.loc[acc_list[begid:endid]]
                tmp = np.zeros((length_each_batch, 1))
                tmp = ctg.loc[acc_list[begid:endid]]
                tmp = np.array(tmp)
                tmp = tmp.reshape(1, -1)
                ctg_total[cur_batch_id, :] = tmp


        else:
            noise = np.random.normal(0, var, [60000 - length, 3])

            # random select node to filling with noise
            insert_data_list = random.sample(range(length), (60000 - length))
            insert_data_list.sort()
            acc_list = range(length_each_batch)
            # insert_data
            insert_data = np.array(data.loc[insert_data_list])
            insert_data = insert_data + noise

            data = data.append(pd.DataFrame(insert_data))  ####!!!!!data=!!!!!!!!!!!

            insert_data = np.array(ctg.loc[insert_data_list])
            ctg = ctg.append(pd.DataFrame(insert_data))
            data = np.array(data)
            ctg = np.array(ctg)

            for j in range(batch_size):
                cur_batch_id = i * batch_size + j
                begid = j * length_each_batch
                endid = (j + 1) * length_each_batch
                data_total[cur_batch_id, :, :] = data[begid:endid, :]
                tmp = np.zeros((length_each_batch, 1))
                tmp = ctg[begid:endid]
                tmp = np.array(tmp)
                tmp = tmp.reshape(1, -1)
                ctg_total[cur_batch_id, :] = tmp

    return (data_total, labels, ctg_total, is_same)
