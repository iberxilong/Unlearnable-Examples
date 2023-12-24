import sys
import h5py
import numpy as np

# 获得数据
def get_data(file):

    try:
        in_file = h5py.File(file)
    except:
        print("错误: 不能打开 '%s' ！" % file)
        sys.exit(-1)

    # Load traces
    X = np.array(in_file['traces'], dtype=np.float32)
    # Load plaintext
    plaintext = np.array(in_file['plaintext'], dtype=np.int32)
    # Load key
    key = np.array(in_file['key'], dtype=np.int32)
    # Load cipher
    cipher = np.array(in_file['cipher'], dtype=np.int32)

    return X, plaintext, key, cipher