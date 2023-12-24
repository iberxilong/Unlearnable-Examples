import numpy as np

# 用于得到label，
def make_label(plaintext, key, target):
    p = np.array(plaintext) #   创建副本的常用方式，为了不改变原始数据
    k = np.array(key)

    label = p[:, target]*16 + k[:, target]  #   因为输入的矩阵一个格子是 4bit 信息，即一个16进制数。所以 *16 等于放在左边
    # 明文加密钥的标签方式
    return label