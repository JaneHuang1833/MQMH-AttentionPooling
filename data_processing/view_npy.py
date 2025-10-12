import numpy as np
import os


def view_npy(npy_path, show_data=True, max_rows=10):
    """
    查看 .npy 文件的基本信息与内容

    参数：
    - npy_path: str，npy 文件路径
    - show_data: bool，是否打印部分数据（默认 True）
    - max_rows: int，最多显示前几行数据

    返回：
    - data: np.ndarray，加载的 numpy 数组
    """

    data = np.load(r'D:\kry_utterance\code\attention_pooling\test.npy', allow_pickle=True)


    print(f"文件路径: {npy_path}")
    print(f"数组形状: {data.shape}")
    print(f"数据类型: {data.dtype}")


    if show_data:
        print("\n 前几行数据:")
        if data.ndim == 1:
            print(data[:max_rows])
        elif data.ndim == 2:
            print(data[:max_rows, :])
        else:
            print(data[:max_rows])

    return data



if __name__ == "__main__":
    view_npy("D:/data/sample.npy")
