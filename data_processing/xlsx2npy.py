import pandas as pd
import numpy as np
import os


def excel_to_npy(excel_path, output_dir=None):
    """
    将 Excel 文件转换为 .npy 文件

    参数：
    - excel_path: str，Excel 文件路径
    - output_dir: str，可选，输出文件夹路径；如果为 None，则输出到原目录下

    返回：
    - npy_path: str，生成的 .npy 文件路径
    """

    df = pd.read_excel('attentionheapmap_BAC_2019Q1.xlsx')


    arr = df.to_numpy()

    base = os.path.splitext(os.path.basename(excel_path))[0]
    npy_name = "test_BAC_2019Q1.npy"
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, npy_name)
    else:
        npy_path = os.path.join(os.path.dirname(excel_path), npy_name)


    np.save(npy_path, arr)
    print(f"✅ 已保存为 {npy_path}, 形状为 {arr.shape}")
    return npy_path



if __name__ == "__main__":
    excel_to_npy("D:/data/sample.xlsx", output_dir="D:/data/npy_files")
