import os
import pandas as pd
import umap
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



DATA_DIR = r'D:\kry_utterance\code\attention_pooling'
NPY_DIR = os.path.join(DATA_DIR, "split_segments_5_no-anoucement")
EXCEL_PATH = os.path.join(DATA_DIR, "full_200label_1_BAC2019Q1.xlsx")

OUT_CSV = os.path.join(DATA_DIR, "umap_results.csv")
OUT_PNG = os.path.join(DATA_DIR, "umap_visualization.png")


df = pd.read_excel(EXCEL_PATH)
print(f"读取 Excel 数据成功！样本数：{len(df)}")
df["filename"] = df["path"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])



# index='停顿'
# print(f"NaN count in {index}:", pd.isna(df[index]).sum())
# print(f"Unique values in {index}:", pd.unique(df[index]))
# 计算每个样本的5秒平均embedding向量
def load_and_average_embedding(file_path):
    arr = np.load(file_path, allow_pickle=False)
    if arr.ndim == 2:
        return arr.mean(axis=0)  # 对5秒做时间均值
    elif arr.ndim == 1:
        return arr
    else:
        raise ValueError(f"不支持的数组形状 {arr.shape} 于文件 {file_path}")


# 根据 Excel 中的 path 列匹配每个样本对应的 .npy 文件
embedding_features = []
for _, row in df.iterrows():
    npy_path = os.path.join(NPY_DIR, f"{row['filename']}.npy")
    if os.path.exists(npy_path):
        embedding_features.append(load_and_average_embedding(npy_path))
    else:
        embedding_features.append(np.zeros(1024))  # 若文件不存在，填充为零向量

X = np.vstack(embedding_features) # (num_samples, feature_dim)
print(f"成功加载 {X.shape[0]} 个样本的嵌入特征，每个样本的维度为 {X.shape[1]}")

# 降维并遍历超参数选择最优结果 ----------
param_grid = {
    "n_neighbors": [5, 10, 15,20,25, 30,35,40,45, 50],
    "min_dist": [0.1, 0.2, 0.3, 0.4, 0.5]
}

# 用于存储不同超参数下的 NMI 结果
best_nmi = -1
best_params = None
best_umap = None


# NMI计算函数
def compute_nmi(Y, labels_true, n_clusters=5):
    # 对Y做KMeans聚类，得到离散的簇标签
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_pred = kmeans.fit_predict(Y)
    # 用真实的Arousal标签与聚类结果计算NMI
    return normalized_mutual_info_score(labels_true, labels_pred)


df['arva'] = df['Arousal'] *df['Valence']

for params in ParameterGrid(param_grid):
    print(f"正在尝试超参数：{params}")
    umap_model = umap.UMAP(n_neighbors=params["n_neighbors"], min_dist=params["min_dist"], metric="cosine",
                           random_state=42)
    Y = umap_model.fit_transform(X)
    nmi = compute_nmi(Y, df['arva'].values, n_clusters=5)
    print(f"NMI：{nmi}")
    if nmi > best_nmi:
        best_nmi = nmi
        best_params = params
        best_umap = umap_model



print(f"最佳超参数：{best_params}")
print(f"最佳 NMI：{best_nmi}")
# 得到的最佳参数是 n_neighbors=15, min_dist=0.2

# 使用最佳参数重新训练 UMAP
best_umap = umap.UMAP(n_neighbors=best_params['n_neighbors'], min_dist=best_params['min_dist'], metric="cosine",
                      random_state=42)
Y_best = best_umap.fit_transform(X)
# df_Y = pd.DataFrame(Y_best, columns=["dim1", "dim2"])
#
# df_Y.to_excel(os.path.join(DATA_DIR, "output_umap_1.xlsx"), index=False)


# 统一 colormap 和固定范围 0~1
cmap = 'magma'
norm = Normalize(vmin=0, vmax=1)

plt.figure(figsize=(10, 8))

shapes = {1:'o', 0:'*'}
colors = df['arva'].values *0.2

# 可视化
for label, shape in shapes.items():
    idx = df["Gender"] == label
    gender_str = "male" if label == 1 else "female"
    plt.scatter(
        Y_best[idx, 0], Y_best[idx, 1],
        label=gender_str,
        marker=shape,
        c=colors[idx],
        alpha=1,
        cmap = 'magma',
        s = 40,
        edgecolors = 'none'
    )

plt.colorbar(label="Arousal")
plt.legend(title="Gender")
plt.title("Embedding Visualization by Gender and Arousal*Valence_UMAP")
plt.xlabel("Dim1")
plt.ylabel("Dim2")
plt.tight_layout()
plt.show()
df["dim1"], df["dim2"] = Y_best[:, 0], Y_best[:, 1]
df.to_csv(OUT_CSV, index=False)


print(f"二维坐标已保存至：{OUT_CSV}")
print(f"可视化图已保存至：{OUT_PNG}")
