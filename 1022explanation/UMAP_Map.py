import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

'''
任务：
用UMAP对1024个embedding特征降维成二维的（对neutral和unknown,happy ,sad样本，只选择前2000条）
1.有网格搜索调参过程
2.生成的二维坐标保存在原文件最后两列
3.生成图像，并用颜色区分不同样本，按照情绪唤醒度匹配对应颜色
'''

input_dir = 'emo' 
embedding_cols = [f'Embedding_{i}' for i in range(1,1025)]

emotion_arousal_map = {
    'sad': 0.1,
    'neutral': 0.2,
    'disgusted': 0.3,
    'fearful': 0.6,
    'happy': 0.7,
    'angry': 0.9,
    'surprised': 1.0,
    'unknown': 0.0
}

# UMAP参数搜索范围
param_grid = {
    'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    'min_dist': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'metric': ['cosine']
}

df_all = pd.read_excel('combined_emotion_data.xlsx')

X = df_all[embedding_cols].values
labels = df_all['emotion'].values


# 寻找最优参数使得余弦距离最小（省略）

reducer = umap.UMAP(n_neighbors=25, min_dist=0.0, metric='cosine', random_state=42)
X_emb = reducer.fit_transform(X)

print("正在保存 UMAP 坐标到原始 Excel 文件...")

df_all['UMAP_1'] = X_emb[:, 0]
df_all['UMAP_2'] = X_emb[:, 1]



df_all.to_excel('2combined_emo.xlsx')  # 总表


print(" 正在绘图...")

# 计算唤醒度颜色
df_all['arousal'] = df_all['emotion'].map(emotion_arousal_map)
plt.figure(figsize=(10, 8))
sc = plt.scatter(df_all['UMAP_1'], df_all['UMAP_2'], c=df_all['arousal'],
                 cmap='coolwarm', s=10, alpha=0.7)

plt.colorbar(sc, label='Arousal Level')
plt.title('UMAP projection of embeddings colored by arousal')
plt.xlabel('UMAP_1')
plt.ylabel('UMAP_2')
plt.tight_layout()
plt.savefig('umap_projection.png', dpi=300)
plt.show()

print("所有任务完成！图像保存为 umap_projection.png")
