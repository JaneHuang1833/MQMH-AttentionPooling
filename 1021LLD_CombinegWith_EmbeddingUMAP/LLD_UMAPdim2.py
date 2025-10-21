import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

def slope(X_raw):
    n, T, d = X_raw.shape
    t = np.arange(T)

    Y = X_raw.transpose(1, 0, 2).reshape(T, -1)  # (5, n*d)

    # 对每一列做一次一次多项式拟合：返回系数形状 (2, n*d)
    # coef[0] 是斜率，coef[1] 是截距
    coef = np.polyfit(t, Y, deg=1)  # (2, n*d)
    slope = coef[0].reshape(n, d)   # (n, d) = (200, 7)

    return slope

# X_raw: (n_samples, 5, 7)  -> 展平 + 可选摘要特征
def make_features(X_raw):
    n, T, F = X_raw.shape  # T=5, F=7
    X1 = X_raw.reshape(n, T*F)  # 35维
    mean = X_raw.mean(axis=1)
    std  = X_raw.std(axis=1)
    t = np.arange(T)
    slope_x = slope(X_raw)
    return np.hstack([X1, mean, std, slope_x])  # 共 35+21=56维    7*5 + 7*3



excel_path_y = "output_umap_1.xlsx"

df_y_noindex = pd.read_excel(excel_path_y)
example_idx = df_y_noindex['Index'].to_numpy()

df_y = pd.read_excel(excel_path_y,index_col='Index')
df_y.index = df_y.index.astype(int)
y_coords = df_y.loc[example_idx, ['dim1', 'dim2']].to_numpy()

y = y_coords  # shape (n, 2)

# df_X = pd.read_excel("1965LLD.xlsx")
# print(df_X.columns)

def build_5s_tensor_from_excel(excel_path, second_col, feature_cols, valid_bins=None):
    df = pd.read_excel(excel_path,index_col='second')
    df  = df[feature_cols].reset_index()

    df["_bin"] = (df[second_col] // 5).astype(int) # 按五秒分样本


    groups, starts, kept_bins = [], [], []
    for b, g in df.groupby("_bin", sort=True):
        if valid_bins is not None and b not in valid_bins:
            continue
        g = g.sort_values(second_col)
        g5 = g.iloc[:5, :].copy()
        secs = g5[second_col].to_numpy()
        groups.append(g5[feature_cols].to_numpy())
        starts.append(int(secs[0]))
        kept_bins.append(b)


    X = np.stack(groups, axis=0)
    starts = np.array(starts, dtype=int)
    kept_bins = np.array(kept_bins, dtype=int)
    return X, kept_bins, starts, feature_cols





feature_cols = ['zero_crossings_rate', 'frame_energy', 'acf', 'spectral_centroids', 'loudness', 'sharpness','mfcc']
valid_bins = df_y_noindex['Index'].astype(int).to_numpy()

X, example_idx, starts, used_feats = build_5s_tensor_from_excel(
    excel_path="1965LLD.xlsx",
    second_col="second",
    feature_cols=feature_cols,
    valid_bins=valid_bins
)



X_raw = X  # (n,5,7)
print(X_raw.shape)


X = make_features(X_raw)

df_X= pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
df_X.to_excel("make_feature_X.xlsx")
print('final X shape:', X.shape)  # 应该是 (n, 56)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", MultiTaskElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=None, cv=5,
                                    max_iter=20000, selection='random', n_jobs=-1))
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipe, X, y, cv=kf)
euclid_mae = np.mean(np.linalg.norm(y_pred - y, axis=1))
print("Euclidean MAE:", euclid_mae)

# 还原
pipe.fit(X, y)
coef = pipe.named_steps["model"].coef_    # shape: (n_outputs=2, n_features)
inter = pipe.named_steps["model"].intercept_


scaler = pipe.named_steps["scaler"]
mu = scaler.mean_          # 各特征均值
sigma = scaler.scale_      # 各特征标准差

coef_scaled = pipe.named_steps["model"].coef_      # (2, n_features)
inter_scaled = pipe.named_steps["model"].intercept_  # (2,)

# 还原到原始空间
coef_orig = coef_scaled / sigma[None, :]           # (2, n_features)
inter_orig = inter_scaled - coef_scaled @ (mu / sigma)


coef_orig_df = pd.DataFrame(coef_orig, columns=[f"feat_{i}" for i in range(coef.shape[1])])
coef_orig_df.to_excel("MultiTaskElasticNetCV_coef_orig.xlsx", index=False)