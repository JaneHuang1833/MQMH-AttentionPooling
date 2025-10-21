import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import shap

X_df = pd.read_excel("make_feature_X.xlsx", index_col='Index')
X = X_df.to_numpy()               # shape (n, p)
feature_names = X_df.columns.tolist()

y = pd.read_excel("output_umap_1.xlsx", index_col='Index').loc[:, ['dim1', 'dim2']].to_numpy()

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("m", ExtraTreesRegressor(
        max_depth=10,    # 从网格搜索结果中选取的超参数,后续还可以往细里调
        min_samples_leaf=1,
        min_samples_split=10,
        n_estimators=50,
        n_jobs=-1,
        random_state=42
    ))
])
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipe, X, y, cv=kf, n_jobs=-1)

euclid_mae = np.mean(np.linalg.norm(y_pred - y, axis=1))  # 欧式距离为准确率评分标准
y_std = y.std(axis=0, ddof=0)
euclid_mae_z = np.mean(np.linalg.norm((y_pred - y) / y_std[None, :], axis=1))
print("CV Euclid MAE:", euclid_mae)
print("CV Euclid MAE (z-space):", euclid_mae_z)

# 置换重要性（Permutation Importance）

# 先 fit 一个最终模型（在全部数据上） ----------
final_est = ExtraTreesRegressor(
    max_depth=10,
    min_samples_leaf=1,
    min_samples_split=10,
    n_estimators=50,
    n_jobs=-1,
    random_state=42
)
final_est.fit(X, y)   # ExtraTrees 支持多输出 y 直接拟合

# 定义合适的 scoring：用 “负 欧几里得 MAE” 作为 score（score 越大越好） ----------
def neg_euclid_mae(y_true, y_pred):
    # 返回值越大越好 —— 所以返回 -MAE（越接近 0 越好，越负越差）
    return -np.mean(np.linalg.norm(y_true - y_pred, axis=1))

scorer = make_scorer(neg_euclid_mae, greater_is_better=True)

r = permutation_importance(
    final_est, X, y,
    scoring=scorer,
    n_repeats=30,
    random_state=42,
    n_jobs=-1
)

'''
r.importances_mean 的含义：baseline_score - score_with_permuted_feature（score 越大越好）
因为我们用的是 neg_euclid_mae（越大越好），如果打乱某特征后 score 降低（更小），
baseline_score - permuted_score 为正 => 该特征是“重要的”。
'''
imp_mean = r.importances_mean
imp_std  = r.importances_std

imp_series = pd.Series(imp_mean, index=feature_names).sort_values(ascending=False)

print("Top 20 permutation importances (higher = more important):")
print(imp_series.head(20))


topk = 20
top_feats = imp_series.head(topk).index
plt.figure(figsize=(8,6))
vals = imp_series.head(topk).values
stds = imp_std[[feature_names.index(f) for f in top_feats]]
plt.barh(range(topk-1, -1, -1), vals, xerr=stds[::-1], align='center')
plt.yticks(range(topk-1, -1, -1), top_feats[::-1])
plt.xlabel("Permutation importance (increase in neg_euclid_mae)")
plt.title("Top-{} permutation importances (ExtraTrees)".format(topk))
plt.tight_layout()
plt.show()



# SHAP

final_est.fit(X, y)

explainer = shap.Explainer(final_est)


shap_values = explainer(X)


i = 1
dim_name = 'dim2'
shap.summary_plot(
    shap_values[..., i],
    X,
    feature_names=feature_names,
    max_display=20
)
plt.title(f"SHAP Summary Plot for {dim_name}")
plt.show()