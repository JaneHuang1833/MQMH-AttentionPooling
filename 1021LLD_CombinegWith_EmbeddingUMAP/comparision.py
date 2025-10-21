from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

X = pd.read_excel("make_feature_X.xlsx",index_col='Index').to_numpy()  #  (n, 56)
y = pd.read_excel("output_umap_1.xlsx", index_col='Index').loc[:, ['dim1', 'dim2']].to_numpy()  # (n, 2)



def eval_model(model, X, y):
    pipe = Pipeline([("scaler", StandardScaler()), ("m", model)])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipe, X, y, cv=kf)
    euclid_mae = np.mean(np.linalg.norm(y_pred - y, axis=1))
    return euclid_mae, pipe.fit(X, y)


# 1) 线性基线
from sklearn.linear_model import Ridge, MultiTaskLasso, MultiTaskElasticNet
mae_ridge,  pipe_ridge  = eval_model(Ridge(alpha=1.0), X, y)
mae_mtl,    pipe_mtl    = eval_model(MultiTaskLasso(alpha=0.01), X, y)
mae_mten,   pipe_mten   = eval_model(MultiTaskElasticNet(alpha=0.01, l1_ratio=0.5), X, y)

# 2) PLS
from sklearn.cross_decomposition import PLSRegression
mae_pls, pipe_pls = eval_model(PLSRegression(n_components=min(10, X.shape[1]//3)), X, y)

# 3) 树模型
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
mae_rf,  pipe_rf  = eval_model(RandomForestRegressor(n_estimators=400, random_state=42), X, y)
mae_et,  pipe_et  = eval_model(ExtraTreesRegressor(n_estimators=400, random_state=42), X, y)

# 4) 核方法（RBF）
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
mae_svr, pipe_svr = eval_model(MultiOutputRegressor(SVR(C=10, gamma="scale", kernel="rbf")), X, y)


# 5) 高斯过程
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0)
mae_gp, pipe_gp = eval_model(MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)), X, y)


maes = {
    "Ridge": mae_ridge, "MultiTaskLasso": mae_mtl, "MultiTaskElasticNet": mae_mten,
    "PLS": mae_pls, "RandomForest": mae_rf, "ExtraTrees": mae_et,
    "SVR(RBF)": mae_svr,  "GaussianProcess": mae_gp
}
print(sorted(maes.items(), key=lambda kv: kv[1]))
