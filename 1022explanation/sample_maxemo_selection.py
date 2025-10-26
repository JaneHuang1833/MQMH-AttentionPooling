import os
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import re

# === 路径设置 ===
path_a = 'D:/emotion_attention/lld_BAC'
path_b = 'D:/emotion_attention/emo_BAC'

# === 配置 ===
low_level_features = ['zero_crossings_rate', 'frame_energy', 'acf',
                      'spectral_centroids', 'loudness', 'sharpness', 'mfcc']
emotion_probs = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'unknown', 'other']
valid_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
embedding_cols = [f'Embedding_{i}' for i in range(1,1025)]

# === 存储所有样本（最终用于标准化） ===
all_data = []

# === 遍历每一个会议 ===
for meeting_folder in tqdm(sorted(os.listdir(path_a))):
    folder_a = os.path.join(path_a, meeting_folder)
    folder_b = os.path.join(path_b, meeting_folder)

    if not os.path.isdir(folder_a) or not os.path.isdir(folder_b):
        continue

    # === 读取B中的情绪+高维数据，并从文件名提取年份季度 ===
    csv_files_b = [f for f in os.listdir(folder_b) if f.endswith('.csv')]
    if not csv_files_b:
        print(f"[跳过] {meeting_folder} 中 B 无 CSV 文件。")
        continue

    filename_b = csv_files_b[0]
    df_b = pd.read_csv(os.path.join(folder_b, filename_b))

    match = re.search(r'BAC[_\-]?(\d{4}Q\d)', filename_b)
    if match:
        year_quarter = match.group(1)
    else:
        print(f"[跳过] 无法从文件名提取年份季度信息：{filename_b}")
        continue

    time_col_b = 'second' if 'second' in df_b.columns else 'Start(s)'
    df_b['second'] = df_b[time_col_b].astype(int)
    df_b_use = df_b[['second'] + emotion_probs + embedding_cols]

    # === 找到A中两个特征文件 ===
    csv_files_a = os.listdir(folder_a)
    file_a1 = next((f for f in csv_files_a if 'normalized_features' in f and f.endswith('.csv')), None)
    file_a2 = next((f for f in csv_files_a if 'normalized_new_features' in f and f.endswith('.csv')), None)

    if not file_a1 or not file_a2:
        print(f"[跳过] {meeting_folder} 缺失A中的特征文件。")
        continue

    df_a1 = pd.read_csv(os.path.join(folder_a, file_a1))
    df_a2 = pd.read_csv(os.path.join(folder_a, file_a2))

    def preprocess_df(df):
        time_col = 'second' if 'second' in df.columns else 'Start(s)'
        df['second'] = df[time_col].astype(int)
        return df[['second'] + [col for col in df.columns if col in low_level_features]]

    df_low1 = preprocess_df(df_a1)
    df_low2 = preprocess_df(df_a2)

    df_low = pd.merge(df_low1, df_low2, on='second', how='outer')
    df_low = df_low.groupby('second').first().reset_index()

    # === 合并A和B
    df_merged = pd.merge(df_b_use, df_low, on='second', how='inner')
    df_merged.dropna(inplace=True)

    if df_merged.empty:
        print(f"[跳过] {meeting_folder} 合并后无有效数据。")
        continue

    df_merged.insert(0, 'id', df_merged['second'].apply(lambda s: f"BAC_{year_quarter}_{s}"))
    df_merged['label'] = df_merged[emotion_probs].idxmax(axis=1)

    all_data.append(df_merged)

# === 拼接所有样本
if not all_data:
    raise ValueError("❌ 没有可用数据，无法继续。")

df_all = pd.concat(all_data, ignore_index=True)

# === 全局标准化低维特征
scaler = StandardScaler()
df_all[low_level_features] = scaler.fit_transform(df_all[low_level_features])

# === 按情绪分类导出 Excel
output_dir = 'output_excels_standardized'
os.makedirs(output_dir, exist_ok=True)

for label in valid_labels:
    df_label = df_all[df_all['label'] == label]
    if not df_label.empty:
        out_path = os.path.join(output_dir, f'{label}.xlsx')
        df_label.to_excel(out_path, index=False)
        print(f"[保存完成] {label}.xlsx 共 {len(df_label)} 条记录")

print("✅ 所有文件处理完成，标准化成功。")
