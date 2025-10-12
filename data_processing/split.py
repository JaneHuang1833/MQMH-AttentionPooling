import os
import pandas as pd
import numpy as np

input_excel = r"C:\Users\IvyHuang\Desktop\BAC_2019Q1_full_audio_normalized_emotion.csv"
output_folder = "split_segments_5_no-anoucement"
output_csv = "embedding_dataset_5_no-anoucement.csv"
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(input_excel)

emotion_cols = ['angry', 'disgusted', 'fearful', 'happy',  'other', 'sad', 'surprised'] # 设置列,排除neutral列
embedding_cols = [col for col in df.columns if str(col).startswith("Embedding")]

summary = []
for i in range(1965, len(df), 5): #BAC_2019_Q1的QA是从1965秒开始的
    chunk = df.iloc[i:i+5]
    if len(chunk) < 5: continue

    avg_emotions = chunk[emotion_cols].mean() # 计算平均情绪分数
    label = avg_emotions.idxmax()

    emb = chunk[embedding_cols].values.astype(np.float32)  #  5×1024
    npy_path = os.path.join(output_folder, f"segment_{i//5:04d}.npy")
    np.save(npy_path, emb)
    summary.append({"path": npy_path, "label": label})

pd.DataFrame(summary).to_csv(output_csv, index=False)
print("处理完成，共保存段数：", len(summary))  # 文件路径，除去neutral列的做平均后的最大概率label，embedding1024