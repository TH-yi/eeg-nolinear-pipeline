import h5py
import os
import shutil

# 原始文件路径
input_path = r'C:\Users\123456\Documents\Github\eeg-nolinear-pipeline\storage\input_data\S1.mat'
output_path = r'C:\Users\123456\Documents\Github\eeg-nolinear-pipeline\storage\input_data\S1_no_RST.mat'

# 先复制一份作为基础
shutil.copy2(input_path, output_path)

# 在副本上删除数据
with h5py.File(output_path, 'a') as f:
    if 'Sig_RST1' in f:
        del f['Sig_RST1']
        print("Removed Sig_RST1")
    if 'Sig_RST2' in f:
        del f['Sig_RST2']
        print("Removed Sig_RST2")

print(f"✅ Cleaned .mat file saved to: {output_path}")
