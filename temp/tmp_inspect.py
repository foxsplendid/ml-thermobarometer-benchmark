# 临时脚本：检查 input.csv 列名结构
import pandas as pd

# 尝试多种编码
for enc in ['utf-8-sig', 'latin-1', 'cp1252']:
    try:
        df = pd.read_csv('input.csv', encoding=enc, nrows=3)
        # 保存列名到文件
        with open('columns_info.txt', 'w', encoding='utf-8') as f:
            f.write(f"Encoding: {enc}\n")
            f.write(f"Shape: {df.shape}\n\n")
            f.write("All columns:\n")
            for i, col in enumerate(df.columns):
                f.write(f"  {i}: {repr(col)}\n")
            f.write("\nFirst row values:\n")
            for col in df.columns:
                f.write(f"  {col}: {df[col].iloc[0]}\n")
        print("Saved to columns_info.txt")
        break
    except Exception as e:
        print(f"Failed with {enc}: {e}")
