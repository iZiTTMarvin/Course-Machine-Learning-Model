#!/usr/bin/env python3
"""
Excel转CSV转换工具
"""

import pandas as pd
import sys
import os

def convert_excel_to_csv(excel_file):
    """
    将Excel文件转换为CSV格式
    
    Args:
        excel_file (str): Excel文件路径
    """
    try:
        # 读取Excel文件
        print(f"正在读取Excel文件: {excel_file}")
        df = pd.read_excel(excel_file)
        
        # 生成CSV文件路径
        csv_file = excel_file.replace('.xlsx', '.csv')
        
        # 保存为CSV
        df.to_csv(csv_file, index=False)
        
        print(f"✅ 转换成功!")
        print(f"Excel文件: {excel_file}")
        print(f"CSV文件: {csv_file}")
        print(f"数据形状: {df.shape}")
        
        return csv_file
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        return None

if __name__ == "__main__":
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    excel_file = os.path.join(current_dir, "data", "Dry_Bean_Dataset.xlsx")
    
    print("=== Excel转CSV转换工具 ===")
    print(f"Excel文件路径: {excel_file}")
    
    if os.path.exists(excel_file):
        convert_excel_to_csv(excel_file)
    else:
        print(f"❌ Excel文件不存在: {excel_file}")
        print("尝试查找其他可能的路径...")
        
        # 尝试其他可能的路径
        possible_paths = [
            os.path.join(current_dir, "..", "data", "Dry_Bean_Dataset.xlsx"),
            "data/Dry_Bean_Dataset.xlsx",
            "../data/Dry_Bean_Dataset.xlsx"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到文件在: {path}")
                convert_excel_to_csv(path)
                break
        else:
            print("未找到Excel文件，请检查文件路径")