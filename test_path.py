#!/usr/bin/env python3
"""
测试路径配置是否正确
"""

import os
import sys

# 添加code目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

try:
    from config import DATASET_PATH, OUTPUT_DIR
    
    print("=== 路径测试 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"配置文件位置: {os.path.abspath(__file__)}")
    print(f"数据文件路径: {DATASET_PATH}")
    print(f"输出目录路径: {OUTPUT_DIR}")
    
    # 检查数据文件是否存在
    if os.path.exists(DATASET_PATH):
        print(f"✅ 数据文件存在: {DATASET_PATH}")
    else:
        print(f"❌ 数据文件不存在: {DATASET_PATH}")
        
    # 检查输出目录是否存在，不存在则创建
    if os.path.exists(OUTPUT_DIR):
        print(f"✅ 输出目录存在: {OUTPUT_DIR}")
    else:
        print(f"⚠️  输出目录不存在，正在创建: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print("✅ 输出目录创建成功")
        
    # 测试读取数据
    import pandas as pd
    try:
        df = pd.read_excel(DATASET_PATH)
        print(f"✅ 数据读取成功，数据集形状: {df.shape}")
    except Exception as e:
        print(f"❌ 数据读取失败: {e}")
        
except ImportError as e:
    print(f"❌ 导入错误: {e}")
except Exception as e:
    print(f"❌ 其他错误: {e}")