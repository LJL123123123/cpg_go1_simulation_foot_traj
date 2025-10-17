#!/usr/bin/env python3
"""
足端轨迹CPG使用示例和总结
Example usage and summary of the Foot Trajectory CPG
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append('/home/cpg_go1_simulation/src')

from cpg_go1_simulation.stein.foot_trajectory_cpg import FootTrajectoryCPG

def demonstrate_foot_cpg_usage():
    """演示足端轨迹CPG的使用方法"""
    
    print("=== 足端轨迹CPG使用示例 ===")
    print()
    
    print("1. 创建足端轨迹CPG实例:")
    print("```python")
    print("foot_cpg = FootTrajectoryCPG(")
    print("    before_ftype=1,        # 步态类型 (1=walk, 2=trot, 3=pace, 4=bound, 5=pronk)")
    print("    after_ftype=1,         # 目标步态类型")
    print("    total_time=3.0,        # 总仿真时间")
    print("    toc=1.5,               # 步态切换时间")
    print("    step_height=0.08,      # 抬腿高度 (米)")
    print("    step_length=0.15,      # 步长 (米)")
    print("    body_height=0.25,      # 机身高度 (米)")
    print("    foot_spacing=0.2       # 足端间距 (米)")
    print(")")
    print("```")
    print()
    
    print("2. 生成实时足端位置:")
    print("```python")
    print("# 获取某一时刻所有足端的3D位置")
    print("t = 1.0  # 时间 (秒)")
    print("for foot_name in ['LF', 'RF', 'LH', 'RH']:")
    print("    pos = foot_cpg.generate_foot_position(foot_name, t)")
    print("    print(f'{foot_name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}')")
    print("```")
    print()
    
    print("3. 导出完整轨迹数据:")
    print("```python")
    print("# 生成并保存完整的足端轨迹数据")
    print("foot_cpg.export_csv()  # 保存为CSV文件")
    print("foot_cpg.plot_foot_trajectories()  # 生成可视化图")
    print("```")
    print()

def show_data_format():
    """展示生成的数据格式"""
    
    print("=== 生成的数据格式 ===")
    
    # 读取示例数据
    data_file = Path('data/cpg_data/gait_data/foot_traj_walk_3.0s.csv')
    if data_file.exists():
        df = pd.read_csv(data_file)
        
        print("数据结构:")
        print(f"- 数据形状: {df.shape}")
        print(f"- 采样频率: 500Hz")
        print(f"- 列名: {list(df.columns)}")
        print()
        
        print("数据含义:")
        print("前12列 - 足端位置坐标:")
        print("  LF_x, LF_y, LF_z: 左前足的3D位置")
        print("  RF_x, RF_y, RF_z: 右前足的3D位置")
        print("  LH_x, LH_y, LH_z: 左后足的3D位置") 
        print("  RH_x, RH_y, RH_z: 右后足的3D位置")
        print()
        print("后12列 - 足端速度:")
        print("  LF_dx, LF_dy, LF_dz: 左前足的3D速度")
        print("  RF_dx, RF_dy, RF_dz: 右前足的3D速度")
        print("  LH_dx, LH_dy, LH_dz: 左后足的3D速度")
        print("  RH_dx, RH_dy, RH_dz: 右后足的3D速度")
        print()
        
        print("示例数据 (前5行):")
        print(df.head())
        print()

def show_applications():
    """展示应用场景"""
    
    print("=== 应用场景 ===")
    print()
    
    print("1. 机器人足端位置控制:")
    print("   - 直接将生成的足端位置发送给逆运动学求解器")
    print("   - 计算关节角度并控制机器人关节")
    print("   - 实现基于足端轨迹的运动控制")
    print()
    
    print("2. 步态规划和分析:")
    print("   - 分析不同步态的运动特征")
    print("   - 优化步态参数 (步长、抬腿高度、步频)")
    print("   - 研究步态转换策略")
    print()
    
    print("3. 仿真和验证:")
    print("   - 在物理仿真环境中验证步态")
    print("   - 分析足端与地面的接触模式")
    print("   - 评估步态的稳定性和效率")
    print()
    
    print("4. 机器学习训练:")
    print("   - 作为监督学习的目标数据")
    print("   - 训练神经网络进行步态预测")
    print("   - 强化学习中的奖励函数设计")
    print()

def show_comparison():
    """展示与关节角度CPG的对比"""
    
    print("=== 与传统关节角度CPG的对比 ===")
    print()
    
    print("传统关节角度CPG:")
    print("  ✓ 直接控制关节")
    print("  ✓ 实现简单")
    print("  ✗ 足端轨迹不直观")
    print("  ✗ 难以指定期望的足端位置")
    print("  ✗ 步态设计复杂")
    print()
    
    print("足端轨迹CPG:")
    print("  ✓ 直观的足端运动控制")
    print("  ✓ 易于调整步态参数")
    print("  ✓ 符合生物学运动模式")
    print("  ✓ 便于步态分析和优化")
    print("  ✗ 需要逆运动学求解")
    print("  ✗ 计算复杂度略高")
    print()
    
    print("适用场景:")
    print("- 足端轨迹CPG: 适合需要精确控制足端位置的应用")
    print("- 关节角度CPG: 适合直接关节控制的简单应用")
    print()

def show_parameters():
    """展示重要参数的含义"""
    
    print("=== 重要参数说明 ===")
    print()
    
    print("几何参数:")
    print("  - step_height: 抬腿高度，影响足端离地的最大高度")
    print("  - step_length: 步长，影响每步的前进距离")
    print("  - body_height: 机身高度，影响足端的基准位置")
    print("  - foot_spacing: 足端间距，影响机器人的稳定性")
    print()
    
    print("动态参数:")
    print("  - frequency: 步频，控制步态的快慢")
    print("  - duty_factor: 支撑相比例，影响足端接地时间")
    print("  - amplitude: 轨迹幅度系数，影响运动的强度")
    print()
    
    print("步态参数:")
    print("  - gait_phases: 各足端的相位差，决定步态模式")
    print("    * Walk: 四拍步态，稳定但较慢")
    print("    * Trot: 对角步态，速度与稳定性平衡")
    print("    * Pace: 同侧步态，适合高速运动")
    print("    * Bound: 跳跃步态，适合越障")
    print("    * Pronk: 同步跳跃，适合垂直跳跃")
    print()

def main():
    """主函数"""
    
    print("🐕 足端轨迹CPG系统完整指南 🐕")
    print("="*50)
    
    # 展示使用方法
    demonstrate_foot_cpg_usage()
    
    # 展示数据格式
    show_data_format()
    
    # 展示应用场景
    show_applications()
    
    # 展示对比
    show_comparison()
    
    # 展示参数说明
    show_parameters()
    
    print("=== 系统优势总结 ===")
    print()
    print("✅ 生物启发: 基于动物中央模式发生器的原理")
    print("✅ 直观控制: 直接生成足端3D轨迹")
    print("✅ 参数化设计: 可调整步长、步高、步频等参数")
    print("✅ 多步态支持: 支持walk、trot、pace、bound、pronk等步态")
    print("✅ 实时生成: 500Hz高频率实时轨迹生成")
    print("✅ 数据完整: 同时提供位置和速度信息")
    print("✅ 易于集成: 可直接用于机器人控制系统")
    print()
    
    print("🚀 下一步建议:")
    print("1. 集成逆运动学求解器，转换足端位置为关节角度")
    print("2. 在物理仿真环境中测试步态效果")
    print("3. 添加地形适应和障碍物避免功能")
    print("4. 优化参数以提高步态效率和稳定性")
    print("5. 开发足端接触检测和反馈控制")
    
    print("\n" + "="*50)
    print("足端轨迹CPG系统已准备就绪! 🎉")

if __name__ == "__main__":
    main()