"""
Foot Trajectory CPG Implementation
足端轨迹中央模式发生器实现

这个模块实现了直接生成足端轨迹的CPG神经网络，
输出信号对应四个足端的3D空间坐标(x,y,z)而不是关节角度。
"""

import math
from typing import Optional, Tuple, List
import numpy as np
from pathlib import Path

from cpg_go1_simulation.config import (
    GAIT_DATA_DIR,
    TRANSITION_DATA_DIR, 
    BACKWARD_DATA_DIR
)
from cpg_go1_simulation.stein.base import CPGBase


class FootTrajectoryCPG(CPGBase):
    """
    足端轨迹CPG网络实现
    
    网络结构:
    - 12个神经元: 4个足端 × 3个坐标(x,y,z)
    - 每个足端有独立的3D轨迹生成器
    - 足端间通过相位耦合实现协调运动
    """

    def __init__(
        self,
        before_ftype: int,
        after_ftype: int, 
        total_time: float,
        toc: float,
        _if_backward: bool = False,
        step_height: float = 0.08,  # 抬腿高度 (m)
        step_length: float = 0.15,  # 步长 (m)
        body_height: float = 0.25,  # 机身高度 (m)
        foot_spacing: float = 0.2,  # 足端间距 (m)
    ):
        super().__init__(before_ftype, after_ftype, total_time, toc)
        
        self._if_backward = _if_backward
        
        # 足端轨迹参数
        self.step_height = step_height
        self.step_length = step_length  
        self.body_height = body_height
        self.foot_spacing = foot_spacing
        
        # 12个神经元: 4个足端 × 3个坐标  
        # 实际状态变量数量: 36 (12个x + 12个y + 12个z)
        self.neuron_nums = 12
        self.state_nums = 36
        
        # 足端名称映射
        self.foot_names = ["LF", "RF", "LH", "RH"]  # 左前、右前、左后、右后
        
        # 初始化网络参数
        self._init_foot_trajectory_params()
        self.set_neuron_params(ftype=self.before_ftype)
        self.set_link_params(ftype=self.before_ftype)
        self.set_pos0()
        self.set_driving_signal()
        
        # 设置步态策略
        self.set_strategy()
        
        # 足端轨迹记录
        self.foot_trajectories = {foot: [] for foot in self.foot_names}
        
    def _init_foot_trajectory_params(self):
        """初始化足端轨迹生成参数"""
        
        # 足端在机体坐标系中的初始位置 (相对于机体中心)
        # 修正: 使用更实际的机器人尺寸和位置
        leg_length = 0.35  # 机器人腿长
        body_width = self.foot_spacing * 2
        body_length = self.foot_spacing * 2.5
        
        self.foot_base_positions = {
            "LF": np.array([body_length/2, body_width/2, -leg_length]),   # 左前
            "RF": np.array([body_length/2, -body_width/2, -leg_length]),  # 右前  
            "LH": np.array([-body_length/2, body_width/2, -leg_length]),  # 左后
            "RH": np.array([-body_length/2, -body_width/2, -leg_length]) # 右后
        }
        
        # 不同步态的相位差 (弧度) - 修正版本
        self.gait_phases = {
            1: {"LF": 0, "RF": np.pi, "LH": np.pi/2, "RH": 3*np.pi/2},     # walk: 四拍步态
            2: {"LF": 0, "RF": np.pi, "LH": np.pi, "RH": 0},               # trot: 对角步态
            3: {"LF": 0, "RF": 0, "LH": np.pi, "RH": np.pi},               # pace: 同侧步态
            4: {"LF": 0, "RF": 0, "LH": np.pi, "RH": np.pi},               # bound: 前后对称跳
            5: {"LF": 0, "RF": 0, "LH": 0, "RH": 0}                        # pronk: 四足同步跳
        }

    def get_foot_trajectory_params(self, ftype: int) -> Tuple[float, float, float]:
        """获取足端轨迹生成参数"""
        
        if ftype == 1:  # walk
            frequency = 1.0    # 步频 (Hz)
            duty_factor = 0.6  # 支撑相比例
            amplitude = 1.0    # 轨迹幅度系数
        elif ftype == 2:  # trot  
            frequency = 2.0
            duty_factor = 0.5
            amplitude = 1.2
        elif ftype == 3:  # pace
            frequency = 1.8
            duty_factor = 0.5
            amplitude = 1.1
        elif ftype == 4:  # bound
            frequency = 2.5
            duty_factor = 0.4
            amplitude = 1.5
        elif ftype == 5:  # pronk
            frequency = 3.0
            duty_factor = 0.3
            amplitude = 2.0
        else:
            frequency = 1.0
            duty_factor = 0.6
            amplitude = 1.0
            
        return frequency, duty_factor, amplitude

    def set_neuron_params(self, ftype: int) -> None:
        """设置神经元参数"""
        super().set_neuron_params(ftype)
        
        # 获取足端轨迹参数
        self.frequency, self.duty_factor, self.amplitude = self.get_foot_trajectory_params(ftype)
        
        # 轨迹生成参数
        self.omega = 2 * np.pi * self.frequency  # 角频率
        
    def set_link_params(self, ftype: int) -> None:
        """设置足端间的耦合参数"""
        
        # 足端间耦合强度矩阵 (36x36)
        self.coupling_matrix = np.zeros((self.state_nums, self.state_nums))
        
        # 设置足端间的相位耦合
        coupling_strength = 0.1
        
        for i in range(4):  # 4个足端
            for j in range(4):
                if i != j:
                    # 足端i和足端j的x,y,z坐标之间的耦合
                    for coord in range(3):
                        self.coupling_matrix[i*3 + coord, j*3 + coord] = coupling_strength
                        
    def set_pos0(self) -> None:
        """设置初始位置"""
        # 12个神经元的初始状态: [x, y, z] × 4个足端
        pos0 = []
        
        for foot_name in self.foot_names:
            base_pos = self.foot_base_positions[foot_name]
            # 添加每个足端的x, y, z坐标
            pos0.extend(base_pos)
                
        # 总共12个状态 (4个足端 × 3个坐标)
        self.pos0 = np.array(pos0)
        
        # 添加y和z状态变量 (总共36个状态: 12个x + 12个y + 12个z)
        self.pos0 = np.hstack([self.pos0, self.pos0 + 0.01, self.pos0 + 0.005])
        
    def set_driving_signal(self) -> None:
        """设置驱动信号"""
        
        # 为每个足端的x,y,z坐标设置驱动参数
        self.drive_amplitudes = np.ones(self.state_nums) * self.amplitude
        self.drive_frequencies = np.ones(self.state_nums) * self.omega
        
        # 不同坐标使用不同的驱动强度
        for i in range(4):
            self.drive_amplitudes[i*3] *= 0.8      # x坐标驱动
            self.drive_amplitudes[i*3 + 1] *= 0.6  # y坐标驱动  
            self.drive_amplitudes[i*3 + 2] *= 1.2  # z坐标驱动(抬腿)
            
    def set_strategy(self) -> None:
        """设置控制策略"""
        if self.before_ftype == self.after_ftype:
            self.strategy = 1  # 单一步态
        else:
            self.strategy = 1  # 简单切换策略 (可以后续扩展)
            
    def generate_foot_position(self, foot_name: str, t: float, phase_offset: float = 0) -> np.ndarray:
        """
        生成单个足端的期望轨迹 - 修正版本
        
        Args:
            foot_name: 足端名称 ("LF", "RF", "LH", "RH")
            t: 当前时间
            phase_offset: 相位偏移
            
        Returns:
            足端3D位置 [x, y, z]
        """
        
        # 获取基础位置 (足端的名义接触位置)
        base_pos = self.foot_base_positions[foot_name].copy()
        
        # 计算步态相位
        gait_phase = self.gait_phases[self.before_ftype][foot_name]
        total_phase = self.omega * t + gait_phase + phase_offset
        
        # 归一化相位到[0, 2π]
        normalized_phase = total_phase % (2 * np.pi)
        phase_ratio = normalized_phase / (2 * np.pi)
        
        # 判断是否在支撑相或摆动相
        is_stance = phase_ratio < self.duty_factor
        
        # 步态方向
        if self._if_backward:
            step_direction = -1.0
        else:
            step_direction = 1.0
            
        if is_stance:
            # 支撑相: 足端在地面滑动，从前向后
            stance_progress = phase_ratio / self.duty_factor  # 0 to 1
            # 从步长的一半位置向后滑动到负一半位置
            x_offset = self.step_length * step_direction * (0.5 - stance_progress)
            y_offset = 0.0
            z_offset = 0.0  # 保持在地面
        else:
            # 摆动相: 足端抬起并向前摆动
            swing_progress = (phase_ratio - self.duty_factor) / (1 - self.duty_factor)  # 0 to 1
            
            # X方向: 椭圆轨迹的水平分量，从后向前
            x_offset = self.step_length * step_direction * (swing_progress - 0.5)
            
            # Y方向: 保持不变
            y_offset = 0.0
            
            # Z方向: 椭圆轨迹的垂直分量，形成抛物线抬腿
            # 使用正弦函数创建平滑的抬腿轨迹
            z_offset = self.step_height * np.sin(np.pi * swing_progress)
            
        # 计算最终足端位置
        foot_position = base_pos + np.array([x_offset, y_offset, z_offset])
        
        return foot_position

    def stein(self, pos: np.ndarray, t: float) -> np.ndarray:
        """直接轨迹跟踪方法 - 简化但更准确的实现"""
        
        # 将36维状态分解: x(12), y(12), z(12)  
        x = pos[:self.neuron_nums]    # 当前足端位置
        y = pos[self.neuron_nums:2*self.neuron_nums]    # 辅助状态1
        z = pos[2*self.neuron_nums:]  # 辅助状态2
        
        # 直接计算每个足端的目标位置
        target_positions = []
        for foot_name in self.foot_names:
            target_pos = self.generate_foot_position(foot_name, t)
            target_positions.extend(target_pos)
        
        target_positions = np.array(target_positions)
        
        # 使用改进的动力学方程，直接跟踪目标轨迹
        # 这种方法更适合足端轨迹生成
        
        # 位置跟踪增益
        kp = 10.0  # 位置增益
        kd = 2.0   # 速度增益 (阻尼)
        
        # 计算位置误差和期望速度
        position_error = target_positions - x
        
        # 计算目标位置的时间导数（期望速度）
        dt = 0.001
        target_positions_next = []
        for foot_name in self.foot_names:
            target_pos_next = self.generate_foot_position(foot_name, t + dt)
            target_positions_next.extend(target_pos_next)
        target_positions_next = np.array(target_positions_next)
        
        desired_velocity = (target_positions_next - target_positions) / dt
        current_velocity = (x - y) * 50.0  # 估算当前速度
        
        velocity_error = desired_velocity - current_velocity
        
        # PD控制器
        dxdt = kp * position_error + kd * velocity_error
        
        # 辅助状态更新（用于记录历史位置）
        dydt = (x - y) * 10.0  # y跟踪x的历史
        dzdt = (y - z) * 5.0   # z跟踪更早的历史
        
        return np.hstack([dxdt, dydt, dzdt])

    def get_output_path(self) -> Path:
        """获取输出文件路径"""
        
        if not self._if_backward:
            if self.before_ftype == self.after_ftype:
                return (
                    GAIT_DATA_DIR / 
                    f"foot_traj_{self.gaitnames[self.before_ftype]}_{self.tspan[1]}s.csv"
                )
            else:
                return (
                    TRANSITION_DATA_DIR /
                    f"foot_traj_{self.gaitnames[self.before_ftype]}_to_{self.gaitnames[self.after_ftype]}_{self.toc:.3f}.csv"
                )
        else:
            return (
                BACKWARD_DATA_DIR /
                f"foot_traj_{self.gaitnames[self.before_ftype]}_backward_{self.tspan[1]}s.csv"
            )
            
    def export_data(self) -> Tuple[np.ndarray, float]:
        """导出足端轨迹数据"""
        
        t = np.arange(self.tspan[0], self.tspan[1] + self.time_step, self.time_step)
        data, _ = self.runge_kutta_x_and_xdot(self.pos0, t)
        
        # 移除最后一行数据
        data = data[:-1, :]
        
        return data, self.toc
        
    def runge_kutta_x_and_xdot(self, pos: np.ndarray, t: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        四阶龙格库塔方法求解足端轨迹
        """
        
        data = []
        t0 = t[0]
        h = self.time_step
        h_half = self.time_step / 2
        x0 = pos
        
        for j in range(len(t)):
            k1 = self.stein(x0, t0)
            k2 = self.stein(x0 + h * k1 / 2, t0 + h_half)
            k3 = self.stein(x0 + h * k2 / 2, t0 + h_half)
            k4 = self.stein(x0 + h * k3, t0 + h)
            x1 = x0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            
            x_dot = (x1 - x0) / h
            
            # 保存足端位置和速度 (只保存前12维，即足端位置)
            foot_positions = x1[:self.neuron_nums]
            foot_velocities = x_dot[:self.neuron_nums] 
            
            data.append(list(foot_positions) + list(foot_velocities))
            current_state = x1
            
            # 记录足端轨迹 (每个足端3个坐标)
            foot_pos_3d = foot_positions.reshape(4, 3)
            for i, foot_name in enumerate(self.foot_names):
                self.foot_trajectories[foot_name].append(foot_pos_3d[i].copy())
            
            # 更新状态
            x0 = x1
            t0 = self.decimal(t0 + h, 5)
            
        return np.array(data), current_state

    def export_csv(self) -> None:
        """导出CSV数据"""
        
        data, _ = self.export_data()
        save_path = self.get_output_path()
        
        # 创建列标题
        headers = []
        for foot in self.foot_names:
            for coord in ['x', 'y', 'z']:
                headers.append(f"{foot}_{coord}")
        for foot in self.foot_names:
            for coord in ['x', 'y', 'z']:
                headers.append(f"{foot}_d{coord}")
                
        self.save_data(file_path=save_path, data=data, headers=headers)
        
        print(f"足端轨迹数据已导出到: {save_path}")
        
    def plot_foot_trajectories(self, save_path: Optional[str] = None) -> None:
        """可视化足端轨迹"""
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("需要安装matplotlib来进行可视化")
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(221, projection='3d')
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, foot_name in enumerate(self.foot_names):
            if foot_name in self.foot_trajectories and self.foot_trajectories[foot_name]:
                traj = np.array(self.foot_trajectories[foot_name])
                ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                        color=colors[i], label=foot_name, linewidth=2)
                
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)') 
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Foot Trajectories')
        ax1.legend()
        
        # X-Z平面投影 (侧视图)
        ax2 = fig.add_subplot(222)
        for i, foot_name in enumerate(self.foot_names):
            if foot_name in self.foot_trajectories and self.foot_trajectories[foot_name]:
                traj = np.array(self.foot_trajectories[foot_name])
                ax2.plot(traj[:, 0], traj[:, 2], color=colors[i], label=foot_name)
                
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title('Side View (X-Z)')
        ax2.legend()
        ax2.grid(True)
        
        # X-Y平面投影 (俯视图)
        ax3 = fig.add_subplot(223)
        for i, foot_name in enumerate(self.foot_names):
            if foot_name in self.foot_trajectories and self.foot_trajectories[foot_name]:
                traj = np.array(self.foot_trajectories[foot_name])
                ax3.plot(traj[:, 0], traj[:, 1], color=colors[i], label=foot_name)
                
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('Top View (X-Y)')
        ax3.legend()
        ax3.grid(True)
        
        # Z坐标时间序列
        ax4 = fig.add_subplot(224)
        time = np.arange(len(self.foot_trajectories[self.foot_names[0]])) * self.time_step
        
        for i, foot_name in enumerate(self.foot_names):
            if foot_name in self.foot_trajectories and self.foot_trajectories[foot_name]:
                traj = np.array(self.foot_trajectories[foot_name])
                ax4.plot(time, traj[:, 2], color=colors[i], label=foot_name)
                
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Z (m)')
        ax4.set_title('Foot Height vs Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图已保存到: {save_path}")
        else:
            plt.savefig('foot_trajectories.png', dpi=300, bbox_inches='tight')
            print("轨迹图已保存到: foot_trajectories.png")
            
        plt.close()


if __name__ == "__main__":
    # 测试足端轨迹CPG
    print("创建足端轨迹CPG...")
    
    # 创建walk步态的足端轨迹CPG
    foot_cpg = FootTrajectoryCPG(
        before_ftype=1,  # walk
        after_ftype=1,
        total_time=3.0,
        toc=1.5,
        step_height=0.08,
        step_length=0.15,
        body_height=0.25
    )
    
    # 导出数据
    print("生成足端轨迹数据...")
    foot_cpg.export_csv()
    
    # 可视化轨迹
    print("生成轨迹可视化...")
    foot_cpg.plot_foot_trajectories("foot_trajectories_walk.png")
    
    print("足端轨迹CPG测试完成!")