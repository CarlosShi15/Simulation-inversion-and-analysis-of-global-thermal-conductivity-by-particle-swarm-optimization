import numpy as np
import matplotlib.pyplot as plt
import os


# 一维热传导方程的计算函数
def evolve(u, u_previous, alpha, dt, dx2):
    u[1:-1] = u_previous[1:-1] + alpha * dt * (u_previous[2:] - 2 * u_previous[1:-1] + u_previous[:-2]) / dx2
    return u


# 初始化numpy矩阵字段
def init_fields(lenX, Tinitial, Tboundary):
    field = np.full(lenX, Tinitial, dtype=np.float64)
    field[0] = Tboundary
    field[-1] = Tboundary
    field0 = field.copy()  # 上一步的温度场数组
    return field, field0


# 绘制二维温度分布图像
def plot_2d_temperature(time_steps, X, temperature_data, save_path):
    fig, ax = plt.subplots()
    c = ax.imshow(temperature_data, aspect='auto', cmap='inferno', extent=[0, max(time_steps), 0, 1], vmin=20,
                  vmax=1000)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('Temperature Distribution Over Time and Position')
    fig.colorbar(c, ax=ax, label='Temperature (C)')

    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()


# 主函数
def main(lenX, timesteps, dt, dx, alpha, save_path):
    # 环境温度
    Tboundary = 20.0
    # 材料初始温度
    Tinitial = 300.0

    # 空间网格
    X = np.linspace(0, 1, lenX)
    dx2 = dx ** 2

    # 初始化温度场
    field, field0 = init_fields(lenX, Tinitial, Tboundary)

    # 记录温度数据
    temperature_data = np.zeros((lenX, timesteps + 1))
    temperature_data[:, 0] = field

    # 迭代
    for step in range(1, timesteps + 1):
        field = evolve(field, field0, alpha, dt, dx2)  # 更新温度场
        field0 = field.copy()  # 将field0设置为field的副本
        temperature_data[:, step] = field

    # 绘制/保存二维温度场图像
    time_steps = np.arange(0, timesteps + 1) * dt
    plot_2d_temperature(time_steps, X, temperature_data, save_path)


if __name__ == '__main__':
    lenX = 100  # 空间网格数
    total_time = 150  # 总时间 (s)
    timesteps = total_time  # 时间步数，假设每步为1秒
    dx = 1 / (lenX - 1)  # 空间步长
    k = 180 # 热导率
    c = 900 # 比热容
    rho = 2700  # 密度
    alpha = k / (c * rho)  # 扩散系数
    dt = 1.0  # 时间步长 (s)

    save_path = r'C:\Users\SYH\Desktop\pic'  # 自定义保存图片路径
    main(lenX, timesteps, dt, dx, alpha, save_path)
