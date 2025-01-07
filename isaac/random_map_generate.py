import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import math


class RandomSceneWithPath:
    def __init__(self, matrix_size=10, obstacle_density=0.3):
        self.matrix_size = matrix_size  # 矩阵大小
        self.obstacle_density = obstacle_density  # 障碍物密度

    def dfs(self, matrix, x, y, n, visited):
        """深度优先搜索，检查是否从 (x, y) 到达 (n-1, n-1)"""
        if x < 0 or x >= n or y < 0 or y >= n or matrix[x][y] == 1 or visited[x][y]:
            return False
        if (x, y) == (n - 1, n - 1):  # 到达终点
            return True

        visited[x][y] = True

        # 上下左右四个方向
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            if self.dfs(matrix, x + dx, y + dy, n, visited):
                return True

        return False

    def bfs(self, matrix, start, end):
        """广度优先搜索，检查从 start 到 end 是否有通路"""
        n = len(matrix)
        visited = [[False] * n for _ in range(n)]
        queue = [start]
        visited[start[0]][start[1]] = True

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.pop(0)
            if (x, y) == end:
                return True

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and not visited[nx][ny] and matrix[nx][ny] == 0:
                    visited[nx][ny] = True
                    queue.append((nx, ny))

        return False

    def generate_random_matrix(self, n, obstacle_density):
        """生成一个随机矩阵，并确保有通路"""
        while True:
            matrix = [[0] * n for _ in range(n)]  # 初始化矩阵，全部为0

            # 保证起点和终点没有障碍物
            matrix[0][0] = 0
            matrix[n - 1][n - 1] = 0

            # 随机生成障碍物，比例控制
            for i in range(n):
                for j in range(n):
                    if (i, j) not in [(0, 0), (n - 1, n - 1)] and random.random() < obstacle_density:
                        matrix[i][j] = 1  # 障碍物为1

            # 确保有路径
            if self.bfs(matrix, (0, 0), (n - 1, n - 1)):  # 如果从起点到终点有路径
                return matrix  # 返回矩阵

    def generate_scene(self):
        """生成场景并返回起点和终点位置"""
        # 生成一个随机矩阵并确保有通路
        matrix = self.generate_random_matrix(self.matrix_size, self.obstacle_density)

        # 找到起点和终点位置
        start_pos = (0, 0)
        end_pos = (self.matrix_size - 1, self.matrix_size - 1)

        return start_pos, end_pos, matrix

    def print_map(self, matrix):
        """输出矩阵的图示，0代表空白，1代表障碍物"""
        for row in matrix:
            print(" ".join(str(cell) for cell in row))
        print()

    def plot_matrix(self, matrix, scale=10, start=None, end=None):
        """根据矩阵生成图像，0为白色，1为黑色，且可以调整图像的尺寸"""
        matrix = np.array(matrix)

        # 图像的大小根据矩阵大小和缩放比例来调整
        plt.figure(figsize=(matrix.shape[1] * scale / 100, matrix.shape[0] * scale / 100))

        # 使用imshow显示矩阵，cmap='binary' 使得 0 为白色，1 为黑色
        plt.imshow(matrix, cmap='binary', interpolation='nearest')

        # 绘制起点（绿色）和终点（红色）
        if start:
            plt.scatter(start[1], start[0], color='green', s=100, label="Start", marker='o')  # 使用o标记start
        if end:
            plt.scatter(end[1], end[0], color='red', s=100, label="End", marker='o')  # 使用o标记end

        # 显示图例
        plt.legend()

        plt.axis('off')  # 不显示坐标轴
        plt.show()

    def upscale_matrix(self, matrix, new_size):
        """通过插值将矩阵放大"""
        matrix = np.array(matrix)
        scale_factor = new_size / matrix.shape[0]  # 计算放大倍数
        upscaled_matrix = zoom(matrix, scale_factor, order=0)  # 使用零阶插值进行上采样
        return upscaled_matrix

    def set_matrix_size(self, new_size):
        """设置新的矩阵大小，并生成新的地图"""
        self.matrix_size = new_size
        return self.generate_random_matrix(self.matrix_size, self.obstacle_density)

    def check_and_generate_matrix(self, original_matrix, new_size):
        """放大矩阵后检查是否有通路，若没有则重新生成"""
        # 放大矩阵
        upscaled_matrix = self.upscale_matrix(original_matrix, new_size)

        # 在放大后的矩阵上进行DFS检查
        if self.bfs(upscaled_matrix, (0, 0), (new_size - 1, new_size - 1)):
            return upscaled_matrix
        else:
            return self.generate_scene()[2]  # 重新生成新的矩阵

    def select_end_position(self, matrix, start, radius):
        """选择一个距离起点 start 半径为 radius 的圆圈外的 0 位置作为 end"""
        x_start, y_start = start
        candidates = []

        # 计算每个点与起点的距离，并筛选出距离大于 radius 的位置
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] == 0:  # 只考虑空白的位置
                    distance = math.sqrt((i - x_start) ** 2 + (j - y_start) ** 2)
                    if distance > radius:
                        candidates.append((i, j))

        if candidates:
            # 随机选择一个候选点作为 end
            return random.choice(candidates)
        else:
            return None  # 如果没有符合条件的点，则返回 None

    def find_valid_end(self, matrix, start, radius):
        """找到一个距离start半径radius外且从start到end有通路的end"""
        while True:
            new_end = self.select_end_position(matrix, start, radius)
            if new_end and self.bfs(matrix, start, new_end):
                return new_end  # 找到一个有效的end
            print("没有有效的end，重新选择")


# 测试
def generate_matrix(matrix_size):
    random_scene = RandomSceneWithPath(matrix_size, obstacle_density=0.5)

    # 生成场景并返回起点、终点和矩阵
    start, end, matrix = random_scene.generate_scene()

    # 放大矩阵并检查是否有通路
    upscaled_matrix = random_scene.check_and_generate_matrix(matrix, 16)

    # 找到一个有效的 end
    new_end = random_scene.find_valid_end(upscaled_matrix, start, 10)

    print(f"新选定的终点位置: {new_end}")

    # 绘制图像，标出起点和终点
    random_scene.plot_matrix(upscaled_matrix, scale=20, start=start, end=new_end)  # 缩放比例设置为20

    return upscaled_matrix, new_end


if __name__ == "__main__":
    print(generate_matrix(12))
