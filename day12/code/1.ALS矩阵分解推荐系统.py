# -*- coding: utf-8 -*-
"""
===============================================================================
ALS矩阵分解推荐系统

一个使用纯 Python 手写矩阵运算库来实现 ALS 矩阵分解算法，用于根据历史评分数据为用户推荐电影

功能描述: 
    实现了一个基于 ALS (Alternating Least Squares, 交替最小二乘法) 
    的协同过滤推荐系统。通过矩阵分解技术，将用户 - 物品评分矩阵分解为两个低秩矩阵
    (用户隐因子矩阵和物品隐因子矩阵)，从而预测用户对未评分物品的喜好程度。

核心算法原理:
    1. 矩阵分解: R ≈ U * V^T
       - R: 用户 - 物品评分矩阵 (稀疏)
       - U: 用户隐因子矩阵 (m x k)
       - V: 物品隐因子矩阵 (n x k)
       - k: 隐因子维度 (潜在特征数)
    2. 交替最小二乘 (ALS):
       - 固定 V，求解 U 的最小二乘解: U = (V^T * V)^-1 * V^T * R
       - 固定 U，求解 V 的最小二乘解: V = (U^T * U)^-1 * U^T * R
       - 交替迭代上述步骤直到收敛或达到最大迭代次数。

执行流程:
    1. [数据加载] 读取 CSV 格式的评分数据 (user_id, item_id, rating)。
    2. [预处理] 构建稀疏字典映射，建立 ID 到矩阵索引的映射关系。
    3. [初始化] 随机生成用户矩阵 U 和物品矩阵 V。
    4. [模型训练] 
       - 循环迭代:
         a. 固定物品矩阵，更新用户矩阵。
         b. 固定用户矩阵，更新物品矩阵。
         c. 计算 RMSE (均方根误差) 监控模型效果。
    5. [预测推荐] 
       - 计算用户向量与所有物品向量的点积得到预测评分。
       - 过滤已评分物品，按分数排序输出 Top-N 推荐列表。

依赖说明:
    - 标准库: itertools, copy, random, collections, math
    - 第三方库: pandas (仅用于类型提示或未实际重度使用), numpy (仅用于随机数生成)
    
注意:
    本代码包含一个手写的 Matrix 类以实现底层线性代数运算，主要用于基础演示算法原理。
    在生产环境中，建议使用 numpy, scipy 或 spark.mllib 以获得更高的性能和数值稳定性。
===============================================================================
"""

from itertools import product, chain
from copy import deepcopy
import random
import numpy as np
from collections import defaultdict

# ==============================================================================
# 第一部分：基础线性代数库 (手写实现，用于演示原理)
# ==============================================================================

class Matrix(object):
    """
    一个简单的矩阵类，封装了基础的矩阵运算操作。
    包括：转置、求逆、乘法、标量乘法等。
    """
    def __init__(self, data):
        self.data = data
        # 确定矩阵形状 (行数, 列数)
        self.shape = (len(data), len(data[0]) if data else 0)

    def row(self, row_no):
        """获取指定行的子矩阵"""
        return Matrix([self.data[row_no]])

    def col(self, col_no):
        """获取指定列的子矩阵"""
        m = self.shape[0]
        return Matrix([[self.data[i][col_no]] for i in range(m)])

    @property
    def is_square(self):
        """判断是否为方阵"""
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        """矩阵转置"""
        data = list(map(list, zip(*self.data)))
        return Matrix(data)

    def _eye(self, n):
        """生成 n x n 的单位矩阵"""
        return [[0 if i != j else 1 for j in range(n)] for i in range(n)]

    @property
    def eye(self):
        """生成与当前矩阵同维度的单位矩阵 (仅限方阵)"""
        assert self.is_square, "只有方阵才能生成同维度单位矩阵!"
        data = self._eye(self.shape[0])
        return Matrix(data)

    def _gaussian_elimination(self, aug_matrix):
        """
        高斯 - 约旦消元法
        将增广矩阵左侧化简为单位对角矩阵，从而求解线性方程组或逆矩阵。
        """
        n = len(aug_matrix)
        m = len(aug_matrix[0])

        # 1. 前向消元 (从上到下)
        for col_idx in range(n):
            # 如果主元为0，寻找下方非零行进行交换/累加
            if aug_matrix[col_idx][col_idx] == 0:
                row_idx = col_idx
                while row_idx < n and aug_matrix[row_idx][col_idx] == 0:
                    row_idx += 1
                # 将找到的非零行加到当前行 (简单处理，非标准行交换)
                if row_idx < n:
                    for i in range(col_idx, m):
                        aug_matrix[col_idx][i] += aug_matrix[row_idx][i]

            # 消去当前列下方的非零元素
            for i in range(col_idx + 1, n):
                if aug_matrix[i][col_idx] == 0:
                    continue
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                for j in range(col_idx, m):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # 2. 后向消元 (从下到上)
        for col_idx in range(n - 1, -1, -1):
            for i in range(col_idx):
                if aug_matrix[i][col_idx] == 0:
                    continue
                k = aug_matrix[i][col_idx] / aug_matrix[col_idx][col_idx]
                # 优化范围：只需处理相关列
                for j in chain(range(i, col_idx + 1), range(n, m)):
                    aug_matrix[i][j] -= k * aug_matrix[col_idx][j]

        # 3. 归一化对角线元素为1
        for i in range(n):
            pivot = aug_matrix[i][i]
            if pivot == 0:
                continue # 避免除以零，虽然理论上不应发生
            k = 1 / pivot
            aug_matrix[i][i] *= k
            for j in range(n, m):
                aug_matrix[i][j] *= k

        return aug_matrix

    def _inverse(self, data):
        """
        求矩阵的逆
        方法：构造增广矩阵 [A | I]，通过高斯消元变为 [I | A^-1]
        """
        n = len(data)
        unit_matrix = self._eye(n)
        # 拼接增广矩阵
        aug_matrix = [a + b for a, b in zip(self.data, unit_matrix)]
        ret = self._gaussian_elimination(aug_matrix)
        # 提取右侧的逆矩阵部分
        return list(map(lambda x: x[n:], ret))

    @property
    def inverse(self):
        """获取逆矩阵对象"""
        assert self.is_square, "只有方阵才有逆矩阵!"
        data = self._inverse(self.data)
        return Matrix(data)

    def _row_mul(self, row_A, row_B):
        """计算两个向量的点积"""
        return sum(x[0] * x[1] for x in zip(row_A, row_B))

    def _mat_mul(self, row_A, B):
        """矩阵乘法辅助函数：计算一行 A 与矩阵 B 的乘积"""
        # 遍历 B 的每一列 (通过转置实现)
        row_pairs = product([row_A], B.transpose.data)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def mat_mul(self, B):
        """
        矩阵乘法: A * B
        检查维度匹配并返回结果矩阵
        """
        error_msg = "矩阵乘法错误：A 的列数必须等于 B 的行数!"
        assert self.shape[1] == B.shape[0], error_msg
        return Matrix([self._mat_mul(row_A, B) for row_A in self.data])

    def _mean(self, data):
        """计算每列的平均值"""
        m = len(data)
        if m == 0: return []
        n = len(data[0])
        ret = [0.0 for _ in range(n)]
        for row in data:
            for j in range(n):
                ret[j] += row[j] / m
        return ret

    def mean(self):
        """返回平均值矩阵"""
        return Matrix(self._mean(self.data))

    def scala_mul(self, scala):
        """标量乘法：矩阵每个元素乘以常数"""
        m, n = self.shape
        data = deepcopy(self.data)
        for i in range(m):
            for j in range(n):
                data[i][j] *= scala
        return Matrix(data)


# ==============================================================================
# 第二部分：ALS 推荐算法核心实现
# ==============================================================================

class ALS(object):
    """
    Alternating Least Squares (ALS) 矩阵分解推荐模型
    """
    def __init__(self):
        self.user_ids = None          # 原始用户ID列表
        self.item_ids = None          # 原始物品ID列表
        self.user_ids_dict = None     # 用户ID -> 索引映射
        self.item_ids_dict = None     # 物品ID -> 索引映射
        self.user_matrix = None       # 用户隐因子矩阵 U (k x m)
        self.item_matrix = None       # 物品隐因子矩阵 V (k x n)
        self.user_items = None        # 记录用户已评分的物品集合 {user_id: set(item_ids)}
        self.shape = None             # (用户数, 物品数)
        self.rmse = None              # 最终训练的 RMSE

    def _process_data(self, X):
        """
        数据预处理：将原始评分列表转换为稀疏字典结构，并建立索引映射
        输入 X: [[user_id, item_id, rating], ...]
        输出: (ratings_dict, ratings_transpose_dict)
        """        
        # 提取唯一用户和物品，并建立 ID 到 0..N 的映射
        self.user_ids = tuple(set(map(lambda x: x[0], X)))
        self.user_ids_dict = {uid: idx for idx, uid in enumerate(self.user_ids)}
     
        self.item_ids = tuple(set(map(lambda x: x[1], X)))
        self.item_ids_dict = {iid: idx for idx, iid in enumerate(self.item_ids)}
     
        self.shape = (len(self.user_ids), len(self.item_ids))
     
        # 构建稀疏邻接表: {user: {item: rating}} 和 {item: {user: rating}}
        ratings = defaultdict(lambda: defaultdict(float))
        ratings_T = defaultdict(lambda: defaultdict(float))
        
        for row in X:
            user_id, item_id, rating = row
            ratings[user_id][item_id] = rating
            ratings_T[item_id][user_id] = rating
     
        # 一致性检查
        assert len(self.user_ids) == len(ratings), "用户数量不匹配!"
        assert len(self.item_ids) == len(ratings_T), "物品数量不匹配!"
        
        return ratings, ratings_T

    def _users_mul_ratings(self, users, ratings_T):
        """
        计算用户矩阵与稀疏评分矩阵的乘积，用于更新物品矩阵
        逻辑：对于每个物品，只计算对其有评分的用户贡献
        返回：新的物品矩阵 (k x n)
        """
        def calc_item_vector(users_row, item_id):
            # 获取对该物品评分的所有用户及其分数
            rated_users = ratings_T[item_id]
            if not rated_users:
                return 0.0
            
            total = 0.0
            for u_id, score in rated_users.items():
                # 通过映射找到用户在矩阵中的列索引
                u_idx = self.user_ids_dict[u_id]
                # 累加：用户隐因子 * 评分
                total += users_row[u_idx] * score
            return total
     
        # 对每个隐因子行 (k) 和每个物品 (n) 进行计算
        ret = [
            [calc_item_vector(users_row, item_id) for item_id in self.item_ids] 
            for users_row in users.data
        ]
        return Matrix(ret)

    def _items_mul_ratings(self, items, ratings):
        """
        计算物品矩阵与稀疏评分矩阵的乘积，用于更新用户矩阵
        逻辑：对于每个用户，只计算其评分过的物品贡献
        返回：新的用户矩阵 (k x m)
        """
        def calc_user_vector(items_row, user_id):
            rated_items = ratings[user_id]
            if not rated_items:
                return 0.0
            
            total = 0.0
            for i_id, score in rated_items.items():
                i_idx = self.item_ids_dict[i_id]
                total += items_row[i_idx] * score
            return total
     
        ret = [
            [calc_user_vector(items_row, user_id) for user_id in self.user_ids] 
            for items_row in items.data
        ]
        return Matrix(ret)

    def _gen_random_matrix(self, n_rows, n_cols):
        """生成随机初始化的矩阵 (使用 numpy 提高效率)"""
        data = np.random.rand(n_rows, n_cols).tolist()
        return Matrix(data)

    def _get_rmse(self, ratings):
        """
        计算均方根误差 (RMSE)
        仅对有评分的数据点进行预测和误差计算
        """
        m, n = self.shape
        mse_sum = 0.0
        count = 0
        
        # 遍历所有已知评分
        for u_id, items in ratings.items():
            u_idx = self.user_ids_dict[u_id]
            # 获取该用户的隐因子向量 (转为行向量 1 x k)
            user_vec = self.user_matrix.col(u_idx).transpose
            
            for i_id, actual_rating in items.items():
                if actual_rating <= 0: continue
                
                i_idx = self.item_ids_dict[i_id]
                # 获取该物品的隐因子向量 (k x 1)
                item_vec = self.item_matrix.col(i_idx)
                
                # 预测评分 = 用户向量 * 物品向量
                predicted_rating = user_vec.mat_mul(item_vec).data[0][0]
                
                error = (actual_rating - predicted_rating) ** 2
                mse_sum += error
                count += 1
        
        if count == 0: return 0.0
        return (mse_sum / count) ** 0.5

    def fit(self, X, k, max_iter=10):
        """
        训练模型
        参数:
            X: 训练数据 [[user, item, rating], ...]
            k: 隐因子维度 (Latent Factors)
            max_iter: 最大迭代次数
        """
        ratings, ratings_T = self._process_data(X)
        # 记录用户已看过的物品，用于后续推荐过滤
        self.user_items = {uid: set(items.keys()) for uid, items in ratings.items()}
        m, n = self.shape
     
        assert k < min(m, n), f"参数 k ({k}) 必须小于矩阵的秩 (min({m}, {n}))"
     
        # 1. 初始化用户矩阵 U (k x m)
        self.user_matrix = self._gen_random_matrix(k, m)
        # 物品矩阵 V 将在第一次迭代中生成
     
        print(f"开始训练 ALS 模型 (k={k}, max_iter={max_iter})...")
        
        for i in range(max_iter):
            # ALS 核心：交替更新
            if i % 2 == 0:
                # 偶数步：固定用户矩阵 U，更新物品矩阵 V
                # 公式推导简化版: V = (U * U^T)^-1 * U * R
                # 代码实现逻辑：先计算 (U * U^T)^-1 * U，再乘以稀疏评分矩阵
                users = self.user_matrix
                # 计算变换矩阵: (U * U^T)^-1 * U
                transform_matrix = users.mat_mul(users.transpose).inverse.mat_mul(users)
                self.item_matrix = self._users_mul_ratings(transform_matrix, ratings_T)
            else:
                # 奇数步：固定物品矩阵 V，更新用户矩阵 U
                # 公式推导简化版: U = (V * V^T)^-1 * V * R^T
                items = self.item_matrix
                transform_matrix = items.mat_mul(items.transpose).inverse.mat_mul(items)
                self.user_matrix = self._items_mul_ratings(transform_matrix, ratings)
            
            # 计算并打印当前 RMSE
            rmse = self._get_rmse(ratings)
            print(f"Iteration {i + 1}/{max_iter}, RMSE: {rmse:.6f}")
     
        self.rmse = rmse
        print("训练完成.")

    def _predict(self, user_id, n_items):
        """
        为单个用户生成 Top-N 推荐
        返回: [(item_id, score), ...]
        """
        if user_id not in self.user_ids_dict:
            return []
            
        u_idx = self.user_ids_dict[user_id]
        # 获取用户隐因子向量 (1 x k)
        users_col = self.user_matrix.col(u_idx).transpose
     
        # 计算该用户对所有物品的预测评分 (1 x n)
        # 结果是一个列表: [score_item_0, score_item_1, ...]
        all_scores = users_col.mat_mul(self.item_matrix).data[0]
        
        # 将索引映射回原始物品 ID，并组成 (item_id, score) 对
        items_scores = [(self.item_ids[idx], score) for idx, score in enumerate(all_scores)]
        
        # 过滤掉用户已经评分过的物品
        viewed_items = self.user_items.get(user_id, set())
        filtered_scores = filter(lambda x: x[0] not in viewed_items, items_scores)
     
        # 按分数降序排序，取前 N 个
        return sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:n_items]

    def predict(self, user_ids, n_items=10):
        """批量预测多个用户的推荐列表"""
        return [self._predict(uid, n_items) for uid in user_ids]


# ==============================================================================
# 第三部分：主程序入口与测试
# ==============================================================================

def format_prediction(item_id, score):
    """格式化输出推荐结果"""
    return f"item_id:{item_id} score:{score:.2f}"

def load_movie_ratings(file_name):
    """
    加载 CSV 评分数据
    假设格式: user_id,item_id,rating (无表头或跳过第一行)
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = iter(f)
            # 尝试读取并打印表头 (如果存在)
            first_line = next(lines, None)
            if first_line:
                # 简单检查是否是表头 (非数字)
                parts = first_line.strip().split(",")
                if not parts[0].isdigit():
                    print(f"The column names are: {', '.join(parts)}.")
                    # 如果是表头，继续读取下一行作为数据
                    data_lines = lines
                else:
                    # 如果没有表头，第一行也是数据
                    data_lines = chain([first_line], lines)
            else:
                data_lines = []

            data = []
            for line in data_lines:
                line = line.strip()
                if not line: continue
                parts = line.split(",")
                # 解析: user(int), item(int), rating(float)
                # 注意处理可能的末尾空字符或多余分隔符
                clean_parts = [p for p in parts if p]
                if len(clean_parts) >= 3:
                    row = [int(clean_parts[0]), int(clean_parts[1]), float(clean_parts[2])]
                    data.append(row)
            return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_name}' 未找到。将使用内置测试数据。")
        return None

if __name__ == "__main__":
    print("="*60)
    print("使用 ALS 算法进行电影推荐")
    print("="*60)
    
    model = ALS()
    
    # 1. 尝试加载外部数据
    X = load_movie_ratings('./ratings_small.csv')
    
    # 2. 如果文件不存在，使用内置的小型测试数据集
    if X is None or len(X) == 0:
        print("未检测到外部数据文件，正在生成模拟数据...")
        # 模拟数据: [user_id, item_id, rating]
        # 构造一些明显的模式以便观察推荐效果
        X = [
            [1,1,5], [1,2,4], [1,3,1],
            [2,1,4], [2,2,5], [2,4,2],
            [3,3,5], [3,4,4], [3,5,1],
            [4,1,1], [4,3,2], [4,5,5],
            [5,2,2], [5,4,1], [5,5,4],
            [6,1,3], [6,2,3], [6,3,3], # 中立用户
            [7,6,5], [7,7,5],          # 新用户群
            [8,6,4], [8,7,5],
            [9,1,5], [9,2,5], [9,6,1], # 喜欢 1,2 不喜欢 6
            [10,6,5], [10,7,4], [10,1,1]
        ]
        print(f"已生成 {len(X)} 条模拟评分数据。")

    # 3. 训练模型
    # k=3: 假设存在 3 种潜在特征 (如：动作、爱情、喜剧)
    # max_iter=10: 迭代 10 次以收敛
    model.fit(X, k=3, max_iter=10)

    print("\n开始生成推荐列表...")
    # 4. 对用户进行 Top-N 推荐
    # 获取数据中出现的所有用户 ID
    unique_users = sorted(list(set(x[0] for x in X)))
    
    # 为前 5 个用户生成推荐 (如果用户少则全部推荐)
    target_users = unique_users[:min(5, len(unique_users))]
    
    predictions = model.predict(target_users, n_items=3)
    
    print("-" * 60)
    for user_id, rec_list in zip(target_users, predictions):
        formatted_recs = [format_prediction(iid, score) for iid, score in rec_list]
        if formatted_recs:
            print(f"User ID: {user_id} => 推荐: {formatted_recs}")
        else:
            print(f"User ID: {user_id} => 暂无合适推荐 (可能已评分所有物品)")
    print("-" * 60)
    print("程序执行完毕。")
