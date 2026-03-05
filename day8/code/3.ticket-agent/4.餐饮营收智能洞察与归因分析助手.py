"""
================================================================================
文件名: 餐饮营收智能洞察与归因分析助手.py
功能描述: 
    本程序是一个基于机器学习的高级餐饮数据分析助手。
    它不仅支持通过自然语言查询餐饮营收数据，还能利用【线性回归】和【决策树】
    算法，深度挖掘影响餐饮总营收和人均消费的关键驱动因素（归因分析）。

核心能力:
    1. Text-to-SQL: 查询基础营收数据、客流结构及活动信息。
    2. 人均消费归因 (Linear Regression): 
       - 针对特定活动（如万圣节）和客群（年卡/散客），构建 LR 模型。
       - 量化不同客群对总营收的边际贡献（系数分析）。
    3. 关键驱动因素分析 (Decision Tree/CART):
       - 构建决策树回归模型，识别影响营收的核心因子（天气、节假日、票价等）。
       - 自动生成决策树可视化图谱和规则文本，直观展示决策路径。
    4. 动态绘图引擎: 支持执行大模型生成的 Python 绘图代码，实现自定义可视化。

核心工作流程:
    1. [用户提问] 用户提出复杂分析问题 (例：“万圣节期间哪类用户对餐饮贡献最大？”)。
    2. [意图路由] Agent 识别意图，调用对应工具：
       - 查数 -> exc_sql
       - 归因分析 -> compute_avg_revenue (LR 模型) 或 analysis_influence_factors (CART 模型)
       - 自定义绘图 -> plot_image
    3. [建模分析 (内部逻辑)]:
       a. 数据提取: 从 MySQL 获取指定条件的数据集。
       b. 特征工程: 
          - 时间特征转换 (Date -> Days Diff)。
          - 衍生特征构造 (如：非北京游客比例 = 100 - 北京占比)。
          - 类别特征数值化 (Weekday -> 1-7)。
       c. 模型训练: 
          - LR: 计算各客群 attendance 的回归系数，判断正向/负向影响。
          - CART: 训练决策树，提取特征重要性，生成树状图。
       d. 结果输出: 返回文字结论 + 决策树图片/规则。
    4. [结果渲染] WebUI 展示分析结论及可视化图表。

依赖环境:
    - Python 3.8+
    - qwen_agent, dashscope, pandas, sqlalchemy, mysql-connector-python
    - scikit-learn (LinearRegression, DecisionTreeRegressor, DictVectorizer)
    - matplotlib, numpy
================================================================================
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text
from sklearn.metrics import mean_squared_error

# Qwen Agent 相关导入
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from dotenv import load_dotenv

load_dotenv()
# ================= 配置区域 =================

# 配置 DashScope API
dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

# 数据库配置
DB_USER = os.getenv('DB_USER', '')
DB_PASS = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', '')
DB_PORT = os.getenv('DB_PORT', '')
DB_NAME = os.getenv('DB_NAME', '')


# 图片保存目录
IMG_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'image_show')
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

# ================= 核心业务逻辑函数 =================

def get_engine():
    """获取数据库连接引擎"""
    conn_str = (
        f'mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
        f'?charset=utf8mb4'
    )
    return create_engine(
        conn_str,
        connect_args={'connect_timeout': 10},
        pool_size=10,
        max_overflow=20
    )

# --- 线性回归分析模块 (compute_avg_revenue 底层逻辑) ---

def get_q3_data(engine, marquee_event: int) -> pd.DataFrame:
    """获取指定活动下的餐饮营收明细数据"""
    with engine.connect() as conn:
        query = text("""
            SELECT date, ap_attendance, ticket_attendance, promotional_ticket_attendance, total_fb_revenue
            FROM ubr_revenue
            WHERE marquee_event = :event_id
        """)
        result = conn.execute(query, {"event_id": marquee_event})
        return pd.DataFrame([dict(row) for row in result.mappings()])

def build_lr_model(df: pd.DataFrame):
    """
    构建线性回归模型，分析不同客群 (AP/Ticket/Promo) 对总营收的贡献系数
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 特征工程：将日期转换为距离参考日期的天数，消除时间序列趋势干扰
    reference_date = pd.to_datetime('2023-01-01')
    df['date_diff'] = (df['date'] - reference_date).dt.days
    df = df.drop(['date'], axis=1)
    
    # 准备特征 (X) 和目标 (y)
    feature_cols = [c for c in df.columns if c not in ['total_fb_revenue']]
    X = df[feature_cols]
    y = df['total_fb_revenue']
    
    # 使用 DictVectorizer 处理混合类型特征 (虽然此处主要是数值，但保持通用性)
    dvec = DictVectorizer(sparse=False)
    X_transformed = dvec.fit_transform(X.to_dict(orient='records'))
    
    model = LinearRegression()
    model.fit(X_transformed, y)
    return model, dvec

def analyze_lr_coefficients(model, dvec) -> str:
    """分析回归系数，生成业务结论"""
    feature_names = dvec.feature_names_
    coefficients = model.coef_
    
    # 构建 DataFrame 便于排序
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
    coef_df = coef_df.sort_values(by='coefficient', ascending=False)
    
    # 筛选正相关且影响最大的前 5 个特征
    positive_df = coef_df[coef_df['coefficient'] > 0].head(5)
    
    if positive_df.empty:
        return "未检测到显著的正向影响因素。"
    
    features_str = ', '.join(positive_df['feature'])
    values_str = ', '.join([f"{v:.2f}" for v in positive_df['coefficient']])
    
    return (f"【线性回归洞察】\n"
            f"影响餐饮总消费的关键正向特征（系数越大，贡献越高）为：{features_str}。\n"
            f"对应的回归系数分别为：{values_str}。")

def compute_avg_revenue_logic(attendance_type: str, marquee_event_str: str, engine) -> str:
    """
    主流程：计算特定场景下的人均消费归因
    注意：虽然函数名叫 compute_avg_revenue，但实际逻辑是通过 LR 模型分析各客群对总营收的边际贡献
    """
    # 映射活动名称到 ID
    event_map = {
        '无活动': 0, 'Chinese New Year': 1, 'Honor Of King': 2, 
        'Cool Summer': 3, 'Halloween Horror Night': 4
    }
    event_id = event_map.get(marquee_event_str, 0)
    
    # 1. 获取数据
    df = get_q3_data(engine, event_id)
    if df.empty:
        return f"未找到活动 '{marquee_event_str}' 的相关数据。"
    
    # 2. 建模
    model, dvec = build_lr_model(df)
    
    # 3. 分析并返回结论
    return analyze_lr_coefficients(model, dvec)

# --- 决策树分析模块 (analysis_influence_factors 底层逻辑) ---

def get_q4_data(engine, target_col: str) -> pd.DataFrame:
    """获取全量特征数据用于决策树建模"""
    with engine.connect() as conn:
        # 动态构建 SQL，确保目标列存在
        query = text(f"""
            SELECT date, ticket_price, operating_hours, total_attendance,
                   ap_attendance, ticket_attendance, promotional_ticket_attendance,
                   media_cost_index, marquee_event, max_temperature, min_temperature,
                   week_days, is_national_holiday, beijing_guest_ratio,
                   age_group_0_3, age_group_4_12, age_group_13_16, age_group_17_18,
                   age_group_19_25, age_group_26_35, age_group_36_45, age_group_46_50,
                   age_group_51_65, age_group_65_plus, {target_col}
            FROM ubr_revenue
        """)
        result = conn.execute(query)
        return pd.DataFrame([dict(row) for row in result.mappings()])

def build_cart_model(df: pd.DataFrame, target_col: str):
    """
    构建决策树回归模型 (CART)
    包含复杂的特征工程：衍生新特征、类别编码、缺失值处理
    """
    df = df.copy()
    
    # 特征工程：构造业务含义更强的衍生特征
    df['POO outside BJ'] = 100 - df['beijing_guest_ratio']  # 非北京游客比例
    # 避免除以零
    df['Pass Revisit %'] = (df['ap_attendance'] / df['total_attendance'].replace(0, 1)) * 100
    
    # 重命名列以简化显示
    rename_map = {
        'total_attendance': 'Att', 
        'operating_hours': 'Park Hrs', 
        'max_temperature': 'Max Temp'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 选择关键特征
    selected_features = [
        'date', 'week_days', 'Att', 'Pass Revisit %', 'Max Temp', 
        'POO outside BJ', 'Park Hrs', target_col
    ]
    # 确保所选列都在 DataFrame 中
    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features]
    
    # 时间特征处理
    df['date'] = pd.to_datetime(df['date'])
    ref_date = pd.to_datetime('2023-01-01')
    df['date_diff'] = (df['date'] - ref_date).dt.days
    df = df.drop(['date'], axis=1)
    
    # 缺失值填充 (用均值填充目标变量)
    df[target_col].fillna(df[target_col].mean(), inplace=True)
    
    # 类别特征数值化 (Weekdays)
    weekday_map = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }
    if 'week_days' in df.columns:
        df['week_days'] = df['week_days'].replace(weekday_map)
    
    # 准备训练数据
    features = [c for c in df.columns if c != target_col]
    X = df[features]
    y = df[target_col]
    
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练决策树 (限制深度以防过拟合，提高可解释性)
    model = DecisionTreeRegressor(random_state=42, max_depth=4)
    model.fit(X_train, y_train)
    
    # 评估
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"[CART Model] MSE: {mse:.2f}")
    
    return model, features

def visualize_cart(model, features) -> Dict[str, Any]:
    """可视化决策树并导出规则文本"""
    plt.figure(figsize=(20, 6))
    plot_tree(model, filled=True, feature_names=features, rounded=True, fontsize=8)
    plt.title("餐饮消费影响因子决策树")
    
    # 保存图片
    timestamp = int(time.time() * 1000)
    filename = f'tree_{timestamp}.png'
    save_path = os.path.join(IMG_SAVE_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # 导出文本规则
    rules_text = export_text(model, feature_names=features)
    
    return {
        'image_url': f'image_show/{filename}',
        'tree_rules': rules_text
    }

def analysis_influence_factors_logic(target_type: str, engine) -> Dict[str, Any]:
    """主流程：分析影响总消费或人均消费的关键因子"""
    target_map = {'总消费': 'total_fb_revenue', '人均消费': 'rev_per_cap'}
    target_col = target_map.get(target_type, 'total_fb_revenue')
    
    # 1. 获取数据
    df = get_q4_data(engine, target_col)
    
    # 2. 建模
    model, features = build_cart_model(df, target_col)
    
    # 3. 可视化与输出
    return visualize_cart(model, features)

# --- 动态绘图模块 (plot_image 底层逻辑) ---

def plot_image_logic(code_input: str) -> str:
    """
    安全执行用户/模型生成的绘图代码
    通过正则修改代码以捕获 Figure 对象，并拦截 plt.show()
    """
    # 1. 修改代码：捕获 figure 对象 (如果有的话)
    # 注意：这里简单替换，实际生产环境需更严格的沙箱机制
    modified_code = re.sub(r'plt\.figure\([^)]*\)', r'fg = \g<0>', code_input)
    modified_code = modified_code.replace("plt.show()", "")
    # 修复可能的三引号格式问题
    modified_code = modified_code.replace('\n""\n', '\n"""\n')
    
    print(f"[Plot Engine] 执行绘图代码...\n{modified_code[:200]}...")
    
    try:
        # 2. 在受限命名空间中执行
        # 仅暴露 plt 模块，防止访问系统文件
        exec_env = {'plt': plt, 'np': np, 'pd': pd}
        exec(modified_code, exec_env)
        
        # 3. 保存图片
        timestamp = int(time.time() * 1000)
        filename = f'plot_{timestamp}.png'
        save_path = os.path.join(IMG_SAVE_DIR, filename)
        plt.savefig(save_path, dpi=150)
        plt.close('all')  # 关闭所有图以防内存泄漏
        
        return f'image_show/{filename}'
    except Exception as e:
        return f"绘图执行失败: {str(e)}"

# ================= 工具类定义 =================

@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """基础 SQL 查询工具"""
    description = '执行 SQL 查询，返回 Markdown 格式的数据表格'
    parameters = [{
        'name': 'sql_input', 'type': 'string', 
        'description': '标准 SQL 查询语句', 'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        try:
            df = pd.read_sql(text(args['sql_input']), get_engine())
            return df.head(10).to_markdown(index=False)
        except Exception as e:
            return f"SQL 执行错误: {str(e)}"

@register_tool('compute_avg_revenue')
class ComputeAvgRevenueTool(BaseTool):
    """人均消费归因分析工具 (基于线性回归)"""
    description = '分析特定活动和客群类型下，各类用户对餐饮总营收的边际贡献（线性回归系数）'
    parameters = [
        {'name': 'attendance_type', 'type': 'string', 'description': '用户类型：ap (年卡), ticket (门票), promotional (促销票)', 'required': True},
        {'name': 'marquee_event', 'type': 'string', 'description': '活动名称：无活动，Chinese New Year, Honor Of King, Cool Summer, Halloween Horror Night', 'required': False}
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        return compute_avg_revenue_logic(
            args['attendance_type'], 
            args.get('marquee_event', '无活动'), 
            get_engine()
        )

@register_tool('analysis_influence_factors')
class AnalysisInfluenceFactorsTool(BaseTool):
    """关键驱动因素分析工具 (基于决策树)"""
    description = '利用决策树模型分析哪些因素（天气、节假日、客流等）对餐饮总消费或人均消费影响最大，并返回决策树图和规则'
    parameters = [
        {'name': 'target_type', 'type': 'string', 'description': '分析目标：总消费 或 人均消费', 'required': True}
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        result = analysis_influence_factors_logic(args['target_type'], get_engine())
        
        if isinstance(result, dict) and 'image_url' in result:
            return f"【决策树分析规则】:\n{result['tree_rules']}\n\n![决策树可视化]({result['image_url']})"
        return str(result)

@register_tool('plot_image')
class PlotImageTool(BaseTool):
    """动态 Python 绘图工具"""
    description = '执行提供的 Python Matplotlib 代码生成图表，适用于自定义可视化需求'
    parameters = [
        {'name': 'code_input', 'type': 'string', 'description': '完整的 Python 绘图代码 (使用 plt 库)', 'required': True}
    ]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        img_path = plot_image_logic(args['code_input'])
        if img_path.startswith("绘图执行失败"):
            return img_path
        return f"![自定义图表]({img_path})"

# ================= 服务启动与交互 =================

system_prompt = """
我是餐饮分析助手，专注于乐园餐饮营收与消费行为分析。
数据表 `ubr_revenue` 包含日期、客流结构、天气、活动、票价及餐饮营收等字段。

核心指令:
1. 若用户查询“万圣节”，请使用 `marquee_event='4'`。
2. 统计金额时保留 2 位小数。
3. **重要**: 当工具返回包含 `![...](image_url)` 的图片链接时，你必须在回复中原样保留该链接，不要省略，以便用户直接看到决策树图或自定义图表。

可用工具:
- exc_sql: 查原始数据。
- compute_avg_revenue: 分析不同客群 (AP/散客) 对营收的贡献度 (线性回归)。
- analysis_influence_factors: 找出影响营收/人均消费的核心因子 (决策树 + 可视化)。
- plot_image: 执行自定义绘图代码。
"""

def init_agent_service():
    """初始化 Agent 服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='餐饮智能洞察助手',
            description='基于机器学习的餐饮营收归因与分析专家',
            system_message=system_prompt,
            function_list=['exc_sql', 'compute_avg_revenue', 'analysis_influence_factors', 'plot_image'],
        )
        print("✅ 助手初始化成功！已加载 LR 归因与 CART 决策树引擎。")
        return bot
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        raise

def app_gui():
    """启动 Web 图形界面"""
    try:
        print("🚀 正在启动 Web 界面...")
        bot = init_agent_service()
        
        chatbot_config = {
            'prompt.suggestions': [
                '请计算万圣节期间，年卡用户、门票用户和促销票用户对餐饮总营收的边际贡献分别是多少？',
                '分析哪些因素（如天气、节假日、票价）对餐饮总消费的影响最大？请画出决策树。',
                '分析影响人均餐饮消费的关键因子，并展示决策规则。',
                '画一张图展示过去一个月每日总营收的趋势。'
            ]
        }
        
        print(f"✅ 服务就绪。图表将保存至: {IMG_SAVE_DIR}")
        WebUI(bot, chatbot_config=chatbot_config).run()
    except Exception as e:
        print(f"❌ Web 启动失败: {e}")

if __name__ == '__main__':
    if not os.getenv('DASHSCOPE_API_KEY'):
        print("⚠️ 警告: 未设置 DASHSCOPE_API_KEY 环境变量")
    app_gui()
