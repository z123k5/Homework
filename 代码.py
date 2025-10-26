import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
import re
from io import StringIO

# --- 数据预处理部分 ---
def preprocess_data(excel_path):
    data_raw = pd.read_excel(excel_path, skiprows=1, header=0)
    data_raw.rename(columns={data_raw.columns[0]: 'Participant_Type'}, inplace=True)
    data_raw['Participant_Type'] = data_raw['Participant_Type'].apply(lambda x: 1 if x == 'Participating' else 0)

    interaction_cols = [col for col in data_raw.columns if any(keyword in col for keyword in ['Dislike', 'Like', 'Share', 'Flag'])]
    for col in interaction_cols:
        data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce').fillna(0).astype(int)

    exclude_cols = ['Participant_Type', 'Session ID', 'Participant ID'] + interaction_cols
    likert_scale_cols = [col for col in data_raw.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data_raw[col])]

    for col in likert_scale_cols:
        if not data_raw[col].isnull().all() and data_raw[col].std() != 0:
            data_raw[col] = (data_raw[col] - data_raw[col].mean()) / data_raw[col].std()
        else:
            data_raw[col] = 0

    for col in data_raw.columns:
        if pd.api.types.is_numeric_dtype(data_raw[col]):
            data_raw[col] = data_raw[col].fillna(0)
            
    return data_raw

# --- GLMER分析部分 ---
# Helper function for LRT for MixedLM
def compare_mixedlm_lr_test(full_model_fit, reduced_model_fit):
    llf_full = full_model_fit.llf
    llf_reduced = reduced_model_fit.llf
    df_full = full_model_fit.df_modelwc
    df_reduced = reduced_model_fit.df_modelwc
    
    lr_statistic = -2 * (llf_reduced - llf_full)
    df_diff = df_full - df_reduced
    
    if df_diff <= 0:
        return np.nan, np.nan

    p_value = chi2.sf(lr_statistic, df_diff)
    return lr_statistic, p_value

# Helper function for LRT for GLM
def compare_glm_lr_test(full_model_fit, reduced_model_fit):
    llf_full = full_model_fit.llf
    llf_reduced = reduced_model_fit.llf
    df_full = full_model_fit.df_model
    df_reduced = reduced_model_fit.df_model
    
    lr_statistic = -2 * (llf_reduced - llf_full)
    df_diff = df_full - df_reduced
    
    if df_diff <= 0:
        return np.nan, np.nan

    p_value = chi2.sf(lr_statistic, df_diff)
    return lr_statistic, p_value

def run_glmer_analysis(data):
    # 识别因变量：购买意向 (Purchase Intention)
    purchase_intention_cols = [col for col in data.columns if (
        '购买意向' in col or 'Purchase Intention' in col
    )]
    
    if not purchase_intention_cols:
        raise ValueError("未找到购买意向相关的列，请检查列名。")
    
    for col in purchase_intention_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data['Purchase_Intention_Avg'] = data[purchase_intention_cols].mean(axis=1)
    
    # 识别问卷星的李科特七分量表数据列
    interaction_col_patterns = ['Dislike', 'Like', 'Share', 'Flag']
    
    all_interaction_cols_in_data = []
    for pattern in interaction_col_patterns:
        if pattern == 'Share': # 'Share ' has a space
            if 'Share ' in data.columns:
                all_interaction_cols_in_data.append('Share ')
        elif pattern == 'Flag': # 'Flag' without suffix
            if 'Flag' in data.columns:
                all_interaction_cols_in_data.append('Flag')
        else: # 'Dislike', 'Like' without suffix
            if pattern in data.columns:
                all_interaction_cols_in_data.append(pattern)
        
        for i in range(1, 10): # Post 2 to Post 10
            if pattern == 'Share':
                col_name = f'Share .{i}'
            elif pattern == 'Flag':
                col_name = f'Flag.{i}'
            else:
                col_name = f'{pattern}.{i}'
            
            if col_name in data.columns:
                all_interaction_cols_in_data.append(col_name)
    
    
    exclude_for_independent = ['Participant_Type', 'Session ID', 'Participant ID', 'Purchase_Intention_Avg'] + \
                              all_interaction_cols_in_data
    
    independent_likert_cols = [col for col in data.columns if col not in exclude_for_independent and pd.api.types.is_numeric_dtype(data[col])]
    
    # 对问卷星的李科特量表数据进行PCA降维
    # 1. 提取李科特量表数据
    likert_data = data[independent_likert_cols].copy()
    
    # 2. 处理缺失值 (PCA前通常需要处理，这里使用均值填充)
    likert_data = likert_data.fillna(likert_data.mean())
    
    # 3. 标准化数据
    scaler = StandardScaler()
    scaled_likert_data = scaler.fit_transform(likert_data)
    
    # 4. 执行PCA
    # 选择主成分数量，例如解释90%的方差
    pca = PCA(n_components=0.9)
    pca_components = pca.fit_transform(scaled_likert_data)
    
    # 将主成分添加到原始数据DataFame中
    pca_df = pd.DataFrame(pca_components, columns=[f'PC_{i+1}' for i in range(pca_components.shape[1])], index=data.index)
    data = pd.concat([data.drop(columns=independent_likert_cols), pca_df], axis=1)
    
    # 更新独立变量列表为新的主成分
    independent_likert_cols_sanitized = list(pca_df.columns)
####
    # 假设上面代码里 pca 是 sklearn.decomposition.PCA() 的结果
    loadings = pd.DataFrame(
        pca.components_.T,  # 转置后行是原始变量
        columns=[f'PC_{i+1}' for i in range(pca.n_components_)],
        index=likert_data.columns  # 原始变量名
    )

    # 按绝对值排序，查看每个主成分的主要构成变量
    for i in range(1, 13):  # 你有 12 个主成分
        print(f'\nTop loadings for PC_{i}:')
        print(loadings[f'PC_{i}'].abs().sort_values(ascending=False).head(10))
###

    # 构建长格式数据
    long_data_list = []
    num_posts = 10
    
    # Create lists of column names for each interaction type across all posts
    dislike_cols = [c for c in data.columns if c == 'Dislike' or (c.startswith('Dislike.') and c[8:].isdigit())]
    dislike_cols.sort(key=lambda x: int(x.split('.')[-1]) if '.' in x else 0)
    
    like_cols = [c for c in data.columns if c == 'Like' or (c.startswith('Like.') and c[5:].isdigit())]
    like_cols.sort(key=lambda x: int(x.split('.')[-1]) if '.' in x else 0)
    
    share_cols = [c for c in data.columns if c == 'Share ' or (c.startswith('Share .') and c[7:].isdigit())]
    share_cols.sort(key=lambda x: int(x.split(' .')[-1]) if ' .' in x else 0)
    
    flag_cols = [c for c in data.columns if c == 'Flag' or (c.startswith('Flag.') and c[5:].isdigit())]
    flag_cols.sort(key=lambda x: int(x.split('.')[-1]) if '.' in x else 0)
    
    
    if len(dislike_cols) == num_posts and len(like_cols) == num_posts and \
       len(share_cols) == num_posts and len(flag_cols) == num_posts:
        for i in range(num_posts):
            temp_df = data[[
                'Session ID', 'Participant ID', 'Participant_Type', 'Purchase_Intention_Avg'
            ] + independent_likert_cols_sanitized].copy()
            temp_df['Post_ID'] = i + 1
            temp_df['Dislike'] = data[dislike_cols[i]]
            temp_df['Like'] = data[like_cols[i]]
            temp_df['Share'] = data[share_cols[i]]
            temp_df['Flag'] = data[flag_cols[i]]
            long_data_list.append(temp_df)
    else:
        raise ValueError(f"互动行为列的数量与预期的帖子数量不匹配。Dislike: {len(dislike_cols)}, Like: {len(like_cols)}, Share: {len(share_cols)}, Flag: {len(flag_cols)}")
    
    if not long_data_list:
        raise ValueError("未能正确构建长格式数据，请检查互动行为列名模式。")
    
    long_data = pd.concat(long_data_list, ignore_index=True)
    
    long_data['Session ID'] = long_data['Session ID'].astype('category')
    long_data['Participant ID'] = long_data['Participant ID'].astype(str).astype('category') # 确保Participant ID是字符串类型
    long_data['Post_ID'] = long_data['Post_ID'].astype('category')
    
    print("\n长格式数据信息：")
    print(long_data.info())
    print(long_data.head())
    
    # --- 线性混合效应模型 ---
    print("\n开始拟合线性混合效应模型...")
    long_data_cleaned_linear = long_data.dropna(subset=[
        'Purchase_Intention_Avg', 'Participant ID', 'Participant_Type'
    ] + independent_likert_cols_sanitized + ['Dislike', 'Like', 'Share', 'Flag'])
    long_data_cleaned_linear['Participant ID'] = long_data_cleaned_linear['Participant ID'].astype('category')
    
    # 构建固定效应公式，使用规范化的列名
    fixed_effects_linear_candidates = independent_likert_cols_sanitized + ['Dislike', 'Like', 'Share', 'Flag', 'Participant_Type']
    fixed_effects_linear_formula_parts = []
    for col in fixed_effects_linear_candidates:
        if col in long_data_cleaned_linear.columns and long_data_cleaned_linear[col].var() != 0:
            fixed_effects_linear_formula_parts.append(col)
        elif col in long_data_cleaned_linear.columns and long_data_cleaned_linear[col].var() == 0:
            print(f"警告: 列 '{col}' 方差为零，将其从线性模型中移除。")
    
    fixed_effects_linear_formula = " + ".join(fixed_effects_linear_formula_parts)
    model_formula_linear = f"Purchase_Intention_Avg ~ {fixed_effects_linear_formula}"
    print(f"线性混合效应模型公式: {model_formula_linear}")
    
    if long_data_cleaned_linear['Participant ID'].nunique() < 2:
        print("警告: Participant ID 组数量不足，无法拟合随机效应模型。将尝试拟合普通线性模型。")
        linear_model = smf.ols(model_formula_linear, data=long_data_cleaned_linear).fit()
        print("普通线性模型拟合完成。")
        print("\n普通线性模型摘要：")
        print(linear_model.summary())
        with open('linear_model_summary.txt', 'w', encoding='utf-8') as f:
            f.write(linear_model.summary().as_text())
        linear_mixed_model_fit = linear_model
    else:
        linear_mixed_model = smf.mixedlm(model_formula_linear, data=long_data_cleaned_linear, groups=long_data_cleaned_linear['Participant ID'])
        linear_mixed_model_fit = linear_mixed_model.fit()
        print("线性混合效应模型拟合完成。")
        print("\n线性混合效应模型摘要：")
        print(linear_mixed_model_fit.summary())
        with open('linear_mixed_model_summary.txt', 'w', encoding='utf-8') as f:
            f.write(linear_mixed_model_fit.summary().as_text())
    
    # --- 逻辑混合效应模型 ---
    # statsmodels的mixedlm不支持family参数，因此我们使用GLM来模拟广义混合效应模型
    # 注意：这并不是一个真正的GLMM，而是GLM。
    # 对于真正的GLMM，需要使用更复杂的库如PyMC3或brms（R中的包）
    long_data['Purchase_Intention_Binary'] = (long_data['Purchase_Intention_Avg'] > long_data['Purchase_Intention_Avg'].median()).astype(int)
    
    print("\n开始拟合逻辑回归模型 (GLM)...")
    long_data_cleaned_logistic = long_data.dropna(subset=[
        'Purchase_Intention_Binary', 'Participant ID', 'Participant_Type'
    ] + independent_likert_cols_sanitized + ['Dislike', 'Like', 'Share', 'Flag'])
    long_data_cleaned_logistic['Participant ID'] = long_data_cleaned_logistic['Participant ID'].astype('category')
    
    # 构建固定效应公式，使用规范化的列名
    fixed_effects_logistic_candidates = independent_likert_cols_sanitized + ['Dislike', 'Like', 'Share', 'Flag', 'Participant_Type']
    fixed_effects_logistic_formula_parts = []
    for col in fixed_effects_logistic_candidates:
        if col in long_data_cleaned_logistic.columns and long_data_cleaned_logistic[col].var() != 0:
            fixed_effects_logistic_formula_parts.append(col)
        elif col in long_data_cleaned_logistic.columns and long_data_cleaned_logistic[col].var() == 0:
            print(f"警告: 列 '{col}' 方差为零，将其从逻辑模型中移除。")
    
    fixed_effects_logistic_formula = " + ".join(fixed_effects_logistic_formula_parts)
    model_formula_logistic = f"Purchase_Intention_Binary ~ {fixed_effects_logistic_formula}"
    print(f"逻辑回归模型公式: {model_formula_logistic}")
    
    logistic_model = smf.glm(model_formula_logistic, data=long_data_cleaned_logistic, family=sm.families.Binomial()).fit()
    print("逻辑回归模型 (GLM) 拟合完成。")
    print("\n逻辑回归模型 (GLM) 摘要：")
    print(logistic_model.summary())
    with open('logistic_glm_summary.txt', 'w', encoding='utf-8') as f:
        f.write(logistic_model.summary().as_text())
    logistic_mixed_model_fit = logistic_model # 为了后续兼容性，仍使用此变量名
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # VIF计算只针对固定效应部分，且不包含因变量
    X_vif_cols = independent_likert_cols_sanitized + ['Dislike', 'Like', 'Share', 'Flag', 'Participant_Type']
    X_vif = long_data[X_vif_cols].dropna()
    
    if not X_vif.empty and X_vif.shape[1] > 0:
        vif_data = pd.DataFrame()
        vif_data['feature'] = X_vif.columns
        if X_vif.shape[0] > X_vif.shape[1]:
            # 确保用于VIF计算的列在X_vif中存在且方差不为零
            vif_features = [col for col in X_vif.columns if X_vif[col].var() != 0]
            if vif_features:
                X_vif_filtered = X_vif[vif_features]
                vif_data = pd.DataFrame({
                    "feature": vif_features,
                    "VIF": [variance_inflation_factor(X_vif_filtered.values, i) for i in range(X_vif_filtered.shape[1])]
                })
                print("\n共线性 (VIF) 结果：")
                print(vif_data)
                vif_data.to_csv('vif_results.csv', index=False)
            else:
                print("\n警告: 所有用于VIF计算的特征方差都为零，无法计算VIF。")
        else:
            print("\n警告: 样本数不足或特征过多，无法可靠计算VIF。")
    else:
        print("\n警告: 用于VIF计算的数据为空或没有特征。")
    
    print("\n方差贡献（似然比检验）:")
    lrt_results = []
    
    # 线性混合效应模型LRT
    if long_data_cleaned_linear['Participant ID'].nunique() >= 2:
        # Full model for LRT
        full_formula_linear = "Purchase_Intention_Avg ~ " + " + ".join(fixed_effects_linear_formula_parts)
        full_model_linear = smf.mixedlm(full_formula_linear, data=long_data_cleaned_linear, groups=long_data_cleaned_linear['Participant ID'])
        full_model_linear_fit = full_model_linear.fit()
    
        for var in fixed_effects_linear_formula_parts:
            reduced_formula_parts = [f for f in fixed_effects_linear_formula_parts if f != var]
            if not reduced_formula_parts:
                reduced_formula_linear = "Purchase_Intention_Avg ~ 1"
            else:
                reduced_formula_linear = "Purchase_Intention_Avg ~ " + " + ".join(reduced_formula_parts)
            
            reduced_model_linear = smf.mixedlm(reduced_formula_linear, data=long_data_cleaned_linear, groups=long_data_cleaned_linear['Participant ID'])
            reduced_model_linear_fit = reduced_model_linear.fit()
            
            lr_statistic, p_value = compare_mixedlm_lr_test(full_model_linear_fit, reduced_model_linear_fit)
            lrt_results.append({'Model': 'Linear Mixed Model', 'Variable': var, 'LRT Statistic': lr_statistic, 'P-value': p_value})
    
    # 逻辑回归模型LRT (GLM)
    # Full model for LRT
    full_formula_logistic = "Purchase_Intention_Binary ~ " + " + ".join(fixed_effects_logistic_formula_parts)
    full_model_logistic = smf.glm(full_formula_logistic, data=long_data_cleaned_logistic, family=sm.families.Binomial()).fit()
    
    for var in fixed_effects_logistic_formula_parts:
        reduced_formula_parts = [f for f in fixed_effects_logistic_formula_parts if f != var]
        if not reduced_formula_parts:
            reduced_formula_logistic = "Purchase_Intention_Binary ~ 1"
        else:
            reduced_formula_logistic = "Purchase_Intention_Binary ~ " + " + ".join(reduced_formula_parts)
        
        reduced_model_logistic = smf.glm(reduced_formula_logistic, data=long_data_cleaned_logistic, family=sm.families.Binomial()).fit()
        
        lr_statistic, p_value = compare_glm_lr_test(full_model_logistic, reduced_model_logistic)
        lrt_results.append({'Model': 'Logistic GLM', 'Variable': var, 'LRT Statistic': lr_statistic, 'P-value': p_value})
    
    lrt_df = pd.DataFrame(lrt_results)
    print(lrt_df)
    lrt_df.to_csv('lrt_results.csv', index=False)
    
    print("GLMER分析完成，结果已保存到文件。")
    return linear_mixed_model_fit, logistic_mixed_model_fit, vif_data, lrt_df

# --- Excel导出部分 ---
def parse_markdown_table(table_string):
    lines = table_string.strip().split("\n")
    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) < 2:
        return pd.DataFrame()

    header_line = lines[0]
    data_lines = lines[2:] # Skip separator line

    column_names = [h.strip() for h in header_line.split("|") if h.strip()]

    data = []
    for line in data_lines:
        row_items = [item.strip() for item in line.split("|")]
        row = [item for item in row_items if item] 
        
        if len(row) == len(column_names):
            data.append(row)

    df = pd.DataFrame(data, columns=column_names)
    return df

def export_tables_to_excel(md_report_path, output_excel_path):
    with open(md_report_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    table_headers = [
        {"header": "| Term        | Coef.   | Std.Err. | z        | P>|z|   | [0.025 | 0.975] |", "sheet_name": "线性混合效应模型摘要"},
        {"header": "| Term        | Coef.     | Std.Err. | z         | P>|z|   | [0.025   | 0.975] |", "sheet_name": "逻辑回归模型摘要"},
        {"header": "| feature | VIF      |", "sheet_name": "共线性诊断_VIF"},
        {"header": "| Model              | Variable         | LRT Statistic | P-value      |", "sheet_name": "似然比检验结果摘要"}
    ]

    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        for table_info in table_headers:
            header = table_info["header"]
            sheet_name = table_info["sheet_name"]
            
            table_string = ""
            start_index = md_content.find(header)
            if start_index != -1:
                table_block = md_content[start_index:]
                lines = table_block.split('\n')
                
                table_lines = []
                if len(lines) > 1:
                    table_lines.append(lines[0]) # Header
                    table_lines.append(lines[1]) # Separator
                    for line in lines[2:]:
                        if line.strip().startswith('|'):
                            table_lines.append(line)
                        else:
                            break
                table_string = "\n".join(table_lines)
            
            if table_string:
                df = parse_markdown_table(table_string)
                if not df.empty:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f'表格已写入到Sheet: {sheet_name}')
                else:
                    print(f'警告: 未能从 {sheet_name} 提取到有效数据。')
            else:
                print(f'警告: 未能从 {sheet_name} 构建表格字符串。')

    print(f"\n所有表格已成功导出到 \"{output_excel_path}\"")

# --- 主执行逻辑 ---
if __name__ == "__main__":
    excel_file_path = "Glmer（Misinfogame加问卷星）.xlsx"
    md_report_path = "数据分析报告.md"
    output_excel_path = "数据分析报告_表格汇总.xlsx"

    # 1. 数据预处理
    preprocessed_data = preprocess_data(excel_file_path)
    preprocessed_data.to_pickle('preprocessed_data.pkl') # 保存预处理后的数据

    # 2. GLMER分析
    linear_model_fit, logistic_model_fit, vif_results, lrt_results = run_glmer_analysis(preprocessed_data.copy())

    # 3. 生成Markdown报告 (简化版，仅为确保文件存在)
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write("## 数据分析报告\n\n")
        f.write("### 线性混合效应模型摘要\n")
        f.write(linear_model_fit.summary().as_text())
        f.write("\n\n### 逻辑回归模型摘要\n")
        f.write(logistic_model_fit.summary().as_text())
        f.write("\n\n### 共线性诊断 (VIF)\n")
        f.write(vif_results.to_markdown(index=False))
        f.write("\n\n### 似然比检验结果摘要\n")
        f.write(lrt_results.to_markdown(index=False))

    # 4. 导出表格到Excel
    export_tables_to_excel(md_report_path, output_excel_path)

    print("所有分析和导出任务已完成。")

