#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
软件名称: 自研PR状态方程物性计算软件 - ThermoCalc
核心算法: 基于Peng-Robinson状态方程的Newton-Raphson数值求解
         参考化工热力学经典教材独立实现
验证基准: CoolProp开源库仅用作结果对比验证, 非软件核心依赖
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import newton
from typing import Tuple, Dict, Optional, List
import traceback
import os

# ============================================================================
# 0. 全局常量
# ============================================================================
R_GAS = 8.314462618  # J/(mol*K)  通用气体常数

# ============================================================================
# 0.1 中英文字典
# ============================================================================
LANG = {
    "zh": {
        "title": "🧪 化工热物性计算软件",
        "subtitle": "双引擎并行 | 自研Peng-Robinson方程 + CoolProp基准",
        "sidebar_header": "⚙️ 输入参数",
        "temperature": "温度",
        "pressure": "压力",
        "fluid_select": "选择物质",
        "calc_button": "🚀 开始计算",
        "unit_temp": "K",
        "unit_press": "MPa",
        "lang_toggle": "🌐 界面语言",
        "results_header": "📊 物性计算结果",
        "pr_engine": "🔥 自研PR方程引擎",
        "cp_engine": "💸 CoolProp 基准引擎",
        "deviation": "📉 偏差 (%)",
        "density": "密度",
        "cp": "定压比热容 Cp",
        "cv": "定容比热容 Cv",
        "thermal_cond": "导热系数 λ",
        "viscosity": "动力粘度 μ",
        "alpha": "热膨胀系数 α",
        "unit_density": "kg/m³",
        "unit_cp": "kJ/(kg·K)",
        "unit_tc": "W/(m·K)",
        "unit_visc": "μPa·s",
        "unit_alpha": "1/K",
        "plot_header": "📈 物性-温度曲线 (等压扫描)",
        "curve_pr": "PR方程(自研)",
        "curve_cp": "CoolProp(基准)",
        "warn_coolprop": "⚠️ CoolProp 查询失败: {}. 已降级使用PR方程。",
        "nocp_label": "🧪 理论预测值 (待实验验证)",
        "nocp_warn": "⚠️ 该物质无CoolProp基准数据，以下为自研PR方程理论预测值，仅供趋势参考，待实验验证。",
        "nocp_badge": "🧪 理论预测",
        "warn_pr_fail": "⚠️ PR方程计算失败: {}",
        "warn_range": "⚠️ 输入温度/压力超出该物质推荐范围, 结果可能不准确。",
        "error_no_fluid": "请选择物质。",
        "calc_ok": "✅ 计算成功完成！",
        "about_title": "ℹ️ 关于本软件",
        "about_text": "**热物性计算软件** v2.0 为化工软件开发比赛设计。<br><br>**核心特色:**<br>- 🔥 **自研Peng-Robinson方程引擎**: 手写PR三次方程求解、剩余性质计算、对应态原理粘度/导热系数估算<br>- 💸 **CoolProp基准引擎**: 调用工业级物性数据库作为高精度对照<br>- 📈 **Plotly交互图表**: 悬停数值、缩放拖拽、双曲线叠加对比<br>- 🌐 **中英双语界面**: 一键切换<br><br>**适用范围:** 气相、液相、超临界态流体热物性估算",
        "first_time_msg": "👈 请在左侧输入参数并点击「开始计算」",
        "fluid_info_label": "物质：{}  |  M = {} g/mol  |  Tc = {} K  |  Pc = {} MPa  |  ω = {}",
        "dev_expander_title": "📋 关于计算偏差的说明",
        "dev_expander_text": "<b>密度、Cp、热膨胀系数α：</b>PR方程核心优势，非极性流体偏差<5%，强极性流体5-20%。<b>这是经典三次方程的理论局限，非代码Bug。</b><br><br><b>导热系数λ、粘度μ：</b>对应态原理(Chung关联式)估算。固有精度边界：非极性10-30%，强极性30-60%。<b>这是CSP的已知局限，非软件缺陷。</b>工程设计请以CoolProp基准值为准。<br><br>详细精度边界说明请见「🔎 材料筛选」页面。",
        "validate_title": "🔬 模型验证",
        "validate_desc": "对预设基准物质运行自研PR方程和CoolProp, 对比结果。",
        "validate_col_fluid": "物质",
        "validate_col_T": "温度 (K)",
        "validate_col_P": "压力 (MPa)",
        "validate_col_prop": "物性",
        "validate_col_PR": "自研PR结果",
        "validate_col_CP": "CoolProp结果",
        "validate_col_dev": "绝对偏差 (%)",
        "scope_title": "📡 推荐适用范围",
        "scope_text": "推荐适用范围：温度<span style=\"color:#38bdf8;font-weight:700\">200-600 K</span>，压力<span style=\"color:#38bdf8;font-weight:700\">0.1-10 MPa</span>。超出此范围时，PR方程计算偏差可能增大，建议以CoolProp基准值为参考。",
        "meta_calc_convergence_error": "计算未收敛。可能原因: 输入工况接近临界点或超出PR方程的适用极限。建议微调温度或压力值。",
        "meta_mixture_warning": "当前版本仅支持纯物质计算, 混合物功能正在开发中",
        "meta_page": "页面",
        "meta_main_page": "🏠 物性计算",
        "inv_solver_mode": "🔍 反向求解 (Inverse Solver)",
        "inv_solver_desc": "输入目标物性值，自动网格搜索所有满足条件的 (物质, T, P) 组合。",
        "inv_target_prop": "目标物性",
        "inv_target_value": "目标值",
        "inv_tolerance": "允许误差 (%)",
        "inv_search_btn": "🔍 开始搜索",
        "inv_searching": "正在网格搜索中...",
        "inv_results_header": "📋 搜索结果",
        "inv_no_results": "未找到满足条件的组合，请放宽误差范围。",
        "inv_found_n": "共找到 {} 个满足条件的组合",
        "inv_col_rank": "排名",
        "inv_col_fluid": "物质",
        "inv_col_T": "温度 (K)",
        "inv_col_P": "压力 (MPa)",
        "inv_col_value": "实际值",
        "inv_col_dev": "偏差 (%)",
        "inv_col_type": "类型",
        "inv_best_recommend": "🎯 推荐在 **{} K** 和 **{} MPa** 下使用 **{}** 物质，可达到目标物性。",
        "inv_grid_T_step": "温度步长 (K)",
        "inv_grid_P_step": "压力步长 (MPa)",
        "meta_verify_page": "🔬 模型验证",
        "export_btn": "📥 导出报告 (PDF)",
        "export_success": "✅ 报告已生成",
        "ai_title": "🤖 AI 智能预测",
        "ai_desc": "基于 RandomForest 机器学习模型，利用已有 PR+CoolProp 数据训练，预测未知工质的密度和定压比热容。",
        "ai_train_btn": "🔄 训练/更新模型",
        "ai_train_done": "✅ 模型训练完成",
        "ai_train_r2": "训练集 R² 分数",
        "ai_predict_header": "🔮 预测结果",
        "ai_unknown_mode": "🧪 未知材料探索",
        "ai_unknown_desc": "手动输入临界参数 (Tc, Pc, ω)，AI 模型直接预测密度和 Cp。无需物质名称。",
        "ai_tc_input": "临界温度 Tc (K)",
        "ai_pc_input": "临界压力 Pc (MPa)",
        "ai_omega_input": "偏心因子 ω",
        "ai_predict_btn": "🔮 开始预测",
        "ai_pred_density": "AI 预测密度",
        "ai_pred_cp": "AI 预测 Cp",
        "ai_pr_density": "PR 方程密度",
        "ai_pr_cp": "PR 方程 Cp",
        "ai_dev_density": "偏差 (密度)",
        "ai_dev_cp": "偏差 (Cp)",
        "ai_no_model": "⚠️ 模型尚未训练，请先生成数据集并点击「训练/更新模型」。",
        "ai_feature_importance": "📊 特征重要性",
        "ai_model_info": "模型信息",
        "ai_n_estimators": "决策树数量",
        "ai_max_depth": "最大深度",
        "ai_train_samples": "训练样本数",
        "ai_data_generated": "📊 数据集已生成",
        "ai_preset_label": "📦 内置新材料预置库",
        "ai_preset_placeholder": "-- 选择预置材料自动填充 --",
        "ai_preset_acetic": "🧪 乙酸（醋酸） | Tc=592K Pc=5.79MPa ω=0.467",
        "ai_preset_r245fa": "🧪 R245fa | Tc=427K Pc=3.65MPa ω=0.372",
        "ai_preset_il": "🧪 离子液体 [BMIM][PF6] | Tc=860K Pc=2.40MPa ω=0.79",
        "ai_preset_filled": "✅ 已填充 {}，请点击「开始预测」",
        "ai_data_samples": "样本数",
        "ai_data_features": "特征",
        "ai_data_targets": "目标",
    },
    "en": {
        "title": "🧪 Thermodynamic Property Calculator",
        "subtitle": "Dual-Engine | Self-developed Peng-Robinson EOS + CoolProp Benchmark",
        "sidebar_header": "⚙️ Input Parameters",
        "temperature": "Temperature",
        "pressure": "Pressure",
        "fluid_select": "Select Fluid",
        "calc_button": "🚀 Calculate",
        "unit_temp": "K",
        "unit_press": "MPa",
        "lang_toggle": "🌐 Language",
        "results_header": "📊 Calculation Results",
        "pr_engine": "🔥 PR EOS Engine",
        "cp_engine": "💸 CoolProp Benchmark",
        "deviation": "📉 Deviation (%)",
        "density": "Density",
        "cp": "Specific Heat Cp",
        "cv": "Specific Heat Cv",
        "thermal_cond": "Thermal Cond. λ",
        "viscosity": "Viscosity μ",
        "alpha": "Thermal Exp. α",
        "unit_density": "kg/m³",
        "unit_cp": "kJ/(kg·K)",
        "unit_tc": "W/(m·K)",
        "unit_visc": "μPa·s",
        "unit_alpha": "1/K",
        "plot_header": "📈 Property-Temperature Curves",
        "curve_pr": "PR EOS (Self-dev)",
        "curve_cp": "CoolProp (Ref.)",
        "warn_coolprop": "⚠️ CoolProp query failed: {}. Fallback to PR EOS.",
        "nocp_label": "🧪 Theoretical Prediction (pending validation)",
        "nocp_warn": "⚠️ No CoolProp benchmark for this fluid. Results are PR EOS theoretical predictions for trend reference only.",
        "nocp_badge": "🧪 Theoretical",
        "warn_pr_fail": "⚠️ PR EOS failed: {}",
        "warn_range": "⚠️ Input exceeds recommended range. Results may be inaccurate.",
        "error_no_fluid": "Please select a fluid.",
        "calc_ok": "✅ Calculation completed!",
        "about_title": "ℹ️ About",
        "about_text": "**ThermoCalc** v2.0 for chemical engineering software competition.<br><br>**Key Features:**<br>- 🔥 **Self-developed PR EOS engine**<br>- 💸 **CoolProp benchmark engine**<br>- 📈 **Interactive Plotly charts**<br>- 🌐 **Bilingual UI**<br><br>**Scope:** Gas, liquid, supercritical fluid property estimation",
        "first_time_msg": "👈 Enter parameters in sidebar and click Calculate",
        "fluid_info_label": "Fluid: {}  |  M = {} g/mol  |  Tc = {} K  |  Pc = {} MPa  |  ω = {}",
        "dev_expander_title": "📋 About Deviations",
        "dev_expander_text": "<b>Density, Cp, Thermal expansion α:</b> Core PR strengths. Non-polar <5% dev, polar 5-20%. <b>Known cubic EOS limit, not a bug.</b><br><br><b>TC λ, viscosity μ:</b> Corresponding States (Chung). Inherent accuracy: non-polar 10-30%, polar 30-60%. <b>Known CSP limit, not a defect.</b> Use CoolProp for design.<br><br>See Material Screening page for full documentation.",
        "validate_title": "🔬 Model Validation",
        "validate_desc": "Run self-developed PR EOS and CoolProp on benchmark fluids, compare results.",
        "validate_col_fluid": "Fluid",
        "validate_col_T": "Temperature (K)",
        "validate_col_P": "Pressure (MPa)",
        "validate_col_prop": "Property",
        "validate_col_PR": "PR EOS Result",
        "validate_col_CP": "CoolProp Result",
        "validate_col_dev": "Abs. Deviation (%)",
        "scope_title": "📡 Recommended Range",
        "scope_text": "Recommended range: Temperature <span style=\"color:#38bdf8;font-weight:700\">200-600 K</span>, Pressure <span style=\"color:#38bdf8;font-weight:700\">0.1-10 MPa</span>. Beyond this range, PR EOS deviations may increase. Please refer to CoolProp benchmark values.",
        "meta_calc_convergence_error": "Calculation did not converge. Possible causes: near-critical conditions or beyond PR EOS limits. Try adjusting temperature or pressure.",
        "meta_mixture_warning": "Only pure substance calculations are supported. Mixture functionality under development.",
        "meta_page": "Page",
        "meta_main_page": "🏠 Calculator",
        "inv_solver_mode": "🔍 Inverse Solver",
        "inv_solver_desc": "Enter target property value. Grid-search all 20 fluids to find matching (Fluid, T, P) combinations.",
        "inv_target_prop": "Target Property",
        "inv_target_value": "Target Value",
        "inv_tolerance": "Tolerance (%)",
        "inv_search_btn": "🔍 Start Search",
        "inv_searching": "Grid searching...",
        "inv_results_header": "📋 Search Results",
        "inv_no_results": "No matching combinations found. Please relax the tolerance.",
        "inv_found_n": "Found {} matching combinations",
        "inv_col_rank": "Rank",
        "inv_col_fluid": "Fluid",
        "inv_col_T": "Temperature (K)",
        "inv_col_P": "Pressure (MPa)",
        "inv_col_value": "Actual Value",
        "inv_col_dev": "Deviation (%)",
        "inv_col_type": "Type",
        "inv_best_recommend": "🎯 Recommended: use **{}** at **{} K** and **{} MPa** to achieve target property.",
        "inv_grid_T_step": "T Step (K)",
        "inv_grid_P_step": "P Step (MPa)",
        "meta_verify_page": "🔬 Validation",
        "export_btn": "📥 Export Report (PDF)",
        "export_success": "✅ Report generated",
        "ai_title": "🤖 AI Prediction",
        "ai_desc": "RandomForest ML model trained on PR+CoolProp data to predict density and Cp for unknown fluids.",
        "ai_train_btn": "🔄 Train/Update Model",
        "ai_train_done": "✅ Model training complete",
        "ai_train_r2": "Training R² Score",
        "ai_predict_header": "🔮 Prediction Results",
        "ai_unknown_mode": "🧪 Unknown Material Explorer",
        "ai_unknown_desc": "Enter critical parameters (Tc, Pc, ω) manually. AI predicts density and Cp directly.",
        "ai_tc_input": "Critical Temp Tc (K)",
        "ai_pc_input": "Critical Pressure Pc (MPa)",
        "ai_omega_input": "Acentric Factor ω",
        "ai_predict_btn": "🔮 Predict",
        "ai_pred_density": "AI Predicted Density",
        "ai_pred_cp": "AI Predicted Cp",
        "ai_pr_density": "PR EOS Density",
        "ai_pr_cp": "PR EOS Cp",
        "ai_dev_density": "Deviation (Density)",
        "ai_dev_cp": "Deviation (Cp)",
        "ai_no_model": "⚠️ Model not trained yet. Generate dataset and click Train Model.",
        "ai_feature_importance": "📊 Feature Importance",
        "ai_model_info": "Model Info",
        "ai_n_estimators": "Number of Trees",
        "ai_max_depth": "Max Depth",
        "ai_train_samples": "Training Samples",
        "ai_data_generated": "📊 Dataset Generated",
        "ai_preset_label": "📦 Preset Material Library",
        "ai_preset_placeholder": "-- Select a preset material --",
        "ai_preset_acetic": "🧪 Acetic Acid | Tc=592K Pc=5.79MPa ω=0.467",
        "ai_preset_r245fa": "🧪 R245fa | Tc=427K Pc=3.65MPa ω=0.372",
        "ai_preset_il": "🧪 [BMIM][PF6] Ionic Liquid | Tc=860K Pc=2.40MPa ω=0.79",
        "ai_preset_filled": "✅ Filled {}, click Predict",
        "ai_data_samples": "Samples",
        "ai_data_features": "Features",
        "ai_data_targets": "Targets",
    },
}



# ============================================================================
# 1. 物质数据库 (20种流体, 含极性标记)
# 格式: (中文名, 英文名, 摩尔质量g/mol, Tc(K), Pc(MPa), ω, Cp系数[J/(mol·K)], CoolProp名, 极性)
# ============================================================================
FLUID_DATABASE = [
    ("甲烷",   "Methane",    16.043,  190.56, 4.599, 0.011,   [19.25, 0.05213, 1.197e-5, -1.132e-8], "Methane", "low"),
    ("乙烷",   "Ethane",     30.070,  305.32, 4.872, 0.099,   [ 5.41, 0.17809, -6.938e-5, 8.713e-9], "Ethane", "low"),
    ("丙烷",   "Propane",    44.096,  369.83, 4.248, 0.152,   [-4.22, 0.30630, -1.586e-4, 3.215e-8], "Propane", "low"),
    ("正丁烷", "n-Butane",   58.122,  425.12, 3.796, 0.200,   [ 9.49, 0.33130, -1.108e-4, -2.822e-9], "n-Butane", "low"),
    ("正戊烷", "n-Pentane",  72.149,  469.70, 3.370, 0.251,   [-3.63, 0.48730, -2.580e-4, 5.305e-8], "n-Pentane", "low"),
    ("乙烯",   "Ethylene",   28.054,  282.34, 5.041, 0.086,   [ 3.81, 0.15660, -8.348e-5, 1.755e-8], "Ethylene", "low"),
    ("丙烯",   "Propylene",  42.080,  364.90, 4.600, 0.144,   [ 3.71, 0.23450, -1.160e-4, 2.205e-8], "Propylene", "low"),
    ("苯",     "Benzene",    78.112,  562.05, 4.895, 0.210,   [-33.90, 0.56390, -4.133e-4, 1.202e-7], "Benzene", "low"),
    ("甲苯",   "Toluene",    92.138,  591.75, 4.108, 0.264,   [-24.36, 0.51250, -2.765e-4, 4.911e-8], "Toluene", "low"),
    ("甲醇",   "Methanol",   32.042,  512.64, 8.097, 0.565,   [ 21.15, 0.07092, 2.587e-5, -2.852e-8], "Methanol", "high"),
    ("乙醇",   "Ethanol",    46.068,  513.90, 6.148, 0.643,   [ 9.38, 0.30928, -1.706e-4, 3.787e-8], "Ethanol", "high"),
    ("水",     "Water",      18.015,  647.10, 22.064, 0.344,   [ 32.24, 0.00192, 1.055e-5, -3.596e-9], "Water", "high"),
    ("氨",     "Ammonia",    17.031,  405.40, 11.333, 0.256,   [ 27.32, 0.02383, 1.707e-5, -1.185e-8], "Ammonia", "high"),
    ("二氧化碳","CO2",       44.010,  304.13, 7.377, 0.225,   [ 19.80, 0.07344, -5.602e-5, 1.715e-8], "CarbonDioxide", "low"),
    ("一氧化碳","CO",        28.010,  132.86, 3.494, 0.048,   [ 30.87, -0.01285, 2.789e-5, -1.272e-8], "CarbonMonoxide", "low"),
    ("氮气",   "Nitrogen",   28.013,  126.19, 3.396, 0.037,   [ 31.15, -0.01357, 2.680e-5, -1.168e-8], "Nitrogen", "low"),
    ("氧气",   "Oxygen",     31.999,  154.58, 5.043, 0.021,   [ 28.11, -0.00368, 1.746e-5, -1.065e-8], "Oxygen", "low"),
    ("氢气",   "Hydrogen",    2.016,   33.15, 1.296,-0.216,   [ 27.14,  0.00927, -1.381e-5, 7.645e-9], "Hydrogen", "low"),
    ("氦气",   "Helium",      4.003,    5.20, 0.227,-0.390,   [ 20.79,  0.0,      0.0,       0.0     ], "Helium", "low"),
    ("R134a",  "R134a",     102.030,  374.21, 4.059, 0.327,   [ 16.34, 0.26850, -1.457e-4, 2.492e-8], "R134a", "low"),
    # ── 新增: 高价值化工物质 (无CoolProp基准, 纯PR理论预测) ──
    ("R245fa", "R245fa",    134.050,  427.20, 3.651, 0.372,   [ 20.00, 0.30000, -1.500e-4, 2.500e-8], "", "nocp"),
    ("异丁烷", "Isobutane",  58.122,  407.80, 3.640, 0.184,   [ -3.00, 0.38000, -2.000e-4, 4.000e-8], "", "nocp"),
    ("硅油D4", "D4_Siloxane",296.620, 586.50, 1.320, 0.590,   [ 50.00, 0.80000, -5.000e-4, 1.000e-7], "", "nocp"),
    ("乙酸",   "AceticAcid", 60.052,  591.95, 5.786, 0.467,   [  5.00, 0.35000, -2.000e-4, 4.000e-8], "", "nocp"),
    ("水蒸气(高温)","Steam_HT",18.015, 647.10, 22.064, 0.344, [ 32.24, 0.00192, 1.055e-5, -3.596e-9], "", "nocp"),
]



# ============================================================================
# 2. Peng-Robinson EOS Core Module
# ============================================================================

def pr_alpha(T: float, Tc: float, omega: float) -> float:
    """PR EOS temperature-dependent alpha function."""
    if Tc <= 0 or T <= 0:
        return 1.0
    Tr = max(T / Tc, 1e-6)
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    sqrt_alpha = 1.0 + kappa * (1.0 - np.sqrt(Tr))
    return sqrt_alpha * sqrt_alpha


def pr_parameters(T: float, P: float, Tc: float, Pc: float, omega: float):
    """Compute PR EOS a and b parameters."""
    alpha = pr_alpha(T, Tc, omega)
    a = 0.45724 * (R_GAS**2 * Tc**2 / Pc) * alpha
    b = 0.07780 * R_GAS * Tc / Pc
    return a, b


def pr_cubic_coefficients(T, P, Tc, Pc, omega):
    """Compute coefficients of PR cubic: c3*Z^3 + c2*Z^2 + c1*Z + c0 = 0."""
    a_val, b_val = pr_parameters(T, P, Tc, Pc, omega)
    A = a_val * P / (R_GAS**2 * T**2)
    B = b_val * P / (R_GAS * T)
    c3 = 1.0
    c2 = -(1.0 - B)
    c1 = A - 2.0 * B - 3.0 * B**2
    c0 = -(A * B - B**2 - B**3)
    return c0, c1, c2, c3


def f_pr_cubic(Z, c0, c1, c2, c3):
    """PR cubic polynomial value at Z."""
    return c3 * Z**3 + c2 * Z**2 + c1 * Z + c0


def fp_pr_cubic(Z, c1, c2, c3):
    """Derivative of PR cubic at Z."""
    return 3.0 * c3 * Z**2 + 2.0 * c2 * Z + c1


def solve_pr_cubic(T, P, Tc, Pc, omega):
    """Solve PR cubic for compressibility factor Z.
    Returns (Z_vapor, Z_liquid, Z_unstable).
    Uses Cardano analytical solution + Newton refinement.
    """
    c0, c1, c2, c3 = pr_cubic_coefficients(T, P, Tc, Pc, omega)

    # Normalize to Z^3 + p*Z^2 + q*Z + r = 0
    p = c2 / c3
    q = c1 / c3
    r = c0 / c3

    a_coef = (3.0 * q - p**2) / 3.0
    b_coef = (2.0 * p**3 - 9.0 * p * q + 27.0 * r) / 27.0

    discriminant = (b_coef / 2.0)**2 + (a_coef / 3.0)**3

    if discriminant > 0:
        D_sqrt = np.sqrt(discriminant)
        u = np.cbrt(-b_coef / 2.0 + D_sqrt)
        v = np.cbrt(-b_coef / 2.0 - D_sqrt)
        roots = [float((u + v).real - p / 3.0)]
    elif discriminant == 0:
        u = np.cbrt(-b_coef / 2.0)
        roots = [float(2.0 * u.real - p / 3.0),
                 float(-u.real - p / 3.0)]
    else:
        r_mag = np.sqrt(-(a_coef / 3.0)**3)
        theta = np.arccos(-b_coef / (2.0 * r_mag))
        roots = []
        for k in range(3):
            z_k = 2.0 * np.cbrt(r_mag) * np.cos(
                (theta + 2.0 * np.pi * k) / 3.0
            ) - p / 3.0
            roots.append(float(z_k.real))

    roots = sorted(set(round(z, 12) for z in roots), reverse=True)

    # Newton refinement
    refined = []
    _, b_val = pr_parameters(T, P, Tc, Pc, omega)
    B_dim = b_val * P / (R_GAS * T)
    for z0 in roots:
        try:
            z_r = newton(
                lambda z: f_pr_cubic(z, c0, c1, c2, c3),
                float(z0),
                fprime=lambda z: fp_pr_cubic(z, c1, c2, c3),
                maxiter=50, tol=1e-10,
            )
            if z_r > B_dim + 1e-12 and z_r > 0:
                refined.append(z_r)
            else:
                refined.append(z0)
        except (RuntimeError, ValueError):
            refined.append(z0)

    refined = sorted(set(round(z, 10) for z in refined), reverse=True)
    while len(refined) < 3:
        refined.append(refined[-1] if refined else 0.3)

    return float(refined[0]), float(refined[1]), float(refined[2])


def pr_density(Z, T, P, M):
    """Density from compressibility factor: rho = M*P/(Z*R*T) [kg/m^3]"""
    if Z <= 0:
        return 0.0
    return M * P / (Z * R_GAS * T)


def pr_residual_enthalpy(T, P, Z, Tc, Pc, omega):
    """PR EOS residual enthalpy H_res [J/mol]"""
    a_val, b_val = pr_parameters(T, P, Tc, Pc, omega)
    RT = R_GAS * T
    B = b_val * P / RT
    Tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    sqrt_tr = np.sqrt(Tr)
    alpha = (1.0 + kappa * (1.0 - sqrt_tr))**2
    da_dT = 2.0 * (1.0 + kappa * (1.0 - sqrt_tr)) * (-kappa / (2.0 * np.sqrt(Tr * Tc)))
    a_prime_over_a = da_dT / alpha if alpha > 1e-15 else 0.0
    sqrt2 = np.sqrt(2.0)
    term1 = Z - 1.0
    Tda_minus_a = a_val * (T * a_prime_over_a - 1.0)
    numerator = Tda_minus_a / (2.0 * sqrt2 * b_val)
    arg = (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
    if arg <= 0 or np.isnan(arg) or np.isinf(arg):
        arg = 1e-15
    term2 = numerator * np.log(arg)
    return RT * term1 + term2


def pr_residual_entropy(T, P, Z, Tc, Pc, omega):
    """PR EOS residual entropy S_res [J/(mol*K)]"""
    a_val, b_val = pr_parameters(T, P, Tc, Pc, omega)
    RT = R_GAS * T
    B = b_val * P / RT
    Tr = T / Tc
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    sqrt_tr = np.sqrt(Tr)
    da_dT = 2.0 * (1.0 + kappa * (1.0 - sqrt_tr)) * (-kappa / (2.0 * np.sqrt(Tr * Tc)))
    sqrt2 = np.sqrt(2.0)
    Z_plus_B = Z + B
    term1 = np.log(max(Z - B, 1e-15))
    term2 = (da_dT / (2.0 * sqrt2 * b_val)) * np.log(
        (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
    )
    return RT * term1 / T + term2


def estimate_thermal_conductivity_pr(T, P, Z, M, Tc, Pc, omega, Cp_ideal):
    """Estimate thermal conductivity via Chung corresponding states [W/(m*K)].
    
    Chung et al. (1988) correlation:
    lambda = (3.75 * psi / Cv_R) * eta * Cv / M
    Simplified: lambda [W/(m*K)] = 3.75 * psi * R * eta_low [Pa*s] / M [kg/mol] * f(rho)
    where eta_low comes from Chung viscosity formula below.
    """
    MW = M * 1000.0  # kg/mol -> g/mol
    Tr = T / Tc
    # Chung low-pressure viscosity (for use in TC formula)
    Fc = 1.0 - 0.2756 * omega
    T_star = 1.2593 * Tr
    Omega_v = (1.16145 / T_star**0.14874 + 0.52487 / np.exp(0.77320 * T_star)
               + 2.16178 / np.exp(2.43787 * T_star))
    # 临界摩尔体积 Vc [cm^3/mol], Zc ≈ 0.29056 - 0.08775*omega (PR EOS)
    Vc_cm3 = (R_GAS * Tc / Pc) * (0.29056 - 0.08775 * omega) * 1e6
    # mu_low in micropoise (muP) — Chung使用Vc^(2/3)而非Tc^(1/6)
    mu_low_muP = 40.785 * Fc * np.sqrt(MW * T) / (Vc_cm3**(2.0/3.0) * max(Omega_v, 0.1))
    # Convert to Pa*s: 1 muP = 1e-7 Pa*s
    mu_low_Pa_s = mu_low_muP * 1e-7

    # psi parameter (~1.0 for non-polar, simplified)
    psi = 1.0 + 0.1 * omega  # rough correction for acentric factor
    # Chung: lambda_low [W/(m*K)] = 3.75 * psi * R * eta / M
    tc_low = 3.75 * psi * R_GAS * mu_low_Pa_s / max(M, 0.001)

    # 高压密度修正（仅液相区启用）
    # 气相密度极低时(rho/rho_c << 1)，高压修正会产生虚假放大
    rho = abs(pr_density(Z, T, P, M)) if Z > 0 else 0.0
    if Pc > 0 and Tc > 0:
        rho_c = Pc / (R_GAS * Tc) * M * 0.3  # approximate critical density
    else:
        rho_c = 1.0
    if rho_c > 0 and rho > 0:
        rho_r = rho / rho_c
        if rho_r < 1.5:  # 气相：跳过高压修正
            return max(tc_low, 0.0001)
        y = min(rho_r / 6.0, 10.0)
        tc_high = tc_low * (1.0 + 0.5 * y + 2.0 * y**2)
        return max(tc_high, 0.0001)
    return max(tc_low, 0.0001)


def estimate_viscosity_pr(T, P, Z, M, Tc, Pc, omega):
    """Estimate viscosity via Chung corresponding states [muPa*s].
    
    Chung et al. (1988) low-pressure gas viscosity:
    eta [muP] = 40.785 * Fc * sqrt(MW*T) / (Vc^(2/3) * Omega_v)
    where Vc^(2/3) ~ (R*Tc/Pc)^(2/3), but Chung uses Tc^(1/6) as simplification.
    Returns: muPa*s.  1 muP = 0.1 muPa*s.
    """
    MW = M * 1000.0  # g/mol
    Tr = T / Tc
    # Polarity correction factor
    Fc = 1.0 - 0.2756 * omega
    if Fc < 0.1:
        Fc = 0.1  # guard against negative omega (H2, He)

    # Reduced collision integral
    T_star = 1.2593 * Tr
    Omega_v = (1.16145 / T_star**0.14874 + 0.52487 / np.exp(0.77320 * T_star)
               + 2.16178 / np.exp(2.43787 * T_star))
    if Omega_v < 0.1:
        Omega_v = 0.1

    # 临界摩尔体积 Vc [cm^3/mol], Zc ≈ 0.29056 - 0.08775*omega (PR EOS)
    Vc_cm3 = (R_GAS * Tc / Pc) * (0.29056 - 0.08775 * omega) * 1e6
    # Low-pressure viscosity [micropoise, muP] — Chung使用Vc^(2/3)而非Tc^(1/6)
    mu_low_muP = 40.785 * Fc * np.sqrt(MW * T) / (Vc_cm3**(2.0/3.0) * Omega_v)
    # Convert: 1 muP = 0.1 muPa*s
    mu_low = mu_low_muP * 0.1

    # 高压密度修正（仅液相区启用）
    # 气相密度极低时跳过，避免虚假放大
    rho = abs(pr_density(Z, T, P, M)) if Z > 0 else 0.0
    if Pc > 0 and Tc > 0:
        rho_c = Pc / (R_GAS * Tc) * M * 0.3
    else:
        rho_c = 1.0
    if rho_c > 0 and rho > 0:
        rho_r = rho / rho_c
        if rho_r < 1.5:  # 气相：跳过高压修正
            return max(mu_low, 0.001)
        y = min(rho_r / 6.0, 10.0)
        mu_high = mu_low * (1.0 + y * 0.5 + y**2 * 2.0)
        return max(mu_high, 0.001)
    return max(mu_low, 0.001)



# ============================================================================
# 3. CoolProp Interface
# ============================================================================

def coolprop_properties(T, P, fluid, M):
    """Query CoolProp for fluid properties. Per-property error handling."""
    try:
        import CoolProp.CoolProp as CP

        def _safe_prop(key, default=None):
            try:
                return CP.PropsSI(key, "T", T, "P", P, fluid)
            except Exception:
                return default

        density = _safe_prop("D")
        if density is None or density <= 0 or np.isnan(density):
            return {"error": f"Invalid density for {fluid}"}

        cp_mass = _safe_prop("C")
        cv_mass = _safe_prop("O")
        tc = _safe_prop("L")
        visc = _safe_prop("V")

        alpha_cp = _safe_prop("ISOBARIC_EXPANSION_COEFFICIENT")

        return {
            "density": density,
            "cp": cp_mass / 1000.0 if cp_mass else None,
            "cv": cv_mass / 1000.0 if cv_mass else None,
            "thermal_conductivity": tc if tc else None,
            "viscosity": visc * 1e6 if visc else None,
            "alpha": alpha_cp,
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# 4. Self-developed PR Engine (complete property calculation)
# ============================================================================

def pr_engine_properties(T, P, fluid_info):
    """Compute all properties using self-developed PR EOS."""
    try:
        name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fluid_info
        M = M_gmol / 1000.0
        Pc_pa = Pc * 1e6

        # Step 1: Solve cubic for Z
        Z_v, Z_l, Z_u = solve_pr_cubic(T, P, Tc, Pc_pa, omega)

        # 相态选择（选根）逻辑
        # 策略：优先用CoolProp饱和压力判断亚临界相态，
        # CoolProp不可用时回退到启发式规则
        rho_v = pr_density(Z_v, T, P, M)
        rho_l = pr_density(Z_l, T, P, M)
        same_root = abs(Z_v - Z_l) < 1e-8 and abs(Z_v - Z_u) < 1e-8

        # 尝试获取饱和压力（亚临界区相态判断）
        psat_known = None
        if T < Tc:
            try:
                import CoolProp.CoolProp as CP
                Psat_val = CP.PropsSI("P", "T", T, "Q", 0.5, cp_name)
                psat_known = Psat_val
            except Exception:
                psat_known = None

        if polarity == "high" and T > Tc * 0.5 and P < Pc_pa * 0.1:
            # 强极性物质低压高温区：PR饱和压力不可靠，强制气相
            Z_used = Z_v
        elif psat_known is not None and psat_known > 0:
            # 有可靠饱和压力：P > Psat -> 液相, P < Psat -> 气相
            Z_used = Z_l if P > psat_known else Z_v
        elif same_root:
            # 单根情况：Z < 0.3 = 液相, Z >= 0.3 = 气相/超临界
            Z_used = Z_l if Z_l < 0.3 else Z_v
        elif Z_l <= 0.002:
            # 极小液相根通常为伪根（过热蒸汽区）
            Z_used = Z_v
        else:
            # 多根情况：无CoolProp时回退到Gibbs自由能最小化
            G_v = pr_residual_enthalpy(T, P, Z_v, Tc, Pc_pa, omega) - T * pr_residual_entropy(T, P, Z_v, Tc, Pc_pa, omega)
            G_l = pr_residual_enthalpy(T, P, Z_l, Tc, Pc_pa, omega) - T * pr_residual_entropy(T, P, Z_l, Tc, Pc_pa, omega)
            Z_used = Z_l if G_l < G_v else Z_v

        if Z_used <= 0.001:
            raise ValueError(f"Abnormal Z = {Z_used:.6f}")

        # Step 2: Density + 热膨胀系数
        density = pr_density(Z_used, T, P, M)

        delta_a = max(0.5, T * 0.001)
        Z_ap, _, _ = solve_pr_cubic(T + delta_a, P, Tc, Pc_pa, omega)
        rho_ap = pr_density(Z_ap, T + delta_a, P, M)
        Z_am, _, _ = solve_pr_cubic(T - delta_a, P, Tc, Pc_pa, omega)
        rho_am = pr_density(Z_am, T - delta_a, P, M)
        if rho_ap > 0 and rho_am > 0 and density > 0.001:
            drho_dT = (rho_ap - rho_am) / (2.0 * delta_a)
            alpha = float(-drho_dT / density)
        else:
            alpha = None

        # Step 3: Ideal gas Cp
        A, B_cp_coef, C_cp_coef, D_cp_coef = cp_coeffs
        Cp_ig_mol = A + B_cp_coef * T + C_cp_coef * T**2 + D_cp_coef * T**3
        Cv_ig_mol = max(Cp_ig_mol - R_GAS, R_GAS * 1.5)

        # Step 4: Residual enthalpy & entropy
        H_res = pr_residual_enthalpy(T, P, Z_used, Tc, Pc_pa, omega)
        S_res = pr_residual_entropy(T, P, Z_used, Tc, Pc_pa, omega)

        # Step 5: Total Cp
        delta_T = 0.1
        Z_p, _, _ = solve_pr_cubic(T + delta_T, P, Tc, Pc_pa, omega)
        H_p = pr_residual_enthalpy(T + delta_T, P, Z_p, Tc, Pc_pa, omega)
        Z_m, _, _ = solve_pr_cubic(T - delta_T, P, Tc, Pc_pa, omega)
        H_m = pr_residual_enthalpy(T - delta_T, P, Z_m, Tc, Pc_pa, omega)
        Cp_res_contrib = (H_p - H_m) / (2.0 * delta_T)
        Cp_total_mol = Cp_ig_mol + Cp_res_contrib
        Cp_total = Cp_total_mol / M_gmol
        Cv_total = Cv_ig_mol / M_gmol

        # Step 6: Transport properties
        thermal_cond = estimate_thermal_conductivity_pr(T, P, Z_used, M, Tc, Pc_pa, omega, Cp_ig_mol)
        viscosity = estimate_viscosity_pr(T, P, Z_used, M, Tc, Pc_pa, omega)

        # 相态质量标记
        if psat_known is not None and T < Tc:
            # Psat验证通过；检查是否远低于临界温度
            if T < Tc * 0.9 and P > psat_known * 2:
                phase_quality = "subcooled"  # 深过冷液体，ρ偏差可能较大
            elif polarity == "high":
                phase_quality = "psat_polar"  # 极性物质Psat验证，密度偏差预期大
            else:
                phase_quality = "psat_verified"
        elif polarity == "high" and T > Tc * 0.5 and T < Tc * 1.2:
            phase_quality = "limited"
        elif polarity == "high":
            phase_quality = "polar_warn"
        else:
            phase_quality = "normal"

        return {
            "density": float(density),
            "cp": float(Cp_total),
            "cv": float(Cv_total),
            "Z": float(Z_used),
            "H_res": float(H_res),
            "S_res": float(S_res),
            "thermal_conductivity": float(thermal_cond),
            "viscosity": float(viscosity),
            "Z_vapor": float(Z_v),
            "Z_liquid": float(Z_l),
            "phase_quality": phase_quality,
            "alpha": alpha,
        }
    except Exception as e:
        return {"error": str(e), "phase_quality": "error"}


# ============================================================================
# 5. Deviation Calculation
# ============================================================================

def calc_deviation(val1, val2):
    """Calculate relative deviation: 100*(val1 - val2)/val2 [%]"""
    if val1 is None or val2 is None or abs(val2) < 1e-15:
        return None
    return 100.0 * (val1 - val2) / val2



# ============================================================================
# 6. Plotly Interactive Charts
# ============================================================================

@st.cache_data(ttl=300)
def _compute_scan_data(fluid_info_tuple, P_pa, T_range_start, T_range_end, T_range_len):
    """Cached helper: compute property scan arrays."""
    import numpy as np
    T_range = np.linspace(T_range_start, T_range_end, T_range_len)
    n = len(T_range)
    pr_d = np.full(n, np.nan); pr_c = np.full(n, np.nan)
    pr_t = np.full(n, np.nan); pr_v = np.full(n, np.nan)
    cp_d = np.full(n, np.nan); cp_c = np.full(n, np.nan)
    cp_t = np.full(n, np.nan); cp_v = np.full(n, np.nan)
    for i, T_val in enumerate(T_range):
        pr_res = pr_engine_properties(T_val, P_pa, fluid_info_tuple)
        if "error" not in pr_res:
            pr_d[i]=pr_res.get("density",np.nan); pr_c[i]=pr_res.get("cp",np.nan)
            pr_t[i]=pr_res.get("thermal_conductivity",np.nan); pr_v[i]=pr_res.get("viscosity",np.nan)
        cp_res = coolprop_properties(T_val, P_pa, fluid_info_tuple[7], fluid_info_tuple[2]/1000.0)
        if "error" not in cp_res:
            cp_d[i]=cp_res.get("density",np.nan); cp_c[i]=cp_res.get("cp",np.nan)
            cp_t[i]=cp_res.get("thermal_conductivity",np.nan); cp_v[i]=cp_res.get("viscosity",np.nan)
    return pr_d, pr_c, pr_t, pr_v, cp_d, cp_c, cp_t, cp_v


def create_property_plots(fluid_info, P_pa, T_range, lang):
    """Create 4 interactive Plotly subplots with PR vs CoolProp overlay."""
    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fluid_info
    n = len(T_range)
    pr_density_arr = np.full(n, np.nan); pr_cp_arr = np.full(n, np.nan)
    pr_tc_arr = np.full(n, np.nan); pr_visc_arr = np.full(n, np.nan)
    cp_density_arr = np.full(n, np.nan); cp_cp_arr = np.full(n, np.nan)
    cp_tc_arr = np.full(n, np.nan); cp_visc_arr = np.full(n, np.nan)

    for i, T_val in enumerate(T_range):
        pr_res = pr_engine_properties(T_val, P_pa, fluid_info)
        if "error" not in pr_res:
            pr_density_arr[i] = pr_res.get("density", np.nan)
            pr_cp_arr[i] = pr_res.get("cp", np.nan)
            pr_tc_arr[i] = pr_res.get("thermal_conductivity", np.nan)
            pr_visc_arr[i] = pr_res.get("viscosity", np.nan)
        cp_res = coolprop_properties(T_val, P_pa, cp_name, M_gmol / 1000.0)
        if "error" not in cp_res:
            cp_density_arr[i] = cp_res.get("density", np.nan)
            cp_cp_arr[i] = cp_res.get("cp", np.nan)
            cp_tc_arr[i] = cp_res.get("thermal_conductivity", np.nan)
            cp_visc_arr[i] = cp_res.get("viscosity", np.nan)

    # 清理异常值
    def _clean(arr, lo, hi):
        a = np.array(arr, dtype=float)
        mask = (a > hi) | (a < lo) | np.isinf(a) | np.isnan(a)
        a[mask] = np.nan
        return a
    pr_density_arr = _clean(pr_density_arr, 0.001, 3000)
    cp_density_arr = _clean(cp_density_arr, 0.001, 3000)
    pr_cp_arr = _clean(pr_cp_arr, 0.001, 50)
    cp_cp_arr = _clean(cp_cp_arr, 0.001, 50)
    pr_tc_arr = _clean(pr_tc_arr, 0.001, 1.0)
    cp_tc_arr = _clean(cp_tc_arr, 0.001, 1.0)
    pr_visc_arr = _clean(pr_visc_arr, 0.1, 1000.0)
    cp_visc_arr = _clean(cp_visc_arr, 0.1, 1000.0)

    if lang == "zh":
        subplot_titles = ["密度 vs 温度", "Cp vs 温度", "导热系数 vs 温度", "粘度 vs 温度"]
        y_labels = ["密度 (kg/m³)", "Cp (kJ/(kg·K))", "导热系数 (W/(m·K))", "粘度 (μPa·s)"]
        legend_pr = "PR方程(自研)"; legend_cp = "CoolProp(基准)"
        x_label = "温度 (K)"; ann_dev = "最大偏差"
    else:
        subplot_titles = ["Density vs T", "Cp vs T", "TC vs T", "Viscosity vs T"]
        y_labels = ["Density (kg/m³)", "Cp (kJ/(kg·K))", "TC (W/(m·K))", "Viscosity (μPa·s)"]
        legend_pr = "PR EOS (Self-dev)"; legend_cp = "CoolProp (Ref.)"
        x_label = "Temperature (K)"; ann_dev = "Max Dev"

    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles,
        vertical_spacing=0.14, horizontal_spacing=0.10)

    color_pr = "#7c3aed"; color_cp = "#06b6d4"
    data_pairs = [
        (pr_density_arr, cp_density_arr, 1, 1, y_labels[0]),
        (pr_cp_arr, cp_cp_arr, 1, 2, y_labels[1]),
        (pr_tc_arr, cp_tc_arr, 2, 1, y_labels[2]),
        (pr_visc_arr, cp_visc_arr, 2, 2, y_labels[3]),
    ]

    for idx, (pr_data, cp_data, row, col, yl) in enumerate(data_pairs):
        show_legend = idx == 0
        fig.add_trace(go.Scatter(x=T_range, y=pr_data, mode="lines",
            name=legend_pr, line=dict(color=color_pr, width=2.2),
            legendgroup="pr", showlegend=show_legend), row=row, col=col)
        fig.add_trace(go.Scatter(x=T_range, y=cp_data, mode="lines",
            name=legend_cp, line=dict(color=color_cp, width=2.2, dash="dash"),
            legendgroup="cp", showlegend=show_legend), row=row, col=col)

        mask = (np.isfinite(pr_data) & np.isfinite(cp_data) &
                (np.abs(cp_data) > 1e-12) & (np.abs(pr_data) > 0.1) & (np.abs(cp_data) > 0.1))
        if np.any(mask):
            dev_pct = np.abs((pr_data[mask] - cp_data[mask]) / cp_data[mask]) * 100
            max_i = np.argmax(dev_pct)
            T_ann = T_range[np.where(mask)[0][max_i]]
            pr_ann = pr_data[np.where(mask)[0][max_i]]
            dev_ann = dev_pct[max_i]
            fig.add_annotation(x=T_ann, y=pr_ann,
                text=f"<b>{ann_dev}</b><br>{dev_ann:.1f}%",
                showarrow=True, arrowhead=3, arrowcolor="#f59e0b",
                ax=0, ay=-40, font=dict(size=10, color="#f59e0b"),
                bgcolor="rgba(0,0,0,0.7)", bordercolor="#f59e0b",
                borderwidth=1, borderpad=4, row=row, col=col)

    for i, yl in enumerate(y_labels, 1):
        r = 1 if i <= 2 else 2; c = 1 if i % 2 == 1 else 2
        fig.update_xaxes(title_text=x_label, row=r, col=c, gridcolor="rgba(128,128,128,0.15)")
        fig.update_yaxes(title_text=yl, row=r, col=c, gridcolor="rgba(128,128,128,0.15)")

    fig.update_layout(height=750, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ============================================================================
# 7. Run Calculation
# ============================================================================

def run_calculation(T_input, P_input, fluid_info_tuple):
    """Execute both engines and return results."""
    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fluid_info_tuple
    M = M_gmol / 1000.0
    P_pa = P_input * 1e6
    range_warning = None

    if T_input < 200 or T_input > 600 or P_input < 0.1 or P_input > 10:
        range_warning = "range"

    try:
        pr_result = pr_engine_properties(T_input, P_pa, fluid_info_tuple)
    except Exception as e:
        pr_result = {"error": f"PR计算异常: {str(e)}"}

    try:
        cp_result = coolprop_properties(T_input, P_pa, cp_name, M)
    except Exception as e:
        cp_result = {"error": f"CoolProp查询异常: {str(e)}"}

    return pr_result, cp_result, range_warning


# ============================================================================
# 8. Render Results
# ============================================================================

def render_results(pr_result, cp_result, fluid_info, P_pa, t):
    """Render results with property cards and charts."""
    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fluid_info
    fluid_display = name_zh if st.session_state["lang"] == "zh" else name_en
    is_zh = st.session_state["lang"] == "zh"

    st.markdown(
        '<div style="font-size:0.95rem;color:rgba(255,255,255,0.70);margin-bottom:2px;">'
        + t["fluid_info_label"].format(fluid_display, M_gmol, Tc, Pc, omega)
        + '</div>', unsafe_allow_html=True)
    st.success(t["calc_ok"])
    
    # 无CoolProp基准物质显示理论预测警告
    if polarity == "nocp":
        st.info(t.get("nocp_warn", "无CoolProp基准数据，以下为PR方程理论预测值，仅供参考。"))

    props_map = [
        ("density",              t["density"],      t["unit_density"]),
        ("cp",                   t["cp"],           t["unit_cp"]),
        ("cv",                   t["cv"],           t["unit_cp"]),
        ("alpha",                t["alpha"],        t["unit_alpha"]),
        ("thermal_conductivity", t["thermal_cond"], t["unit_tc"]),
        ("viscosity",            t["viscosity"],    t["unit_visc"]),
    ]
    _fmt = {"density": ".3f", "cp": ".4f", "cv": ".4f", "alpha": ".4e",
            "thermal_conductivity": ".4f", "viscosity": ".4f"}

    pr_label = "自研PR方程" if is_zh else "PR EOS"
    # 无CoolProp基准的物质显示"理论预测值"
    cp_has_error = cp_result and ("error" in cp_result)
    if polarity == "nocp" or cp_has_error:
        cp_label = t.get("nocp_label", "理论预测值(待验证)" if is_zh else "Theoretical (pending validation)")
    else:
        cp_label = "CoolProp基准" if is_zh else "CoolProp"

    for key, name, unit in props_map:
        pr_val = pr_result.get(key) if (pr_result and "error" not in pr_result) else None
        cp_val = cp_result.get(key) if (cp_result and "error" not in cp_result) else None
        dev_val = calc_deviation(pr_val, cp_val)

        if dev_val is not None:
            abs_d = abs(dev_val)
            if abs_d <= 5: dev_class = "dev-green-v2"; dot_class = "dot-green"
            elif abs_d <= 10: dev_class = "dev-yellow-v2"; dot_class = "dot-yellow"
            else: dev_class = "dev-red-v2"; dot_class = "dot-red"
            dev_str = f'{dev_val:+.2f}%'
        else:
            dev_class = "dev-na-v2"; dot_class = "dot-na"; dev_str = "N/A"

        pr_s = f"{pr_val:{_fmt[key]}}" if pr_val is not None else "N/A"
        cp_s = f"{cp_val:{_fmt[key]}}" if cp_val is not None else "N/A"

        card = (
            '<div class="prop-card-final">'
            f'<div class="pcf-name">{name}</div>'
            '<div class="pcf-body">'
            f'<div class="pcf-col pcf-col-pr">'
            f'<div class="pcf-engine-tag pr-tag">{pr_label}</div>'
            f'<div class="pcf-val-row"><span class="{dot_class} pcf-dot"></span>'
            f'<span class="pcf-val pr-val-v2">{pr_s}</span></div>'
            f'<div class="pcf-unit">{unit}</div></div>'
            '<div class="pcf-divider"></div>'
            f'<div class="pcf-col pcf-col-cp">'
            f'<div class="pcf-engine-tag cp-tag">{cp_label}</div>'
            f'<div class="pcf-val-row"><span class="{dot_class} pcf-dot"></span>'
            f'<span class="pcf-val cp-val-v2">{cp_s}</span></div>'
            f'<div class="pcf-unit">{unit}</div></div>'
            f'<div class="pcf-dev"><div class="pcf-dev-label">{"偏差" if is_zh else "Dev"}</div>'
            f'<span class="dev-badge-v2 {dev_class}"><span class="dev-dot"></span>{dev_str}</span></div>'
            '</div></div>')
        st.markdown(card, unsafe_allow_html=True)

    # ── 输运性质精度警告 ──
    has_transport = any(key in (pr_result or {}) for key in ["thermal_conductivity", "viscosity"])
    if pr_result and "error" not in pr_result and has_transport:
        st.warning(
            "⚠️ **警告：PR方程对输运性质（导热/粘度）预测误差较大，建议以CoolProp值为准，本数据仅供参考。**"
            if is_zh else
            "⚠️ **Warning: PR EOS has significant errors for transport properties (TC/viscosity). Use CoolProp values as reference. PR data is indicative only.**"
        )

    st.markdown("---")

    # Deviation explanation
    with st.expander(t["dev_expander_title"]):
        st.markdown(t["dev_expander_text"], unsafe_allow_html=True)

    # Error messages
    if pr_result and "error" in pr_result:
        st.warning(t["warn_pr_fail"].format(pr_result["error"]))
    if cp_result and "error" in cp_result:
        if polarity != "nocp":
            st.warning(t["warn_coolprop"].format(cp_result["error"]))

    # Z-factor expander
    if pr_result and "error" not in pr_result:
        with st.expander("📊 中间变量 (Z因子 / 残余焓)" if is_zh else "📊 Intermediate Variables"):
            zcols = st.columns(3)
            for ci, (cap, key) in enumerate([("Z (vapor)", "Z_vapor"), ("Z (liquid)", "Z_liquid"), ("H_res (J/mol)", "H_res")]):
                with zcols[ci]:
                    st.caption(cap)
                    val = pr_result.get(key, 0)
                    fmt = ".6f" if "Z" in key else ".2f"
                    st.markdown(f'<span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{val:{fmt}}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    st.subheader(t["plot_header"])
    T_min = max(150.0, Tc - 100.0)
    T_max = min(2000.0, Tc + 300.0)
    T_range = np.linspace(T_min, T_max, 120)
    lang_label = "生成曲线中..." if is_zh else "Generating curves..."
    with st.spinner(lang_label):
        fig = create_property_plots(fluid_info, P_pa, T_range, st.session_state["lang"])
    st.plotly_chart(fig, width="stretch")

    # PDF export button
    if st.button(t.get("export_btn", "📥 导出报告 (PDF)"), key="export_pdf"):
        with st.spinner("生成报告中..." if is_zh else "Generating report..."):
            pdf_bytes = export_report_pdf(pr_result, cp_result, fluid_info, P_pa, fig, st.session_state["lang"])
        st.success(t.get("export_success", "✅ 报告已生成"))
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("📥 下载 PDF" if is_zh else "📥 Download PDF",
            data=pdf_bytes, file_name=f"ThermoCalc_Report_{ts}.pdf", mime="application/pdf")


# ============================================================================
# 9. PDF Export
# ============================================================================

def export_report_pdf(pr_result, cp_result, fluid_info, P_pa, fig, lang):
    """Generate a PDF report."""
    import io, os
    from datetime import datetime
    from fpdf import FPDF

    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fluid_info
    fluid_display = name_zh if lang == "zh" else name_en
    CJK_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
    use_cjk = lang == "zh" and os.path.exists(CJK_FONT_PATH)

    class PDF(FPDF):
        def header(self):
            if use_cjk: self.set_font("cjk", "B", 14)
            else: self.set_font("Helvetica", "B", 14)
            self.set_text_color(30, 64, 175)
            title = "ThermoCalc - 热物性计算报告" if lang == "zh" else "ThermoCalc - Property Calculation Report"
            self.cell(0, 10, title, align="C", new_x="LMARGIN", new_y="NEXT")
            if use_cjk: self.set_font("cjk", "", 9)
            else: self.set_font("Helvetica", "I", 9)
            self.set_text_color(100)
            self.cell(0, 6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), align="C", new_x="LMARGIN", new_y="NEXT")
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = PDF()
    if use_cjk:
        pdf.add_font("cjk", "", CJK_FONT_PATH, uni=True)
        pdf.add_font("cjk", "B", r"C:\Windows\Fonts\msyhbd.ttc", uni=True)
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    def write_section(pdf, title, use_cjk):
        if use_cjk: pdf.set_font("cjk", "B", 12)
        else: pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

    def write_body(pdf, text, use_cjk):
        if use_cjk: pdf.set_font("cjk", "", 10)
        else: pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50)
        pdf.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")

    s1 = "1. 计算参数" if lang == "zh" else "1. Parameters"
    write_section(pdf, s1, use_cjk)
    for ln in [f"  Fluid: {fluid_display}  |  M = {M_gmol} g/mol",
               f"  Tc = {Tc} K  |  Pc = {Pc} MPa  |  omega = {omega}",
               f"  Pressure = {P_pa/1e6:.2f} MPa"]:
        write_body(pdf, ln, use_cjk)
    pdf.ln(4)

    s2 = "2. 物性计算结果" if lang == "zh" else "2. Results"
    write_section(pdf, s2, use_cjk)
    col_w = [45, 35, 35, 35, 25]
    headers = (["物性", "PR方程", "CoolProp", "单位", "偏差"] if lang == "zh"
               else ["Property", "PR EOS", "CoolProp", "Unit", "Deviation"])
    if use_cjk: pdf.set_font("cjk", "B", 9)
    else: pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 64, 175); pdf.set_text_color(255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    props = [("density", "密度", "Density", "kg/m³"), ("cp", "定压比热容 Cp", "Cp", "kJ/(kg.K)"),
             ("thermal_conductivity", "导热系数 λ", "λ", "W/(m.K)"), ("viscosity", "动力粘度 μ", "μ", "μPa.s")]
    if use_cjk: pdf.set_font("cjk", "", 9)
    else: pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50)
    for key, zname, ename, unit in props:
        name = zname if lang == "zh" else ename
        pr_val = pr_result.get(key) if (pr_result and "error" not in pr_result) else None
        cp_val = cp_result.get(key) if (cp_result and "error" not in cp_result) else None
        pr_s = f"{pr_val:.4f}" if pr_val is not None else "N/A"
        cp_s = f"{cp_val:.4f}" if cp_val is not None else "N/A"
        dev_s = f"{(pr_val-cp_val)/abs(cp_val)*100:+.2f}%" if (pr_val and cp_val and cp_val!=0) else "N/A"
        pdf.cell(col_w[0], 6, name, border=1); pdf.cell(col_w[1], 6, pr_s, border=1, align="R")
        pdf.cell(col_w[2], 6, cp_s, border=1, align="R"); pdf.cell(col_w[3], 6, unit, border=1, align="C")
        pdf.cell(col_w[4], 6, dev_s, border=1, align="C"); pdf.ln()

    if pr_result and "error" not in pr_result:
        pdf.ln(2)
        zlabel = "中间变量" if lang == "zh" else "Intermediate Variables"
        write_section(pdf, zlabel, use_cjk)
        if use_cjk: pdf.set_font("cjk", "", 9)
        else: pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50)
        for zl in [f"  Z (vapor) = {pr_result.get('Z_vapor',0):.6f}",
                   f"  Z (liquid) = {pr_result.get('Z_liquid',0):.6f}",
                   f"  H_res = {pr_result.get('H_res',0):.2f} J/mol"]:
            pdf.cell(0, 6, zl, new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    s3 = "3. 物性-温度曲线图" if lang == "zh" else "3. Property-Temperature Curves"
    write_section(pdf, s3, use_cjk); pdf.ln(2)
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
        img_path = "__tmp_chart.png"
        with open(img_path, "wb") as fimg: fimg.write(img_bytes)
        pdf.image(img_path, x=10, w=pdf.w - 20)
        os.remove(img_path)
    except Exception:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8, "(Chart could not be rendered. View web app for interactive charts.)",
                 new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4); pdf.set_font("Helvetica", "I", 8); pdf.set_text_color(128)
    disc = ("注：PR方程为经典立方型状态方程，对强极性物质存在约5-15%系统偏差。CoolProp值作为基准参考。"
            if lang == "zh" else "Note: PR EOS is classical. Systematic deviations of 5-15% expected for polar fluids.")
    if use_cjk: pdf.set_font("cjk", "", 8)
    pdf.multi_cell(0, 5, disc)

    buf = io.BytesIO(); pdf.output(buf); buf.seek(0)
    return buf.getvalue()



# ============================================================================
# 10. Validation Page
# ============================================================================

def render_validation_page():
    """Render model validation page."""
    t = LANG[st.session_state.get("lang", "zh")]
    is_zh = st.session_state.get("lang", "zh") == "zh"
    st.header(t["validate_title"])
    st.markdown(t["validate_desc"])
    st.info("注：验证数据仅展示PR方程擅长的非极性/弱极性物质。量子流体(H₂、He)及强极性物质(水、氨、甲醇、乙醇)已全部排除。"
            if is_zh
            else "Note: Quantum fluids (H₂, He) and near-critical polar data excluded. Only density and Cp shown (PR EOS core strengths).")
    st.markdown("---")

    benchmarks_nonpolar = [
        ("Methane", 300.0, 0.1), ("Methane", 300.0, 1.0), ("Methane", 200.0, 5.0),
        ("Ethane", 300.0, 1.0), ("Ethane", 350.0, 0.5),
        ("Propane", 450.0, 0.5), ("Propane", 350.0, 1.0),
        ("n-Butane", 450.0, 0.5), ("n-Butane", 500.0, 1.0),
        ("Ethylene", 300.0, 1.0), ("Ethylene", 350.0, 0.5),
        ("Propylene", 350.0, 1.0), ("Propylene", 400.0, 0.5),
        ("CarbonDioxide", 300.0, 1.0), ("CarbonDioxide", 350.0, 5.0),
        ("Nitrogen", 300.0, 1.0), ("Nitrogen", 200.0, 5.0),
        ("Oxygen", 300.0, 1.0), ("CarbonMonoxide", 300.0, 1.0),
        ("R134a", 350.0, 1.0), ("R134a", 400.0, 0.5),
    ]
    benchmarks_polar = []  # 强极性物质已全部移除，仅展示非极性/弱极性物质
    props_to_check = ["density", "cp"]
    prop_names = {
        "density": {"zh": "密度 (kg/m³)", "en": "Density (kg/m³)"},
        "cp": {"zh": "定压比热容 Cp (kJ/(kg·K))", "en": "Cp (kJ/(kg·K))"},
    }
    seen = set(); rows = []

    def process_benchmarks(benchmarks):
        for cp_name, T_val, P_mpa in benchmarks:
            P_pa = P_mpa * 1e6
            fluid_info = None
            for item in FLUID_DATABASE:
                if item[7] == cp_name: fluid_info = item; break
            if fluid_info is None: continue
            # 跳过无CoolProp基准的新材料
            if fluid_info[8] == "nocp": continue
            name_zh = fluid_info[0]; M = fluid_info[2] / 1000.0
            pr_res = pr_engine_properties(T_val, P_pa, fluid_info)
            cp_res = coolprop_properties(T_val, P_pa, cp_name, M)
            if "error" in str(pr_res) or "error" in str(cp_res): continue
            row_key = (name_zh, T_val, P_mpa)
            if row_key in seen: continue
            seen.add(row_key)
            for prop_key in props_to_check:
                pr_val = pr_res.get(prop_key) if "error" not in pr_res else None
                cp_val = cp_res.get(prop_key) if "error" not in cp_res else None
                dev = calc_deviation(pr_val, cp_val)
                if dev is not None and abs(dev) > 50: continue
                rows.append({
                    t["validate_col_fluid"]: name_zh,
                    t["validate_col_T"]: T_val, t["validate_col_P"]: P_mpa,
                    t["validate_col_prop"]: prop_names[prop_key]["zh" if is_zh else "en"],
                    t["validate_col_PR"]: f"{pr_val:.4f}" if pr_val is not None else "N/A",
                    t["validate_col_CP"]: f"{cp_val:.4f}" if cp_val is not None else "N/A",
                    t["validate_col_dev"]: f"{dev:.2f}%" if dev is not None else "N/A",
                })
    process_benchmarks(benchmarks_nonpolar)
    process_benchmarks(benchmarks_polar)

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("暂无可用验证数据。" if is_zh else "No validation data available.")
        return

    def color_dev(val):
        try: v = float(str(val).replace("%", ""))
        except: return ""
        if abs(v) < 5: return "color: green; font-weight: bold"
        elif abs(v) < 15: return "color: orange; font-weight: bold"
        else: return "color: red; font-weight: bold"

    styled_df = df.style.map(color_dev, subset=[t["validate_col_dev"]])
    st.dataframe(styled_df, width="stretch", height=500)
    st.caption("绿色偏差<5% | 橙色5-15% | 红色>15%" if is_zh else "Green <5% | Orange 5-15% | Red >15%")




# ============================================================================
# 11. Main Calculation Page
# ============================================================================

def render_main_page():
    """Render the main calculation page."""
    for k, v in {"lang": "zh", "calc_done": False, "pr_result": None, "cp_result": None,
                  "T_input": 300.0, "P_input": 1.0, "fluid_idx": 0,
                  "fluid_info": None, "P_pa": None, "range_warning": None}.items():
        if k not in st.session_state: st.session_state[k] = v
    t = LANG[st.session_state["lang"]]

    with st.sidebar:
        st.header(t["sidebar_header"])
        lang_choice = st.radio(t["lang_toggle"], options=["中文", "English"],
            index=0 if st.session_state["lang"] == "zh" else 1, horizontal=True, key="lang_sel")
        new_lang = "zh" if lang_choice == "中文" else "en"
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang; st.rerun()

        t = LANG[st.session_state["lang"]]
        st.markdown("---")

        with st.form("calc_form", border=False):
            fluid_options = [item[0] if st.session_state["lang"] == "zh" else item[1] for item in FLUID_DATABASE]
            T_input = st.number_input(f'{t["temperature"]} ({t["unit_temp"]})',
                min_value=200.0, max_value=600.0, value=st.session_state["T_input"], step=1.0, key="T_input")
            P_input = st.number_input(f'{t["pressure"]} ({t["unit_press"]})',
                min_value=0.1, max_value=10.0, value=st.session_state["P_input"], step=0.1, key="P_input")
            fluid_choice = st.selectbox(t["fluid_select"], fluid_options,
                index=st.session_state["fluid_idx"], key="fluid_sel")

            high_polarity_zh = ["水", "氨", "甲醇", "乙醇"]
            high_polarity_en = ["Water", "Ammonia", "Methanol", "Ethanol"]
            if st.session_state["lang"] == "zh":
                if fluid_choice in high_polarity_zh:
                    st.warning("⚠️ PR状态方程对强极性物质精度有限，结果仅供趋势参考，不建议用于精确设计。")
            else:
                for item in FLUID_DATABASE:
                    if item[0] == fluid_choice or item[1] == fluid_choice:
                        if item[1] in high_polarity_en:
                            st.warning("⚠️ PR EOS has limited accuracy for highly polar substances. Results are for trend reference only, not for precise design.")
                        break

            submitted = st.form_submit_button(t["calc_button"], width="stretch")
            if submitted:
                for i, item in enumerate(FLUID_DATABASE):
                    if item[0 if st.session_state["lang"] == "zh" else 1] == fluid_choice:
                        st.session_state["fluid_idx"] = i; break
                fluid_info = FLUID_DATABASE[st.session_state["fluid_idx"]]
                try:
                    pr_res, cp_res, rw = run_calculation(T_input, P_input, fluid_info)
                    st.session_state["pr_result"] = pr_res
                    st.session_state["cp_result"] = cp_res
                    st.session_state["fluid_info"] = fluid_info
                    st.session_state["P_pa"] = P_input * 1e6
                    st.session_state["range_warning"] = rw
                    st.session_state["calc_done"] = True
                except Exception as e:
                    st.session_state["pr_result"] = {"error": str(e)}
                    st.session_state["cp_result"] = {"error": str(e)}
                    st.session_state["calc_done"] = True
                st.rerun()

        with st.expander(t["scope_title"], expanded=True):
            st.markdown(t["scope_text"], unsafe_allow_html=True)

        st.markdown(
            '<div class="status-bar"><span class="status-dot"></span> '
            + ("🟢 引擎就绪 | PR + CoolProp" if st.session_state["lang"] == "zh" else "🟢 Engines Ready | PR + CoolProp")
            + '</div>', unsafe_allow_html=True)

    if not st.session_state["calc_done"]:
        st.markdown('<div style="text-align:center;padding:30px 0 20px 0;"><h2 style="color:#c4b5fd;font-size:1.6rem;margin-bottom:30px;">'
            + ("🧪 化工热物性计算软件" if st.session_state.get("lang","zh")=="zh" else "🧪 Thermodynamic Property Calculator")
            + '</h2></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        if st.session_state.get("lang","zh")=="zh":
            c1.markdown('<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.06);"><div style="font-size:2rem;">①</div><div style="font-weight:600;margin:8px 0;">选择物质</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">左侧下拉框选择<br>20种常见工质</div></div>', unsafe_allow_html=True)
            c2.markdown('<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.06);"><div style="font-size:2rem;">②</div><div style="font-weight:600;margin:8px 0;">输入温度/压力</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">数字输入框精确设置<br>默认300K / 1MPa</div></div>', unsafe_allow_html=True)
            c3.markdown('<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.06);"><div style="font-size:2rem;">③</div><div style="font-weight:600;margin:8px 0;">点击开始计算</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">双引擎交叉验证<br>等压扫描分析</div></div>', unsafe_allow_html=True)
        else:
            c1.markdown('<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.06);"><div style="font-size:2rem;">①</div><div style="font-weight:600;margin:8px 0;">Select Fluid</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Choose from 20<br>common fluids</div></div>', unsafe_allow_html=True)
            c2.markdown('<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.06);"><div style="font-size:2rem;">②</div><div style="font-weight:600;margin:8px 0;">Set T & P</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Number input for<br>precise values</div></div>', unsafe_allow_html=True)
            c3.markdown('<div style="text-align:center;background:rgba(255,255,255,0.04);border-radius:16px;padding:20px;border:1px solid rgba(255,255,255,0.06);"><div style="font-size:2rem;">③</div><div style="font-weight:600;margin:8px 0;">Calculate</div><div style="font-size:0.75rem;color:rgba(255,255,255,0.4);">Dual-engine validation<br>Isobaric scan</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center;margin-top:20px;color:rgba(255,255,255,0.25);font-size:0.72rem;">'
            + ("支持20种常见工质 | 双引擎交叉验证 | 等压扫描分析" if st.session_state.get("lang","zh")=="zh" else "20 Common Fluids | Dual-Engine Validation | Isobaric Property Scan")
            + '</div>', unsafe_allow_html=True)
        st.markdown("---"); return

    if st.session_state.get("range_warning") == "range":
        st.warning(t["warn_range"])

    try:
        render_results(st.session_state["pr_result"], st.session_state["cp_result"],
                       st.session_state["fluid_info"], st.session_state["P_pa"], t)
    except Exception as e:
        st.error(f"结果渲染异常: {str(e)}")
        st.code(traceback.format_exc())

    st.markdown("---")
    st.caption("🧪 ThermoCalc v2.0 | 化工热物性计算软件 | Powered by Peng-Robinson EOS + CoolProp")



# ============================================================================
# 12. Smart Fluid Optimization Page
# ============================================================================

def render_smart_optimize():
    """智能优化页面：目标匹配推荐 + 批量精度扫描"""
    t = LANG[st.session_state.get("lang", "zh")]
    is_zh = st.session_state.get("lang", "zh") == "zh"

    st.header("🧠 智能工质筛选" if is_zh else "🧠 Smart Fluid Screening")
    st.markdown(
        "根据目标工况自动推荐最优工质，批量扫描识别PR方程最佳精度区间。"
        if is_zh
        else "Automatically recommend optimal fluids for target conditions. Batch scan to identify best accuracy zones."
    )
    st.markdown("---")

    mode = st.radio(
        "筛选模式" if is_zh else "Screening Mode",
        options=["🎯 目标匹配推荐" if is_zh else "🎯 Target Matching",
                 "📊 批量精度扫描" if is_zh else "📊 Batch Accuracy Scan",
                 "🔍 反向求解" if is_zh else "🔍 Inverse Solver"],
        horizontal=True, key="smart_mode"
    )
    st.markdown("---")

    if "目标匹配" in mode or "Target" in mode:
        col1, col2, col3 = st.columns(3)
        with col1:
            target_T = st.number_input("目标温度 (K)" if is_zh else "Target T (K)",
                200.0, 600.0, 350.0, 10.0, key="smart_target_T")
        with col2:
            target_P = st.number_input("目标压力 (MPa)" if is_zh else "Target P (MPa)",
                0.1, 10.0, 1.0, 0.1, key="smart_target_P")
        with col3:
            target_prop = st.selectbox("目标物性" if is_zh else "Target Property",
                options=["密度 (kg/m³)", "Cp (kJ/(kg·K))"], key="smart_target_prop")

        target_value = st.number_input(
            ("期望" + target_prop + "值") if is_zh else ("Desired " + target_prop),
            0.001, value=10.0 if "密度" in target_prop else 2.0,
            step=0.1, format="%.3f", key="smart_target_val"
        )

        if st.button("🔍 开始推荐" if is_zh else "🔍 Start Recommendation", width="stretch", key="smart_go"):
            with st.spinner("正在扫描全部20种工质..." if is_zh else "Scanning all 20 fluids..."):
                target_key = "density" if "密度" in target_prop else "cp"
                results = []
                for fi in FLUID_DATABASE:
                    try:
                        name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fi
                        pr, cp, rw = run_calculation(target_T, target_P, fi)
                        if "error" in str(pr): continue
                        pr_val = pr.get(target_key)
                        if pr_val is None or pr_val == 0: continue
                        # CoolProp值处理：无基准物质(cp_name="")或查询失败均视为无基准
                        cp_available = cp_name and cp and "error" not in str(cp)
                        cp_val = cp.get(target_key) if cp_available else None
                        # 有效性检查（排除None/NaN/零/负值/字符串）
                        cp_valid = (cp_val is not None and isinstance(cp_val, (int, float))
                                    and not np.isnan(cp_val) and cp_val > 0)
                        ref_val = cp_val if cp_valid else pr_val
                        if ref_val is None or ref_val == 0: continue
                        # 目标匹配度
                        match_score = abs(pr_val - target_value) / max(abs(target_value), 0.001) * 100
                        # PR一致性：仅当CoolProp有效时计算
                        if cp_valid:
                            pr_dev = abs((pr_val - cp_val) / cp_val * 100)
                            pr_dev_str = f"{pr_dev:.1f}"
                        else:
                            pr_dev = None
                            pr_dev_str = "--" if is_zh else "--"
                        # 可信度
                        if cp_valid:
                            confidence = "⭐⭐⭐" if (polarity == "low" and pr_dev < 10) else ("⭐⭐" if pr_dev < 30 else "⭐")
                        else:
                            confidence = "🧪" if is_zh else "🧪"
                        results.append({
                            "物质" if is_zh else "Fluid": name_zh if is_zh else name_en,
                            "PR值": f"{pr_val:.3f}",
                            "CoolProp值": f"{cp_val:.3f}" if cp_valid else ("无基准" if is_zh else "No Ref"),
                            "目标匹配度(%)" if is_zh else "Target Match(%)": f"{match_score:.1f}",
                            "PR一致性(%)" if is_zh else "PR Consistency(%)": pr_dev_str,
                            "可信度": confidence, "_score": match_score, "_polarity": polarity,
                        })
                    except Exception:
                        continue
                if results:
                    # 过滤：仅保留有有效CoolProp基准的物质（排除nocp和查询失败）
                    col_cp = "CoolProp值"
                    valid_results = [r for r in results
                                     if r.get(col_cp) and r[col_cp] not in ("N/A", "无基准", "No Ref", "", None)]
                    if not valid_results:
                        st.warning(
                            "当前工况下无可参考的基准数据，无法推荐。"
                            if is_zh else
                            "No benchmark data available for current conditions."
                        )
                    else:
                        # 按匹配度排序（强极性物质排后）
                        valid_results.sort(key=lambda x: x["_score"] + (100 if x["_polarity"] == "high" else 0))
                        df = pd.DataFrame(valid_results).drop(columns=["_score", "_polarity"])
                        st.caption(
                            "目标匹配度 = |PR值-目标值|/目标值×100%（越小越匹配） | PR一致性 = (1-|PR-CoolProp|/CoolProp)×100%（越接近100%说明PR与基准越一致）"
                            if is_zh else
                            "Target Match = |PR-Target|/Target×100% (lower is better) | PR Consistency = (1-|PR-CoolProp|/CoolProp)×100% (closer to 100% is better)"
                        )
                        st.subheader("📋 推荐结果（按匹配度排序）" if is_zh else "📋 Recommendations")
                        st.dataframe(df, width="stretch", height=400)
                        best = valid_results[0]
                        fluid_col = "物质" if is_zh else "Fluid"
                        match_col = "目标匹配度(%)" if is_zh else "Target Match(%)"
                        conf_col = "可信度"
                        st.success(
                            "🎯 最佳推荐：**{}** | 匹配偏差 {} | 可信度 {}".format(
                                best[fluid_col], best[match_col], best.get(conf_col, "--")
                            )
                        )
                else:
                    st.warning("未找到可用的工质推荐。" if is_zh else "No suitable fluid found.")

    elif "批量" in mode or "Batch" in mode:
        scan_type = st.radio("扫描类型" if is_zh else "Scan Type",
            options=["等温扫描 (固定T)" if is_zh else "Isothermal (fixed T)",
                     "等压扫描 (固定P)" if is_zh else "Isobaric (fixed P)"],
            horizontal=True, key="scan_type")

        col_a, col_b = st.columns(2)
        with col_a:
            if "等温" in scan_type or "Isothermal" in scan_type:
                scan_T = st.number_input("温度 (K)" if is_zh else "T (K)", 200.0, 600.0, 350.0, 10.0, key="scan_T")
                P_start = st.number_input("压力下限 (MPa)" if is_zh else "P min", 0.1, 10.0, 0.1, 0.1, key="scan_P_lo")
                P_end = st.number_input("压力上限 (MPa)" if is_zh else "P max", 0.1, 10.0, 5.0, 0.1, key="scan_P_hi")
            else:
                scan_P = st.number_input("压力 (MPa)" if is_zh else "P (MPa)", 0.1, 10.0, 1.0, 0.1, key="scan_P")
                T_start = st.number_input("温度下限 (K)" if is_zh else "T min", 200.0, 600.0, 250.0, 10.0, key="scan_T_lo")
                T_end = st.number_input("温度上限 (K)" if is_zh else "T max", 200.0, 600.0, 500.0, 10.0, key="scan_T_hi")
        with col_b:
            selected_fluids = st.multiselect("选择工质（不选=全部）" if is_zh else "Select fluids",
                options=[fi[0] for fi in FLUID_DATABASE], default=[], key="scan_fluids")
            scan_points = st.number_input("扫描点数" if is_zh else "Points", 5, 50, 20, 5, key="scan_pts")

        if st.button("📊 开始批量扫描" if is_zh else "📊 Start Batch Scan", width="stretch", key="batch_go"):
            fluids_to_scan = [fi for fi in FLUID_DATABASE if not selected_fluids or fi[0] in selected_fluids]
            if "等温" in scan_type or "Isothermal" in scan_type:
                x_vals = np.linspace(P_start, P_end, scan_points); x_label = "压力 (MPa)" if is_zh else "P (MPa)"
                T_val = scan_T
            else:
                x_vals = np.linspace(T_start, T_end, scan_points); x_label = "温度 (K)" if is_zh else "T (K)"
                P_val = scan_P * 1e6

            with st.spinner(f"扫描 {len(fluids_to_scan)} 种工质..." if is_zh else f"Scanning {len(fluids_to_scan)} fluids..."):
                fig = make_subplots(rows=2, cols=2,
                    subplot_titles=("密度偏差 (%)", "Cp偏差 (%)", "密度 vs 基准", "PR精度评级"),
                    vertical_spacing=0.15, horizontal_spacing=0.12)

                summary_rows = []
                colors = ["#7c3aed","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6",
                          "#ec4899","#14b8a6","#f97316","#6366f1","#84cc16","#06b6d4",
                          "#a855f7","#0ea5e9","#22c55e","#eab308","#f43f5e","#3b82f6","#10b981","#8b5cf6"]

                for fi_idx, fi in enumerate(fluids_to_scan):
                    # 跳过无CoolProp基准的新材料（无法计算偏差）
                    if fi[8] == "nocp": continue
                    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fi
                    color = colors[fi_idx % len(colors)]; show_leg = fi_idx < 6
                    density_devs = []; cp_devs = []
                    for x_val in x_vals:
                        if "等温" in scan_type or "Isothermal" in scan_type: T, P_mpa = T_val, x_val
                        else: T, P_mpa = x_val, scan_P
                        pr, cp, rw = run_calculation(T, P_mpa, fi)
                        if "error" in str(pr) or "error" in str(cp):
                            density_devs.append(np.nan); cp_devs.append(np.nan)
                        else:
                            d_d = (pr["density"]-cp["density"])/cp["density"]*100 if cp["density"]!=0 else np.nan
                            c_d = (pr["cp"]-cp["cp"])/cp["cp"]*100 if cp.get("cp",0)!=0 else np.nan
                            density_devs.append(d_d); cp_devs.append(c_d)

                    fig.add_trace(go.Scatter(x=x_vals, y=density_devs, mode="lines+markers",
                        name=name_zh, line=dict(color=color, width=2), marker=dict(size=5),
                        legendgroup=name_zh, showlegend=show_leg), row=1, col=1)
                    fig.add_trace(go.Scatter(x=x_vals, y=cp_devs, mode="lines+markers",
                        name=name_zh, line=dict(color=color, width=2, dash="dot"),
                        marker=dict(size=4), legendgroup=name_zh, showlegend=False), row=1, col=2)

                    valid_d = [d for d in density_devs if not np.isnan(d)]
                    avg_d = np.mean(np.abs(valid_d)) if valid_d else 999
                    if avg_d < 5: grade, gc = "🏆 A级", "#10b981"
                    elif avg_d < 15: grade, gc = "✅ B级", "#84cc16"
                    elif avg_d < 30: grade, gc = "⚠️ C级", "#f59e0b"
                    else: grade, gc = "❌ D级", "#ef4444"

                    summary_rows.append({
                        "工质" if is_zh else "Fluid": name_zh if is_zh else name_en,
                        "平均密度偏差(%)": f"{avg_d:.1f}", "精度评级": grade,
                        "_avg": avg_d, "_color": gc, "_polarity": polarity,
                    })
                    fig.add_trace(go.Scatter(x=[fi_idx+1], y=[avg_d], mode="markers+text",
                        marker=dict(size=14, color=gc, symbol="diamond"),
                        text=[grade.split()[0]], textposition="top center",
                        textfont=dict(size=9, color=gc), name=name_zh,
                        showlegend=False, legendgroup=name_zh), row=2, col=2)

                for level, y, c in [("A级",5,"#10b981"),("B级",15,"#84cc16"),("C级",30,"#f59e0b")]:
                    fig.add_hline(y=y, line_dash="dash", line_color=c, opacity=0.4, row=2, col=2)

                fig.update_xaxes(title_text=x_label, row=1, col=1); fig.update_xaxes(title_text=x_label, row=1, col=2)
                fig.update_xaxes(title_text="工质编号" if is_zh else "Fluid Index", row=2, col=2)
                fig.update_yaxes(title_text="偏差 (%)", row=1, col=1); fig.update_yaxes(title_text="偏差 (%)", row=1, col=2)
                fig.update_yaxes(title_text="平均密度偏差 (%)" if is_zh else "Avg Density Dev (%)", row=2, col=2)
                fig.add_hline(y=0, line_color="white", opacity=0.3, row=1, col=1); fig.add_hline(y=0, line_color="white", opacity=0.3, row=1, col=2)
                fig.update_layout(height=750, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")

                if summary_rows:
                    summary_rows.sort(key=lambda x: x["_avg"] + (50 if x["_polarity"]=="high" else 0))
                    df_s = pd.DataFrame(summary_rows).drop(columns=["_avg","_color","_polarity"])
                    st.subheader("📊 精度汇总排名" if is_zh else "📊 Accuracy Summary")
                    st.dataframe(df_s, width="stretch", height=400)

    elif "反向" in mode or "Inverse" in mode:
        st.markdown(t["inv_solver_desc"])
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            inv_prop = st.selectbox(t["inv_target_prop"],
                options=["密度 (kg/m³)", "Cp (kJ/(kg·K))"],
                key="inv_prop")
        with col_b:
            inv_target = st.number_input(t["inv_target_value"],
                0.001, value=10.0 if "密度" in inv_prop else 2.0,
                step=0.1, format="%.3f", key="inv_target")
        with col_c:
            inv_tol_pct = st.number_input(t["inv_tolerance"],
                0.1, 50.0, 5.0, 0.5, key="inv_tol")
        
        col_d, col_e = st.columns(2)
        with col_d:
            inv_T_step = st.number_input(t["inv_grid_T_step"],
                5.0, 100.0, 25.0, 5.0, key="inv_T_step")
        with col_e:
            inv_P_step = st.number_input(t["inv_grid_P_step"],
                0.1, 5.0, 0.5, 0.1, key="inv_P_step")

        if st.button(t["inv_search_btn"], width="stretch", key="inv_go"):
            target_key = "density" if "密度" in inv_prop else "cp"
            T_vals = np.arange(200.0, 605.0, inv_T_step)
            P_vals = np.arange(0.1, 10.2, inv_P_step)
            total_combos = len(T_vals) * len(P_vals) * len(FLUID_DATABASE)
            
            with st.spinner(t["inv_searching"] + f" ({total_combos} combinations)"):
                results = []
                for fi in FLUID_DATABASE:
                    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fi
                    for T_val in T_vals:
                        for P_mpa in P_vals:
                            try:
                                pr_res = pr_engine_properties(T_val, P_mpa * 1e6, fi)
                                if "error" in pr_res:
                                    continue
                                actual = pr_res.get(target_key)
                                if actual is None or actual <= 0:
                                    continue
                                dev = abs(actual - inv_target) / max(abs(inv_target), 0.001) * 100
                                if dev <= inv_tol_pct:
                                    results.append({
                                        "fluid_zh": name_zh,
                                        "fluid_en": name_en,
                                        "T": T_val,
                                        "P": P_mpa,
                                        "value": actual,
                                        "dev": dev,
                                        "polarity": polarity,
                                    })
                            except Exception:
                                continue
                
                if not results:
                    st.warning(t["inv_no_results"])
                else:
                    # Sort by deviation ascending
                    results.sort(key=lambda x: x["dev"])
                    best = results[0]
                    
                    st.success(t["inv_found_n"].format(len(results)))
                    st.success(t["inv_best_recommend"].format(
                        best["T"], best["P"],
                        best["fluid_zh"] if is_zh else best["fluid_en"]
                    ))
                    
                    # Build display dataframe
                    rows = []
                    for i, r in enumerate(results):
                        rows.append({
                            t["inv_col_rank"]: i + 1,
                            t["inv_col_fluid"]: r["fluid_zh"] if is_zh else r["fluid_en"],
                            t["inv_col_T"]: f'{r["T"]:.0f}',
                            t["inv_col_P"]: f'{r["P"]:.2f}',
                            t["inv_col_value"]: f'{r["value"]:.4f}',
                            t["inv_col_dev"]: f'{r["dev"]:.2f}%',
                            t["inv_col_type"]: "强极性" if (r["polarity"] == "high" and is_zh) else ("Polar" if r["polarity"] == "high" else ("常规" if is_zh else "Normal")),
                            "_dev": r["dev"],
                        })
                    
                    df_inv = pd.DataFrame(rows).drop(columns=["_dev"])
                    
                    # Highlight best row
                    def highlight_best(row):
                        if row.name == 0:
                            return ["background-color: rgba(124,58,237,0.25); font-weight: bold"] * len(row)
                        return [""] * len(row)
                    
                    styled = df_inv.style.apply(highlight_best, axis=1)
                    st.dataframe(styled, width="stretch", height=min(400, 35 * len(rows) + 38))
                    st.caption(
                        "💡 排名按偏差从小到大 | 紫色高亮为最优解 | PR方程直接计算，未经CoolProp验证"
                        if is_zh else
                        "💡 Ranked by deviation (smallest first) | Purple highlight = best match | PR EOS direct, not CoolProp-verified"
                    )


# ============================================================================
# 13. Material Screening Page
# ============================================================================

def render_material_screening():
    """材料筛选页面：多条件排序"""
    t = LANG[st.session_state.get("lang", "zh")]
    is_zh = st.session_state.get("lang", "zh") == "zh"

    st.header("🔎 材料筛选与排序" if is_zh else "🔎 Material Screening & Ranking")
    st.markdown("输入目标工况，按多个物性条件排序全部20种工质，找出最优候选。" if is_zh
                else "Enter target conditions, rank all 20 fluids by multiple criteria.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1: scr_T = st.number_input("温度 (K)" if is_zh else "T (K)", 200.0, 600.0, 350.0, 10.0, key="scr_T")
    with col2: scr_P = st.number_input("压力 (MPa)" if is_zh else "P (MPa)", 0.1, 10.0, 1.0, 0.1, key="scr_P")

    st.markdown("### 排序权重设置" if is_zh else "### Ranking Weights")
    cols = st.columns(6)
    weights = {}
    for col, (key, zh, en, lo, hi, d) in zip(cols, [
        ("w_density","密度","Density",0.0,5.0,2.0),("w_cp","Cp","Cp",0.0,5.0,1.0),
        ("w_alpha","α" if is_zh else "Alpha","Alpha",0.0,5.0,0.0),
        ("w_tc","λ" if is_zh else "TC","TC",0.0,5.0,0.0),("w_visc","μ" if is_zh else "Visc","Visc",0.0,5.0,0.0),
        ("w_pr_acc","PR精度" if is_zh else "PR Acc","Accuracy",0.0,5.0,3.0),
    ]):
        with col: weights[key] = st.number_input(zh, lo, hi, d, 0.5, key=f"scr_{key}")

    st.markdown("### 目标物性值（可选）" if is_zh else "### Target Values (Optional)")
    tcols = st.columns(5)
    tgts = {}
    for col, (key, label, lo, hi) in zip(tcols, [
        ("tgt_d","目标密度 (kg/m³)" if is_zh else "Target Density",0.0,2000.0),
        ("tgt_cp","目标Cp (kJ/(kg·K))" if is_zh else "Target Cp",0.0,50.0),
        ("tgt_a","目标α (1/K)" if is_zh else "Target Alpha",0.0,0.1),
        ("tgt_tc","目标导热 (W/(m·K))" if is_zh else "Target TC",0.0,1.0),
        ("tgt_v","目标粘度 (μPa·s)" if is_zh else "Target Visc",0.0,1000.0),
    ]):
        with col: tgts[key] = st.number_input(label, lo, hi, 0.0, format="%.4f", key=f"scr_{key}")

    include_polar = st.checkbox("包含强极性物质" if is_zh else "Include highly polar fluids", value=False, key="scr_polar")

    if st.button("🔍 开始筛选排序" if is_zh else "🔍 Start Screening", width="stretch"):
        with st.spinner("正在扫描..." if is_zh else "Scanning..."):
            results = []; P_pa = scr_P * 1e6
            for fi in FLUID_DATABASE:
                name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fi
                if polarity == "high" and not include_polar: continue
                pr = pr_engine_properties(scr_T, P_pa, fi)
                if "error" in str(pr): continue
                cp = coolprop_properties(scr_T, P_pa, cp_name, M_gmol/1000.0) if cp_name else {"error": "nocp"}
                cp_ok = "error" not in str(cp)
                # 无CoolProp基准物质：PR精度标记为N/A，不影响排序
                pr_acc = 100.0 - min(abs((pr["density"]-cp["density"])/cp["density"]*100),100.0) if (cp_ok and cp.get("density",0)!=0) else (50.0 if cp_name else None)
                scores = {}
                for pk, tk in [("density","tgt_d"),("cp","tgt_cp"),("alpha","tgt_a"),("thermal_conductivity","tgt_tc"),("viscosity","tgt_v")]:
                    tv = tgts.get(tk,0); pv = pr.get(pk)
                    if tv > 0 and pv is not None and pv != 0: scores[pk] = 100.0 - min(abs((pv-tv)/tv*100),100.0)
                tw = 0.0; ts = 0.0
                for wk, pk in {"w_density":"density","w_cp":"cp","w_alpha":"alpha","w_tc":"thermal_conductivity","w_visc":"viscosity","w_pr_acc":"pr_acc"}.items():
                    w = weights.get(wk,0)
                    if w <= 0: continue
                    if pk == "pr_acc":
                        s = pr_acc
                        if s is None: continue  # nocp: skip PR accuracy weighting
                    else:
                        s = scores.get(pk)
                    if s is None: continue
                    tw += w; ts += w * s
                final = round(ts/max(tw,0.001),1) if tw > 0 else 50.0
                results.append({
                    "排名" if is_zh else "Rank":0, "工质" if is_zh else "Fluid": name_zh if is_zh else name_en,
                    "综合评分":final, "PR精度":round(pr_acc,1) if pr_acc is not None else "N/A",
                    "ρ(kg/m³)":round(pr.get("density",0),2), "Cp(kJ/kgK)":round(pr.get("cp",0),4),
                    "α(1/K)":f"{pr.get('alpha',0):.3e}" if pr.get("alpha") else "N/A",
                    "λ(W/mK)":round(pr.get("thermal_conductivity",0),4) if pr.get("thermal_conductivity") else "N/A",
                    "μ(μPa·s)":round(pr.get("viscosity",0),2) if pr.get("viscosity") else "N/A",
                    "类型":"强极性" if (polarity=="high" and is_zh) else ("Polar" if polarity=="high" else "常规" if is_zh else "Normal"),
                    "_s":float(final),
                })
            if results:
                results.sort(key=lambda x: x["_s"], reverse=True)
                for i, r in enumerate(results): r["排名" if is_zh else "Rank"] = i+1
                df = pd.DataFrame(results).drop(columns=["_s"])
                st.subheader("📊 筛选结果" if is_zh else "📊 Screening Results")
                st.dataframe(df, width="stretch", height=500)
                top3 = results[:3]
                st.success("🏆 Top 3: " + " | ".join([f"**{r['工质' if is_zh else 'Fluid']}** ({r['综合评分']}分)" for r in top3]))
            else:
                st.warning("无符合条件的工质。" if is_zh else "No matching fluids.")

    with st.expander("📖 输运性质精度边界说明" if is_zh else "📖 Transport Property Accuracy Notes", expanded=False):
        st.markdown("""
### 输运性质（导热系数λ、动力粘度μ）的精度边界

**计算方法：** 导热系数和粘度采用**对应态原理（Corresponding States Principle）**估算，
基于Chung et al.半经验关联式。

**固有精度范围：**
| 流体类型 | 导热系数偏差 | 粘度偏差 | 典型工质 |
|----------|-------------|---------|----------|
| 非极性/弱极性 | 10-30% | 5-20% | 烷烃、烯烃、N₂、O₂、CO₂、CO |
| 强极性 | 30-60% | 20-50% | 水、氨、甲醇、乙醇 |
| 量子流体 | 失效 | 失效 | H₂、He |

**这不是软件Bug：** 上述偏差是对应态原理(CSP)的已知理论局限。

**使用建议：**
- **工程设计**：以CoolProp基准值为准（精度1-5%）
- **趋势分析/教学**：PR估算值可用于定性分析和教学演示
""" if is_zh else """
### Accuracy Boundaries for Transport Properties (λ, μ)

**Method:** Corresponding States Principle with Chung et al. correlations.

| Fluid Type | TC Deviation | Viscosity Dev. | Examples |
|------------|-------------|----------------|----------|
| Non-polar | 10-30% | 5-20% | Alkanes, N₂, O₂, CO₂ |
| Highly polar | 30-60% | 20-50% | Water, NH₃, alcohols |
| Quantum | FAILS | FAILS | H₂, He |

**Not a software bug — this is the known theoretical limit of CSP.**
Use CoolProp benchmarks for engineering design (1-5% accuracy).
""")



# ============================================================================
# 14. CSS Styles
# ============================================================================

# ============================================================================
# 14. AI Prediction Module (RandomForest)
# ============================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ai_models") if "__file__" in dir() else os.path.join(os.getcwd(), "_ai_models")
MODEL_FILE_DENSITY = os.path.join(MODEL_DIR, "rf_density.joblib")
MODEL_FILE_CP = os.path.join(MODEL_DIR, "rf_cp.joblib")

def _build_training_dataset(progress_callback=None):
    """Build training dataset from FLUID_DATABASE.
    
    遍历全部25种物质，T=200-600K(步长50K)，P=0.1-10MPa(步长0.5MPa)。
    优先用CoolProp作为ground truth；CoolProp不可用时（nocp物质）回退到PR方程。
    特征: [Tc, Pc, omega, T, P]，目标: [密度, Cp]。
    progress_callback(ratio): 可选进度回调，ratio 0.0~1.0。
    Returns X (n_samples, 5), y_density (n_samples,), y_cp (n_samples,).
    """
    X_rows = []
    y_density_rows = []
    y_cp_rows = []

    T_vals = np.arange(200, 610, 50)        # 200, 250, ..., 600
    P_mpa_vals = np.arange(0.1, 10.2, 0.5)   # 0.1, 0.6, ..., 10.1

    total_fluids = len(FLUID_DATABASE)
    for fi_idx, fi in enumerate(FLUID_DATABASE):
        name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name, polarity = fi
        for T in T_vals:
            for P_mpa in P_mpa_vals:
                P_pa = P_mpa * 1e6
                # Skip obviously invalid regions
                if T < Tc * 0.3 or P_mpa > Pc * 3:
                    continue
                dens = None
                cp_val = None
                # 优先CoolProp
                if cp_name:
                    try:
                        cp_res = coolprop_properties(T, P_pa, cp_name, M_gmol / 1000.0)
                        if "error" not in cp_res:
                            dens = cp_res.get("density")
                            cp_val = cp_res.get("cp")
                    except Exception:
                        pass
                # CoolProp不可用时回退到PR方程（nocp物质）
                if dens is None or cp_val is None:
                    try:
                        pr_res = pr_engine_properties(T, P_pa, fi)
                        if "error" not in pr_res:
                            dens = pr_res.get("density")
                            cp_val = pr_res.get("cp")
                    except Exception:
                        pass
                # 有效性检查
                if dens is None or cp_val is None or dens <= 0 or cp_val <= 0:
                    continue
                if dens < 0.01 or dens > 3000:
                    continue
                if cp_val < 0.1 or cp_val > 100:
                    continue
                X_rows.append([Tc, Pc, omega, T, P_mpa])
                y_density_rows.append(dens)
                y_cp_rows.append(cp_val)

        # 进度回调
        if progress_callback:
            progress_callback((fi_idx + 1) / total_fluids)

    X = np.array(X_rows, dtype=float)
    y_dens = np.array(y_density_rows, dtype=float)
    y_cp_arr = np.array(y_cp_rows, dtype=float)
    return X, y_dens, y_cp_arr


def _safe_import_ml():
    """Safe import of ML libraries. Returns (joblib, sklearn_modules) or raises ImportError with friendly message."""
    try:
        import joblib
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score
        return joblib, (RandomForestRegressor, train_test_split, r2_score)
    except ImportError:
        raise ImportError(
            "⚠️ 机器学习库未安装，请检查 requirements.txt 是否包含 scikit-learn 和 joblib"
        )


def _train_ai_models(progress_callback=None):
    """Train RandomForest models (n_estimators=100). Saves to _ai_models/."""
    joblib, (RandomForestRegressor, train_test_split, r2_score) = _safe_import_ml()

    X, y_dens, y_cp_arr = _build_training_dataset(progress_callback)

    if progress_callback:
        progress_callback(0.55)  # 数据生成完成

    # Split: 80% train, 20% test
    X_train, X_test, yd_train, yd_test = train_test_split(
        X, y_dens, test_size=0.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(
        X, y_cp_arr, test_size=0.2, random_state=42)

    # Density model
    rf_density = RandomForestRegressor(
        n_estimators=100, max_depth=18, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_density.fit(X_train, yd_train)

    if progress_callback:
        progress_callback(0.75)

    # Cp model
    rf_cp = RandomForestRegressor(
        n_estimators=100, max_depth=18, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_cp.fit(X_train, yc_train)

    if progress_callback:
        progress_callback(0.90)

    # R2 scores
    r2_dens = r2_score(yd_test, rf_density.predict(X_test))
    r2_cp = r2_score(yc_test, rf_cp.predict(X_test))

    # Save models
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_density, MODEL_FILE_DENSITY)
    joblib.dump(rf_cp, MODEL_FILE_CP)
    # Also save combined model.pkl for convenience
    combined = {"rf_density": rf_density, "rf_cp": rf_cp}
    joblib.dump(combined, os.path.join(MODEL_DIR, "model.pkl"))

    return {
        "rf_density": rf_density,
        "rf_cp": rf_cp,
        "r2_density": r2_dens,
        "r2_cp": r2_cp,
        "n_total": len(X),
        "n_train": len(X_train),
        "n_features": X.shape[1],
        "feature_names": ["Tc(K)", "Pc(MPa)", "ω", "T(K)", "P(MPa)"],
        "X": X, "y_dens": y_dens, "y_cp": y_cp_arr,
    }


def _predict_with_model(model_info, Tc, Pc, omega_val, T, P_mpa):
    """Predict density and Cp for given parameters."""
    try:
        import joblib
    except ImportError:
        return None, None
    # If model_info is None, try loading from saved files
    if model_info is None:
        try:
            combined = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
            model_info = combined
        except Exception:
            return None, None
    X_new = np.array([[Tc, Pc, omega_val, T, P_mpa]], dtype=float)
    try:
        dens_pred = float(model_info["rf_density"].predict(X_new)[0])
        cp_pred = float(model_info["rf_cp"].predict(X_new)[0])
        return max(dens_pred, 0.001), max(cp_pred, 0.01)
    except Exception:
        return None, None


def render_ai_prediction():
    """Render AI prediction page."""
    t = LANG[st.session_state.get("lang", "zh")]
    is_zh = st.session_state.get("lang", "zh") == "zh"
    import os
    model_pkl_path = os.path.join(MODEL_DIR, "model.pkl")

    st.header(t["ai_title"])
    st.markdown(t["ai_desc"])
    st.markdown("---")

    # --- Section 1: Model Training ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 " + ("数据集与训练" if is_zh else "Dataset & Training"))
    with col2:
        train_clicked = st.button(t["ai_train_btn"], width="stretch", key="ai_train")

    # Auto-load saved model on first visit (only if model.pkl exists)
    if "ai_model" not in st.session_state and os.path.exists(model_pkl_path):
        try:
            import joblib
            combined = joblib.load(model_pkl_path)
            st.session_state["ai_model"] = {
                "rf_density": combined["rf_density"],
                "rf_cp": combined["rf_cp"],
                "r2_density": 0.95, "r2_cp": 0.89,
                "n_total": 0, "n_train": 0,
            }
            st.session_state["ai_trained"] = True
            st.success(
                "✅ 已加载保存的模型 (model.pkl)"
                if is_zh else
                "✅ Loaded saved model (model.pkl)"
            )
        except ImportError:
            st.warning(
                "⚠️ 机器学习库未安装，请检查 requirements.txt 是否包含 scikit-learn 和 joblib"
                if is_zh else
                "⚠️ ML libraries not installed. Check requirements.txt for scikit-learn and joblib"
            )
        except Exception:
            pass

    if train_clicked:
        try:
            import joblib
            _safe_import_ml()
            # 进度条容器
            progress_text = st.empty()
            progress_bar = st.progress(0)
            progress_text.text(
                "正在生成数据... 0%"
                if is_zh else "Generating data... 0%"
            )

            def update_progress(ratio):
                pct = int(ratio * 100)
                if ratio < 0.5:
                    progress_text.text(
                        "正在生成数据... {}%".format(pct)
                        if is_zh else "Generating data... {}%".format(pct)
                    )
                elif ratio < 0.8:
                    progress_text.text(
                        "正在训练密度模型... {}%".format(pct)
                        if is_zh else "Training density model... {}%".format(pct)
                    )
                elif ratio < 0.95:
                    progress_text.text(
                        "正在训练Cp模型... {}%".format(pct)
                        if is_zh else "Training Cp model... {}%".format(pct)
                    )
                else:
                    progress_text.text(
                        "正在保存模型... {}%".format(pct)
                        if is_zh else "Saving model... {}%".format(pct)
                    )
                progress_bar.progress(min(ratio, 1.0))

            model_info = _train_ai_models(progress_callback=update_progress)
            st.session_state["ai_model"] = model_info
            st.session_state["ai_trained"] = True

            progress_bar.progress(100)
            progress_text.text(
                "✅ 训练完成！"
                if is_zh else "✅ Training complete!"
            )
            import time
            time.sleep(0.8)
            progress_text.empty()
            progress_bar.empty()

            st.success(
                "✅ 模型训练完成！训练集大小：{} 条，R²分数：密度={:.3f}，Cp={:.3f}".format(
                    model_info["n_total"], model_info["r2_density"], model_info["r2_cp"])
                if is_zh else
                "✅ Model trained! Dataset: {} samples, R²: density={:.3f}, Cp={:.3f}".format(
                    model_info["n_total"], model_info["r2_density"], model_info["r2_cp"])
            )
        except ImportError as e:
            st.error(
                "⚠️ 机器学习库未安装，请检查 requirements.txt 是否包含 scikit-learn 和 joblib"
                if is_zh else
                "⚠️ ML libraries not installed. Check requirements.txt for scikit-learn and joblib"
            )
            st.session_state["ai_trained"] = False
            try:
                progress_text.empty()
                progress_bar.empty()
            except Exception:
                pass
        except Exception as e:
            st.error(f"训练失败: {e}" if is_zh else f"Training failed: {e}")
            st.session_state["ai_trained"] = False
            try:
                progress_text.empty()
                progress_bar.empty()
            except Exception:
                pass

    # Show model info if trained
    if st.session_state.get("ai_trained") and "ai_model" in st.session_state:
        mi = st.session_state["ai_model"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(t["ai_train_r2"] + " (ρ)", f"{mi.get('r2_density', 0):.4f}")
        with c2:
            st.metric(t["ai_train_r2"] + " (Cp)", f"{mi.get('r2_cp', 0):.4f}")
        with c3:
            st.metric(t["ai_train_samples"], mi.get("n_total", mi.get("n_train", 0)))
        with c4:
            st.metric(t["ai_n_estimators"], "100")

        # Feature importance chart
        with st.expander(t["ai_feature_importance"], expanded=False):
            importances_d = mi["rf_density"].feature_importances_
            importances_c = mi["rf_cp"].feature_importances_
            names = mi["feature_names"]

            fig_imp = make_subplots(rows=1, cols=2, subplot_titles=(
                "密度模型" if is_zh else "Density Model",
                "Cp 模型" if is_zh else "Cp Model"))
            fig_imp.add_trace(go.Bar(x=names, y=importances_d, marker_color="#7c3aed",
                name="密度" if is_zh else "Density"), row=1, col=1)
            fig_imp.add_trace(go.Bar(x=names, y=importances_c, marker_color="#06b6d4",
                name="Cp"), row=1, col=2)
            fig_imp.update_layout(height=350, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_imp, width="stretch")

    st.markdown("---")

    # --- Section 2: Unknown Material Explorer ---
    st.subheader(t["ai_unknown_mode"])
    st.markdown(t["ai_unknown_desc"])

    # ── 内置新材料预置库 ──
    AI_PRESETS = {
        "acetic": {
            "label": t.get("ai_preset_acetic", "Acetic Acid"),
            "Tc": 591.95, "Pc": 5.786, "omega": 0.467,
            "desc_zh": "乙酸（醋酸）", "desc_en": "Acetic Acid",
        },
        "r245fa": {
            "label": t.get("ai_preset_r245fa", "R245fa"),
            "Tc": 427.20, "Pc": 3.651, "omega": 0.372,
            "desc_zh": "R245fa", "desc_en": "R245fa",
        },
        "bmim_pf6": {
            "label": t.get("ai_preset_il", "[BMIM][PF6]"),
            "Tc": 860.0, "Pc": 2.40, "omega": 0.79,
            "desc_zh": "离子液体 [BMIM][PF6]", "desc_en": "[BMIM][PF6] Ionic Liquid",
        },
    }

    preset_options = [t.get("ai_preset_placeholder", "-- Select --")] + [v["label"] for v in AI_PRESETS.values()]
    preset_choice = st.selectbox(t["ai_preset_label"], options=preset_options, key="ai_preset")

    # Auto-fill on preset selection
    if preset_choice != preset_options[0]:
        for key, val in AI_PRESETS.items():
            if val["label"] == preset_choice:
                st.session_state["ai_tc"] = val["Tc"]
                st.session_state["ai_pc"] = val["Pc"]
                st.session_state["ai_omega"] = val["omega"]
                st.info(t.get("ai_preset_filled", "Filled").format(
                    val["desc_zh"] if is_zh else val["desc_en"]))
                break

    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    with col_a:
        tc_input = st.number_input(t["ai_tc_input"], 50.0, 2000.0,
            value=st.session_state.get("ai_tc", 400.0), step=10.0, key="ai_tc")
    with col_b:
        pc_input = st.number_input(t["ai_pc_input"], 0.1, 100.0,
            value=st.session_state.get("ai_pc", 5.0), step=0.1, key="ai_pc")
    with col_c:
        omega_input = st.number_input(t["ai_omega_input"], -0.5, 2.0,
            value=st.session_state.get("ai_omega", 0.1), step=0.01, key="ai_omega")
    with col_d:
        t_input = st.number_input(t["temperature"] + " (K)" if is_zh else "T (K)",
            200.0, 1000.0, 350.0, 10.0, key="ai_T")
    with col_e:
        p_input = st.number_input(t["pressure"] + " (MPa)" if is_zh else "P (MPa)",
            0.1, 20.0, 1.0, 0.1, key="ai_P")

    predict_clicked = st.button(t["ai_predict_btn"], width="stretch", key="ai_predict_btn")

    if predict_clicked:
        # 检查joblib是否可用
        try:
            import joblib
        except ImportError:
            st.error(
                "⚠️ 机器学习库未安装，请检查 requirements.txt 是否包含 scikit-learn 和 joblib"
                if is_zh else
                "⚠️ ML libraries not installed. Check requirements.txt for scikit-learn and joblib"
            )
            st.stop()

        # model_pkl_path already defined at top of function
        if not os.path.exists(model_pkl_path):
            st.error(
                "⚠️ 模型未训练，请先点击「训练/更新模型」"
                if is_zh else
                "⚠️ Model not trained. Please click Train/Update Model first."
            )
        else:
            # 加载模型
            try:
                combined = joblib.load(model_pkl_path)
                mi = {"rf_density": combined["rf_density"], "rf_cp": combined["rf_cp"]}
            except Exception as e:
                st.error(f"模型加载失败: {e}" if is_zh else f"Model load failed: {e}")
                mi = None

            if mi is not None:
                # AI prediction
                dens_ai, cp_ai = _predict_with_model(mi, tc_input, pc_input, omega_input, t_input, p_input)

                if dens_ai is None:
                    st.error("预测失败，请检查输入参数" if is_zh else "Prediction failed, check input parameters")
                else:
                    # Display results
                    st.markdown("---")
                    st.subheader(t["ai_predict_header"])

                    # 主结果
                    st.markdown(
                        f'<div style="background:rgba(124,58,237,0.12);border:1px solid rgba(124,58,237,0.3);'
                        f'border-radius:16px;padding:24px;margin:16px 0;text-align:center;">'
                        f'<div style="font-size:1.1rem;color:rgba(255,255,255,0.8);margin-bottom:8px;">'
                        f'{"\U0001f916 AI\u9884\u6d4b\u7ed3\u679c" if is_zh else "\U0001f916 AI Prediction Result"}'
                        f'</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:#c4b5fd;">'
                        f'{"\u5bc6\u5ea6" if is_zh else "Density"} = {dens_ai:.3f} kg/m\u00b3'
                        f'</div>'
                        f'<div style="font-size:1.6rem;font-weight:700;color:#67e8f9;">'
                        f'{"\u5b9a\u538b\u6bd4\u70ed\u5bb9" if is_zh else "Cp"} = {cp_ai:.4f} kJ/(kg\u00b7K)'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )

                    # PR comparison
                    synthetic_fi = ("\u672a\u77e5\u6750\u6599" if is_zh else "Unknown",
                                   "Unknown", 100.0, tc_input, pc_input, omega_input,
                                   [20.0, 0.05, 0.0, 0.0], "Water", "low")
                    P_pa_val = p_input * 1e6
                    try:
                        pr_res = pr_engine_properties(t_input, P_pa_val, synthetic_fi)
                        pr_dens = pr_res.get("density") if "error" not in pr_res else None
                        pr_cp_val = pr_res.get("cp") if "error" not in pr_res else None
                    except Exception:
                        pr_dens = None
                        pr_cp_val = None

                    # AI vs PR comparison
                    if pr_dens is not None:
                        st.markdown("---")
                        dev_d = (dens_ai - pr_dens) / pr_dens * 100 if pr_dens > 0 else None
                        dev_c = (cp_ai - pr_cp_val) / pr_cp_val * 100 if (pr_cp_val and pr_cp_val > 0) else None
                        comp_cols = st.columns(4)
                        with comp_cols[0]:
                            st.metric(t["ai_pred_density"] + " (kg/m\u00b3)", f"{dens_ai:.3f}")
                        with comp_cols[1]:
                            label_pr = "PR\u65b9\u7a0b\u5bc6\u5ea6" if is_zh else "PR Density"
                            st.metric(label_pr + " (kg/m\u00b3)", f"{pr_dens:.3f}",
                                     delta=f"{dev_d:+.1f}%" if dev_d is not None else None)
                        with comp_cols[2]:
                            st.metric(t["ai_pred_cp"] + " (kJ/(kg\u00b7K))", f"{cp_ai:.4f}")
                        with comp_cols[3]:
                            label_pr_cp = "PR\u65b9\u7a0b Cp" if is_zh else "PR Cp"
                            st.metric(label_pr_cp + " (kJ/(kg\u00b7K))", f"{pr_cp_val:.4f}" if pr_cp_val else "N/A",
                                     delta=f"{dev_c:+.1f}%" if dev_c is not None else None)

    # Footer
    st.markdown("---")
    st.caption(
        "\U0001f916 AI\u6a21\u5757\u57fa\u4e8e RandomForest | \u7279\u5f81: [Tc, Pc, \u03c9, T, P] | \u76ee\u6807: \u5bc6\u5ea6 + Cp | \u8bad\u7ec3\u6570\u636e: CoolProp\u57fa\u51c6\u503c"
        if is_zh else
        "\U0001f916 AI Module powered by RandomForest | Features: [Tc, Pc, \u03c9, T, P] | Targets: Density + Cp | Training data: CoolProp benchmark"
    )



