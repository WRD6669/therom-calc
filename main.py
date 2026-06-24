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
import re

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
        "pr_engine": "🔩 自研PR方程引擎",
        "cp_engine": "📎 CoolProp 基准引擎",
        "deviation": "📉 偏差 (%)",
        "density": "密度",
        "cp": "定压比热容 Cp",
        "cv": "定容比热容 Cv",        "thermal_cond": "导热系数 λ",
        "viscosity": "动力粘度 μ",
        "tc_note": "??????????Chung???????10-20%???????????????",
        "mu_note": "??????????Chung???????10-20%???????????????",
        "thermal_expansion": "????? ?",
        "unit_alpha": "1/K",
        "unit_density": "kg/m³",
        "unit_cp": "kJ/(kg·K)",
        "unit_tc": "W/(m·K)",
        "unit_visc": "μPa·s",
        "plot_header": "📈 物性-温度曲线 (等压扫描)",
        "curve_pr": "PR方程(自研)",
        "curve_cp": "CoolProp(基准)",
        "warn_coolprop": "⚠️ CoolProp 查询失败: {}. 已降级使用PR方程。",
        "warn_pr_fail": "⚠️ PR方程计算失败: {}",
        "warn_range": "⚠️ 输入温度/压力超出该物质推荐范围, 结果可能不准确。",
        "error_no_fluid": "请选择物质。",
        "calc_failed": "计算失败",
        "calc_ok": "✅ 计算成功完成！",
        "phase_label": "当前状态",
        "phase_vapor": "气相",
        "phase_liquid": "液相",
        "phase_supercritical": "超临界流体",
        "phase_two_phase": "两相区",
        "warn_two_phase": "⚠️ 当前工况处于两相区，PR方程精度有限，建议参考CoolProp基准值。",
        "about_title": "ℹ️ 关于本软件",
        "about_text": "**热物性计算软件** v2.0 为化工软件开发比赛设计。<br><br>**核心特色:**<br>- 🔩 **自研Peng-Robinson方程引擎**: 手写PR三次方程求解、剩余性质计算、对应态原理粘度/导热系数估算<br>- 📎 **CoolProp基准引擎**: 调用工业级物性数据库作为高精度对照<br>- 📈 **Plotly交互图表**: 悬停数值、缩放拖拽、双曲线叠加对比<br>- 🌐 **中英双语界面**: 一键切换<br><br>**适用范围:** 气相、液相、超临界态流体热物性估算",
        "first_time_msg": "⏳ 请在左侧输入参数并点击「开始计算」",
        "fluid_info_label": "物质：{}  |  M = {} g/mol  |  Tc = {} K  |  Pc = {} MPa  |  ω = {}",
        "dev_expander_title": "📖 关于计算偏差的说明",
        "dev_expander_text": "对于水、醇类等强极性物质, 经典PR方程本身存在约5-15%的系统偏差, 这是模型的已知理论局限, 而非代码错误。具体原因包括:<br>- 未引入氢键缔合修正项<br>- 偏心因子ω对极性分子的描述能力有限<br>- 对应态原理的导热系数/粘度估算是半经验近似<br><br>如需更高精度, 建议参考CoolProp基准值。",
        "validate_title": "🔬 模型验证",
        "validate_desc": "对预设基准物质运行自研PR方程和CoolProp, 对比结果。",
        "validate_col_fluid": "物质",
        "validate_col_T": "温度 (K)",
        "validate_col_P": "压力 (MPa)",
        "validate_col_prop": "物性",
        "validate_col_PR": "自研PR结果",
        "validate_col_CP": "CoolProp结果",
        "validate_col_dev": "绝对偏差 (%)",
        "scope_title": "📋 推荐适用范围",
        "scope_text": "推荐适用范围：温度 <span style=\"color:#38bdf8;font-weight:700\">200-600 K</span>，压力 <span style=\"color:#38bdf8;font-weight:700\">0.1-10 MPa</span>。超出此范围时，PR方程计算偏差可能增大，建议以CoolProp基准值为参考。",
        "meta_calc_convergence_error": "计算未收敛。可能原因: 输入工况接近临界点或超出了PR方程的适用极限。建议微调温度或压力值。",
        "meta_mixture_warning": "当前版本仅支持纯物质计算, 混合物功能正在开发中",
        "meta_page": "页面",
        "meta_main_page": "🏠 物性计算",
        "meta_verify_page": "🔬 模型验证",
        "mode_label": "模式",
        "mode_calc": "物性计算",
        "mode_screening": "材料筛选",
        "screening_title": "🔍 材料筛选器",
        "screening_desc": "遍历所有20种物质，按指定条件筛选最优材料",
        "screening_target_prop": "目标物性",
        "screening_condition": "筛选条件",
        "screening_threshold": "阈值",
        "screening_btn": "🔍 开始筛选",
        "screening_col_fluid": "物质",
        "screening_col_value": "物性值",
        "screening_col_meet": "是否满足",
        "screening_meet": "✓",
        "screening_not_meet": "✗",
        "screening_no_results": "没有物质满足筛选条件，请调整参数。",
        "screening_results_count": "共 {} 种物质满足条件",
        "screening_error": "筛选计算出错: {}",

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
        "pr_engine": "🔩 PR EOS Engine",
        "cp_engine": "📎 CoolProp Benchmark",
        "deviation": "📉 Deviation (%)",
        "density": "Density",
        "cp": "Specific Heat Cp",
        "cv": "Specific Heat Cv",        "thermal_cond": "Thermal Conductivity λ",
        "viscosity": "Viscosity μ",
        "tc_note": "Estimated via corresponding states principle (Chung method), typical deviation 10-20%, suitable for trend comparison during material screening",
        "mu_note": "Estimated via corresponding states principle (Chung method), typical deviation 10-20%, suitable for trend comparison during material screening",
        "thermal_expansion": "Thermal Expansion ?",
        "unit_alpha": "1/K",
        "unit_density": "kg/m³",
        "unit_cp": "kJ/(kg·K)",
        "unit_tc": "W/(m·K)",
        "unit_visc": "μPa·s",
        "plot_header": "📈 Property-Temperature Curves (Isobaric Scan)",
        "curve_pr": "PR EOS (Self-dev)",
        "curve_cp": "CoolProp (Ref.)",
        "warn_coolprop": "⚠️ CoolProp query failed: {}. Falling back to PR EOS.",
        "warn_pr_fail": "⚠️ PR EOS calculation failed: {}",
        "warn_range": "⚠️ Input T/P may be out of recommended range for this fluid.",
        "error_no_fluid": "Please select a fluid.",
        "calc_failed": "Calculation failed",
        "calc_ok": "Calculation completed!",
        "phase_label": "Current State",
        "phase_vapor": "Vapor",
        "phase_liquid": "Liquid",
        "phase_supercritical": "Supercritical Fluid",
        "phase_two_phase": "Two-phase",
        "warn_two_phase": "Currently in two-phase region. PR accuracy limited.",
        "about_title": "ℹ️ About",
        "about_text": "**Thermodynamic Property Calculator** v2.0 - Built for chemical engineering software competition.<br><br>**Key Features:**<br>- 🔩 **Self-developed PR EOS Engine**: Handwritten cubic equation solver, residual properties, corresponding-state transport properties<br>- 📎 **CoolProp Benchmark Engine**: Industrial-grade thermodynamic database as reference<br>- 📈 **Plotly Interactive Charts**: Hover values, zoom/pan, dual-curve overlay comparison<br>- 🌐 **Bilingual Interface**: One-click Chinese/English switch<br><br>**Scope:** Gas, liquid, and supercritical fluid property estimation",
        "first_time_msg": "⏳ Enter parameters in the sidebar and click Calculate",
        "fluid_info_label": "**Fluid:** {}  |  M = {} g/mol  |  Tc = {} K  |  Pc = {} MPa  |  ω = {}",
        "dev_expander_title": "📖 About Calculation Deviations",
        "dev_expander_text": "For highly polar substances such as water and alcohols, the classical PR equation inherently exhibits systematic deviations of approximately 5-15%. This is a known theoretical limitation of the model, not a code error. Specific reasons include:<br>- No hydrogen bonding correction term<br>- Limited ability of acentric factor ω to describe polar molecules<br>- Transport property estimates via corresponding states are semi-empirical approximations<br><br>For higher accuracy, refer to the CoolProp benchmark values.",
        "validate_title": "🔬 Model Validation",
        "validate_desc": "Run self-developed PR EOS and CoolProp on preset benchmark fluids and compare results.",
        "validate_col_fluid": "Fluid",
        "validate_col_T": "Temp (K)",
        "validate_col_P": "Pressure (MPa)",
        "validate_col_prop": "Property",
        "validate_col_PR": "PR Result",
        "validate_col_CP": "CoolProp Result",
        "validate_col_dev": "Abs. Deviation (%)",
        "scope_title": "📋 Recommended Operating Range",
        "scope_text": "Recommended range: Temperature 200-600 K, Pressure 0.1-10 MPa. Beyond this range, PR equation deviations may increase. Refer to CoolProp benchmark values.",
        "meta_calc_convergence_error": "Calculation did not converge. Possible reasons: conditions near critical point or beyond PR EOS applicability. Try slightly adjusting temperature or pressure.",
        "meta_mixture_warning": "Current version supports pure substances only. Mixture functionality is under development.",
        "meta_page": "Page",
        "meta_main_page": "🏠 Property Calc",
        "meta_verify_page": "🔬 Model Validation",
        "mode_label": "Mode",
        "mode_calc": "Property Calc",
        "mode_screening": "Material Screening",
        "screening_title": "🔍 Material Screening",
        "screening_desc": "Scan all 20 substances and filter by specified criteria",
        "screening_target_prop": "Target Property",
        "screening_condition": "Condition",
        "screening_threshold": "Threshold",
        "screening_btn": "🔍 Start Screening",
        "screening_col_fluid": "Substance",
        "screening_col_value": "Property Value",
        "screening_col_meet": "Meets Condition",
        "screening_meet": "✓",
        "screening_not_meet": "✗",
        "screening_no_results": "No substances match the screening criteria. Please adjust parameters.",
        "screening_results_count": "{} substances meet the criteria",
        "screening_error": "Screening error: {}",

    },
}

FLUID_DATABASE = [
    ("甲烷", "Methane",    90.7,   16.043,  190.56,  4.599,  0.011, [19.25, 0.05213, 1.197e-05, -1.132e-08]      , "Methane"),
    ("乙烷", "Ethane",    90.4,   30.070,  305.32,  4.872,  0.099, [5.41, 0.17809, -6.938e-05, 8.713e-09]       , "Ethane"),
    ("丙烷", "Propane",    85.5,   44.096,  369.83,  4.248,  0.152, [-4.22, 0.3063, -0.0001586, 3.215e-08]       , "Propane"),
    ("正丁烷", "n-Butane",   134.9,   58.122,  425.12,  3.796,  0.200, [9.49, 0.3313, -0.0001108, -2.822e-09]       , "n-Butane"),
    ("正戊烷", "n-Pentane",   143.5,   72.149,  469.70,  3.370,  0.251, [-3.63, 0.4873, -0.000258, 5.305e-08]        , "n-Pentane"),
    ("乙烯", "Ethylene",   104.0,   28.054,  282.34,  5.041,  0.086, [3.81, 0.1566, -8.348e-05, 1.755e-08]        , "Ethylene"),
    ("丙烯", "Propylene",    87.9,   42.080,  364.90,  4.600,  0.144, [3.71, 0.2345, -0.000116, 2.205e-08]         , "Propylene"),
    ("苯", "Benzene",   278.7,   78.112,  562.05,  4.895,  0.210, [-33.9, 0.5639, -0.0004133, 1.202e-07]       , "Benzene"),
    ("甲苯", "Toluene",   178.2,   92.138,  591.75,  4.108,  0.264, [-24.36, 0.5125, -0.0002765, 4.911e-08]      , "Toluene"),
    ("甲醇", "Methanol",   175.5,   32.042,  512.64,  8.097,  0.565, [21.15, 0.07092, 2.587e-05, -2.852e-08]      , "Methanol"),
    ("乙醇", "Ethanol",   159.1,   46.068,  513.90,  6.148,  0.643, [9.38, 0.30928, -0.0001706, 3.787e-08]       , "Ethanol"),
    ("水", "Water",   273.2,   18.015,  647.10, 22.064,  0.344, [32.24, 0.00192, 1.055e-05, -3.596e-09]      , "Water"),
    ("氨", "Ammonia",   195.4,   17.031,  405.40, 11.333,  0.256, [27.32, 0.02383, 1.707e-05, -1.185e-08]      , "Ammonia"),
    ("二氧化碳", "CO2",   216.6,   44.010,  304.13,  7.377,  0.225, [19.8, 0.07344, -5.602e-05, 1.715e-08]       , "CarbonDioxide"),
    ("一氧化碳", "CO",    68.1,   28.010,  132.86,  3.494,  0.048, [30.87, -0.01285, 2.789e-05, -1.272e-08]     , "CarbonMonoxide"),
    ("氮气", "Nitrogen",    63.2,   28.013,  126.19,  3.396,  0.037, [31.15, -0.01357, 2.68e-05, -1.168e-08]      , "Nitrogen"),
    ("氧气", "Oxygen",    54.4,   31.999,  154.58,  5.043,  0.021, [28.11, -0.00368, 1.746e-05, -1.065e-08]     , "Oxygen"),
    ("氢气", "Hydrogen",    14.0,    2.016,   33.15,  1.296, -0.216, [27.14, 0.00927, -1.381e-05, 7.645e-09]      , "Hydrogen"),
    ("氦气", "Helium",     2.2,    4.003,    5.20,  0.227, -0.390, [20.79, 0.0, 0.0, 0.0]                       , "Helium"),
    ("R134a", "R134a",   170.0,  102.030,  374.21,  4.059,  0.327, [16.34, 0.2685, -0.0001457, 2.492e-08]       , "R134a"),
]


# ============================================================================
# 2. Peng-Robinson EOS Core Module
# ============================================================================

def pr_alpha(T: float, Tc: float, omega: float) -> float:
    """PR equation alpha(T) function"""
    Tr = T / Tc
    if Tr <= 0:
        raise ValueError(f"Tr={Tr:.4f} <= 0")
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    sqrt_alpha = 1.0 + kappa * (1.0 - np.sqrt(Tr))
    return sqrt_alpha**2


def pr_parameters(T: float, P: float, Tc: float, Pc: float, omega: float):
    """Compute PR EOS parameters a(T) and b in SI units"""
    alpha = pr_alpha(T, Tc, omega)
    a = 0.45724 * (R_GAS**2) * (Tc**2) / Pc * alpha
    b = 0.07780 * R_GAS * Tc / Pc
    return a, b


def pr_cubic_coefficients(T, P, Tc, Pc, omega):
    """Build PR cubic: Z^3 + c2*Z^2 + c1*Z + c0 = 0"""
    a_val, b_val = pr_parameters(T, P, Tc, Pc, omega)
    RT = R_GAS * T
    A_dim = a_val * P / (RT**2)
    B_dim = b_val * P / RT
    c3 = 1.0
    c2 = -(1.0 - B_dim)
    c1 = A_dim - 2.0 * B_dim - 3.0 * B_dim**2
    c0 = -(A_dim * B_dim - B_dim**2 - B_dim**3)
    return c0, c1, c2, c3


def f_pr_cubic(Z, c0, c1, c2, c3):
    """Cubic function value f(Z)"""
    return ((c3 * Z + c2) * Z + c1) * Z + c0


def fp_pr_cubic(Z, c1, c2, c3):
    """Derivative f''(Z)"""
    return (3.0 * c3 * Z + 2.0 * c2) * Z + c1


def solve_pr_cubic(T, P, Tc, Pc, omega):
    """Solve PR cubic for compressibility factor Z.
    Returns (Z_small, Z_mid, Z_large) — three real roots sorted ascending.
    Uses Cardano analytical solution + Newton refinement.
    """
    import cmath

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

    # Return roots sorted ascending (smallest first).
    # The caller will determine which is vapor/liquid based on density.
    refined_asc = sorted(set(round(z, 10) for z in refined))
    while len(refined_asc) < 3:
        refined_asc.append(refined_asc[-1] if refined_asc else 0.3)
    # Return: (smallest, middle, largest)
    return float(refined_asc[0]), float(refined_asc[1]), float(refined_asc[2])


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
    # T*da/dT - a (correct PR residual enthalpy formula)
    Tda_minus_a = a_val * (T * a_prime_over_a - 1.0)
    numerator = Tda_minus_a / (2.0 * sqrt2 * b_val)
    arg = (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
    if arg <= 0:
        arg = abs(arg) + 1e-15
    term2 = numerator * np.log(arg)
    # H_res = RT*(Z-1) + (T*da/dT-a)/(2*sqrt2*b) * ln[(Z+(1+sqrt2)B)/(Z+(1-sqrt2)B)]
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
    ln_z_b = np.log(max(Z - B, 1e-15))
    arg = (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
    if arg <= 0:
        arg = abs(arg) + 1e-15
    term2 = (da_dT / (2.0 * sqrt2 * b_val)) * np.log(arg)
    return R_GAS * (ln_z_b - term2)



# ============================================================================
# 3. Corresponding States - Transport Properties
# ============================================================================

def estimate_thermal_conductivity_pr(T, P, Z, M, Tc, Pc, omega, Cp_ideal):
    """Estimate thermal conductivity via Chung+Eucken method [W/(m*K)]"""
    M_g = M * 1000.0
    Tr = T / Tc
    Cv_ideal = max(Cp_ideal - R_GAS, R_GAS * 1.5)
    Zc = 0.288
    Vc_cm3 = Zc * R_GAS * Tc / Pc * 1e6
    sigma_a = 0.809 * np.cbrt(Vc_cm3)
    Omega_v = (
        1.16145 / (Tr ** 0.14874)
        + 0.52487 * np.exp(-0.77320 * Tr)
        + 2.16178 * np.exp(-2.43787 * Tr)
    )
    Fc = 1.0 - 0.2756 * omega
    mu0_muP = 26.69 * Fc * np.sqrt(M_g * T) / (sigma_a ** 2 * Omega_v)
    mu0_Pas = mu0_muP * 1e-7
    alpha_c = Cv_ideal / R_GAS - 1.5
    beta_c = 0.7862 - 0.7109 * omega + 1.3168 * omega ** 2
    psi_c = 1.0 + alpha_c * (
        (0.215 + 0.28288 * alpha_c - 1.061 * beta_c + 0.26665 * Zc)
        / (0.6366 + beta_c * Zc + 1.061 * alpha_c * beta_c)
    )
    lambda0 = 3.75 * psi_c / (Cv_ideal / R_GAS) * mu0_Pas * Cv_ideal / (M_g / 1000.0)
    rho_actual = pr_density(Z, T, P, M)
    rho_c = M * Pc / (R_GAS * Tc * Zc)
    rho_r = max(rho_actual / (rho_c + 1e-15), 0.0)
    if rho_r > 0.02:
        lambda0 *= (1.0 + 0.4 * rho_r ** 0.7)
    return max(lambda0, 0.001)

def estimate_viscosity_pr(T, P, Z, M, Tc, Pc, omega):
    """Estimate dynamic viscosity via Chung et al. (1988) [muPa*s]"""
    M_g = M * 1000.0
    Tr = T / Tc
    Zc_v = 0.288
    Vc_cm3 = Zc_v * R_GAS * Tc / Pc * 1e6
    sigma_a = 0.809 * np.cbrt(Vc_cm3)
    Omega_v = (
        1.16145 / (Tr ** 0.14874)
        + 0.52487 * np.exp(-0.77320 * Tr)
        + 2.16178 * np.exp(-2.43787 * Tr)
    )
    Fc = 1.0 - 0.2756 * omega
    mu0_muP = 26.69 * Fc * np.sqrt(M_g * T) / (sigma_a ** 2 * Omega_v)
    if omega > 0.01:
        mu0_muP *= (1.0 + 0.8 * omega)
    mu0_muPas = mu0_muP / 10.0
    rho_actual = pr_density(Z, T, P, M)
    rho_c = M * Pc / (R_GAS * Tc * Zc_v)
    rho_r = max(rho_actual / (rho_c + 1e-15), 0.0)
    if rho_r > 0.01:
        FK = 1.0 + 1.114 * rho_r ** 0.894 / (Tr + 0.912)
        mu_total = mu0_muPas * FK
    else:
        mu_total = mu0_muPas
    return max(mu_total, 0.5)



def calc_thermal_expansion_pr(T, P, Z, M, Tc, Pc, omega):
    """Calculate thermal expansion coefficient alpha via numerical derivative.
    
    alpha = -(rho(T+dT) - rho(T-dT)) / (rho(T) * 2*dT)
    where dT = 0.5 K
    
    Returns:
        float: alpha in 1/K
    """
    dT = 0.5
    try:
        # Get density at T+dT
        Z_plus, _, _ = solve_pr_cubic(T + dT, P, Tc, Pc, omega)
        rho_plus = pr_density(Z_plus, T + dT, P, M)
        
        # Get density at T-dT
        Z_minus, _, _ = solve_pr_cubic(T - dT, P, Tc, Pc, omega)
        rho_minus = pr_density(Z_minus, T - dT, P, M)
        
        # Get density at T
        rho_T = pr_density(Z, T, P, M)
        
        if rho_T <= 0 or rho_plus <= 0 or rho_minus <= 0:
            return 0.0
        
        # alpha = -(drho/dT)_P / rho
        alpha = -(rho_plus - rho_minus) / (rho_T * 2.0 * dT)
        return max(alpha, -1.0)  # clamp unreasonable negative values
    except Exception:
        return 0.0

def coolprop_properties(T, P, fluid, M):
    """Query CoolProp for fluid properties.

    Args:
        T: Temperature [K]
        P: Pressure [Pa]
        fluid: CoolProp fluid name string
        M: Molar mass [kg/mol]

    Returns:
        dict with density [kg/m^3], cp [kJ/(kg*K)], cv [kJ/(kg*K)],
        thermal_conductivity [W/(m*K)], viscosity [muPa*s].
        Or {"error": msg} on failure.
    """
    try:
        import CoolProp.CoolProp as CP

        density = CP.PropsSI("D", "T", T, "P", P, fluid)
        cp_mass = CP.PropsSI("C", "T", T, "P", P, fluid)   # J/(kg*K)
        cv_mass = CP.PropsSI("O", "T", T, "P", P, fluid)   # J/(kg*K)
        tc = CP.PropsSI("L", "T", T, "P", P, fluid)        # W/(m*K)
        visc = CP.PropsSI("V", "T", T, "P", P, fluid)      # Pa*s
        alpha_cp = CP.PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "T", T, "P", P, fluid)  # 1/K

        if density <= 0 or np.isnan(density):
            raise ValueError(f"Invalid density: {density}")

        return {
            "density": density,                  # kg/m^3
            "cp": cp_mass / 1000.0,              # kJ/(kg*K)
            "cv": cv_mass / 1000.0,              # kJ/(kg*K)
            "thermal_conductivity": tc,          # W/(m*K)
            "viscosity": visc * 1e6,             # muPa*s
            "thermal_expansion": alpha_cp,           # 1/K
        }
    except Exception as e:
        return {"error": str(e)}



def pr_fugacity_coeff(Z, T, P, Tc, Pc, omega):
    """PR EOS fugacity coefficient ln(phi).
    
    ln(phi) = Z - 1 - ln(Z - B) - A/(2*sqrt(2)*B) * ln((Z + (1+sqrt(2))B)/(Z + (1-sqrt(2))B))
    """
    a_val, b_val = pr_parameters(T, P, Tc, Pc, omega)
    RT = R_GAS * T
    A = a_val * P / (RT ** 2)
    B = b_val * P / RT
    sqrt2 = np.sqrt(2.0)
    if Z <= B:
        return -1e10  # invalid, heavily penalize
    term1 = Z - 1.0
    term2 = -np.log(Z - B)
    term3 = -A / (2.0 * sqrt2 * B) * np.log((Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B))
    return term1 + term2 + term3


def estimate_psat_pr(T, Tc, Pc, omega, max_iter=50, tol=1e-4):
    """Estimate saturated vapor pressure using PR EOS fugacity equality.
    
    Bisection on ln(phi_vapor) - ln(phi_liquid) = 0.
    Uses Pitzer correlation as initial guess, refined by fugacity balance.
    Returns Psat in Pa.
    """
    if T >= Tc:
        return Pc

    Tr = T / Tc

    def fug_diff(P_test):
        """ln(phi_v) - ln(phi_l). Should cross 0 at Psat."""
        try:
            Z_s, Z_m, Z_l = solve_pr_cubic(T, P_test, Tc, Pc, omega)
            # If only one real root (Z_s == Z_l), return a large negative to push search lower
            if abs(Z_s - Z_l) < 1e-6:
                return -10.0
            ln_phi_v = pr_fugacity_coeff(Z_l, T, P_test, Tc, Pc, omega)
            ln_phi_l = pr_fugacity_coeff(Z_s, T, P_test, Tc, Pc, omega)
            return ln_phi_v - ln_phi_l
        except Exception:
            return 1.0

    # Pitzer initial guess
    log10_Pr = (7.0 / 3.0) * (1.0 + omega) * (1.0 - 1.0 / Tr)
    Psat_pitzer = Pc * (10.0 ** log10_Pr)

    # Set bounds: search around Pitzer estimate
    P_lo = max(Psat_pitzer * 0.1, Pc * 1e-8)
    P_hi = min(Psat_pitzer * 10.0, Pc * 0.95)

    # Ensure P_hi is in the three-root region (fug_diff meaningful)
    # At pressures near Pc, only one root exists; find the max P with 3 distinct roots
    for _ in range(20):
        Z_s, Z_m, Z_l = solve_pr_cubic(T, P_hi, Tc, Pc, omega)
        if abs(Z_s - Z_l) > 1e-6:
            break
        P_hi *= 0.8
    P_hi = max(P_hi, P_lo * 2.0)

    f_lo = fug_diff(P_lo)
    f_hi = fug_diff(P_hi)

    # If no sign change, the Pitzer estimate is already good
    if f_lo * f_hi >= 0:
        return Psat_pitzer

    # Bisection
    for _ in range(max_iter):
        P_mid = (P_lo + P_hi) / 2.0
        f_mid = fug_diff(P_mid)
        if abs(f_mid) < tol:
            return P_mid
        if f_lo * f_mid < 0:
            P_hi = P_mid; f_hi = f_mid
        else:
            P_lo = P_mid; f_lo = f_mid

    return (P_lo + P_hi) / 2.0

def pr_engine_properties(T, P, fluid_info):
    """Compute all properties using self-developed PR EOS.

    Root selection based on Psat from PR fugacity equality iteration:
    1. Estimate Psat via Pitzer, refine by bisection on ln(phi_v)-ln(phi_l)
    2. T > Tc -> supercritical, use Z_large (vapor-like)
    3. T <= Tc: P < 0.95*Psat -> vapor (Z_large)
                P > 1.05*Psat -> liquid (Z_small)
                otherwise -> two-phase (Z_large + warning)
    """
    try:
        name_zh, name_en, T_triple, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info
        M = M_gmol / 1000.0
        Pc_pa = Pc * 1e6

        # ---- Step 1: Solve cubic for Z (ascending order) ----
        Z_s, Z_m, Z_l = solve_pr_cubic(T, P, Tc, Pc_pa, omega)

        # ---- Step 2: Estimate Psat via PR fugacity iteration ----
        Psat_est = estimate_psat_pr(T, Tc, Pc_pa, omega)

        # ---- Step 3: Phase detection and root selection ----
        two_phase_flag = False
        phase = "unknown"

        if T > Tc:
            phase = "supercritical"
            Z_used = Z_l
        elif P < Psat_est * 0.95:
            phase = "vapor"
            Z_used = Z_l
        elif P > Psat_est * 1.05:
            phase = "liquid"
            Z_used = Z_s
        else:
            phase = "two-phase"
            two_phase_flag = True
            Z_used = Z_l

        # Never use the middle root
        if Z_used == Z_m:
            Z_used = Z_l

        if Z_used <= 0.001 or Z_used > 5.0:
            raise ValueError(f"Abnormal Z={Z_used:.6f} at T={T:.1f}K P={P/1e6:.4f}MPa")

        Z_vapor_out = Z_l
        Z_liquid_out = Z_s

        # ---- Step 4: Density ----
        density = pr_density(Z_used, T, P, M)

        # ---- Step 5: Ideal gas heat capacity ----
        A, B_cp, C_cp, D_cp = cp_coeffs
        Cp_ig_mol = A + B_cp * T + C_cp * T**2 + D_cp * T**3
        Cv_ig_mol = max(Cp_ig_mol - R_GAS, R_GAS * 1.5)

        # ---- Step 6: Residual enthalpy & entropy ----
        H_res = pr_residual_enthalpy(T, P, Z_used, Tc, Pc_pa, omega)
        S_res = pr_residual_entropy(T, P, Z_used, Tc, Pc_pa, omega)

        # ---- Step 7: Cp via numerical derivative ----
        dT = 0.1
        Zp, _, _ = solve_pr_cubic(T + dT, P, Tc, Pc_pa, omega)
        Hp = pr_residual_enthalpy(T + dT, P, Zp, Tc, Pc_pa, omega)
        Zm, _, _ = solve_pr_cubic(T - dT, P, Tc, Pc_pa, omega)
        Hm = pr_residual_enthalpy(T - dT, P, Zm, Tc, Pc_pa, omega)
        Cp_res = (Hp - Hm) / (2.0 * dT)
        Cp_total = (Cp_ig_mol + Cp_res) / M_gmol
        Cv_total = Cv_ig_mol / M_gmol

        # ---- Step 8: Transport properties ----
        tc = estimate_thermal_conductivity_pr(T, P, Z_used, M, Tc, Pc_pa, omega, Cp_ig_mol)
        mu = estimate_viscosity_pr(T, P, Z_used, M, Tc, Pc_pa, omega)

        # ---- Step 8.5: Thermal expansion coefficient ----
        alpha = calc_thermal_expansion_pr(T, P, Z_used, M, Tc, Pc_pa, omega)

# ---- Step 9: Sanity checks ----
        rho_v = float(density); cp_v = float(Cp_total); cv_v = float(Cv_total)
        tc_v = float(tc); mu_v = float(mu)

        if not (0.01 <= rho_v <= 3000):
            return {"error": f"Density out of range: {rho_v:.1f} kg/m3"}
        if not (0.5 <= cp_v <= 50):
            return {"error": f"Cp out of range: {cp_v:.2f} kJ/(kg.K)"}
        if not (0.001 <= tc_v <= 10):
            return {"error": f"TC out of range: {tc_v:.4f} W/(m.K)"}
        if not (0.1 <= mu_v <= 10000):
            return {"error": f"Viscosity out of range: {mu_v:.2f} muPa.s"}

        return {
            "density": rho_v, "cp": cp_v, "cv": cv_v,
            "Z": float(Z_used), "H_res": float(H_res), "S_res": float(S_res),
            "thermal_conductivity": tc_v, "viscosity": mu_v,
            "thermal_expansion": float(alpha),
            "Z_vapor": float(Z_vapor_out), "Z_liquid": float(Z_liquid_out),
            "two_phase": two_phase_flag, "phase": phase,
        }
    except Exception as e:
        return {"error": str(e)}


def calc_deviation(val1, val2):
    """Calculate relative deviation: 100*(val1 - val2)/val2 [%]"""
    if val1 is None or val2 is None or abs(val2) < 1e-15:
        return None
    return 100.0 * (val1 - val2) / val2



# ============================================================================
# 7. Plotly Interactive Charts
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
    """Generate 4 Plotly subplots with PR/CoolProp overlay, phase-aware."""
    name_zh, name_en, T_triple, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info

    # [FIX P0-1] Smart temperature lower bound from triple point
    T_min = max(T_triple + 30.0, 150.0) if T_triple is not None else 150.0
    T_max = min(1000.0, Tc + 300.0)
    T_scan = np.linspace(T_min, T_max, 80)
    n = len(T_scan)

    pr_density_arr  = np.full(n, np.nan)
    pr_cp_arr       = np.full(n, np.nan)
    pr_tc_arr       = np.full(n, np.nan)
    pr_visc_arr     = np.full(n, np.nan)
    cp_density_arr  = np.full(n, np.nan)
    cp_cp_arr       = np.full(n, np.nan)
    cp_tc_arr       = np.full(n, np.nan)
    cp_visc_arr     = np.full(n, np.nan)
    two_phase_mask  = np.full(n, False)  # [FIX P0-1]

    for i, T_val in enumerate(T_scan):
        pr_res = pr_engine_properties(T_val, P_pa, fluid_info)
        if "error" not in pr_res:
            # [FIX P0-1] Detect two-phase via Z-factor gap
            # Check if PR result indicates two-phase
            if pr_res.get("two_phase", False):
                two_phase_mask[i] = True
            else:
                pr_density_arr[i] = pr_res.get("density", np.nan)
                pr_cp_arr[i]      = pr_res.get("cp", np.nan)
                pr_tc_arr[i]      = pr_res.get("thermal_conductivity", np.nan)
                pr_visc_arr[i]    = pr_res.get("viscosity", np.nan)
        cp_res = coolprop_properties(T_val, P_pa, cp_name, M_gmol / 1000.0)
        if "error" not in cp_res:
            cp_density_arr[i] = cp_res.get("density", np.nan)
            cp_cp_arr[i]      = cp_res.get("cp", np.nan)
            cp_tc_arr[i]      = cp_res.get("thermal_conductivity", np.nan)
            cp_visc_arr[i]    = cp_res.get("viscosity", np.nan)

    # [FIX P0-1] Fixed y-axis ranges
    y_ranges = [(0, 600), (0, 15), (0, 0.5), (0, 60)]

    if lang == "zh":
        y_labels = ["密度 (kg/m³)","Cp (kJ/(kg·K))","导热系数 (W/(m·K))","粘度 (µPa·s)"]
        legend_pr = "PR方程(自研)"; legend_cp = "CoolProp(基准)"
        x_label = "温度 (K)"; ann_dev = "最大偏差"
        tp_label = "两相区(PR不适用)"
    else:
        y_labels = ["Density (kg/m³)","Cp (kJ/(kg·K))","TC (W/(m·K))","Viscosity (µPa·s)"]
        legend_pr = "PR EOS (Self-dev)"; legend_cp = "CoolProp (Ref.)"
        x_label = "Temperature (K)"; ann_dev = "Max Dev"
        tp_label = "Two-phase (PR N/A)"

    fig = make_subplots(rows=2, cols=2, subplot_titles=y_labels,
                         vertical_spacing=0.14, horizontal_spacing=0.10)
    fig.update_annotations(font=dict(family="Microsoft YaHei, SimHei, sans-serif", size=13, color="#e2e8f0"))

    color_pr = "#c4b5fd"; color_cp = "#38bdf8"

    data_pairs = [
        (pr_density_arr, cp_density_arr, 1, 1, y_labels[0], y_ranges[0]),
        (pr_cp_arr, cp_cp_arr, 1, 2, y_labels[1], y_ranges[1]),
        (pr_tc_arr, cp_tc_arr, 2, 1, y_labels[2], y_ranges[2]),
        (pr_visc_arr, cp_visc_arr, 2, 2, y_labels[3], y_ranges[3]),
    ]

    annotations = []

    # [FIX P0-1] Two-phase zone boundaries
    if np.any(two_phase_mask):
        tp_idx = np.where(two_phase_mask)[0]
        tp_start = T_scan[tp_idx[0]]; tp_end = T_scan[tp_idx[-1]]

    for idx, (pr_data, cp_data, row, col, yl, (ylo, yhi)) in enumerate(data_pairs):
        show_legend = idx == 0
        fig.add_trace(go.Scatter(x=T_scan, y=pr_data, mode="lines", name=legend_pr,
            line=dict(color=color_pr, width=2.2), legendgroup="pr", showlegend=show_legend), row=row, col=col)
        fig.add_trace(go.Scatter(x=T_scan, y=cp_data, mode="lines", name=legend_cp,
            line=dict(color=color_cp, width=2.2, dash="dash"), legendgroup="cp", showlegend=show_legend), row=row, col=col)

        # [FIX P0-1] Two-phase vertical lines
        if np.any(two_phase_mask):
            fig.add_vline(x=tp_start, line_dash="dot", line_color="rgba(255,255,255,0.25)", row=row, col=col)
            fig.add_vline(x=tp_end, line_dash="dot", line_color="rgba(255,255,255,0.25)", row=row, col=col)
            if idx == 0:
                annotations.append(dict(x=(tp_start+tp_end)/2, y=yhi*0.55,
                    xref=f"x{idx+1}", yref=f"y{idx+1}", text=tp_label,
                    showarrow=False, font=dict(color="rgba(255,255,255,0.45)", size=10)))

        # [FIX P0-1] Deviation stats exclude NaN
        mask = np.isfinite(pr_data) & np.isfinite(cp_data) & (np.abs(cp_data) > 1e-12)
        if np.any(mask):
            dev_pct = np.abs((pr_data[mask] - cp_data[mask]) / cp_data[mask]) * 100
            max_idx_local = np.argmax(dev_pct)
            max_idx = np.where(mask)[0][max_idx_local]
            annotations.append(dict(x=T_scan[max_idx], y=pr_data[max_idx],
                xref=f"x{idx+1}", yref=f"y{idx+1}",
                text=f"<b>{ann_dev}</b><br>{dev_pct[max_idx_local]:.1f}%",
                showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=1.5,
                arrowcolor="#f59e0b", ax=0, ay=-40,
                font=dict(size=10, color="#f59e0b"),
                bgcolor="rgba(0,0,0,0.7)", bordercolor="#f59e0b", borderwidth=1, borderpad=4))

        # [FIX P0-1] Fixed y-axis
        fig.update_yaxes(range=[ylo, yhi], row=row, col=col, gridcolor="rgba(128,128,128,0.15)")

    for i in range(1, 5):
        r = 1 if i <= 2 else 2; cv = 1 if i % 2 == 1 else 2
        fig.update_xaxes(title_text=x_label, row=r, col=cv, gridcolor="rgba(128,128,128,0.15)")

    fig.update_layout(height=750, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5,
            itemclick="toggle", itemdoubleclick="toggleothers",
            bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#e2e8f0", size=12)),
        annotations=annotations, margin=dict(l=50, r=30, t=80, b=50),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Microsoft YaHei, SimHei, Segoe UI, Arial, sans-serif", size=13, color="#e2e8f0"))
    return fig
def run_calculation(T_input, P_input, fluid_info_tuple):
    """Execute both engines and return results. Cached for performance."""
    name_zh, name_en, T_triple, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info_tuple
    M = M_gmol / 1000.0
    P_pa = P_input * 1e6

    pr_result = None
    cp_result = None
    range_warning = None

    # Range check
    if T_input < 200 or T_input > 600 or P_input < 0.1 or P_input > 10:
        range_warning = "range"

    # PR Engine
    try:
        pr_result = pr_engine_properties(T_input, P_pa, fluid_info_tuple)
    except Exception as e:
        pr_result = {"error": f"PR\u8ba1\u7b97\u5f02\u5e38: {str(e)}"}

    # CoolProp Engine
    try:
        cp_result = coolprop_properties(T_input, P_pa, cp_name, M)
    except Exception as e:
        cp_result = {"error": f"CoolProp\u67e5\u8be2\u5f02\u5e38: {str(e)}"}

    return pr_result, cp_result, range_warning


def render_results(pr_result, cp_result, fluid_info, P_pa, t):
    """Render results: one card per property with PR-vs-CoolProp side-by-side."""
    name_zh, name_en, T_triple, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info
    fluid_display = name_zh if st.session_state["lang"] == "zh" else name_en
    is_zh = st.session_state["lang"] == "zh"

    # ---- Fluid info + results header ----
    st.markdown(
        '<div style="font-size:0.95rem;color:rgba(255,255,255,0.70);margin-bottom:2px;">'
        + t["fluid_info_label"].format(fluid_display, M_gmol, Tc, Pc, omega)
        + '</div>',
        unsafe_allow_html=True,
    )
    st.success(t["calc_ok"])

    # Phase indicator
    if pr_result and "error" not in pr_result:
        ph = pr_result.get("phase", "")
        if ph:
            ph_label = {"vapor": t["phase_vapor"], "liquid": t["phase_liquid"],
                        "supercritical": t["phase_supercritical"], "two-phase": t["phase_two_phase"]}.get(ph, ph)
            st.caption(f'{t["phase_label"]}: {ph_label}')
            if ph == "two-phase":
                st.warning(t["warn_two_phase"])



    props_map = [
        ("density",              t["density"],      t["unit_density"]),
        ("cp",                   t["cp"],           t["unit_cp"]),
        ("cv",                   t["cv"],           t["unit_cp"]),
        ("thermal_conductivity", t["thermal_cond"], t["unit_tc"]),
        ("viscosity",            t["viscosity"],    t["unit_visc"]),
        ("thermal_expansion",    t["thermal_expansion"], t["unit_alpha"]),
    ]
    _fmt = {"density": ".3f", "cp": ".4f", "cv": ".4f", "thermal_conductivity": ".4f", "viscosity": ".4f", "thermal_expansion": ".4e"}

    pr_label = "\u81ea\u7814PR\u65b9\u7a0b" if is_zh else "PR EOS"
    cp_label = "CoolProp基准" if is_zh else "CoolProp"

    # ---- 5 property cards, one per row ----
    for key, name, unit in props_map:
        pr_val = pr_result.get(key) if (pr_result and "error" not in pr_result) else None
        cp_val = cp_result.get(key) if (cp_result and "error" not in cp_result) else None
        dev_val = calc_deviation(pr_val, cp_val)

        if dev_val is not None:
            abs_d = abs(dev_val)
            if abs_d <= 5:
                dev_class = "dev-green-v2"; dot_class = "dot-green"
            elif abs_d <= 10:
                dev_class = "dev-yellow-v2"; dot_class = "dot-yellow"
            else:
                dev_class = "dev-red-v2"; dot_class = "dot-red"
            dev_str = f'{dev_val:+.2f}%'
        else:
            dev_class = "dev-na-v2"; dot_class = "dot-na"
            dev_str = "N/A"

        pr_s = f"{pr_val:{_fmt[key]}}" if pr_val is not None else "N/A"
        cp_s = f"{cp_val:{_fmt[key]}}" if cp_val is not None else "N/A"

        dev_label_text = "\u504f\u5dee" if is_zh else "Dev"
        card = (
            '<div class="prop-card-final">'
            # ---- Name row ----
            f'<div class="pcf-name">{name}</div>'
            # ---- Body: two columns + deviation ----
            '<div class="pcf-body">'
            # PR column
            '<div class="pcf-col pcf-col-pr">'
            f'<div class="pcf-engine-tag pr-tag">{pr_label}</div>'
            f'<div class="pcf-val-row"><span class="{dot_class} pcf-dot"></span>'
            f'<span class="pcf-val pr-val-v2">{pr_s}</span></div>'
            f'<div class="pcf-unit">{unit}</div>'
            '</div>'
            # Divider
            '<div class="pcf-divider"></div>'
            # CP column
            '<div class="pcf-col pcf-col-cp">'
            f'<div class="pcf-engine-tag cp-tag">{cp_label}</div>'
            f'<div class="pcf-val-row"><span class="{dot_class} pcf-dot"></span>'
            f'<span class="pcf-val cp-val-v2">{cp_s}</span></div>'
            f'<div class="pcf-unit">{unit}</div>'
            '</div>'
            # Deviation (rightmost)
            '<div class="pcf-dev">'
            f'<div class="pcf-dev-label">{dev_label_text}</div>'
            f'<span class="dev-badge-v2 {dev_class}"><span class="dev-dot"></span>{dev_str}</span>'
            '</div>'
            '</div>'
            '</div>'
        )
        # Transport property note for tc and viscosity
        if key == "thermal_conductivity" and "tc_note" in t:
            card += f'<div class="prop-note">{t["tc_note"]}</div>'
        if key == "viscosity" and "mu_note" in t:
            card += f'<div class="prop-note">{t["mu_note"]}</div>'
        st.markdown(card, unsafe_allow_html=True)

    st.markdown("---")

    # Deviation explanation
    with st.expander(t["dev_expander_title"]):
        st.markdown(t["dev_expander_text"], unsafe_allow_html=True)

    # Error messages
    if pr_result and "error" in pr_result:
        st.warning(t["warn_pr_fail"].format(pr_result["error"]))
    if cp_result and "error" in cp_result:
        st.warning(t["warn_coolprop"])

    # Z-factor
    if pr_result and "error" not in pr_result:
        with st.expander("\U0001f4ca \u4e2d\u95f4\u53d8\u91cf (Z\u56e0\u5b50 / \u6b8b\u4f59\u7113)" if is_zh else "\U0001f4ca Intermediate Variables (Z-factor / Residual H)"):
            zcols = st.columns(3)
            for ci, (cap, key) in enumerate([("Z (vapor)", "Z_vapor"), ("Z (liquid)", "Z_liquid"), ("H_res (J/mol)", "H_res")]):
                with zcols[ci]:
                    st.caption(cap)
                    val = pr_result.get(key, 0)
                    fmt_spec = ".6f" if "Z" in key else ".2f"
                    st.markdown(f'<span style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">{val:{fmt_spec}}</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    st.subheader(t["plot_header"])
    T_min = max(50.0, Tc - 200.0)
    T_max = min(2000.0, Tc + 300.0)
    T_range = np.linspace(T_min, T_max, 80)
    lang_label = "\u751f\u6210\u66f2\u7ebf\u4e2d..." if is_zh else "Generating curves..."
    with st.spinner(lang_label):
        fig = create_property_plots(fluid_info, P_pa, T_range, st.session_state["lang"])
    st.plotly_chart(fig, width='stretch')

    # Export PDF
    st.session_state["_fig"] = fig
    st.session_state["_pr_result"] = pr_result
    st.session_state["_cp_result"] = cp_result
    st.session_state["_fluid_info"] = fluid_info
    st.session_state["_P_pa"] = P_pa

    if st.button(t.get("export_btn", "\U0001f4e5 \u5bfc\u51fa\u62a5\u544a (PDF)"), key="export_pdf"):
        with st.spinner("\u751f\u6210\u62a5\u544a\u4e2d..." if is_zh else "Generating report..."):
            pdf_bytes = export_report_pdf(pr_result, cp_result, fluid_info, P_pa, st.session_state["_fig"], st.session_state["lang"])
        st.success(t.get("export_success", "\u2705 \u62a5\u544a\u5df2\u751f\u6210"))
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="\U0001f4e5 \u4e0b\u8f7d PDF" if is_zh else "\U0001f4e5 Download PDF",
            data=pdf_bytes, file_name=f"ThermoCalc_Report_{ts}.pdf",
            mime="application/pdf", key="dl_pdf"
        )

def export_report_pdf(pr_result, cp_result, fluid_info, P_pa, fig, lang):
    """Generate a PDF report with results table and charts."""
    import io, os
    from datetime import datetime
    from fpdf import FPDF

    name_zh, name_en, T_triple, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info
    fluid_display = name_zh if lang == "zh" else name_en

    # Try to use CJK font for Chinese; fall back to built-in for English
    CJK_FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
    use_cjk = lang == "zh" and os.path.exists(CJK_FONT_PATH)

    class PDF(FPDF):
        def header(self):
            if use_cjk:
                self.set_font("cjk", "B", 14)
            else:
                self.set_font("Helvetica", "B", 14)
            self.set_text_color(30, 64, 175)
            title = "ThermoCalc - \u70ed\u7269\u6027\u8ba1\u7b97\u62a5\u544a" if lang == "zh" else "ThermoCalc - Property Calculation Report"
            self.cell(0, 10, title, align="C", new_x="LMARGIN", new_y="NEXT")
            if use_cjk:
                self.set_font("cjk", "", 9)
            else:
                self.set_font("Helvetica", "I", 9)
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
        if use_cjk:
            pdf.set_font("cjk", "B", 12)
        else:
            pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")

    def write_body(pdf, text, use_cjk):
        if use_cjk:
            pdf.set_font("cjk", "", 10)
        else:
            pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50)
        pdf.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")        # Transport property note for tc and viscosity
        if key == "thermal_conductivity" and "tc_note" in t:
            card_html += f'<div class="prop-note">{t["tc_note"]}</div>'
        if key == "viscosity" and "mu_note" in t:
            card_html += f'<div class="prop-note">{t["mu_note"]}</div>'
            st.markdown(card_html, unsafe_allow_html=True)

    # ---- Section 1 ----
    s1 = "1. \u8ba1\u7b97\u53c2\u6570" if lang == "zh" else "1. Calculation Parameters"
    write_section(pdf, s1, use_cjk)

    info_lines = [
        f"  Fluid: {fluid_display}  |  M = {M_gmol} g/mol",
        f"  Tc = {Tc} K  |  Pc = {Pc} MPa  |  omega = {omega}",
        f"  Pressure = {P_pa/1e6:.2f} MPa",
    ]
    for ln in info_lines:
        write_body(pdf, ln, use_cjk)
        if key == "thermal_conductivity" and "tc_note" in t:
            card_html += f'<div class="prop-note">{t["tc_note"]}</div>'
        if key == "viscosity" and "mu_note" in t:
            card_html += f'<div class="prop-note">{t["mu_note"]}</div>'
    st.markdown(card_html, unsafe_allow_html=True)

    # ---- Section 2: Results Table ----
    s2 = "2. \u7269\u6027\u8ba1\u7b97\u7ed3\u679c" if lang == "zh" else "2. Property Results"
    write_section(pdf, s2, use_cjk)

    col_w = [45, 35, 35, 35, 25]
    headers = (
        ["\u7269\u6027", "PR\u65b9\u7a0b", "CoolProp", "\u5355\u4f4d", "\u504f\u5dee"]
        if lang == "zh"
        else ["Property", "PR EOS", "CoolProp", "Unit", "Deviation"]
    )
    if use_cjk:
        pdf.set_font("cjk", "B", 9)
    else:
        pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(30, 64, 175)
    pdf.set_text_color(255)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    props = [
        ("density", "\u5bc6\u5ea6", "Density", "kg/m³"),
        ("cp", "\u5b9a\u538b\u6bd4\u70ed\u5bb9 Cp", "Cp", "kJ/(kg.K)"),
        ("cv", "\u5b9a\u5bb9\u6bd4\u70ed\u5bb9 Cv", "Cv", "kJ/(kg.K)"),
        ("thermal_conductivity", "\u5bfc\u70ed\u7cfb\u6570 \u03bb", "\u03bb", "W/(m.K)"),
        ("viscosity", "\u52a8\u529b\u7c98\u5ea6 \u03bc", "\u03bc", "\u03bcPa.s"),
    ]

    if use_cjk:
        pdf.set_font("cjk", "", 9)
    else:
        pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50)
    for key, zname, ename, unit in props:
        name = zname if lang == "zh" else ename
        pr_val = pr_result.get(key) if (pr_result and "error" not in pr_result) else None
        cp_val = cp_result.get(key) if (cp_result and "error" not in cp_result) else None
        pr_s = f"{pr_val:.4f}" if pr_val is not None else "N/A"
        cp_s = f"{cp_val:.4f}" if cp_val is not None else "N/A"
        if pr_val is not None and cp_val is not None and cp_val != 0:
            dev = (pr_val - cp_val) / abs(cp_val) * 100
            dev_s = f"{dev:+.2f}%"
        else:
            dev_s = "N/A"
        pdf.cell(col_w[0], 6, name, border=1)
        pdf.cell(col_w[1], 6, pr_s, border=1, align="R")
        pdf.cell(col_w[2], 6, cp_s, border=1, align="R")
        pdf.cell(col_w[3], 6, unit, border=1, align="C")
        pdf.cell(col_w[4], 6, dev_s, border=1, align="C")
        pdf.ln()

    # Z-factor
    if pr_result and "error" not in pr_result:
        pdf.ln(2)
        zlabel = "\u4e2d\u95f4\u53d8\u91cf" if lang == "zh" else "Intermediate Variables"
        write_section(pdf, zlabel, use_cjk)
        if use_cjk:
            pdf.set_font("cjk", "", 9)
        else:
            pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50)
        for zl in [
            f"  Z (vapor) = {pr_result.get('Z_vapor', 0):.6f}",
            f"  Z (liquid) = {pr_result.get('Z_liquid', 0):.6f}",
            f"  H_res = {pr_result.get('H_res', 0):.2f} J/mol",
        ]:
            pdf.cell(0, 6, zl, new_x="LMARGIN", new_y="NEXT")        # Transport property note for tc and viscosity
        if key == "thermal_conductivity" and "tc_note" in t:
            card_html += f'<div class="prop-note">{t["tc_note"]}</div>'
        if key == "viscosity" and "mu_note" in t:
            card_html += f'<div class="prop-note">{t["mu_note"]}</div>'
    st.markdown(card_html, unsafe_allow_html=True)

    # ---- Section 3: Charts ----
    pdf.add_page()
    s3 = "3. \u7269\u6027-\u6e29\u5ea6\u66f2\u7ebf\u56fe" if lang == "zh" else "3. Property-Temperature Curves"
    write_section(pdf, s3, use_cjk)
    pdf.ln(2)

    try:
        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
        img_path = "__tmp_chart.png"
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        pdf.image(img_path, x=10, w=pdf.w - 20)
        os.remove(img_path)
    except Exception:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8, "(Chart image could not be rendered. Please view the web app for interactive charts.)", new_x="LMARGIN", new_y="NEXT")        # Transport property note for tc and viscosity
        if key == "thermal_conductivity" and "tc_note" in t:
            card_html += f'<div class="prop-note">{t["tc_note"]}</div>'
        if key == "viscosity" and "mu_note" in t:
            card_html += f'<div class="prop-note">{t["mu_note"]}</div>'
    st.markdown(card_html, unsafe_allow_html=True)

    # ---- Section 4: Disclaimer ----
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(128)
    disc_en = "Note: PR EOS is a classical cubic equation of state. For highly polar substances, systematic deviations of 5-15% are expected. CoolProp values serve as reference benchmarks."
    disc_zh = "\u6ce8\uff1aPR\u65b9\u7a0b\u4e3a\u7ecf\u5178\u7acb\u65b9\u578b\u72b6\u6001\u65b9\u7a0b\uff0c\u5bf9\u5f3a\u6781\u6027\u7269\u8d28\u5b58\u5728\u7ea65-15%\u7cfb\u7edf\u504f\u5dee\u3002CoolProp\u503c\u4f5c\u4e3a\u57fa\u51c6\u53c2\u8003\u3002"
    disc = disc_zh if lang == "zh" else disc_en
    if use_cjk:
        pdf.set_font("cjk", "", 8)
    pdf.multi_cell(0, 5, disc)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf.getvalue()


def render_validation_page(t):
    """Render the model validation page."""
    st.header(t["validate_title"])
    st.markdown(t["validate_desc"])
    st.markdown("---")

    benchmarks = [
        ("Methane",        300.0, 0.1),
        ("Methane",        300.0, 1.0),
        ("Water",          400.0, 0.1),
        ("Water",          500.0, 1.0),
        ("CarbonDioxide",  300.0, 1.0),
        ("CarbonDioxide",  350.0, 5.0),
    ]

    props_to_check = ["density", "cp", "thermal_conductivity", "viscosity"]

    rows = []
    for cp_name, T_val, P_mpa in benchmarks:
        P_pa = P_mpa * 1e6
        fluid_info = None
        for item in FLUID_DATABASE:
            if item[7] == cp_name:
                fluid_info = item
                break
        if fluid_info is None:
            continue

        name_zh = fluid_info[0]
        M = fluid_info[2] / 1000.0

        pr_res = pr_engine_properties(T_val, P_pa, fluid_info)
        cp_res = coolprop_properties(T_val, P_pa, cp_name, M)

        for prop_key in props_to_check:
            pr_val = pr_res.get(prop_key) if "error" not in pr_res else None
            cp_val = cp_res.get(prop_key) if "error" not in cp_res else None
            dev = calc_deviation(pr_val, cp_val)
            rows.append({
                t["validate_col_fluid"]: name_zh,
                t["validate_col_T"]: T_val,
                t["validate_col_P"]: P_mpa,
                t["validate_col_prop"]: prop_key,
                t["validate_col_PR"]: f"{pr_val:.4f}" if pr_val else "N/A",
                t["validate_col_CP"]: f"{cp_val:.4f}" if cp_val else "N/A",
                t["validate_col_dev"]: f"{dev:.2f}%" if dev else "N/A",
            })

    df = pd.DataFrame(rows)

    def color_dev(val):
        try:
            v = float(str(val).replace("%", ""))
        except:
            return ""
        if abs(v) < 5:
            return "color: green; font-weight: bold"
        elif abs(v) < 20:
            return "color: orange; font-weight: bold"
        else:
            return "color: red; font-weight: bold"

    styled_df = df.style.map(color_dev, subset=[t["validate_col_dev"]])
    st.dataframe(styled_df, width='stretch', height=500)

    st.caption("\u7eff\u8272\u504f\u5dee<5% | \u6a59\u82725-20% | \u7ea2\u8272>20%")




def material_screening_page(t):
    """Material screening page: scan all 20 substances and filter by property criteria."""
    st.subheader(t["screening_title"])
    st.caption(t["screening_desc"])
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        T_screen = st.number_input(
            f'{t["temperature"]} ({t["unit_temp"]})',
            min_value=200.0, max_value=600.0, value=300.0, step=1.0,
            key="screen_T"
        )
    with col2:
        P_screen = st.number_input(
            f'{t["pressure"]} ({t["unit_press"]})',
            min_value=0.1, max_value=10.0, value=1.0, step=0.1,
            key="screen_P"
        )
    
    prop_options = {
        t["density"]: "density",
        t["cp"]: "cp",
        t["thermal_cond"]: "thermal_conductivity",
        t["viscosity"]: "viscosity",
        t["thermal_expansion"]: "thermal_expansion",
    }
    
    col_a, col_b, col_c = st.columns([1.5, 1, 1.5])
    with col_a:
        prop_display = st.selectbox(
            t["screening_target_prop"],
            list(prop_options.keys()),
            key="screen_prop"
        )
    with col_b:
        condition = st.selectbox(
            t["screening_condition"],
            [">", "<"],
            key="screen_cond"
        )
    with col_c:
        threshold = st.number_input(
            t["screening_threshold"],
            value=0.0, step=0.001, format="%.6e",
            key="screen_threshold"
        )
    
    if st.button(t["screening_btn"], use_container_width=True, key="screen_btn"):
        target_key = prop_options[prop_display]
        P_pa = P_screen * 1e6
        is_greater = (condition == ">")
        
        results = []
        errors = []
        
        with st.spinner("Calculating all 20 substances..."):
            for fluid_info in FLUID_DATABASE:
                name_display = fluid_info[0] if st.session_state.get("lang", "zh") == "zh" else fluid_info[1]
                try:
                    pr_res = pr_engine_properties(T_screen, P_pa, fluid_info)
                    if "error" in pr_res:
                        errors.append(f"{name_display}: {pr_res['error']}")
                        results.append({
                            t["screening_col_fluid"]: name_display,
                            t["screening_col_value"]: "N/A",
                            t["screening_col_meet"]: t["screening_not_meet"],
                        })
                        continue
                    
                    val = pr_res.get(target_key)
                    if val is None:
                        results.append({
                            t["screening_col_fluid"]: name_display,
                            t["screening_col_value"]: "N/A",
                            t["screening_col_meet"]: t["screening_not_meet"],
                        })
                        continue
                    
                    meets = (val > threshold) if is_greater else (val < threshold)
                    results.append({
                        t["screening_col_fluid"]: name_display,
                        t["screening_col_value"]: val,
                        t["screening_col_meet"]: t["screening_meet"] if meets else t["screening_not_meet"],
                    })
                except Exception as e:
                    errors.append(f"{name_display}: {str(e)}")
                    results.append({
                        t["screening_col_fluid"]: name_display,
                        t["screening_col_value"]: "Error",
                        t["screening_col_meet"]: t["screening_not_meet"],
                    })
        
        meet_str = t["screening_meet"]
        met_results = [r for r in results if r[t["screening_col_meet"]] == meet_str]
        not_met = [r for r in results if r[t["screening_col_meet"]] != meet_str]
        
        if is_greater:
            met_results.sort(key=lambda x: x[t["screening_col_value"]] if isinstance(x[t["screening_col_value"]], (int, float)) else float('-inf'), reverse=True)
            not_met.sort(key=lambda x: x[t["screening_col_value"]] if isinstance(x[t["screening_col_value"]], (int, float)) else float('-inf'), reverse=True)
        else:
            met_results.sort(key=lambda x: x[t["screening_col_value"]] if isinstance(x[t["screening_col_value"]], (int, float)) else float('inf'))
            not_met.sort(key=lambda x: x[t["screening_col_value"]] if isinstance(x[t["screening_col_value"]], (int, float)) else float('inf'))
        
        all_sorted = met_results + not_met
        
        if met_results:
            st.success(t["screening_results_count"].format(len(met_results)))
        else:
            st.warning(t["screening_no_results"])
        
        display_rows = []
        for r in all_sorted:
            val = r[t["screening_col_value"]]
            if isinstance(val, float):
                if abs(val) < 0.001 or abs(val) >= 10000:
                    val_str = f"{val:.4e}"
                else:
                    val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            display_rows.append({
                t["screening_col_fluid"]: r[t["screening_col_fluid"]],
                t["screening_col_value"]: val_str,
                t["screening_col_meet"]: r[t["screening_col_meet"]],
            })
        
        df = pd.DataFrame(display_rows)
        
        def highlight_meet(val):
            if val == meet_str:
                return 'color: #6ee7b7; font-weight: bold'
            return 'color: rgba(255,255,255,0.3)'
        
        styled = df.style.map(highlight_meet, subset=[t["screening_col_meet"]])
        st.dataframe(styled, width='stretch', height=600)
        
        if errors:
            with st.expander(f"\u26a0 {len(errors)} calculation errors"):
                for e in errors:
                    st.caption(e)


def main():
    """Main Streamlit application entry point."""

    st.set_page_config(
        page_title="ThermoCalc",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS (dark sci-tech + property cards)
    st.markdown("""<style>
    /* ============================================================
       iOS Glassmorphism - Balanced Cards (v2.3)
       ============================================================ */

    .stApp {
        background: linear-gradient(160deg, #0f0c29 0%, #1a1744 30%, #24243e 70%, #0f0c29 100%);
        color: #e2e8f0; min-height: 100vh;
    }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: rgba(20, 18, 50, 0.75) !important;
        backdrop-filter: blur(40px) saturate(200%);
        -webkit-backdrop-filter: blur(40px) saturate(200%);
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.4);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* --- Buttons --- */
    .stButton > button {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #e2e8f0 !important; border-radius: 14px !important;
        font-weight: 600 !important;
        transition: all 0.30s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .stButton > button:hover {
        background: rgba(56, 189, 248, 0.12) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.22), 0 8px 25px rgba(0, 0, 0, 0.5);
        transform: translateY(-2px) scale(1.01); color: #fff !important;
    }
    .stButton > button:active { transform: scale(0.97) !important; }

    /* --- Inputs --- */
    input, .stNumberInput input {
        background: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        color: #e2e8f0 !important; border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    input:focus {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.12) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important; border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
    }
    .stSlider > div > div > div > div { background: rgba(56, 189, 248, 0.35) !important; }

    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
    header[data-testid="stHeader"] {
        background: transparent !important; box-shadow: none !important; border-bottom: none !important;
    }

    /* ============================================================
       PROPERTY CARD (balanced, ~150px height)
       ============================================================ */
    .prop-card-final {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 12px 20px 14px;
        margin-bottom: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255,255,255,0.03);
        transition: all 0.35s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    .prop-card-final:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.22);
        background: rgba(255, 255, 255, 0.07);
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.40), 0 0 25px rgba(56, 189, 248, 0.04);
    }

    /* --- Card name (top row, subtle) --- */
    .pcf-name {
        font-size: 0.70rem;
        text-transform: uppercase;
        letter-spacing: 2.0px;
        color: rgba(255, 255, 255, 0.38);
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* --- Body: flex row of columns --- */
    .pcf-body {
        display: flex;
        align-items: center;
        gap: 0;
    }

    /* --- PR / CP columns (50% each) --- */
    .pcf-col {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 4px 8px;
    }

    /* --- Engine tag --- */
    .pcf-engine-tag {
        font-size: 0.56rem;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        padding: 2px 9px;
        border-radius: 7px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .pr-tag {
        color: #c4b5fd;
        background: rgba(167, 139, 250, 0.08);
        border: 1px solid rgba(167, 139, 250, 0.14);
    }
    .cp-tag {
        color: #67e8f9;
        background: rgba(34, 211, 238, 0.08);
        border: 1px solid rgba(34, 211, 238, 0.14);
    }

    /* --- Value row (dot + value) --- */
    .pcf-val-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 2px;
    }
    .pcf-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .dot-green  { background: #6ee7b7; box-shadow: 0 0 8px rgba(110,231,183,0.45); }
    .dot-yellow { background: #fbbf24; box-shadow: 0 0 8px rgba(251,191,36,0.45); }
    .dot-red    { background: #fb923c; box-shadow: 0 0 8px rgba(251,146,60,0.45); }
    .dot-na     { background: rgba(255,255,255,0.18); }

    .pcf-val {
        font-size: 1.65rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
        line-height: 1.1;
    }
    .pr-val-v2 {
        background: linear-gradient(135deg, #c4b5fd, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .cp-val-v2 {
        background: linear-gradient(135deg, #67e8f9, #22d3ee);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* --- Unit --- */
    .pcf-unit {
        font-size: 0.64rem;
        color: rgba(255, 255, 255, 0.28);
        letter-spacing: 0.2px;
    }

    /* --- Vertical divider --- */
    .pcf-divider {
        width: 1px;
        height: 65px;
        background: linear-gradient(180deg, transparent, rgba(255,255,255,0.12), transparent);
        flex-shrink: 0;
        margin: 0 4px;
    }

    /* --- Deviation section (right side) --- */
    .pcf-dev {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        min-width: 80px;
        flex-shrink: 0;
        padding-left: 4px;
    }
    .pcf-dev-label {
        font-size: 0.58rem;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: rgba(255, 255, 255, 0.30);
        font-weight: 600;
    }

    /* --- Deviation badges --- */
    .dev-badge-v2 {
        font-size: 1.2rem; font-weight: 700;
        padding: 3px 12px; border-radius: 14px;
        display: inline-flex; align-items: center; gap: 5px;
        backdrop-filter: blur(8px);
    }
    .dev-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
    .dev-green-v2  { color: #6ee7b7; background: rgba(16, 185, 129, 0.10); border: 1px solid rgba(16,185,129,0.16); }
    .dev-green-v2 .dev-dot { background: #6ee7b7; }
    .dev-yellow-v2 { color: #fbbf24; background: rgba(245, 158, 11, 0.10); border: 1px solid rgba(245,158,11,0.16); }
    .dev-yellow-v2 .dev-dot { background: #fbbf24; }
    .dev-red-v2    { color: #fb923c; background: rgba(249, 115, 22, 0.10); border: 1px solid rgba(249,115,22,0.16); }
    .dev-red-v2 .dev-dot { background: #fb923c; }
    .dev-na-v2     { color: rgba(255,255,255,0.30); background: rgba(100,116,139,0.06); }

    /* --- Transport property note --- */
    .prop-note {
        font-size: 0.58rem;
        color: rgba(255, 255, 255, 0.25);
        margin-top: 6px;
        font-style: italic;
        line-height: 1.3;
    }

    /* --- Status bar --- */
    .status-bar { white-space: nowrap;
        display: flex; align-items: center; gap: 10px;
        padding: 10px 14px; border-radius: 12px;
        background: rgba(255, 255, 255, 0.04); backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 14px; font-size: 0.73rem; color: rgba(255, 255, 255, 0.50);
    }
    .status-dot {
        width: 8px; height: 8px; border-radius: 50%;
        background: #10b981; box-shadow: 0 0 12px #10b981;
        animation: statusPulse 2s ease-in-out infinite;
    }
    @keyframes statusPulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* --- Title --- */
    .title-glow {
        font-size: 1.7rem; font-weight: 800;
        background: linear-gradient(135deg, #c4b5fd, #38bdf8, #67e8f9, #c4b5fd);
        background-size: 300% 300%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 6s ease infinite; letter-spacing: -0.5px;
    }
    @keyframes gradientShift { 0%{background-position:0 50%} 50%{background-position:100% 50%} 100%{background-position:0 50%} }
    .version-chip {
        display: inline-block; font-size: 0.62rem; padding: 3px 12px; border-radius: 12px;
        background: rgba(56, 189, 248, 0.10); backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.18); color: #38bdf8;
        font-weight: 600; letter-spacing: 1.5px; vertical-align: middle; margin-left: 10px;
    }

    /* --- Expander --- */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.03) !important;
        backdrop-filter: blur(8px);
    }
    .stExpander:hover { border-color: rgba(255,255,255,0.15) !important; }

    /* --- Scrollbar --- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.18); }

    /* --- Fluid info line --- */
    .fluid-info-line {
        font-size: 0.76rem;
        color: rgba(255, 255, 255, 0.38);
        margin-bottom: 4px;
    }
    

</style>""", unsafe_allow_html=True)
    # Session state
    for k, v in {"lang": "zh", "calc_done": False, "pr_result": None, "cp_result": None,
                  "T_input": 300.0, "P_input": 1.0, "fluid_idx": 0,
                  "fluid_info": None, "P_pa": None, "range_warning": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    t = LANG[st.session_state["lang"]]

    with st.sidebar:
        st.header(t["sidebar_header"])

        # ---- Mode switch (Material Screening / Property Calc) ----
        if "app_mode" not in st.session_state:
            st.session_state["app_mode"] = "calc"
        
        mode_choice = st.radio(
            t["mode_label"],
            options=[t["mode_calc"], t["mode_screening"]],
            index=0 if st.session_state["app_mode"] == "calc" else 1,
            horizontal=True,
            key="mode_sel"
        )
        new_mode = "calc" if mode_choice == t["mode_calc"] else "screening"
        if new_mode != st.session_state["app_mode"]:
            st.session_state["app_mode"] = new_mode
            st.rerun()
        
        st.markdown("---")

        # Language toggle
        lang_choice = st.radio(
            t["lang_toggle"],
            options=["中文", "English"],
            index=0 if st.session_state["lang"] == "zh" else 1,
            horizontal=True,
            key="lang_sel"
        )
        new_lang = "zh" if lang_choice == "中文" else "en"
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
            st.rerun()

        t = LANG[st.session_state["lang"]]
        st.markdown("---")

        fluid_options = [item[0] if st.session_state["lang"] == "zh" else item[1] for item in FLUID_DATABASE]

        # Temperature input
        T_input = st.number_input(
            f'{t["temperature"]} ({t["unit_temp"]})',
            min_value=200.0, max_value=600.0,
            value=st.session_state.get("T_input", 300.0), step=1.0,
            key="T_input"
        )
        # Pressure input
        P_input = st.number_input(
            f'{t["pressure"]} ({t["unit_press"]})',
            min_value=0.1, max_value=10.0,
            value=st.session_state.get("P_input", 1.0), step=0.1,
            key="P_input"
        )

        # Fluid selection
        fluid_choice = st.selectbox(
            t["fluid_select"],
            fluid_options,
            index=st.session_state.get("fluid_idx", 0),
            key="fluid_sel"
        )

        # Calculate button
        if st.button(t["calc_button"], use_container_width=True):
            for i, item in enumerate(FLUID_DATABASE):
                if item[0 if st.session_state["lang"] == "zh" else 1] == fluid_choice:
                    st.session_state["fluid_idx"] = i
                    break
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

        # Scope
        with st.expander(t["scope_title"], expanded=True):
            st.markdown(t["scope_text"], unsafe_allow_html=True)

        # Engine status
        st.markdown(
            '<div class="status-bar">'
            '<span class="status-dot"></span> '
            + ("🟢 引擎就绪 | PR + CoolProp" if st.session_state["lang"] == "zh" else "🟢 Engines Ready | PR + CoolProp")
            + '</div>',
            unsafe_allow_html=True,
        )

    # ---- Main content area (outside sidebar) ----
    if st.session_state.get("app_mode", "calc") == "screening":
        material_screening_page(t)
    elif st.session_state.get("calc_done", False) and st.session_state.get("fluid_info"):
        # Render property calculation results
        fluid_info = st.session_state["fluid_info"]
        pr_result = st.session_state.get("pr_result")
        cp_result = st.session_state.get("cp_result")
        P_pa = st.session_state.get("P_pa")
        range_warning = st.session_state.get("range_warning")
        
        if range_warning == "range":
            st.warning(t["warn_range"])
        
        render_results(pr_result, cp_result, fluid_info, P_pa, t)
    else:
        # First visit: show welcome message
        st.info(t["first_time_msg"])
