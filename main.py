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
        "cv": "定容比热容 Cv",
        "thermal_cond": "导热系数 λ",
        "viscosity": "动力粘度 μ",
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
        "calc_ok": "✅ 计算成功完成！",
        "about_title": "ℹ️ 关于本软件",
        "about_text": "**热物性计算软件** v1.0 为化工软件开发比赛设计。<br><br>**核心特色:**<br>- 🔩 **自研Peng-Robinson方程引擎**: 手写PR三次方程求解、剩余性质计算、对应态原理粘度/导热系数估算<br>- 📎 **CoolProp基准引擎**: 调用工业级物性数据库作为高精度对照<br>- 📈 **Plotly交互图表**: 悬停数值、缩放拖拽、双曲线叠加对比<br>- 🌐 **中英双语界面**: 一键切换<br><br>**适用范围:** 气相、液相、超临界态流体热物性估算",
        "first_time_msg": "⏳ 请在左侧输入参数并点击「开始计算」",
        "fluid_info_label": "**物质:** {}  |  M = {} g/mol  |  Tc = {} K  |  Pc = {} MPa  |  ω = {}",
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
        "scope_text": "推荐适用范围：温度 200-600 K，压力 0.1-10 MPa。超出此范围时，PR方程计算偏差可能增大，建议以CoolProp基准值为参考。",
        "meta_calc_convergence_error": "计算未收敛。可能原因: 输入工况接近临界点或超出了PR方程的适用极限。建议微调温度或压力值。",
        "meta_mixture_warning": "当前版本仅支持纯物质计算, 混合物功能正在开发中",
        "meta_page": "页面",
        "meta_main_page": "🏠 物性计算",
        "meta_verify_page": "🔬 模型验证",
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
        "cv": "Specific Heat Cv",
        "thermal_cond": "Thermal Conductivity λ",
        "viscosity": "Viscosity μ",
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
        "calc_ok": "✅ Calculation completed successfully!",
        "about_title": "ℹ️ About",
        "about_text": "**Thermodynamic Property Calculator** v1.0 - Built for chemical engineering software competition.<br><br>**Key Features:**<br>- 🔩 **Self-developed PR EOS Engine**: Handwritten cubic equation solver, residual properties, corresponding-state transport properties<br>- 📎 **CoolProp Benchmark Engine**: Industrial-grade thermodynamic database as reference<br>- 📈 **Plotly Interactive Charts**: Hover values, zoom/pan, dual-curve overlay comparison<br>- 🌐 **Bilingual Interface**: One-click Chinese/English switch<br><br>**Scope:** Gas, liquid, and supercritical fluid property estimation",
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
    },
}

FLUID_DATABASE = [
    ("甲烷",   "Methane",    16.043,  190.56, 4.599, 0.011,   [19.25, 0.05213, 1.197e-5, -1.132e-8], "Methane"),
    ("乙烷",   "Ethane",     30.070,  305.32, 4.872, 0.099,   [ 5.41, 0.17809, -6.938e-5, 8.713e-9], "Ethane"),
    ("丙烷",   "Propane",    44.096,  369.83, 4.248, 0.152,   [-4.22, 0.30630, -1.586e-4, 3.215e-8], "Propane"),
    ("正丁烷", "n-Butane",   58.122,  425.12, 3.796, 0.200,   [ 9.49, 0.33130, -1.108e-4, -2.822e-9], "n-Butane"),
    ("正戊烷", "n-Pentane",  72.149,  469.70, 3.370, 0.251,   [-3.63, 0.48730, -2.580e-4, 5.305e-8], "n-Pentane"),
    ("乙烯",   "Ethylene",   28.054,  282.34, 5.041, 0.086,   [ 3.81, 0.15660, -8.348e-5, 1.755e-8], "Ethylene"),
    ("丙烯",   "Propylene",  42.080,  364.90, 4.600, 0.144,   [ 3.71, 0.23450, -1.160e-4, 2.205e-8], "Propylene"),
    ("苯",     "Benzene",    78.112,  562.05, 4.895, 0.210,   [-33.90, 0.56390, -4.133e-4, 1.202e-7], "Benzene"),
    ("甲苯",   "Toluene",    92.138,  591.75, 4.108, 0.264,   [-24.36, 0.51250, -2.765e-4, 4.911e-8], "Toluene"),
    ("甲醇",   "Methanol",   32.042,  512.64, 8.097, 0.565,   [ 21.15, 0.07092, 2.587e-5, -2.852e-8], "Methanol"),
    ("乙醇",   "Ethanol",    46.068,  513.90, 6.148, 0.643,   [ 9.38, 0.30928, -1.706e-4, 3.787e-8], "Ethanol"),
    ("水",     "Water",      18.015,  647.10, 22.064, 0.344,   [ 32.24, 0.00192, 1.055e-5, -3.596e-9], "Water"),
    ("氨",     "Ammonia",    17.031,  405.40, 11.333, 0.256,   [ 27.32, 0.02383, 1.707e-5, -1.185e-8], "Ammonia"),
    ("二氧化碳","CO2",       44.010,  304.13, 7.377, 0.225,   [ 19.80, 0.07344, -5.602e-5, 1.715e-8], "CarbonDioxide"),
    ("一氧化碳","CO",        28.010,  132.86, 3.494, 0.048,   [ 30.87, -0.01285, 2.789e-5, -1.272e-8], "CarbonMonoxide"),
    ("氮气",   "Nitrogen",   28.013,  126.19, 3.396, 0.037,   [ 31.15, -0.01357, 2.680e-5, -1.168e-8], "Nitrogen"),
    ("氧气",   "Oxygen",     31.999,  154.58, 5.043, 0.021,   [ 28.11, -0.00368, 1.746e-5, -1.065e-8], "Oxygen"),
    ("氢气",   "Hydrogen",    2.016,   33.15, 1.296,-0.216,   [ 27.14,  0.00927, -1.381e-5, 7.645e-9], "Hydrogen"),
    ("氦气",   "Helium",      4.003,    5.20, 0.227,-0.390,   [ 20.79,  0.0,      0.0,       0.0     ], "Helium"),
    ("R134a",  "R134a",     102.030,  374.21, 4.059, 0.327,   [ 16.34, 0.26850, -1.457e-4, 2.492e-8], "R134a"),
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
    Returns (Z_vapor, Z_liquid, Z_unstable).
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
    """Estimate thermal conductivity via Eucken/Chung method [W/(m*K)]"""
    M_g = M * 1000.0  # g/mol
    Tr = T / Tc
    Cv_ideal = max(Cp_ideal - R_GAS, R_GAS * 1.5)

    # Collision integral
    Omega_v = (
        1.16145 * Tr**(-0.14874)
        + 0.52487 * np.exp(-0.77320 * Tr)
        + 2.16178 * np.exp(-2.43787 * Tr)
    )
    Fc = 1.0 - 0.2756 * omega

    Vc_est = M_g * R_GAS * Tc / Pc
    sigma = np.cbrt(0.809 * Vc_est)
    mu0_muPas = 40.785 * Fc * np.sqrt(M_g * T) / (sigma**2 * Omega_v) * 1e-3
    mu0_muPas *= (1.0 + 2.5 * omega)
    mu0_Pas = mu0_muPas * 1e-6

    # Modified Eucken correlation
    lambda0 = mu0_Pas * Cv_ideal / (M_g / 1000.0) * (1.32 + 1.77 * (R_GAS / Cv_ideal)) * 0.15

    # Dense fluid enhancement
    rho_actual = pr_density(Z, T, P, M)
    rho_c = M * Pc / (R_GAS * Tc * 0.28)
    rho_r = max(rho_actual / (rho_c + 1e-15), 0.0)
    if rho_r > 0.05:
        lambda0 *= (1.0 + 0.5 * rho_r**0.8)

    return max(lambda0, 0.001)


def estimate_viscosity_pr(T, P, Z, M, Tc, Pc, omega):
    """Estimate dynamic viscosity via Chung et al. method [muPa*s]"""
    M_g = M * 1000.0
    Tr = T / Tc

    Omega_v = (
        1.16145 * Tr**(-0.14874)
        + 0.52487 * np.exp(-0.77320 * Tr)
        + 2.16178 * np.exp(-2.43787 * Tr)
    )
    Fc = 1.0 - 0.2756 * omega

    Vc_est = M_g * R_GAS * Tc / Pc
    sigma = np.cbrt(0.809 * Vc_est)
    mu0 = 40.785 * Fc * np.sqrt(M_g * T) / (sigma**2 * Omega_v) * 1e-3 * 0.12
    mu0 *= (1.0 + 2.5 * omega)

    # Dense gas correction
    rho_actual = pr_density(Z, T, P, M)
    rho_c = M * Pc / (R_GAS * Tc * 0.28)
    rho_r = max(rho_actual / (rho_c + 1e-15), 0.0)

    if rho_r > 0.01:
        FK = np.exp(1.15 * rho_r**0.85 / (Tr + 0.1))
        mu_total = mu0 * FK
    else:
        mu_total = mu0

    return max(mu_total, 0.5)



# ============================================================================
# 4. CoolProp Engine Service
# ============================================================================

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

        if density <= 0 or np.isnan(density):
            raise ValueError(f"Invalid density: {density}")

        return {
            "density": density,                  # kg/m^3
            "cp": cp_mass / 1000.0,              # kJ/(kg*K)
            "cv": cv_mass / 1000.0,              # kJ/(kg*K)
            "thermal_conductivity": tc,          # W/(m*K)
            "viscosity": visc * 1e6,             # muPa*s
        }
    except Exception as e:
        return {"error": str(e)}


def pr_engine_properties(T, P, fluid_info):
    """Compute all properties using self-developed PR EOS.

    Args:
        T: Temperature in K
        P: Pressure in Pa
        fluid_info: Tuple from FLUID_DATABASE

    Returns:
        dict with: density, cp, cv, Z, H_res, S_res,
                   thermal_conductivity, viscosity,
                   Z_vapor, Z_liquid
    """
    try:
        name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info
        M = M_gmol / 1000.0  # kg/mol
        Pc_pa = Pc * 1e6     # MPa -> Pa

        # Step 1: Solve cubic for Z
        Z_v, Z_l, Z_u = solve_pr_cubic(T, P, Tc, Pc_pa, omega)

        # Select physically meaningful root
        if T < Tc:
            Z_used = Z_v  # Largest root = vapor-like
        else:
            Z_used = Z_v  # Supercritical

        if Z_used <= 0.01:
            raise ValueError(f"Abnormal Z = {Z_used:.6f}")

        # Step 2: Density
        density = pr_density(Z_used, T, P, M)

        # Step 3: Ideal gas heat capacity
        A, B_cp_coef, C_cp_coef, D_cp_coef = cp_coeffs
        Cp_ig_mol = (
            A
            + B_cp_coef * T
            + C_cp_coef * T**2
            + D_cp_coef * T**3
        )  # J/(mol*K)
        Cv_ig_mol = max(Cp_ig_mol - R_GAS, R_GAS * 1.5)

        # Step 4: Residual enthalpy & entropy
        H_res = pr_residual_enthalpy(T, P, Z_used, Tc, Pc_pa, omega)
        S_res = pr_residual_entropy(T, P, Z_used, Tc, Pc_pa, omega)

        # Step 5: Total Cp via real gas Cp = Cp_ig + Cp_res
        # Cp_res from PR EOS: Cp_res = R*(term_Cp) where
        # term_Cp involves d^2a/dT^2 which we approximate numerically
        # Use centered difference on (H_res + H_ig) instead of just H_res
        # But since H_ig cancels, we just need to compute H_res at T +/- delta
        # with Z re-solved at each temperature
        delta_T = 0.1
        # Re-solve Z at T+delta
        Z_p, _, _ = solve_pr_cubic(T + delta_T, P, Tc, Pc_pa, omega)
        H_p = pr_residual_enthalpy(T + delta_T, P, Z_p, Tc, Pc_pa, omega)
        # Re-solve Z at T-delta
        Z_m, _, _ = solve_pr_cubic(T - delta_T, P, Tc, Pc_pa, omega)
        H_m = pr_residual_enthalpy(T - delta_T, P, Z_m, Tc, Pc_pa, omega)
        Cp_res_contrib = (H_p - H_m) / (2.0 * delta_T)  # J/(mol*K)
        Cp_total_mol = Cp_ig_mol + Cp_res_contrib  # J/(mol*K)
        # Cp_total_mol is J/(mol*K), M_gmol is g/mol
        # Convert: J/(mol*K) / (kg/mol) / 1000 = kJ/(kg*K)
        # J/(mol*K) / (M_gmol/1000 kg/mol) / 1000 J/kJ = Cp_mol * 1000 / (M_gmol * 1000) = Cp_mol / M_gmol
        Cp_total = Cp_total_mol / M_gmol  # kJ/(kg*K) [since J/g = kJ/kg]
        Cv_total = Cv_ig_mol / M_gmol  # kJ/(kg*K)

        # Step 6: Transport properties (corresponding states)
        thermal_cond = estimate_thermal_conductivity_pr(
            T, P, Z_used, M, Tc, Pc_pa, omega, Cp_ig_mol
        )
        viscosity = estimate_viscosity_pr(
            T, P, Z_used, M, Tc, Pc_pa, omega
        )

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
        }
    except Exception as e:
        return {"error": str(e)}



# ============================================================================
# 6. Deviation Calculation
# ============================================================================

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
    """Create 4 interactive Plotly subplots with PR vs CoolProp overlay.

    Subplots: Density, Cp, Thermal Conductivity, Viscosity vs Temperature.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info

    # Pre-allocate arrays
    n = len(T_range)
    pr_density_arr = np.full(n, np.nan)
    pr_cp_arr = np.full(n, np.nan)
    pr_tc_arr = np.full(n, np.nan)
    pr_visc_arr = np.full(n, np.nan)
    cp_density_arr = np.full(n, np.nan)
    cp_cp_arr = np.full(n, np.nan)
    cp_tc_arr = np.full(n, np.nan)
    cp_visc_arr = np.full(n, np.nan)

    for i, T_val in enumerate(T_range):
        # PR Engine
        pr_res = pr_engine_properties(T_val, P_pa, fluid_info)
        if "error" not in pr_res:
            pr_density_arr[i] = pr_res.get("density", np.nan)
            pr_cp_arr[i] = pr_res.get("cp", np.nan)
            pr_tc_arr[i] = pr_res.get("thermal_conductivity", np.nan)
            pr_visc_arr[i] = pr_res.get("viscosity", np.nan)

        # CoolProp Engine
        cp_res = coolprop_properties(T_val, P_pa, cp_name, M_gmol / 1000.0)
        if "error" not in cp_res:
            cp_density_arr[i] = cp_res.get("density", np.nan)
            cp_cp_arr[i] = cp_res.get("cp", np.nan)
            cp_tc_arr[i] = cp_res.get("thermal_conductivity", np.nan)
            cp_visc_arr[i] = cp_res.get("viscosity", np.nan)

    # Labels
    if lang == "zh":
        subplot_titles = [
            "\u5bc6\u5ea6 vs \u6e29\u5ea6",
            "Cp vs \u6e29\u5ea6",
            "\u5bfc\u70ed\u7cfb\u6570 vs \u6e29\u5ea6",
            "\u7c98\u5ea6 vs \u6e29\u5ea6",
        ]
        y_labels = [
            "\u5bc6\u5ea6 (kg/m\u00b3)",
            "Cp (kJ/(kg\u00b7K))",
            "\u5bfc\u70ed\u7cfb\u6570 (W/(m\u00b7K))",
            "\u7c98\u5ea6 (\u00b5Pa\u00b7s)",
        ]
        legend_pr = "PR\u65b9\u7a0b(\u81ea\u7814)"
        legend_cp = "CoolProp(\u57fa\u51c6)"
        x_label = "\u6e29\u5ea6 (K)"
    else:
        subplot_titles = [
            "Density vs T",
            "Cp vs T",
            "TC vs T",
            "Viscosity vs T",
        ]
        y_labels = [
            "Density (kg/m\u00b3)",
            "Cp (kJ/(kg\u00b7K))",
            "TC (W/(m\u00b7K))",
            "Viscosity (\u00b5Pa\u00b7s)",
        ]
        legend_pr = "PR EOS (Self-dev)"
        legend_cp = "CoolProp (Ref.)"
        x_label = "Temperature (K)"

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    color_pr = "#1f77b4"
    color_cp = "#ff7f0e"

    data_pairs = [
        (pr_density_arr, cp_density_arr, 1, 1, y_labels[0]),
        (pr_cp_arr, cp_cp_arr, 1, 2, y_labels[1]),
        (pr_tc_arr, cp_tc_arr, 2, 1, y_labels[2]),
        (pr_visc_arr, cp_visc_arr, 2, 2, y_labels[3]),
    ]

    for idx, (pr_data, cp_data, row, col, yl) in enumerate(data_pairs):
        show_legend = idx == 0
        fig.add_trace(
            go.Scatter(
                x=T_range,
                y=pr_data,
                mode="lines",
                name=legend_pr,
                line=dict(color=color_pr, width=2),
                hovertemplate=(
                    f"{x_label}: %{{x:.1f}}<br>"
                    f"{yl}: %{{y:.3f}}<extra></extra>"
                ),
                legendgroup="pr",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=T_range,
                y=cp_data,
                mode="lines",
                name=legend_cp,
                line=dict(color=color_cp, width=2, dash="dash"),
                hovertemplate=(
                    f"{x_label}: %{{x:.1f}}<br>"
                    f"{yl}: %{{y:.3f}}<extra></extra>"
                ),
                legendgroup="cp",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

    for i, yl in enumerate(y_labels, 1):
        row = 1 if i <= 2 else 2
        col = 1 if i % 2 == 1 else 2
        fig.update_xaxes(title_text=x_label, row=row, col=col)
        fig.update_yaxes(title_text=yl, row=row, col=col)

    fig.update_layout(
        height=700,
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        margin=dict(l=50, r=30, t=80, b=50),
        template="plotly_white",
    )

    return fig



# ============================================================================
# 8. Streamlit UI - Main Application
# ============================================================================



# ============================================================================
# 8. Streamlit UI - Main Application
# ============================================================================

def run_calculation(T_input, P_input, fluid_info_tuple):
    """Execute both engines and return results. Cached for performance."""
    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info_tuple
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
    """Render the results section in the main content area."""
    name_zh, name_en, M_gmol, Tc, Pc, omega, cp_coeffs, cp_name = fluid_info

    fluid_display = name_zh if st.session_state["lang"] == "zh" else name_en
    st.markdown(
        t["fluid_info_label"].format(fluid_display, M_gmol, Tc, Pc, omega)
    )
    st.markdown("---")

    st.subheader(t["results_header"])
    st.success(t["calc_ok"])

    props_map = [
        ("density",          t["density"],          t["unit_density"]),
        ("cp",               t["cp"],               t["unit_cp"]),
        ("cv",               t["cv"],               t["unit_cp"]),
        ("thermal_conductivity", t["thermal_cond"], t["unit_tc"]),
        ("viscosity",        t["viscosity"],        t["unit_visc"]),
    ]

    col_pr, col_cp, col_dev = st.columns(3)

    with col_pr:
        st.markdown(f"**{t['pr_engine']}**")
        for key, label, unit in props_map:
            val = None
            if pr_result and "error" not in pr_result:
                val = pr_result.get(key)
            if val is not None:
                st.metric(label=label, value=f"{val:.4f} {unit}")
            else:
                st.metric(label=label, value="N/A")

    with col_cp:
        st.markdown(f"**{t['cp_engine']}**")
        for key, label, unit in props_map:
            val = None
            if cp_result and "error" not in cp_result:
                val = cp_result.get(key)
            if val is not None:
                st.metric(label=label, value=f"{val:.4f} {unit}")
            else:
                st.metric(label=label, value="N/A")

    with col_dev:
        st.markdown(f"**{t['deviation']}**")
        for key, label, unit in props_map:
            pr_val = cp_val = None
            if pr_result and "error" not in pr_result:
                pr_val = pr_result.get(key)
            if cp_result and "error" not in cp_result:
                cp_val = cp_result.get(key)
            dev = calc_deviation(pr_val, cp_val)
            if dev is not None:
                st.metric(label=label, value=f"{dev:+.2f}%")
            else:
                st.metric(label=label, value="N/A")

    # Deviation explanation
    with st.expander(t["dev_expander_title"]):
        st.markdown(t["dev_expander_text"], unsafe_allow_html=True)

    # Error messages
    if pr_result and "error" in pr_result:
        st.warning(t["warn_pr_fail"].format(pr_result["error"]))
    if cp_result and "error" in cp_result:
        st.warning(t["warn_coolprop"].format(cp_result["error"]))

    # Z-factor info
    if pr_result and "error" not in pr_result:
        col_z1, col_z2, col_z3 = st.columns(3)
        with col_z1:
            st.metric("Z (vapor)", f'{pr_result.get("Z_vapor", 0):.6f}')
        with col_z2:
            st.metric("Z (liquid)", f'{pr_result.get("Z_liquid", 0):.6f}')
        with col_z3:
            st.metric("H_res (J/mol)", f'{pr_result.get("H_res", 0):.2f}')

    st.markdown("---")

    # Charts
    st.subheader(t["plot_header"])
    T_min = max(50.0, Tc - 200.0)
    T_max = min(2000.0, Tc + 300.0)
    T_range = np.linspace(T_min, T_max, 80)

    lang_label = "\u751f\u6210\u66f2\u7ebf\u4e2d..." if st.session_state["lang"] == "zh" else "Generating curves..."
    with st.spinner(lang_label):
        fig = create_property_plots(fluid_info, P_pa, T_range, st.session_state["lang"])
    st.plotly_chart(fig, width='stretch')


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



def main():
    """Main Streamlit application entry point."""

    st.set_page_config(
        page_title="ThermoCalc",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS
    st.markdown("""<style>
    .stApp { background: linear-gradient(145deg, #0b1120 0%, #15203a 100%); color: #e2e8f0; }
    section[data-testid="stSidebar"] { background: rgba(20, 30, 50, 0.85) !important; backdrop-filter: blur(10px); border-right: 1px solid rgba(255,255,255,0.08); }
    .stButton > button { background: linear-gradient(135deg, #1e293b, #0f172a) !important; border: 1px solid #38bdf8 !important; color: white !important; border-radius: 8px !important; width: 100%; transition: all 0.3s ease !important; }
    .stButton > button:hover { background: #38bdf8 !important; color: #0b1120 !important; box-shadow: 0 0 25px rgba(56, 189, 248, 0.5); }
    input { background-color: rgba(255,255,255,0.05) !important; border: 1px solid #334155 !important; color: white !important; border-radius: 8px !important; }
    .stSelectbox > div > div { background-color: rgba(255,255,255,0.05) !important; border-radius: 8px !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; border-bottom: none !important; }
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

        # ---- INPUT WIDGETS (always shown) ----
        fluid_options = [item[0] if st.session_state["lang"] == "zh" else item[1] for item in FLUID_DATABASE]

        T_input = st.number_input(
            f'{t["temperature"]} ({t["unit_temp"]})',
            min_value=50.0, max_value=2000.0,
            value=st.session_state["T_input"], step=1.0, format="%.1f",
            key="T_num"
        )
        P_input = st.number_input(
            f'{t["pressure"]} ({t["unit_press"]})',
            min_value=0.01, max_value=100.0,
            value=st.session_state["P_input"], step=0.1, format="%.2f",
            key="P_num"
        )
        fluid_choice = st.selectbox(
            t["fluid_select"], fluid_options,
            index=st.session_state["fluid_idx"], key="fluid_sel"
        )

        if st.button(t["calc_button"], width='stretch', key="calc_btn"):
            st.session_state["T_input"] = T_input
            st.session_state["P_input"] = P_input
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

        st.markdown("---")

        with st.expander(t["scope_title"]):
            st.info(t["scope_text"], icon="📋")


    

    if not st.session_state["calc_done"]:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(t["first_time_msg"])
        with col2:
            with st.expander(t["about_title"]):
                st.markdown(t["about_text"], unsafe_allow_html=True)
        st.markdown("---")
        st.caption("\U0001f9ea ThermoCalc v1.0 | \u5316\u5de5\u70ed\u7269\u6027\u8ba1\u7b97\u8f6f\u4ef6 | Powered by Peng-Robinson EOS + CoolProp")
        return

    if st.session_state.get("range_warning") == "range":
        st.warning(t["warn_range"])

    try:
        render_results(st.session_state["pr_result"], st.session_state["cp_result"],
                       st.session_state["fluid_info"], st.session_state["P_pa"], t)
    except Exception as e:
        st.error(f"\u7ed3\u679c\u6e32\u67d3\u5f02\u5e38: {str(e)}")
        st.code(traceback.format_exc())

    st.markdown("---")
    st.caption("\U0001f9ea ThermoCalc v1.0 | \u5316\u5de5\u70ed\u7269\u6027\u8ba1\u7b97\u8f6f\u4ef6 | Powered by Peng-Robinson EOS + CoolProp")


if __name__ == "__main__":
    main()
