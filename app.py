import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math


# 1. 参数设定
normalization_params = {
    'age':       {'label': '年龄 (Age)',          'mean': 58.722008,   'sd': 8.791730,    'log': False},
    'feq':       {'label': 'Freq',               'mean': 16.960981,   'sd': 6.283532,    'log': False},
    'ax':        {'label': 'AX',                 'mean': 1.005278,    'sd': 1.180524,    'log': False},
    'volume':    {'label': '肺容积 (Volume)',     'mean': 4818.949258, 'sd': 1243.133053, 'log': False},
    'laa950':    {'label': '吸气相 LAA950%',      'mean': 20.545781,   'sd': 7.500869,    'log': False},
    'ex_laa856': {'label': '呼气相 LAA856%',      'mean': 40.030276,   'sd': 14.264588,   'log': False},
    'ex_mld_HU': {'label': '呼气相 MLD (HU)',     'mean': -762.596008, 'sd': 52.068138,   'log': False},
    'GALE':      {'label': 'GALE',               'mean': 15.493274,   'sd': 2.371538,    'log': True},
    'MORC1':     {'label': 'MORC1',              'mean': 17.687595,   'sd': 2.476704,    'log': True},
    'EIF5':      {'label': 'EIF5',               'mean': 20.085333,   'sd': 0.762238,    'log': True},
    'NT5DC2':    {'label': 'NT5DC2',             'mean': 14.304475,   'sd': 1.197168,    'log': True},
    'FBLN5':     {'label': 'FBLN5',              'mean': 14.161654,   'sd': 1.282076,    'log': True},
    'ENO1':      {'label': 'ENO1',               'mean': 13.338325,   'sd': 1.497263,    'log': True},
    'CES1':      {'label': 'CES1',               'mean': 12.930673,   'sd': 1.420015,    'log': True},
    'OSMR':      {'label': 'OSMR',               'mean': 12.576449,   'sd': 1.161471,    'log': True}
}

# 2. 模型加载
st.set_page_config(page_title="慢阻肺病早期诊断预测系统", layout="wide")

@st.cache_resource
def load_resources():
    # model = joblib.load('svm_model_final.pkl')
    # features = joblib.load('feature_names.pkl')
    model = joblib.load('xgboost_model_final.pkl')
    features = joblib.load('feature_namesv2.pkl')
    return model, features

try:
    model, feature_names = load_resources()
except FileNotFoundError:
    st.error("未找到文件")
    st.stop()

# 3. 侧边栏

st.sidebar.title("🩺 患者指标录入")
st.sidebar.markdown("请输入原始临床数值")

user_inputs = {}

with st.sidebar.form("patient_data_form"):
    st.subheader("临床与影像学指标")
    for col in feature_names:
        if col in normalization_params and not normalization_params[col]['log']:
            config = normalization_params[col]
            user_inputs[col] = st.number_input(
                f"{config['label']}", 
                value=float(config['mean']), 
                format="%.2f"
            )
    st.subheader("蛋白组学")
    st.caption("请输入原始检测值")
    for col in feature_names:
        if col in normalization_params and normalization_params[col]['log']:
            config = normalization_params[col]
            # 这里需要给一个合理的初始值：因为 mean 是 log 后的，所以还原回去展示给用户大概是 2^mean，我这里添了1，如果有更合适的你自己改
            user_inputs[col] = st.number_input(
                f"{config['label']}", 
                value=1.0, 
                format="%.2f",
                help="请输入原始表达量"
            )
            
    # 提交按钮
    submitted = st.form_submit_button("开始风险预测", use_container_width=True)


# 4. 预测结果展示
st.title("慢阻肺病早期诊断预测系统")

if submitted:
    processed_data = []
    for col in feature_names:
        original_val = user_inputs[col]
        params = normalization_params.get(col)
        if params:
            val_to_normalize = original_val
            if params['log']:
                if original_val <= 0:
                    st.toast(f" {col} 的值必须大于 0 才能进行 Log 变换，已自动按最小值处理。")
                    val_to_normalize = 0 # 或者设置一个极小值
                else:
                    val_to_normalize = math.log2(original_val)
            if params['sd'] != 0:
                norm_val = (val_to_normalize - params['mean']) / params['sd']
            else:
                norm_val = val_to_normalize
        else:
            norm_val = original_val
            
        processed_data.append(norm_val)
    
    final_input = np.array([processed_data])
    with st.spinner('正在进行特征分析与风险计算...'):
        try:
            proba = model.predict_proba(final_input)[0] 
            risk_score = proba[1]
            prediction = model.predict(final_input)[0]
        except Exception as e:
            st.error(f"预测计算出错: {e}")
            st.stop()
    
    # --- 结果可视化 ---
    st.divider()
    c1, c2 = st.columns([1, 1.5])
    
    with c1:
        if risk_score > 0.5:
            st.error("🔴 高风险")
            st.markdown(f"建议\n请结合临床症状进行进一步检查。")
        else:
            st.success("🟢 低风险")
            st.markdown(f"当前指标未显示明显异常。")
            
        st.metric("慢阻肺病患病可能", f"{risk_score:.1%}")

    with c2:
        st.write("风险评估详情")
        bar_color = "red" if risk_score > 0.5 else "green"
        st.progress(risk_score, text=f"风险指数: {risk_score:.4f}")
        with st.expander("查看模型输入详情"):
            df_display = pd.DataFrame([processed_data], columns=feature_names)
            st.dataframe(df_display.style.format("{:.4f}"))


else:
    # 初始欢迎界面
    st.info("请在左侧侧边栏输入患者数据，点击“开始风险评估”即可获得结果。")
