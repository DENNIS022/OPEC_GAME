import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st

file_path = 'OPEC_CODE/high_demand_even.xlsx'


xl = pd.ExcelFile(file_path)
first_sheet_name = xl.sheet_names[0]
high_demand = pd.read_excel(file_path, sheet_name=first_sheet_name)
high_demand['OPEC Demand(thousand bbl/day)'] = high_demand['Estimated World Demand(thousand bbl/day)'] - high_demand['Estimated ROW Demand(thousand bbl/day)']

file_path = 'OPEC_CODE/low_demand_odd.xlsx'


xl = pd.ExcelFile(file_path)
first_sheet_name = xl.sheet_names[0]
low_demand = pd.read_excel(file_path, sheet_name=first_sheet_name)
low_demand['OPEC Demand(thousand bbl/day)'] = low_demand['Estimated World Demand(thousand bbl/day)'] - low_demand['Estimated ROW Demand(thousand bbl/day)']

x1 = low_demand['OPEC Demand(thousand bbl/day)']
y1 = low_demand['World Price($/bbl)']
x2 = high_demand['OPEC Demand(thousand bbl/day)']
y2 = high_demand['World Price($/bbl)']
# 計算第一組數據的斜率與截距
slope_low, intercept_low = np.polyfit(x1, y1, 1)
# 計算第二組數據的斜率與截距
slope_high, intercept_high = np.polyfit(x2, y2, 1)

file_path = 'OPEC_CODE/country_data.xlsx'

xl = pd.ExcelFile(file_path)
first_sheet_name = xl.sheet_names[0]
country_data = pd.read_excel(file_path, sheet_name=first_sheet_name)
    
    
    
interest_rate = 0.05
backstop_price = 70 # $/bbl
n = 10 #幾期
# 定義國家列表
countries = ['Saudi Arabia', 'Iran', 'Iraq', 'Kuwait',
             'UAE', 'Venezuela', 'Nigeria']

# 儲存每個國家的邊際成本
marginal_costs = {country: country_data.loc[3, country] for country in countries}
# 主函數
def main():
    st.title("OPEC 產量優化模型")

    st.sidebar.header("調整每個國家每一期的產量")

    n_countries = len(countries)

    # **添加：讓使用者選擇希望達到最高獲利的國家（可多選），並添加「全選」選項**
    select_all = st.sidebar.checkbox("全選")

    if select_all:
        objective_countries = st.sidebar.multiselect(
            "選擇希望達到最高獲利的國家（可多選）", countries, default=countries
        )
    else:
        objective_countries = st.sidebar.multiselect(
            "選擇希望達到最高獲利的國家（可多選）", countries
        )

    if not objective_countries:
        st.error("請至少選擇一個目標國家。")
        return

    # 初始化變數
    initial_guess = np.zeros((n_countries, n))
    fixed_indices = []
    free_indices = []
    fixed_values = []
    free_initial_guess = []
    free_bounds = []

    # 儲存每個國家期望的產量
    expected_productions = np.zeros((n_countries, n))

    # 使用 tabs 分頁顯示
    tab1, tab2 = st.tabs(["OPEC 總生產量及販售金額", "各國產量調整"])

    with tab1:
        # 讓用戶輸入每期的總生產量和販售金額
        st.markdown("### 每期 OPEC 總生產量及販售金額")
        total_world_productions = []
        total_opec_productions = []
        total_sales = []
        for t in range(n):
            st.markdown(f"#### 第 {t+1} 期")

            total_production = st.number_input(
                f"第 {t+1} 期全球總生產量（千桶）",
                min_value=0.00,
                value=74961.64,  # 預設值，可根據需要調整
                step=0.01,
                key=f"total_production_{t}"
            )
            sales_value = st.number_input(
                f"第 {t+1} 期販售金額（USD）",
                min_value=0.00,
                value=79.08,  # 預設值，可根據需要調整
                step=0.01,
                key=f"total_sales_{t}"
            )
            if (t + 1) % 2 == 1:
                slope = slope_low
                intercept = intercept_low
            else:
                slope = slope_high
                intercept = intercept_high
            opec_productions = (sales_value - intercept)/slope

            # 將每期的總生產量和販售金額加入列表
            total_opec_productions.append(opec_productions)
            total_world_productions.append(total_production)
            total_sales.append(sales_value)

        # 顯示用戶輸入的每期總生產量及總販售金額
        st.markdown("### 總生產量及販售金額回顯")
        for t in range(n):
            st.write(f"第 {t+1} 期總生產量: {total_world_productions[t]} 千桶")
            st.write(f"第 {t+1} 期OPEC總生產量: {total_opec_productions[t]} 千桶")
            st.write(f"第 {t+1} 期販售金額: ${total_sales[t]}/桶")

    with tab2:
        st.subheader("產量調整")

        for i, country in enumerate(countries):
            st.markdown(f"**{country}**")

            capacity = country_data.loc[2, country]

            cols = st.columns(n)

            for t in range(n):
                with cols[t]:
                    # 用 number_input 代替 slider，允許用戶直接輸入值
                    val = st.number_input(
                        f"{country} - P{t+1}",
                        min_value=0.0,
                        max_value=float(capacity),
                        value=float(capacity)/2,
                        step=0.1,
                        key=f"{country}_{t}"
                    )
                    expected_productions[i, t] = val

                    manual_key = f"{country}_{t}_manual"
                    is_manual = st.checkbox(
                        f"手動調整",
                        value=False,
                        key=manual_key
                    )

                idx = i * n + t  # 展開後的索引

                if is_manual:
                    # 手動調整為固定變數
                    value = expected_productions[i, t]
                    fixed_indices.append(idx)
                    fixed_values.append(value)
                else:
                    # 自由變數
                    initial_guess[i, t] = expected_productions[i, t]  
                    free_indices.append(idx)
                    free_initial_guess.append(initial_guess[i, t])
                    free_bounds.append((0, capacity))


    # 將固定值轉換為字典，以便在目標函數中使用
    fixed_dict = dict(zip(fixed_indices, fixed_values))

    # 定義目標函數
    def objective(free_vars):
        # 重建完整的 Q 矩陣
        Q_flat = np.zeros(n_countries * n)
        # 設置自由變數的值
        for idx, val in zip(free_indices, free_vars):
            Q_flat[idx] = val
        # 設置固定變數的值
        for idx, val in fixed_dict.items():
            Q_flat[idx] = val
        # 將 Q_flat 重塑為 (n_countries, n)
        Q = Q_flat.reshape(n_countries, n)
        profit, _ = country_total_profit(Q, objective_countries)
        return profit

    if len(free_initial_guess) == 0:
        st.error("所有變數都被固定，無法進行優化。")
        return

    # 執行優化
    result = minimize(
        objective,
        free_initial_guess,
        bounds=free_bounds,
        method='SLSQP',
        options={'maxiter': 2147483647}
    )

    if not result.success:
        st.error(f"優化失敗: {result.message}")
        return

    # 重建最優解的完整 Q 矩陣
    optimized_Q_flat = np.zeros(n_countries * n)
    # 設置自由變數的最優值
    for idx, val in zip(free_indices, result.x):
        optimized_Q_flat[idx] = val
    # 設置固定變數的值
    for idx, val in fixed_dict.items():
        optimized_Q_flat[idx] = val
    # 重塑為 (n_countries, n)
    optimized_Q = optimized_Q_flat.reshape(n_countries, n)

    # 計算利潤和價格
    _, period_prices = country_total_profit(optimized_Q, objective_countries)
    
    # 計算每個國家的總利潤
    country_profits = {}
    for i, country in enumerate(countries):
        profit = compute_country_profit(country, optimized_Q, period_prices)
        country_profits[country] = profit

    # 顯示結果
    st.header("結果顯示")
    for i, country in enumerate(countries):
        total_reserve = country_data.loc[1, country]
        capacity = country_data.loc[2, country]
        mc = marginal_costs[country]
        profit = country_profits[country]
        total_production = np.sum(optimized_Q[i, :])
        leftover = total_reserve - total_production
        production_details = ', '.join(
            [f"Period {t+1}: {optimized_Q[i, t]:.2f}" for t in range(n)]
        )
        st.subheader(f"{country} (產能: {capacity}, 邊際成本: {mc})")
        st.write(f"總產量(不包含11期以70元出售): {total_production:.2f}")
        st.write(f"剩餘產量(11期以70元出售): {leftover:.2f}")
        st.write(f"各期產量: {production_details}")
        st.write(f"總利潤(於第十一期): {profit:.2f}")

    # 計算 OPEC 總產量和總利潤
    total_opec_reserve = country_data.loc[1, 'OPEC']
    total_opec_production = np.sum(optimized_Q)
    opec_leftover = total_opec_reserve - total_opec_production
    total_profit = sum(country_profits.values())
    opec_production_details = ', '.join(
        [f"Period {t+1}: {np.sum(optimized_Q[:, t]):.2f}" for t in range(n)]
    )
    st.subheader("OPEC 結果")
    st.write(f"OPEC 總產量: {total_opec_production:.2f}")
    st.write(f"OPEC 剩餘產量(11期以70元出售): {opec_leftover:.2f}")
    st.write(f"OPEC 總利潤: {total_profit:.2f}")
    st.write(f"各期總產量: {opec_production_details}")

    # 顯示價格資訊
    price_details = ', '.join([f"Period {t+1}: {period_prices[t]:.2f}" for t in range(n)])
    st.subheader("每期價格")
    st.write(price_details)

# 修改 country_total_profit 函數，將 objective_countries 作為參數
def country_total_profit(Q, objective_countries):
    country_profit_value = 0
    period_prices = []

    for t in range(n):
        Q_t = Q[:, t]
        total_Q_t = np.sum(Q_t)

        if (t + 1) % 2 == 1:
            slope = slope_low
            intercept = intercept_low
        else:
            slope = slope_high
            intercept = intercept_high

        price = slope * total_Q_t + intercept  # 每期的價格
        period_prices.append(price)  # 將價格加入列表
        
        # 計算目標國家的利潤總和
        for country in objective_countries:
            i = countries.index(country)
            country_mc = marginal_costs[country]
            capacity = country_data.loc[2, country]
            country_profit = (price - country_mc) * Q[i, t]
            country_future_value = country_profit * (1 + interest_rate) ** (n - t)
            backstop_value = (backstop_price - country_mc) * (capacity - Q_t[i])
            country_future_value = country_future_value + backstop_value
            country_profit_value += country_future_value
            # 計算目標國家的利潤
           
    return -country_profit_value, period_prices  # 返回負的目標國家利潤總和和價格列表

# 計算單個國家的總利潤（不改變價格計算）
def compute_country_profit(country, Q, period_prices):
    total_profit_value = 0
    for t in range(n):
        price = period_prices[t]
        i = countries.index(country)  # 獲取該國家的索引
        mc = marginal_costs[country]  # 該國家的邊際成本
        capacity = country_data.loc[2, country]
        profit = (price - mc) * Q[i, t]  # 該國家每期的利潤
        future_value = profit * (1 + interest_rate) ** (n - t)
        backstop_value = (backstop_price - mc) * (capacity - Q[i, t]) #第11期賣掉的  
        future_value = future_value + backstop_value
        total_profit_value += future_value
        
    return total_profit_value

if __name__ == "__main__":
    main()
