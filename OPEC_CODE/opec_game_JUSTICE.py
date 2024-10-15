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
n_countries = len(countries)
capacities = np.array([country_data.loc[2, country] for country in countries])
total_capacity = np.sum(capacities)
marginal_costs = {country: country_data.loc[3, country] for country in countries}

def main():
    st.title("OPEC 產量優化模型")

    st.sidebar.header("調整每個國家每一期的產量")

    # 選擇目標國家
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

    # 選擇分配方式
    proportionate_allocation = st.sidebar.checkbox("按照產能比例分配產量（立足點平等）")
    fix_allocation = st.sidebar.checkbox("設定固定每期每國相同產量（齊頭式平等）")

    if proportionate_allocation and fix_allocation:
        st.error("請勿同時選擇兩個分配方式，請選擇一個。")
        return

    # 定義分配比例
    if fix_allocation:
        allocation_proportions = np.ones(n_countries) / n_countries
    elif proportionate_allocation:
        allocation_proportions = capacities / total_capacity
    else:
        st.error("請選擇一種分配方式：齊頭式平等或按產能比例。")
        return

    # 定義約束條件列表
    constraints = []

    # 每個國家的每期產量不能超過其產能
    for i in range(n_countries):
        capacity = capacities[i]
        proportion = allocation_proportions[i]
        for t in range(n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, t=t: capacity - x[t] * proportion
            })

    # 每個國家的總產量不能超過其儲量
    for i in range(n_countries):
        reserve = country_data.loc[1, countries[i]]
        proportion = allocation_proportions[i]
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: reserve - np.sum(x * proportion)
        })

    # 定義目標函數
    def objective(x):
        total_profit = 0
        period_prices = []
        for t in range(n):
            # 計算總產量 Q_t
            Q_t = x[t] * allocation_proportions
            total_Q_t = np.sum(Q_t)

            # 根據需求場景選擇價格函數
            if (t + 1) % 2 == 1:
                slope = slope_low
                intercept = intercept_low
            else:
                slope = slope_high
                intercept = intercept_high

            # 計算價格
            price = slope * total_Q_t + intercept
            period_prices.append(price)

            # 計算目標國家的利潤
            for country in objective_countries:
                i = countries.index(country)
                mc = marginal_costs[country]
                capacity = country_data.loc[2, country]
                profit = (price - mc) * Q_t[i]
                future_value = profit  * (1 + interest_rate) ** (n - t)
                backstop_value = (backstop_price - mc) * (capacity - Q_t[i])
                future_total_value = future_value + backstop_value
                total_profit += future_total_value
        return -total_profit  # 最大化利潤

    # 初始猜測和邊界
    initial_guess = np.array([np.sum(capacities) / n_countries] * n)
    bounds = [(0, None) for _ in range(n)]

    # 優化
    result = minimize(
        objective,
        initial_guess,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 2147483647, 'ftol': 1e-6}
    )

    if not result.success:
        st.error(f"優化失敗: {result.message}")
        return

    # 取得最優化的 x[t]
    optimized_x = result.x
    # 計算各國的產量矩陣 Q[i,t]
    Q = np.outer(allocation_proportions, optimized_x)

    # 計算各國的利潤
    country_profits = {}
    period_prices = []
    for t in range(n):
        Q_t = Q[:, t]
        total_Q_t = np.sum(Q_t)

        # 根據需求場景選擇價格函數
        if (t + 1) % 2 == 1:
            slope = slope_low
            intercept = intercept_low
        else:
            slope = slope_high
            intercept = intercept_high

        # 計算價格
        price = slope * total_Q_t + intercept
        period_prices.append(price)

        for i, country in enumerate(countries):
            mc = marginal_costs[country]
            capacity = country_data.loc[2, country]
            profit = (price - mc) * Q_t[i]
            future_value = profit  * (1 + interest_rate) ** (n - t)
            backstop_value = (backstop_price - mc) * (capacity - Q_t[i])
            if country not in country_profits:
                country_profits[country] = 0
            future_total_value = future_value + backstop_value
            country_profits[country] += future_total_value
            
            

    # 計算 OPEC 總利潤
    total_opec_reserve = country_data.loc[1, 'OPEC']
    total_opec_production = np.sum(Q)
    opec_leftover = total_opec_reserve - total_opec_production
    total_profit = sum(country_profits.values())
    opec_actual_profit = total_profit

    # 顯示結果
    st.header("結果顯示")
    for i, country in enumerate(countries):
        total_reserve = country_data.loc[1, country]
        capacity = capacities[i]
        mc = marginal_costs[country]
        profit = country_profits[country]
        total_production = np.sum(Q[i, :])
        leftover = total_reserve - total_production
        production_details = ', '.join(
            [f"Period {t+1}: {Q[i, t]:.2f}" for t in range(n)]
        )
        st.subheader(f"{country} (產能: {capacity}, 邊際成本: {mc})")
        st.write(f"總產量(不包含第11期以70元出售): {total_production:.2f}")
        st.write(f"剩餘產量(第11期以70元出售): {leftover:.2f}")
        st.write(f"各期產量: {production_details}")
        st.write(f"總利潤(於第11期): {profit:.2f} ")

    # 顯示 OPEC 結果
    st.subheader("OPEC 結果")
    st.write(f"OPEC 總產量: {total_opec_production:.2f}")
    st.write(f"OPEC 剩餘產量(第11期以70元出售): {opec_leftover:.2f}")
    st.write(f"OPEC 總利潤: {total_profit:.2f}")
    st.write(f"各期總產量: {', '.join([f'Period {t+1}: {np.sum(Q[:, t]):.2f}' for t in range(n)])}")

    # 顯示價格資訊
    price_details = ', '.join([f"Period {t+1}: {period_prices[t]:.2f}" for t in range(n)])
    st.subheader("每期價格")
    st.write(price_details)

if __name__ == "__main__":
    main()
