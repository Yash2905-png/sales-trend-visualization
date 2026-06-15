# ============================================================
# SALES TREND VISUALIZATION - Advanced Data Analysis Project
# CodTech Internship | Data Analytics Domain
# Intern: Yash Gamare
# Task 1: Sales Trend Visualization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# GLOBAL STYLE CONFIG
# ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'text.color': '#e2e8f0',
    'grid.color': '#334155',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e', '#a78bfa']
ACCENT = '#6366f1'

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  SALES TREND VISUALIZATION — ADVANCED ANALYSIS")
print("=" * 60)

df = pd.read_csv('data/sales_data.csv', parse_dates=['Date'])
print(f"\n[DATA LOADED] Shape: {df.shape}")
print(f"Date Range : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Columns    : {list(df.columns)}")

# --- Data Cleaning ---
print("\n── Data Cleaning ──")
print(f"Missing values:\n{df.isnull().sum()}")
df.dropna(inplace=True)
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
df.dropna(subset=['Revenue','Units_Sold'], inplace=True)

# Feature Engineering
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%b')
df['Quarter'] = df['Date'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Revenue_Lakh'] = df['Revenue'] / 100000
df['Effective_Price'] = df['Unit_Price'] * (1 - df['Discount_Pct']/100)
df['Profit_Margin'] = ((df['Effective_Price'] - df['Unit_Price'] * 0.6) / df['Effective_Price']) * 100

print(f"\n[CLEAN] {len(df)} records ready for analysis.")
print(f"\nRevenue Summary (₹ Lakhs):\n{df['Revenue_Lakh'].describe().round(2)}")

# ─────────────────────────────────────────────
# 2. MONTHLY REVENUE TREND + FORECAST
# ─────────────────────────────────────────────
monthly = df.groupby('Month').agg(
    Revenue=('Revenue_Lakh','sum'),
    Units=('Units_Sold','sum'),
    Transactions=('Date','count')
).reset_index()
monthly['Month_Name'] = pd.to_datetime(monthly['Month'], format='%m').dt.strftime('%b')

X = monthly['Month'].values.reshape(-1, 1)
y = monthly['Revenue'].values

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_pred = model.predict(X_poly)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='#0f172a')
fig.suptitle('MONTHLY REVENUE TREND & FORECAST', fontsize=18, fontweight='bold',
             color='white', y=1.02)

# LEFT: Revenue trend
ax1 = axes[0]
ax1.fill_between(monthly['Month'], monthly['Revenue'], alpha=0.2, color=ACCENT)
ax1.plot(monthly['Month'], monthly['Revenue'], color=ACCENT, linewidth=2.5,
         marker='o', markersize=8, markerfacecolor='white', markeredgecolor=ACCENT, label='Actual Revenue')
ax1.plot(monthly['Month'], y_pred, '--', color='#f59e0b', linewidth=2,
         label=f'Poly Forecast (R²={r2:.3f})')
for i, row in monthly.iterrows():
    ax1.annotate(f"₹{row['Revenue']:.0f}L", (row['Month'], row['Revenue']),
                 textcoords='offset points', xytext=(0, 10), ha='center',
                 fontsize=8, color='#94a3b8')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Revenue (₹ Lakhs)', fontsize=12)
ax1.set_title('Revenue Trend with Polynomial Forecast', fontsize=13, color='#94a3b8', pad=12)
ax1.set_xticks(monthly['Month'])
ax1.set_xticklabels(monthly['Month_Name'])
ax1.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white')
ax1.grid(True, alpha=0.3)

# RIGHT: Units sold bar
ax2 = axes[1]
bars = ax2.bar(monthly['Month_Name'], monthly['Units'], color=COLORS[:len(monthly)], edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, monthly['Units']):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val),
             ha='center', va='bottom', fontsize=8, color='white')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Total Units Sold', fontsize=12)
ax2.set_title('Monthly Units Sold', fontsize=13, color='#94a3b8', pad=12)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/01_monthly_revenue_trend.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("\n[CHART 1] Monthly Revenue Trend saved.")

# ─────────────────────────────────────────────
# 3. CATEGORY & REGION ANALYSIS
# ─────────────────────────────────────────────
cat_rev = df.groupby('Product_Category')['Revenue_Lakh'].sum().sort_values(ascending=False)
region_rev = df.groupby('Region')['Revenue_Lakh'].sum().sort_values(ascending=False)
region_cat = df.pivot_table(index='Region', columns='Product_Category',
                             values='Revenue_Lakh', aggfunc='sum', fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='#0f172a')
fig.suptitle('CATEGORY & REGION REVENUE BREAKDOWN', fontsize=18, fontweight='bold', color='white', y=1.02)

# Donut — Category
ax1 = axes[0]
wedges, texts, autotexts = ax1.pie(cat_rev.values, labels=cat_rev.index,
    autopct='%1.1f%%', startangle=90,
    colors=COLORS[:len(cat_rev)],
    pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor='#0f172a', linewidth=2))
for at in autotexts: at.set_color('white'); at.set_fontsize(9)
for t in texts: t.set_color('#cbd5e1'); t.set_fontsize(10)
ax1.set_title('Revenue by Category', fontsize=13, color='#94a3b8', pad=12)
ax1.text(0, 0, f'₹{cat_rev.sum():.0f}L\nTotal', ha='center', va='center',
          fontsize=11, color='white', fontweight='bold')

# Horizontal bar — Region
ax2 = axes[1]
colors_region = COLORS[:len(region_rev)]
bars = ax2.barh(region_rev.index, region_rev.values, color=colors_region, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, region_rev.values):
    ax2.text(val + 10, bar.get_y() + bar.get_height()/2, f'₹{val:.0f}L',
             va='center', fontsize=10, color='white')
ax2.set_xlabel('Revenue (₹ Lakhs)', fontsize=12)
ax2.set_title('Revenue by Region', fontsize=13, color='#94a3b8', pad=12)
ax2.grid(True, axis='x', alpha=0.3)

# Stacked bar — Region x Category
ax3 = axes[2]
region_cat.plot(kind='bar', stacked=True, ax=ax3, color=COLORS[:len(region_cat.columns)],
                edgecolor='white', linewidth=0.5)
ax3.set_xlabel('Region', fontsize=12)
ax3.set_ylabel('Revenue (₹ Lakhs)', fontsize=12)
ax3.set_title('Region × Category Stacked', fontsize=13, color='#94a3b8', pad=12)
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
ax3.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=8)
ax3.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/02_category_region_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("[CHART 2] Category & Region Analysis saved.")

# ─────────────────────────────────────────────
# 4. HEATMAP — MONTH × CATEGORY
# ─────────────────────────────────────────────
pivot = df.pivot_table(index='Product_Category', columns='Month_Name',
                        values='Revenue_Lakh', aggfunc='sum', fill_value=0)
month_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

fig, ax = plt.subplots(figsize=(16, 6), facecolor='#0f172a')
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=0.5, linecolor='#0f172a',
            cbar_kws={'label': 'Revenue (₹ Lakhs)', 'shrink': 0.8},
            annot_kws={'size': 9, 'color': 'black'})
ax.set_title('Revenue Heatmap: Product Category × Month', fontsize=16, fontweight='bold',
             color='white', pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Product Category', fontsize=12)
ax.tick_params(axis='both', colors='#94a3b8')
plt.tight_layout()
plt.savefig('outputs/03_revenue_heatmap.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("[CHART 3] Revenue Heatmap saved.")

# ─────────────────────────────────────────────
# 5. SALES REP PERFORMANCE + CUSTOMER SEGMENT
# ─────────────────────────────────────────────
rep_perf = df.groupby('Sales_Rep').agg(
    Revenue=('Revenue_Lakh','sum'),
    Units=('Units_Sold','sum'),
    Deals=('Date','count')
).sort_values('Revenue', ascending=False).reset_index()

seg_rev = df.groupby('Customer_Segment')['Revenue_Lakh'].sum()

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0f172a')
fig.suptitle('SALES REP PERFORMANCE & CUSTOMER SEGMENT', fontsize=18, fontweight='bold',
             color='white', y=1.02)

ax1 = axes[0]
x = np.arange(len(rep_perf))
w = 0.35
b1 = ax1.bar(x - w/2, rep_perf['Revenue'], w, label='Revenue (₹L)', color=COLORS[0], edgecolor='white', linewidth=0.5)
ax1_r = ax1.twinx()
b2 = ax1_r.bar(x + w/2, rep_perf['Deals'], w, label='No. of Deals', color=COLORS[1], edgecolor='white', linewidth=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(rep_perf['Sales_Rep'], rotation=15, ha='right')
ax1.set_ylabel('Revenue (₹ Lakhs)', color=COLORS[0], fontsize=11)
ax1_r.set_ylabel('Number of Deals', color=COLORS[1], fontsize=11)
ax1.set_title('Sales Rep: Revenue vs Deals', fontsize=13, color='#94a3b8', pad=12)
ax1.grid(True, axis='y', alpha=0.3)
lines = [mpatches.Patch(color=COLORS[0], label='Revenue (₹L)'),
         mpatches.Patch(color=COLORS[1], label='No. of Deals')]
ax1.legend(handles=lines, facecolor='#1e293b', edgecolor='#334155', labelcolor='white')

ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(seg_rev.values, labels=seg_rev.index, autopct='%1.1f%%',
    startangle=140, colors=COLORS[:3], pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor='#0f172a', linewidth=2))
for at in autotexts: at.set_color('white'); at.set_fontsize(11)
for t in texts: t.set_color('#cbd5e1'); t.set_fontsize(12)
ax2.set_title('Revenue by Customer Segment', fontsize=13, color='#94a3b8', pad=12)

plt.tight_layout()
plt.savefig('outputs/04_sales_rep_segment.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("[CHART 4] Sales Rep & Segment Analysis saved.")

# ─────────────────────────────────────────────
# 6. QUARTERLY + TOP PRODUCTS
# ─────────────────────────────────────────────
quarterly = df.groupby('Quarter').agg(
    Revenue=('Revenue_Lakh','sum'),
    Units=('Units_Sold','sum')
).reset_index()

top_products = df.groupby('Product_Name').agg(
    Revenue=('Revenue_Lakh','sum'),
    Units=('Units_Sold','sum')
).sort_values('Revenue', ascending=True).tail(8)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0f172a')
fig.suptitle('QUARTERLY PERFORMANCE & TOP PRODUCTS', fontsize=18, fontweight='bold',
             color='white', y=1.02)

ax1 = axes[0]
q_colors = [COLORS[0], COLORS[1], COLORS[2], COLORS[3]]
bars = ax1.bar(quarterly['Quarter'], quarterly['Revenue'], color=q_colors, edgecolor='white',
               linewidth=0.5, width=0.5)
for bar, val in zip(bars, quarterly['Revenue']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'₹{val:.0f}L', ha='center', fontsize=11, fontweight='bold', color='white')
ax1.plot(quarterly['Quarter'], quarterly['Revenue'], 'o--', color='#f59e0b',
         linewidth=2, markersize=8, markerfacecolor='white')
ax1.set_ylabel('Revenue (₹ Lakhs)', fontsize=12)
ax1.set_title('Quarterly Revenue', fontsize=13, color='#94a3b8', pad=12)
ax1.grid(True, axis='y', alpha=0.3)

# QoQ growth annotations
for i in range(1, len(quarterly)):
    growth = ((quarterly['Revenue'].iloc[i] - quarterly['Revenue'].iloc[i-1])
               / quarterly['Revenue'].iloc[i-1] * 100)
    ax1.annotate(f'{growth:+.1f}%',
        xy=(quarterly['Quarter'].iloc[i], quarterly['Revenue'].iloc[i]),
        xytext=(0, -30), textcoords='offset points',
        ha='center', color='#22d3ee', fontsize=10, fontweight='bold')

ax2 = axes[1]
colors_prod = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top_products)))
bars = ax2.barh(top_products.index, top_products['Revenue'],
                color=colors_prod, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top_products['Revenue']):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f'₹{val:.0f}L',
             va='center', fontsize=9, color='white')
ax2.set_xlabel('Revenue (₹ Lakhs)', fontsize=12)
ax2.set_title('Top Products by Revenue', fontsize=13, color='#94a3b8', pad=12)
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_quarterly_top_products.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("[CHART 5] Quarterly & Top Products saved.")

# ─────────────────────────────────────────────
# 7. CORRELATION & SCATTER — UNITS vs REVENUE
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0f172a')
fig.suptitle('CORRELATION ANALYSIS & SCATTER PLOTS', fontsize=18, fontweight='bold',
             color='white', y=1.02)

ax1 = axes[0]
corr_cols = ['Units_Sold', 'Unit_Price', 'Revenue_Lakh', 'Discount_Pct', 'Profit_Margin']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax1, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.5, linecolor='#0f172a',
            annot_kws={'size': 10},
            cbar_kws={'shrink': 0.8})
ax1.set_title('Correlation Matrix', fontsize=13, color='#94a3b8', pad=12)
ax1.tick_params(axis='both', colors='#94a3b8', labelsize=9)

ax2 = axes[1]
cats = df['Product_Category'].unique()
cat_colors = dict(zip(cats, COLORS[:len(cats)]))
for cat in cats:
    mask = df['Product_Category'] == cat
    ax2.scatter(df[mask]['Units_Sold'], df[mask]['Revenue_Lakh'],
                label=cat, color=cat_colors[cat], alpha=0.7, s=60, edgecolors='white', linewidth=0.3)
z = np.polyfit(df['Units_Sold'], df['Revenue_Lakh'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['Units_Sold'].min(), df['Units_Sold'].max(), 100)
ax2.plot(x_line, p(x_line), '--', color='#f59e0b', linewidth=2, label='Trend Line')
ax2.set_xlabel('Units Sold', fontsize=12)
ax2.set_ylabel('Revenue (₹ Lakhs)', fontsize=12)
ax2.set_title('Units Sold vs Revenue by Category', fontsize=13, color='#94a3b8', pad=12)
ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/06_correlation_scatter.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("[CHART 6] Correlation & Scatter saved.")

# ─────────────────────────────────────────────
# 8. EXECUTIVE SUMMARY DASHBOARD
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12), facecolor='#0f172a')
fig.suptitle('SALES PERFORMANCE EXECUTIVE DASHBOARD — 2023',
             fontsize=22, fontweight='bold', color='white', y=0.98)

# KPI Cards (top row)
kpis = [
    ('Total Revenue', f"₹{df['Revenue_Lakh'].sum():,.0f}L", '#6366f1'),
    ('Total Units Sold', f"{df['Units_Sold'].sum():,}", '#22d3ee'),
    ('Avg Order Value', f"₹{df['Revenue_Lakh'].mean():.1f}L", '#f59e0b'),
    ('Top Region', df.groupby('Region')['Revenue_Lakh'].sum().idxmax(), '#10b981'),
    ('Best Category', df.groupby('Product_Category')['Revenue_Lakh'].sum().idxmax(), '#f43f5e'),
    ('Peak Month', df.groupby('Month_Name')['Revenue_Lakh'].sum().idxmax(), '#a78bfa'),
]

for i, (title, val, col) in enumerate(kpis):
    ax = fig.add_axes([0.03 + i*0.163, 0.80, 0.145, 0.13])
    ax.set_facecolor(col + '22')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(col); spine.set_linewidth(2)
    ax.text(0.5, 0.65, val, transform=ax.transAxes, ha='center', va='center',
            fontsize=14, fontweight='bold', color=col)
    ax.text(0.5, 0.25, title, transform=ax.transAxes, ha='center', va='center',
            fontsize=9, color='#94a3b8')

# Monthly trend
ax_main = fig.add_axes([0.03, 0.42, 0.44, 0.32])
ax_main.set_facecolor('#1e293b')
ax_main.fill_between(monthly['Month'], monthly['Revenue'], alpha=0.15, color=ACCENT)
ax_main.plot(monthly['Month'], monthly['Revenue'], color=ACCENT, linewidth=2.5,
             marker='o', markersize=7, markerfacecolor='white', markeredgecolor=ACCENT)
ax_main.plot(monthly['Month'], y_pred, '--', color='#f59e0b', linewidth=1.5, alpha=0.8)
ax_main.set_title('Monthly Revenue Trend + Forecast', color='#94a3b8', fontsize=11, pad=8)
ax_main.set_xticks(monthly['Month']); ax_main.set_xticklabels(monthly['Month_Name'])
ax_main.grid(True, alpha=0.3)

# Category pie
ax_pie = fig.add_axes([0.50, 0.42, 0.22, 0.32])
ax_pie.set_facecolor('#1e293b')
wedges, texts, autotexts = ax_pie.pie(cat_rev.values, labels=cat_rev.index,
    autopct='%1.0f%%', startangle=90, colors=COLORS[:len(cat_rev)],
    pctdistance=0.8, wedgeprops=dict(width=0.5, edgecolor='#0f172a', linewidth=1.5),
    textprops={'fontsize': 8})
for at in autotexts: at.set_color('white'); at.set_fontsize(8)
for t in texts: t.set_color('#94a3b8'); t.set_fontsize(8)
ax_pie.set_title('Revenue by Category', color='#94a3b8', fontsize=11, pad=8)

# Quarterly bar
ax_q = fig.add_axes([0.75, 0.42, 0.22, 0.32])
ax_q.set_facecolor('#1e293b')
ax_q.bar(quarterly['Quarter'], quarterly['Revenue'], color=COLORS[:4], edgecolor='white', linewidth=0.5)
ax_q.set_title('Quarterly Revenue', color='#94a3b8', fontsize=11, pad=8)
ax_q.grid(True, axis='y', alpha=0.3)

# Region heatmap
ax_heat = fig.add_axes([0.03, 0.05, 0.44, 0.30])
ax_heat.set_facecolor('#1e293b')
rh = df.pivot_table(index='Region', columns='Month_Name', values='Revenue_Lakh', aggfunc='sum', fill_value=0)
rh = rh.reindex(columns=[m for m in month_order if m in rh.columns])
sns.heatmap(rh, ax=ax_heat, cmap='Blues', annot=True, fmt='.0f', linewidths=0.5,
            linecolor='#0f172a', annot_kws={'size': 8}, cbar=False)
ax_heat.set_title('Region × Month Revenue Heatmap', color='#94a3b8', fontsize=11, pad=8)
ax_heat.tick_params(colors='#94a3b8', labelsize=8)

# Top products
ax_prod = fig.add_axes([0.50, 0.05, 0.47, 0.30])
ax_prod.set_facecolor('#1e293b')
tp = df.groupby('Product_Name')['Revenue_Lakh'].sum().sort_values(ascending=True)
prod_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(tp)))
ax_prod.barh(tp.index, tp.values, color=prod_colors, edgecolor='white', linewidth=0.3)
ax_prod.set_title('Revenue by Product', color='#94a3b8', fontsize=11, pad=8)
ax_prod.grid(True, axis='x', alpha=0.3)
ax_prod.tick_params(colors='#94a3b8', labelsize=8)

plt.savefig('outputs/07_executive_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print("[CHART 7] Executive Dashboard saved.")

# ─────────────────────────────────────────────
# 9. PRINT FINAL INSIGHTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY BUSINESS INSIGHTS")
print("=" * 60)
print(f"\n  Total Revenue     : ₹{df['Revenue_Lakh'].sum():,.0f} Lakhs")
print(f"  Total Units Sold  : {df['Units_Sold'].sum():,} units")
print(f"  Total Transactions: {len(df)}")
print(f"  Avg Revenue/Deal  : ₹{df['Revenue_Lakh'].mean():.1f} Lakhs")
print(f"\n  Best Region       : {df.groupby('Region')['Revenue_Lakh'].sum().idxmax()}")
print(f"  Best Category     : {df.groupby('Product_Category')['Revenue_Lakh'].sum().idxmax()}")
print(f"  Best Product      : {df.groupby('Product_Name')['Revenue_Lakh'].sum().idxmax()}")
print(f"  Best Sales Rep    : {df.groupby('Sales_Rep')['Revenue_Lakh'].sum().idxmax()}")
print(f"  Peak Month        : {df.groupby('Month_Name')['Revenue_Lakh'].sum().idxmax()}")
print(f"  Forecast Model R² : {r2:.4f} | MAE: ₹{mae:.1f}L")
print(f"\n  Quarterly Growth:")
for i in range(1, len(quarterly)):
    g = ((quarterly['Revenue'].iloc[i] - quarterly['Revenue'].iloc[i-1]) / quarterly['Revenue'].iloc[i-1]) * 100
    print(f"    {quarterly['Quarter'].iloc[i-1]} → {quarterly['Quarter'].iloc[i]}: {g:+.1f}%")

print("\n  All charts saved to outputs/ folder.")
print("=" * 60)
