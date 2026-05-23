# ============================================================
#  CodeAlpha Internship — Task 2: Unemployment Analysis
#  Dataset : Unemployment in India.csv
#            Unemployment_Rate_upto_11_2020.csv
#  Place both CSV files in the same folder as this script
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 2. Load Datasets ─────────────────────────────────────────
df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Clean column names (remove extra spaces)
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

print("✅ Datasets loaded!")
print("\n── Dataset 1 ──")
print(df1.head())
print("Shape:", df1.shape)
print("Columns:", df1.columns.tolist())

print("\n── Dataset 2 ──")
print(df2.head())
print("Shape:", df2.shape)
print("Columns:", df2.columns.tolist())

# ── 3. Data Cleaning ─────────────────────────────────────────
print("\nMissing values (Dataset 1):")
print(df1.isnull().sum())
df1.dropna(inplace=True)
df2.dropna(inplace=True)

# Convert Date column to datetime
df1['Date'] = pd.to_datetime(df1['Date'].str.strip(), dayfirst=True)
df2['Date'] = pd.to_datetime(df2['Date'].str.strip(), dayfirst=True)

# Rename columns for easier use
df1.rename(columns={
    'Region'                              : 'Region',
    'Estimated Unemployment Rate (%)'     : 'Unemployment_Rate',
    'Estimated Employed'                  : 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation'
}, inplace=True)

df2.rename(columns={
    'Region'                              : 'Region',
    'Estimated Unemployment Rate (%)'     : 'Unemployment_Rate',
    'Estimated Employed'                  : 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour_Participation'
}, inplace=True)

print("\n✅ Data cleaned!")
print("Date range (Dataset 1):", df1['Date'].min(), "to", df1['Date'].max())
print("Date range (Dataset 2):", df2['Date'].min(), "to", df2['Date'].max())

# ── 4. EDA & Visualizations ──────────────────────────────────

# 4a. Overall Unemployment Rate Over Time
plt.figure(figsize=(12, 5))
monthly = df1.groupby('Date')['Unemployment_Rate'].mean().reset_index()
plt.plot(monthly['Date'], monthly['Unemployment_Rate'],
         color='steelblue', linewidth=2, marker='o', markersize=4)
plt.axvline(pd.to_datetime('2020-03-01'), color='red',
            linestyle='--', linewidth=1.5, label='Covid-19 Lockdown (Mar 2020)')
plt.fill_between(monthly['Date'], monthly['Unemployment_Rate'],
                 alpha=0.15, color='steelblue')
plt.title("Unemployment Rate Over Time in India", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("unemployment_over_time.png", dpi=150)
plt.show()
print("📊 Saved: unemployment_over_time.png")

# 4b. Covid Impact — Before vs During Lockdown
df1['Period'] = df1['Date'].apply(
    lambda x: 'During Covid\n(Mar-Jun 2020)'
    if pd.Timestamp('2020-03-01') <= x <= pd.Timestamp('2020-06-30')
    else 'Before Covid\n(Before Mar 2020)'
)
covid_compare = df1.groupby('Period')['Unemployment_Rate'].mean().reset_index()

plt.figure(figsize=(7, 5))
colors = ['#55A868', '#C44E52']
bars = plt.bar(covid_compare['Period'], covid_compare['Unemployment_Rate'],
               color=colors, width=0.4, edgecolor='white')
for bar, val in zip(bars, covid_compare['Unemployment_Rate']):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
plt.title("Covid-19 Impact on Unemployment Rate", fontsize=14)
plt.ylabel("Average Unemployment Rate (%)")
plt.ylim(0, covid_compare['Unemployment_Rate'].max() + 5)
plt.tight_layout()
plt.savefig("covid_impact.png", dpi=150)
plt.show()
print("📊 Saved: covid_impact.png")

# 4c. Region-wise Unemployment Rate
plt.figure(figsize=(14, 6))
region_avg = df1.groupby('Region')['Unemployment_Rate'].mean().sort_values(ascending=False)
sns.barplot(x=region_avg.index, y=region_avg.values, palette="coolwarm")
plt.title("Average Unemployment Rate by Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("region_unemployment.png", dpi=150)
plt.show()
print("📊 Saved: region_unemployment.png")

# 4d. Top 5 Most Affected Regions during Covid
covid_period = df1[df1['Period'] == 'During Covid\n(Mar-Jun 2020)']
top5 = covid_period.groupby('Region')['Unemployment_Rate'].mean().nlargest(5)

plt.figure(figsize=(8, 5))
sns.barplot(x=top5.values, y=top5.index, palette="Reds_r")
plt.title("Top 5 Most Affected Regions During Covid-19", fontsize=14)
plt.xlabel("Average Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("top5_covid_regions.png", dpi=150)
plt.show()
print("📊 Saved: top5_covid_regions.png")

# 4e. Heatmap — Region vs Month
df1['Month'] = df1['Date'].dt.strftime('%b %Y')
pivot = df1.pivot_table(values='Unemployment_Rate',
                         index='Region', columns='Month', aggfunc='mean')
plt.figure(figsize=(16, 10))
sns.heatmap(pivot, cmap='YlOrRd', annot=False, linewidths=0.3)
plt.title("Unemployment Rate Heatmap — Region vs Month", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Region")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("heatmap_region_month.png", dpi=150)
plt.show()
print("📊 Saved: heatmap_region_month.png")

# 4f. Labour Participation Rate vs Unemployment Rate
plt.figure(figsize=(8, 5))
plt.scatter(df1['Labour_Participation'], df1['Unemployment_Rate'],
            alpha=0.5, color='purple', edgecolors='white', linewidth=0.5)
plt.title("Labour Participation Rate vs Unemployment Rate", fontsize=14)
plt.xlabel("Labour Participation Rate (%)")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.savefig("labour_vs_unemployment.png", dpi=150)
plt.show()
print("📊 Saved: labour_vs_unemployment.png")

# ── 5. Key Insights ──────────────────────────────────────────
print("\n" + "="*55)
print("  📌 KEY INSIGHTS")
print("="*55)

before = df1[df1['Period'] == 'Before Covid\n(Before Mar 2020)']['Unemployment_Rate'].mean()
during = df1[df1['Period'] == 'During Covid\n(Mar-Jun 2020)']['Unemployment_Rate'].mean()

print(f"  Average Unemployment Before Covid : {before:.2f}%")
print(f"  Average Unemployment During Covid : {during:.2f}%")
print(f"  Increase due to Covid             : {during - before:.2f}%")
print(f"\n  Highest Unemployment Region : {region_avg.index[0]} ({region_avg.values[0]:.2f}%)")
print(f"  Lowest  Unemployment Region : {region_avg.index[-1]} ({region_avg.values[-1]:.2f}%)")
print("="*55)

print("\n✅ Task 2 complete! Saare charts save ho gaye hain.")
print("\n📁 GitHub par ye files upload karo (Task2 folder mein):")
print("   - task2_unemployment_analysis.py")
print("   - Unemployment in India.csv")
print("   - Unemployment_Rate_upto_11_2020.csv")
print("   - unemployment_over_time.png")
print("   - covid_impact.png")
print("   - region_unemployment.png")
print("   - top5_covid_regions.png")
print("   - heatmap_region_month.png")
print("   - labour_vs_unemployment.png")
