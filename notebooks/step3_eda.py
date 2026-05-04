import pandas as pd
import matplotlib.pyplot as plt
import os

# Create folder to save charts
os.makedirs('../charts', exist_ok=True)

print("Loading clean data...")
df = pd.read_csv('data/clean_flights.csv')
print("Data loaded! Shape:", df.shape)

# ---- Chart 1: Top 10 Airlines by Delay Rate ----
print("\nMaking Chart 1...")
airline_delay = df.groupby('AIRLINE')['IS_DELAYED'].mean() * 100
airline_delay = airline_delay.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
bars = plt.bar(airline_delay.index, airline_delay.values, color='steelblue')
plt.title('Top 10 Airlines by Delay Rate (%)', fontsize=16)
plt.xlabel('Airline')
plt.ylabel('Delay Rate (%)')
plt.xticks(rotation=45, ha='right')
for bar, val in zip(bars, airline_delay.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('../charts/chart1_airline_delay.png')
plt.close()
print("Chart 1 saved!")

# ---- Chart 2: Delay by Month ----
print("Making Chart 2...")
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
monthly = df.groupby('MONTH')['IS_DELAYED'].mean() * 100
monthly.index = [month_names[m] for m in monthly.index]

plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly.values, marker='o', 
         color='tomato', linewidth=2.5, markersize=8)
plt.fill_between(range(len(monthly)), monthly.values, alpha=0.1, color='tomato')
plt.title('Flight Delay Rate by Month (%)', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Delay Rate (%)')
plt.xticks(range(len(monthly)), monthly.index)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('../charts/chart2_monthly_delay.png')
plt.close()
print("Chart 2 saved!")

# ---- Chart 3: Delay Causes Breakdown ----
print("Making Chart 3...")
causes = {
    'Carrier': df['DELAY_DUE_CARRIER'].sum(),
    'Weather': df['DELAY_DUE_WEATHER'].sum(),
    'NAS': df['DELAY_DUE_NAS'].sum(),
    'Security': df['DELAY_DUE_SECURITY'].sum(),
    'Late Aircraft': df['DELAY_DUE_LATE_AIRCRAFT'].sum()
}
causes = {k: v for k, v in sorted(causes.items(), 
          key=lambda x: x[1], reverse=True)}

plt.figure(figsize=(8, 8))
colors = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6']
wedges, texts, autotexts = plt.pie(
    causes.values(), labels=causes.keys(),
    autopct='%1.1f%%', colors=colors,
    startangle=140, textprops={'fontsize': 12})
plt.title('Flight Delay Causes Breakdown', fontsize=16)
plt.tight_layout()
plt.savefig('../charts/chart3_delay_causes.png')
plt.close()
print("Chart 3 saved!")

# ---- Chart 4: Delay by Day of Week ----
print("Making Chart 4...")
day_names = {0:'Monday',1:'Tuesday',2:'Wednesday',
             3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
daily = df.groupby('DAY_OF_WEEK')['IS_DELAYED'].mean() * 100
daily.index = [day_names[d] for d in daily.index]

plt.figure(figsize=(12, 6))
colors = ['#e74c3c' if v == daily.max() else 'steelblue' 
          for v in daily.values]
bars = plt.bar(daily.index, daily.values, color=colors)
plt.title('Flight Delay Rate by Day of Week (%)', fontsize=16)
plt.xlabel('Day')
plt.ylabel('Delay Rate (%)')
plt.xticks(rotation=30, ha='right')
for bar, val in zip(bars, daily.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('../charts/chart4_day_of_week.png')
plt.close()
print("Chart 4 saved!")

# ---- Chart 5: Top 10 Worst Origin Airports ----
print("Making Chart 5...")
airport_delay = df.groupby('ORIGIN')['IS_DELAYED'].mean() * 100
airport_delay = airport_delay.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
bars = plt.barh(airport_delay.index, airport_delay.values, color='darkorange')
plt.title('Top 10 Worst Airports by Delay Rate (%)', fontsize=16)
plt.xlabel('Delay Rate (%)')
plt.ylabel('Airport Code')
for bar, val in zip(bars, airport_delay.values):
    plt.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('../charts/chart5_worst_airports.png')
plt.close()
print("Chart 5 saved!")

print("\n All 5 charts saved in charts/ folder!")
print("Open your flight_project/charts/ folder to see them!")
# ```

# Press `Ctrl + S` then run:
# ```
# python step3_eda.py