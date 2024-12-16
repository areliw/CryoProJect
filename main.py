import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, spearmanr, kruskal
import numpy as np

# ------------------------------------------------------
# อ่านข้อมูลจากลิงก์ CSV
# ------------------------------------------------------
url = "https://docs.google.com/spreadsheets/d/13oJHWG48vVJrh5WwkFzI0T0j9niPxvx3Gz4rwFX8js4/export?format=csv&gid=1940769691"
df = pd.read_csv(url)

# ------------------------------------------------------
# เลือกเฉพาะคอลัมน์ตัวเลขเพื่อวิเคราะห์ข้อมูลเชิงพรรณนา
# ------------------------------------------------------
numeric_df = df.select_dtypes(include='number')

# คำนวณสถิติเชิงพรรณนา (mean, sd, min, median, max)
summary = numeric_df.describe().T
summary['median'] = numeric_df.median()
summary = summary[['mean', 'std', 'min', '50%', 'max', 'median']]
summary.columns = ['mean', 'sd', 'min', '50%', 'max', 'median']

# ------------------------------------------------------
# สร้างตารางแสดงค่าสถิติเชิงพรรณนา
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table_data = summary.reset_index()
table = ax.table(cellText=table_data.values,
                 colLabels=table_data.columns,
                 cellLoc='center',
                 loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.show()

# ------------------------------------------------------
# สร้าง Histogram ของ Fibrinogen (mg/dL)
# ------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.hist(df['Fibrinogen (mg/dL)'].dropna(), bins=20, edgecolor='black')
plt.title('Histogram of Fibrinogen (mg/dL)', fontsize=14)
plt.xlabel('Fibrinogen (mg/dL)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------
# สร้าง Q-Q Plot ของ Fibrinogen (mg/dL)
# ------------------------------------------------------
plt.figure(figsize=(12, 6))
stats.probplot(df['Fibrinogen (mg/dL)'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Fibrinogen (mg/dL)', fontsize=14)
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------
# ทดสอบการแจกแจงแบบปกติ (Shapiro-Wilk) สำหรับ Fibrinogen
# ------------------------------------------------------
fibrinogen_data = df['Fibrinogen (mg/dL)'].dropna()
stat, p_value = shapiro(fibrinogen_data)
print("Shapiro-Wilk Test for Fibrinogen:")
print(f"Test Statistic: {stat:.10f}")
print(f"P-value: {p_value:.10f}")

if p_value > 0.05:
    print("Conclusion: Data appears to follow a normal distribution (Fail to reject H0).")
else:
    print("Conclusion: Data does not follow a normal distribution (Reject H0).")

# ------------------------------------------------------
# คำนวณ Spearman Correlation สำหรับตัวแปรเชิงตัวเลขทั้งหมด
# ------------------------------------------------------
spearman_corr = numeric_df.corr(method='spearman')

# ------------------------------------------------------
# สร้าง Heatmap ของ Spearman Correlation ด้วย Matplotlib
# พร้อมแสดงตัวเลขกำกับบน Heatmap
# ------------------------------------------------------
plt.figure(figsize=(10, 8))
im = plt.imshow(spearman_corr, aspect='auto', interpolation='none', cmap='coolwarm')
plt.title('Spearman Correlation Heatmap', fontsize=16)

# กำหนดตำแหน่งของ Label แกน x และ y
plt.xticks(np.arange(len(spearman_corr.columns)), spearman_corr.columns, rotation=45, ha='right', fontsize=10)
plt.yticks(np.arange(len(spearman_corr.index)), spearman_corr.index, fontsize=10)

# เพิ่มตัวเลขกำกับใน Heatmap
for i in range(len(spearman_corr.index)):
    for j in range(len(spearman_corr.columns)):
        text = f"{spearman_corr.iloc[i, j]:.2f}"
        plt.text(j, i, text, ha='center', va='center', color='black', fontsize=9)

# เพิ่ม Colorbar
plt.colorbar(im)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# คำนวณค่า Spearman Correlation ระหว่าง Cryo Volume (ml/unit) กับ Fibrinogen (mg/dL)
# ------------------------------------------------------
cryo_column = 'Cryo Volume (ml/unit)'
fibrinogen_column = 'Fibrinogen (mg/dL)'

corr, pval = spearmanr(df[cryo_column].dropna(), df[fibrinogen_column].dropna())

print(f"Spearman Correlation between {cryo_column} and {fibrinogen_column}:")
print(f"Correlation: {corr:.4f}")
print(f"P-value: {pval:.15f}")

# ------------------------------------------------------
# เปลี่ยนช่วงของ FFP Storage Duration (Days) เป็น:
# 1 วัน (เก็บในระยะเวลาสั้นที่สุด)
# 2-7 วัน (ช่วง 1 สัปดาห์แรก)
# 8-14 วัน (ช่วงสัปดาห์ที่สอง)
# 15-30 วัน (ช่วง 1 เดือน)
# 30+ วัน (ระยะเวลานานกว่า 1 เดือน)
# ------------------------------------------------------
bins = [0, 1, 7, 14, 30, float('inf')]
labels = ['1 Day', '2-7', '8-14', '15-30', '30+']
df['FFP Storage Group'] = pd.cut(df['FFP Storage Duration (Days)'], bins=bins, labels=labels)

# ------------------------------------------------------
# คำนวณค่าเฉลี่ยของ Fibrinogen (mg/dL) สำหรับแต่ละกลุ่ม FFP Storage Duration
# ------------------------------------------------------
grouped_data = df.groupby('FFP Storage Group')['Fibrinogen (mg/dL)'].mean().reset_index()

# ------------------------------------------------------
# สร้างกราฟแสดงค่าเฉลี่ยของ Fibrinogen ในแต่ละกลุ่ม FFP Storage Duration
# ไม่ตั้งค่าการระบายสีด้วยตนเอง
# ------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.bar(grouped_data['FFP Storage Group'], grouped_data['Fibrinogen (mg/dL)'], edgecolor='black')
plt.title('Mean Fibrinogen (mg/dL) by FFP Storage Duration Group', fontsize=14)
plt.xlabel('FFP Storage Duration Group (Days)', fontsize=12)
plt.ylabel('Mean Fibrinogen (mg/dL)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------
# ทดสอบการแจกแจงแบบปกติสำหรับแต่ละกลุ่ม FFP Storage Group
# ------------------------------------------------------
normality_results = []
for group_name, data_group in df.groupby('FFP Storage Group'):
    fibrinogen_values = data_group['Fibrinogen (mg/dL)'].dropna()
    if len(fibrinogen_values) > 0:
        stat, p_val = shapiro(fibrinogen_values)
        normality_results.append((group_name, stat, p_val))
    else:
        normality_results.append((group_name, np.nan, np.nan))

normality_df = pd.DataFrame(normality_results, columns=['FFP Storage Group', 'Shapiro-Wilk Statistic', 'P-value'])
print("Normality Test Results for Each FFP Storage Group:")
print(normality_df)

# ------------------------------------------------------
# ทดสอบ Kruskal-Wallis เพื่อดูความแตกต่างระหว่างกลุ่ม
# ------------------------------------------------------
groups_for_test = [g['Fibrinogen (mg/dL)'].dropna() for _, g in df.groupby('FFP Storage Group') if len(g['Fibrinogen (mg/dL)'].dropna()) > 0]

if len(groups_for_test) > 1:
    h_stat, p_value = kruskal(*groups_for_test)
    print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("Not enough groups with data for Kruskal-Wallis test.")

# ------------------------------------------------------
# นับจำนวนตัวอย่างในแต่ละกลุ่ม FFP Storage Group และจัดเรียงตามลำดับกลุ่ม
# ------------------------------------------------------
sample_counts = df['FFP Storage Group'].value_counts().reindex(labels)
print("Sample Sizes in Each FFP Storage Group:")
print(sample_counts)
