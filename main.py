import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, spearmanr, kruskal
import numpy as np

# ------------------------------------------------------
# 1) อ่านข้อมูลจากลิงก์ CSV (Google Sheets)
# ------------------------------------------------------
url = "https://docs.google.com/spreadsheets/d/13oJHWG48vVJrh5WwkFzI0T0j9niPxvx3Gz4rwFX8js4/export?format=csv&gid=1940769691"
df = pd.read_csv(url)

# ------------------------------------------------------
# 2) เลือกเฉพาะคอลัมน์ตัวเลขเพื่อวิเคราะห์ข้อมูลเชิงพรรณนา (Overall)
# ------------------------------------------------------
numeric_df = df.select_dtypes(include='number')

# ------------------------------------------------------
# 3) คำนวณสถิติเชิงพรรณนาแบบภาพรวม (mean, sd, min, median, max)
# ------------------------------------------------------
summary_overall = numeric_df.describe().T
summary_overall['median'] = numeric_df.median()  # เพิ่ม median แยกออกมา
summary_overall = summary_overall[['mean', 'std', 'min', '50%', 'max', 'median']]
summary_overall.columns = ['mean', 'sd', 'min', '50%', 'max', 'median']

# ------------------------------------------------------
# 4) แสดงตารางสถิติเชิงพรรณนา (Overall)
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table_data = summary_overall.reset_index()
table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.title('Summary Statistics (Overall)')
plt.show()

# ------------------------------------------------------
# 5) คำนวณสถิติเชิงพรรณนาแยกเพศ: M กับ F
#    แล้ววาดตารางสรุปออกมาแยกตาราง
# ------------------------------------------------------
# สร้างฟังก์ชันย่อยสำหรับทำตารางสถิติ + plot
def plot_descriptive_table(df_in, title_text):
    """ สร้างตาราง descriptive statistic และ plot ออกมาในรูปใหม่ """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # คำนวณสถิติเชิงพรรณนา
    _summary = df_in.describe().T
    _summary['median'] = df_in.median()
    _summary = _summary[['mean', 'std', 'min', '50%', 'max', 'median']]
    _summary.columns = ['mean', 'sd', 'min', '50%', 'max', 'median']
    
    # แปลงเป็นตาราง
    table_data_local = _summary.reset_index()
    _table = ax.table(
        cellText=table_data_local.values,
        colLabels=table_data_local.columns,
        cellLoc='center',
        loc='center'
    )
    _table.auto_set_font_size(False)
    _table.set_fontsize(10)
    _table.scale(1.2, 1.2)
    plt.title(title_text)
    plt.show()

# แยกเพศชาย (M)
df_male = df[df['Gender'] == 'M'].select_dtypes(include='number')
plot_descriptive_table(df_male, "Summary Statistics for M")

# แยกเพศหญิง (F)
df_female = df[df['Gender'] == 'F'].select_dtypes(include='number')
plot_descriptive_table(df_female, "Summary Statistics for F")

# ------------------------------------------------------
# 6) สร้าง Histogram ของ Fibrinogen (mg/dL)
# ------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.hist(df['Fibrinogen (mg/dL)'].dropna(), bins=20, edgecolor='black')
plt.title('Histogram of Fibrinogen (mg/dL)', fontsize=14)
plt.xlabel('Fibrinogen (mg/dL)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------
# 7) สร้าง Q-Q Plot ของ Fibrinogen (mg/dL) เพื่อดูการกระจาย
# ------------------------------------------------------
plt.figure(figsize=(12, 6))
stats.probplot(df['Fibrinogen (mg/dL)'].dropna(), dist="norm", plot=plt)
plt.title('Q-Q Plot of Fibrinogen (mg/dL)', fontsize=14)
plt.xlabel('Theoretical Quantiles', fontsize=12)
plt.ylabel('Sample Quantiles', fontsize=12)
plt.grid(linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------
# 8) ทดสอบการแจกแจงแบบปกติ (Shapiro-Wilk) ของ Fibrinogen
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
print("-" * 50)

# ------------------------------------------------------
# 9) คำนวณ Spearman Correlation สำหรับตัวแปรเชิงตัวเลขทั้งหมด
# ------------------------------------------------------
spearman_corr = numeric_df.corr(method='spearman')

# ------------------------------------------------------
# 10) สร้าง Heatmap ของ Spearman Correlation พร้อมตัวเลขกำกับ
# ------------------------------------------------------
plt.figure(figsize=(10, 8))
im = plt.imshow(spearman_corr, aspect='auto', interpolation='none', cmap='coolwarm')
plt.title('Spearman Correlation Heatmap', fontsize=16)

# กำหนดตำแหน่ง Label แกน x, y
plt.xticks(np.arange(len(spearman_corr.columns)), spearman_corr.columns, 
           rotation=45, ha='right', fontsize=10)
plt.yticks(np.arange(len(spearman_corr.index)), spearman_corr.index, fontsize=10)

# ใส่ค่า correlation ลงไปตรงกลาง cell
for i in range(len(spearman_corr.index)):
    for j in range(len(spearman_corr.columns)):
        text = f"{spearman_corr.iloc[i, j]:.2f}"
        plt.text(j, i, text, ha='center', va='center', color='black', fontsize=9)

plt.colorbar(im)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 11) Spearman Correlation ระหว่าง Cryo Volume (ml/unit) และ Fibrinogen (mg/dL)
# ------------------------------------------------------
cryo_column = 'Cryo Volume (ml/unit)'
fibrinogen_column = 'Fibrinogen (mg/dL)'

corr, pval = spearmanr(df[cryo_column].dropna(), df[fibrinogen_column].dropna())
print(f"Spearman Correlation between {cryo_column} and {fibrinogen_column}:")
print(f"Correlation: {corr:.4f}")
print(f"P-value: {pval:.15f}")
print("-" * 50)

# ------------------------------------------------------
# 12) สร้างกลุ่ม FFP Storage Duration (Days) ตามช่วง
#     (1 Day, 2-7, 8-14, 15-30, 30+)
# ------------------------------------------------------
bins = [0, 1, 7, 14, 30, float('inf')]
labels = ['1 Day', '2-7', '8-14', '15-30', '30+']
df['FFP Storage Group'] = pd.cut(df['FFP Storage Duration (Days)'], 
                                 bins=bins, labels=labels)

# ------------------------------------------------------
# 13) คำนวณค่าเฉลี่ย Fibrinogen (mg/dL) ในแต่ละ FFP Storage Group (Bar Chart)
# ------------------------------------------------------
grouped_data = df.groupby('FFP Storage Group')['Fibrinogen (mg/dL)'].mean().reset_index()

plt.figure(figsize=(8, 6))
plt.bar(grouped_data['FFP Storage Group'], grouped_data['Fibrinogen (mg/dL)'], 
        edgecolor='black')
plt.title('Mean Fibrinogen (mg/dL) by FFP Storage Duration Group', fontsize=14)
plt.xlabel('FFP Storage Duration Group (Days)', fontsize=12)
plt.ylabel('Mean Fibrinogen (mg/dL)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ------------------------------------------------------
# 14) ทดสอบ Normality (Shapiro) ของ Fibrinogen แยกแต่ละ FFP Storage Group
# ------------------------------------------------------
normality_results = []
for group_name, data_group in df.groupby('FFP Storage Group'):
    fibrinogen_values = data_group['Fibrinogen (mg/dL)'].dropna()
    if len(fibrinogen_values) > 0:
        stat, p_val = shapiro(fibrinogen_values)
        normality_results.append((group_name, stat, p_val))
    else:
        normality_results.append((group_name, np.nan, np.nan))

normality_df = pd.DataFrame(normality_results, 
                            columns=['FFP Storage Group', 'Shapiro-Wilk Statistic', 'P-value'])
print("Normality Test Results for Each FFP Storage Group:")
print(normality_df)
print("-" * 50)

# ------------------------------------------------------
# 15) ทดสอบ Kruskal-Wallis ดูความแตกต่างระหว่างหลายกลุ่ม
# ------------------------------------------------------
groups_for_test = [
    g['Fibrinogen (mg/dL)'].dropna() 
    for _, g in df.groupby('FFP Storage Group') 
    if len(g['Fibrinogen (mg/dL)'].dropna()) > 0
]

if len(groups_for_test) > 1:
    h_stat, p_value = kruskal(*groups_for_test)
    print(f"Kruskal-Wallis H-statistic: {h_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
else:
    print("Not enough groups with data for Kruskal-Wallis test.")
print("-" * 50)

# ------------------------------------------------------
# 16) นับจำนวนตัวอย่างในแต่ละกลุ่ม FFP Storage Group
# ------------------------------------------------------
sample_counts = df['FFP Storage Group'].value_counts().reindex(labels)
print("Sample Sizes in Each FFP Storage Group:")
print(sample_counts)
print("-" * 50)

# ------------------------------------------------------
# 17) แยกตามเพศ (Gender) + สร้างกราฟ Grouped Bar Chart
# ------------------------------------------------------
grouped_data_gender = df.groupby(['FFP Storage Group','Gender'])['Fibrinogen (mg/dL)'].mean().reset_index()
pivot_table = grouped_data_gender.pivot(index='FFP Storage Group', columns='Gender', values='Fibrinogen (mg/dL)')

plt.figure(figsize=(8, 6))
x = np.arange(len(pivot_table.index))
bar_width = 0.35

# ดึงค่า Fibrinogen ของ M และ F
m_values = pivot_table['M'] if 'M' in pivot_table.columns else []
f_values = pivot_table['F'] if 'F' in pivot_table.columns else []

# วาดแท่งของ M
plt.bar(x - bar_width/2, m_values, width=bar_width, label='M', alpha=0.7, edgecolor='black')
# วาดแท่งของ F
plt.bar(x + bar_width/2, f_values, width=bar_width, label='F', alpha=0.7, edgecolor='black')

plt.title('Mean Fibrinogen (mg/dL) by FFP Storage Group and Gender', fontsize=14)
plt.xlabel('FFP Storage Group (Days)', fontsize=12)
plt.ylabel('Mean Fibrinogen (mg/dL)', fontsize=12)
plt.xticks(x, pivot_table.index)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ------------------------------------------------------
# 18) (ตัวอย่างเสริม) Kruskal-Wallis ภายในแต่ละ Group เปรียบเทียบ M vs F
# ------------------------------------------------------
all_results = []
for group_name, data_group in df.groupby('FFP Storage Group'):
    m_group = data_group[data_group['Gender'] == 'M']['Fibrinogen (mg/dL)'].dropna()
    f_group = data_group[data_group['Gender'] == 'F']['Fibrinogen (mg/dL)'].dropna()
    
    if len(m_group) > 0 and len(f_group) > 0:
        h_stat, p_val = kruskal(m_group, f_group)
        all_results.append((group_name, h_stat, p_val))
    else:
        all_results.append((group_name, np.nan, np.nan))

results_df = pd.DataFrame(all_results, columns=['FFP Storage Group', 'Kruskal-H', 'P-value'])
print("Kruskal-Wallis (M vs F) in each FFP Storage Group:")
print(results_df)
