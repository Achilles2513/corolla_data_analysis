#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()
os.chdir('D:/_研究生/_研究生作业/数据挖掘/任务8_狗熊会')
os.getcwd()


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
from scipy.stats import shapiro, anderson
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, max_error, r2_score
from sklearn.model_selection import GridSearchCV


# In[3]:


corolla=pd.read_csv('ToyotaCorolla.csv')


# In[4]:


corolla.head()


# In[5]:


corolla.info()


# In[6]:


corolla.describe()


# In[7]:


'''corolla=corolla.rename(columns={
    'Age_08_04':'汽车车龄-2004/08',
    'Mfg_Month':'制造月份',
    'Mfg_Year':'制造年份',
    'KM':'公里数',
    'Fuel_type':'燃料类型',
    'HP':'马力',
    'Met_Color':'金属漆',
    'Color':'颜色',
   'Automatic': '自动变速器',
    'CC': '发动机尺寸',
    'Doors': '车门数量',
    'Cylinders': '气缸数',
    'Gears': '齿轮数',
    'Quarterly': '季度税',
    'Weight': '重量',
    'Mfr_Guarantee': '制造商保证',
    'BOVAG_Guarantee': 'BOVAG 保证',
    'Guarantee_Period': '保修期限',
    'ABS': '防抱死制动系统',
    'Airbag_1': '安全气囊1',
    'Airbag_2': '安全气囊2',
    'Airco': '空调',
    'Automatic_airco': '自动空调',
    'Boardcomputer': '车机',
    'CD_Player': 'CD播放器',
    'Central_Lock': '中控锁系统',
    'Powered_Windows': '通电窗口',
    'Power_Steering': '动力转向',
    'Radio': '收音机',
    'Mistlamps': '雾灯',
    'Sport_Model': '运动模型',
    'Backseat_Divider': '后座隔板',
    'Metallic_Rim': '金属轮辋',
    'Radio_cassette': '磁带式收音机',
    'Parking_Assistant': '停车助手',
    'Tow_Bar': '牵引杆'
})'''


# In[8]:


'''corolla.head()'''


# In[9]:


corolla_corr=corolla.drop(labels=['Model','Id','Fuel_Type'],axis=1,inplace=False)


# In[10]:


corolla_corr.head()


# In[11]:


#
correlation_matrix = corolla_corr.corr()
price_correlation = corolla_corr.corr()['Price']
print(price_correlation)


# In[12]:


price_correlation_matrix = corolla_corr.corr()['Price']
plt.figure(figsize=(8, 10))
sns.heatmap(price_correlation_matrix.to_frame(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap for Price')
plt.show()


# In[13]:


plt.figure(figsize=(10, 6))
sns.histplot(corolla['Price'], kde=True)
plt.title('Histogram and KDE for Price')
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
probplot(corolla['Price'], plot=plt)
plt.title('Q-Q Plot for Price')
plt.show()


# In[15]:


# 或者，如果你想要删除包含任何异常值的整个行，可以使用 drop 方法
corolla = corolla.drop(corolla[corolla['Price'] > 30000].index)

# 打印一些统计信息，确认异常值已被删除
corolla['Price'].head()


# In[16]:


stat_sw, p_sw = shapiro(corolla['Price'])
print(f'Shapiro-Wilk Test - Statistic: {stat_sw}, p-value: {p_sw}')

# Anderson-Darling 测试
result_ad = anderson(corolla['Price'])
print(f'Anderson-Darling Test - Statistic: {result_ad.statistic}, Critical Values: {result_ad.critical_values}')


# In[17]:


# 选择相关系数绝对值大于 0.5 的列（排除 'Price' 本身）
high_corr_columns = correlation_matrix[abs(correlation_matrix['Price']) > 0.5].index.tolist()

# 保留相关性较高的指标
selected_data = corolla[high_corr_columns]

# 输出结果
selected_data.head()


# In[18]:


plt.figure(figsize=(10, 6))
sns.regplot(x='Age_08_04', y='Price', data=selected_data, scatter_kws={'s': 15}, line_kws={'color': 'red'})

# 添加标题和标签
plt.title('Scatter Plot of Price and Age_08_04 with Trendline')
plt.xlabel('Age_08_04')
plt.ylabel('Price')

# 显示图形
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.regplot(x='KM', y='Price', data=selected_data, scatter_kws={'s': 15}, line_kws={'color': 'red'})

# 添加标题和标签
plt.title('Scatter Plot of Price and KM with Trendline')
plt.xlabel('KM')
plt.ylabel('Price')

# 显示图形
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.histplot(selected_data['KM'], kde=True)
plt.title('Histogram and KDE for KM')
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.regplot(x='Weight', y='Price', data=selected_data, scatter_kws={'s': 15}, line_kws={'color': 'red'})

# 添加标题和标签
plt.title('Scatter Plot of Price and Weight with Trendline')
plt.xlabel('Weight')
plt.ylabel('Price')

# 显示图形
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Automatic_airco', y='Price', data=selected_data)

# 添加标题和标签
plt.title('Boxplot of Price and Automatic_airco')
plt.xlabel('Automatic_airco')
plt.ylabel('Price')

# 显示图形
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Boardcomputer', y='Price', data=selected_data)

# 添加标题和标签
plt.title('Boxplot of Price and Boardcomputer')
plt.xlabel('Boardcomputer')
plt.ylabel('Price')

# 显示图形
plt.show()


# In[24]:


selected_data.columns.tolist()


# In[25]:


# 选择自变量和因变量
features = ['Age_08_04', 'Mfg_Year', 'KM', 'Weight', 'Automatic_airco', 'Boardcomputer']
target = 'Price'


# In[26]:


# 提取特征和目标变量
X = corolla[features]
y = corolla[target]


# In[45]:


# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)


# In[46]:


# 在测试集上进行预测
y_pred = rf_model.predict(X_test)


# In[47]:


# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Explained Variance Score (EVS): {evs}')
print(f'Maximum Error: {max_err}')
print(f'R-squared (R2): {r2}')


# In[48]:


# 预测新样本
new_data = np.array([[3, 2002, 80000, 1200, 1, 1]])  # 请根据实际情况提供新样本的特征
predicted_price = rf_model.predict(new_data)
print(f'Predicted Price for the new sample: {predicted_price}')


# In[36]:


#优化参数
# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10]
}

# 创建随机森林回归模型
rf_model = RandomForestRegressor(random_state=42)

# 使用 GridSearchCV 进行超参数搜索
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# 输出最佳参数
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# 使用最佳参数的模型进行预测
best_rf_model = grid_search.best_estimator_
y_pred_optimized = best_rf_model.predict(X_test)

# 评估优化后模型的性能
mse_optimized = mean_squared_error(y_test, y_pred_optimized)
print(f'Mean Squared Error (Optimized): {mse_optimized}')

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Explained Variance Score (EVS): {evs}')
print(f'Maximum Error: {max_err}')
print(f'R-squared (R2): {r2}')

# 预测新样本
new_data = np.array([[3, 2002, 80000, 1200, 1, 1]])  # 请根据实际情况提供新样本的特征
predicted_price_optimized = best_rf_model.predict(new_data)
print(f'Predicted Price for the new sample (Optimized): {predicted_price_optimized}')


# In[37]:


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Explained Variance Score (EVS): {evs}')
print(f'Maximum Error: {max_err}')
print(f'R-squared (R2): {r2}')


# In[38]:


corolla['Predicted_Price'] = best_rf_model.predict(corolla[features])


# In[39]:


corolla.head()


# In[40]:


corolla.to_csv('modified_corolla.csv', index=False)


# In[41]:


# 获取特征的重要性分数
feature_importances = best_rf_model.feature_importances_

# 创建一个包含特征名和对应重要性分数的DataFrame
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# 按重要性降序排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 绘制特征重要性图表
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance Plot')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[ ]:




