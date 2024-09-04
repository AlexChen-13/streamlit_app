import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import io
import zipfile

st.title('Приложение по чаевым')
st.write('Приложение призванное визуализировать датасет по чаевым')


uploaded_file = st.sidebar.file_uploader('Загружаем файл', type='csv')

if uploaded_file is not None:
    tips = pd.read_csv(uploaded_file)
    st.write(tips.head(5))
else:
    st.stop()

@st.cache_data
def generate_random_dates(size):
    dates = pd.date_range(start='2023-01-01', end='2023-01-31').tolist()
    random_dates = np.random.choice(dates, size=size, replace=True)
    return random_dates

tips['time_order'] = generate_random_dates(size=len(tips))

zip_buffer = io.BytesIO()
zip_file = zipfile.ZipFile(zip_buffer, 'w')

def save_plot_to_zip(title, buf):
    buf.seek(0)
    zip_file.writestr(title, buf.getvalue())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='time_order', y='tip', color='blue', s=150, marker='o')
plt.title('Динамика чаевых во времени через скатерплот')
plt.xlabel('Даты')
plt.ylabel('Чаевые')
st.pyplot(plt)

buf = io.BytesIO()
plt.savefig(buf, format='png') 
save_plot_to_zip('Динамика чаевых во времени через скатерплот.png', buf)

plt.figure(figsize=(10, 6))
sns.relplot(data=tips, x='time_order', y='tip')
plt.title('Динамика чаевых во времени через реплот')
plt.xlabel('Даты')
plt.ylabel('Чаевые')
st.pyplot(plt)

buf1 = io.BytesIO()
plt.savefig(buf1, format='png') 
save_plot_to_zip('Динамика чаевых во времени через реплот.png', buf1)

plt.figure(figsize=(10, 6))
plt.hist(tips['total_bill'], bins=10)
plt.title('гистограмма total_bill через hist')
st.pyplot(plt)

buf2 = io.BytesIO()
plt.savefig(buf2, format='png') 
save_plot_to_zip('гистограмма total_bill через hist.png', buf2)

plt.figure(figsize=(10, 8))
sns.displot(data = tips, x = 'total_bill')
plt.title('гистограмма total_bill через displot')
st.pyplot(plt)

buf3 = io.BytesIO()
plt.savefig(buf3, format='png') 
save_plot_to_zip('гистограмма total_bill через displot.png', buf3)

plt.figure(figsize=(20, 8))
sns.scatterplot(data=tips, x='total_bill', y='tip', color='blue', s=150, marker='o')
plt.title('Зависимость размера чека и чаевых через скатер')
plt.xlabel('total_bill')
plt.ylabel('tip')
st.pyplot(plt)

buf4 = io.BytesIO()
plt.savefig(buf4, format='png') 
save_plot_to_zip('Зависимость размера чека и чаевых через скатер.png', buf4)

plt.figure(figsize=(20, 8))
sns.relplot(data=tips, x='total_bill', y='tip', color='blue', s=150, marker='o')
plt.title('Зависимость размера чека и чаевых через реплот')
plt.xlabel('total_bill')
plt.ylabel('tip')
st.pyplot(plt)

buf5 = io.BytesIO()
plt.savefig(buf5, format='png') 
save_plot_to_zip('Зависимость размера чека и чаевых через реплот.png', buf5)

# plt.figure(figsize=(20, 8))
# sns.scatterplot(data=tips, x='total_bill', y='tip', size = 'size', sizes = (20, 300), color='blue', s=150, marker='o')
# plt.title('Зависимость размера чека и чаевых и size')
# plt.xlabel('total_bill')
# plt.ylabel('tip')
# st.pyplot(plt)

st.subheader("Зависимость размера чека и чаевых и size")
fig = px.scatter(tips, x='total_bill', y='tip', size = 'size')
st.plotly_chart(fig)

buf6 = io.BytesIO()
plt.savefig(buf6, format='png') 
save_plot_to_zip('Зависимость размера чека и чаевых и size.png', buf6)

plt.figure(figsize=(20, 8))
plt.bar(tips['day'], tips['total_bill'], label = 'Размер счёта',  width = 0.5, color ='brown')
plt.title('Размер счёта по дням недели')
plt.xlabel('Дни недели')
plt.ylabel('Размер счёта')
plt.legend()
st.pyplot(plt)

buf7 = io.BytesIO()
plt.savefig(buf7, format='png') 
save_plot_to_zip('Размер счёта по дням недели.png', buf7)

# plt.figure(figsize=(20, 6))
st.subheader("Чаевые по дням недели")
fig = px.scatter(tips, x='tip', y='day', color='sex')
# plt.title('Чаевые по дням недели')
# plt.xlabel('размер чаевых')
# plt.ylabel('Дни недели')
# st.pyplot(plt)
st.plotly_chart(fig)

buf8 = io.BytesIO()
plt.savefig(buf8, format='png') 
save_plot_to_zip('Чаевые по дням недели.png', buf8)

df = pd.DataFrame(tips.groupby(['time_order', 'time'])['total_bill'].sum()).reset_index()
plt.figure(figsize=(20, 6))
sns.boxplot(x="time", y="total_bill", color = 'red', data=df)
plt.xlabel('Причины смертей')
plt.ylabel('Срок правления')
plt.title('Зависимость причины смерти и длительности периода правления через boxplot')
st.pyplot(plt)

buf9 = io.BytesIO()
plt.savefig(buf9, format='png') 
save_plot_to_zip('Зависимость причины смерти и длительности периода правления через boxplot.png', buf9)

plt.figure(figsize=(20, 6))
sns.catplot(x="time", y="total_bill", color = 'red', data=df,  kind="violin")
plt.xlabel('Причины смертей')
plt.ylabel('Срок правления')
plt.title('Зависимость причины смерти и длительности периода правления через catplot')
st.pyplot(plt)

buf10 = io.BytesIO()
plt.savefig(buf10, format='png') 
save_plot_to_zip('Зависимость причины смерти и длительности периода правления через catplot.png', buf10)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(tips[tips['time'] == 'Dinner']['tip'], bins=10)
axs[0].set_title('распределение чаевых по Dinner')
axs[1].hist(tips[tips['time'] == 'Lunch']['tip'], bins=10)
axs[1].set_title('распределение чаевых по Lunch')
st.pyplot(plt)

buf11 = io.BytesIO()
plt.savefig(buf11, format='png') 
save_plot_to_zip('Распределение чаевых по Dinner и Lunch.png', buf11)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sns.scatterplot(data = tips[tips['sex'] == 'Male'], x = 'total_bill', y = 'tip', hue = 'smoker', ax=axs[0])
axs[0].set_title('Связь размера счёта и чаевых для мужчин')
sns.scatterplot(data = tips[tips['sex'] == 'Female'], x = 'total_bill', y = 'tip', hue = 'smoker', ax=axs[1])
axs[0].set_title('Связь размера счёта и чаевых для женщин')
st.pyplot(plt)

buf12 = io.BytesIO()
plt.savefig(buf12, format='png') 
save_plot_to_zip('Связь размера счёта и чаевых для мужчин и женщин.png', buf12)

df_digit = tips.select_dtypes(include=['int64', 'float64'])
df_corr = df_digit.corr(method='pearson')
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot = True)
st.pyplot(plt)

buf13 = io.BytesIO()
plt.savefig(buf13, format='png') 
save_plot_to_zip('Тепловая карта зависимостей численных переменных.png', buf13)

zip_file.close()  
zip_buffer.seek(0)

st.sidebar.download_button(
    label="Скачать все графики",
    data=zip_buffer.getvalue(),
    file_name='графики.zip',
    mime='application/zip')