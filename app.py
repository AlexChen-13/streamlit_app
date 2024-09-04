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

plt.figure(figsize=(10, 6))
sns.scatterplot(data=tips, x='time_order', y='tip', color='blue', s=150, marker='o')
plt.title('Динамика чаевых во времени через скатерплот')
plt.xlabel('Даты')
plt.ylabel('Чаевые')
st.pyplot(plt)

buf = io.BytesIO()
plt.savefig(buf, format='png') 
buf.seek(0)
zip_file.writestr('Динамика чаевых во времени через скатерплот', buf.getvalue())

plt.figure(figsize=(10, 6))
sns.relplot(data=tips, x='time_order', y='tip')
plt.title('Динамика чаевых во времени через реплот')
plt.xlabel('Даты')
plt.ylabel('Чаевые')
st.pyplot(plt)

buf1 = io.BytesIO()
plt.savefig(buf1, format='png') 
buf1.seek(0)
zip_file.writestr('Динамика чаевых во времени через реплот', buf1.getvalue())

plt.figure(figsize=(10, 6))
plt.hist(tips['total_bill'], bins=10)
plt.title('гистограмма total_bill через hist')
st.pyplot(plt)

buf2 = io.BytesIO()
plt.savefig(buf2, format='png') 
buf2.seek(0)
zip_file.writestr('гистограмма total_bill через hist', buf2.getvalue())

plt.figure(figsize=(10, 8))
sns.displot(data = tips, x = 'total_bill')
plt.title('гистограмма total_bill через displot')
st.pyplot(plt)

buf3 = io.BytesIO()
plt.savefig(buf3, format='png') 
buf3.seek(0)
zip_file.writestr('гистограмма total_bill через displot', buf3.getvalue())

plt.figure(figsize=(20, 8))
sns.scatterplot(data=tips, x='total_bill', y='tip', color='blue', s=150, marker='o')
plt.title('Зависимость размера чека и чаевых через скатер')
plt.xlabel('total_bill')
plt.ylabel('tip')
st.pyplot(plt)

buf4 = io.BytesIO()
plt.savefig(buf4, format='png') 
buf4.seek(0)
zip_file.writestr('Зависимость размера чека и чаевых через скатер', buf4.getvalue())

plt.figure(figsize=(20, 8))
sns.relplot(data=tips, x='total_bill', y='tip', color='blue', s=150, marker='o')
plt.title('Зависимость размера чека и чаевых через реплот')
plt.xlabel('total_bill')
plt.ylabel('tip')
st.pyplot(plt)

buf5 = io.BytesIO()
plt.savefig(buf5, format='png') 
buf5.seek(0)
zip_file.writestr('Зависимость размера чека и чаевых через реплот', buf5.getvalue())

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
buf6.seek(0)
zip_file.writestr('Зависимость размера чека и чаевых и size', buf6.getvalue())

plt.figure(figsize=(20, 8))
plt.bar(tips['day'], tips['total_bill'], label = 'Размер счёта',  width = 0.5, color ='brown')
plt.title('Размер счёта по дням недели')
plt.xlabel('Дни недели')
plt.ylabel('Размер счёта')
plt.legend()
st.pyplot(plt)

buf7 = io.BytesIO()
plt.savefig(buf7, format='png') 
buf7.seek(0)
zip_file.writestr('Размер счёта по дням недели', buf7.getvalue())

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
buf8.seek(0)
zip_file.writestr('Чаевые по дням недели', buf8.getvalue())

df = pd.DataFrame(tips.groupby(['time_order', 'time'])['total_bill'].sum()).reset_index()
plt.figure(figsize=(20, 6))
sns.boxplot(x="time", y="total_bill", color = 'red', data=df)
plt.xlabel('Причины смертей')
plt.ylabel('Срок правления')
plt.title('Зависимость причины смерти и длительности периода правления через boxplot')
st.pyplot(plt)

buf9 = io.BytesIO()
plt.savefig(buf9, format='png') 
buf9.seek(0)
zip_file.writestr('Зависимость причины смерти и длительности периода правления через boxplot', buf9.getvalue())

plt.figure(figsize=(20, 6))
sns.catplot(x="time", y="total_bill", color = 'red', data=df,  kind="violin")
plt.xlabel('Причины смертей')
plt.ylabel('Срок правления')
plt.title('Зависимость причины смерти и длительности периода правления через catplot')
st.pyplot(plt)

buf10 = io.BytesIO()
plt.savefig(buf10, format='png') 
buf10.seek(0)
zip_file.writestr('Зависимость причины смерти и длительности периода правления через catplot', buf10.getvalue())

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(tips[tips['time'] == 'Dinner']['tip'], bins=10)
axs[0].set_title('распределение чаевых по Dinner')
axs[1].hist(tips[tips['time'] == 'Lunch']['tip'], bins=10)
axs[1].set_title('распределение чаевых по Lunch')
st.pyplot(plt)

buf11 = io.BytesIO()
plt.savefig(buf11, format='png') 
buf11.seek(0)
zip_file.writestr('Распределение чаевых по Dinner и Lunch', buf11.getvalue())

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sns.scatterplot(data = tips[tips['sex'] == 'Male'], x = 'total_bill', y = 'tip', hue = 'smoker', ax=axs[0])
axs[0].set_title('Связь размера счёта и чаевых для мужчин')
sns.scatterplot(data = tips[tips['sex'] == 'Female'], x = 'total_bill', y = 'tip', hue = 'smoker', ax=axs[1])
axs[0].set_title('Связь размера счёта и чаевых для женщин')
st.pyplot(plt)

buf12 = io.BytesIO()
plt.savefig(buf12, format='png') 
buf12.seek(0)
zip_file.writestr('Связь размера счёта и чаевых для мужчин и женщин', buf12.getvalue())

df_digit = tips.select_dtypes(include=['int64', 'float64'])
df_corr = df_digit.corr(method='pearson')
plt.figure(figsize=(10, 6))
sns.heatmap(df_corr, annot = True)
st.pyplot(plt)

buf13 = io.BytesIO()
plt.savefig(buf13, format='png') 
buf13.seek(0)
zip_file.writestr('Тепловая карта зависимостей численных переменных', buf13.getvalue())

zip_file.close()  
zip_buffer.seek(0)

st.sidebar.download_button(
    label="Скачать все графики",
    data=buf,
    file_name='графики.png',
    mime='image/png')