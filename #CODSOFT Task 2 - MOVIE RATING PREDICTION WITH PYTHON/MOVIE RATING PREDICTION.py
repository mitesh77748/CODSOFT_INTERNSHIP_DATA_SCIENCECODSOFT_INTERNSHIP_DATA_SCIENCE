import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn .tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r'E:\lab\project\Data_Science\Codsoft_task 2\IMDb Movies India.csv',encoding='latin1').dropna()
print(data.columns)
print('Info:','\n')
print(data.info(),"\n\n\n\n\n")
print('summary of the dataframe:','\n',data.describe,'\n\n\n\n\n')
print('nunique:','\n',data['Genre'].nunique(),'\n\n\n\n\n')
print('unique:',"\n",data['Year'].unique(),'\n\n\n\n\n')
print('Rating.unique:',"\n",data.Rating.unique(),'\n\n\n\n\n')
print("unique:",'\n',data['Duration'].unique(),"\n\n\n\n\n")
print("groupby(['Genre]):","\n",data.groupby(['Genre']).count(),"\n\n\n\n\n")
print("value_counts:","\n",data['Director'].value_counts().head(6),"\n\n\n\n\n")
print("isnull().any():",'\n',data.isnull().any(),"\n\n\n\n\n")

def missing_values_percentage(data):
    missing_values=data.isna().sum()
    percentage_missing = (missing_values / len(data) * 100).round(2)
    result_movie=pd.DataFrame({'missing Values':missing_values})
    return result_movie
result=missing_values_percentage(data)
print(result)
print(data.describe())

data['Year'] = data['Year'].str.replace(r'\D', '', regex=True).astype(int)
data['Duration'] = data['Duration'].str.replace(r'\D', '', regex=True).astype(int)
data['Votes'] = data['Votes'].str.replace(',', '').astype(int)
print(data.info())

years=data['Year']
ratings=data['Rating']

plt.figure(figsize=(10,6))
plt.scatter(years,ratings,alpha=0.5,color='blue')
plt.title('Scatter plot of movie Based on year')
plt.xlabel('year')
plt.ylabel('rating')
plt.show()

years=data['Year']
ratings=data['Rating']

df_year_rating=pd.DataFrame({'Year':years,'Rating':ratings})
#print(df_year_rating)
count_movies=df_year_rating.groupby(['Year',"Rating"]).size().reset_index(name='Count')

plt.figure(figsize=(12,8))
sns.barplot(x='Year',y='Count',hue='Rating',data=count_movies, palette='viridis')
plt.title('Numberof movies within a year Based on rating')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.legend(title='Rating',loc='upper right')
plt.show()

print(data['Genre'].unique())

genre_column=['Genre']
filtered_data=data[(data['Year']>=2000)&(data['Year']<=2023)]

years=sorted(filtered_data['Year'].unique())
num_years=len(years)

plt.figure(figsize=(15,5*num_years))
for i, year in enumerate(years,1):
    plt.subplot(num_years,1,i)
    year_data=filtered_data[filtered_data["Year"]==year]
    genre_counts=year_data['Genre'].value_counts()
    plt.pie(genre_counts,labels=genre_counts.index,autopct='%1.1f%%',startangle=90,colors=plt.cm.Paired(range(len(genre_counts))))
    plt.title(f'Genre Distribution-{year}')
plt.tight_layout()
plt.show()

genre_mean_rating=data.groupby("Genre")['Rating'].transform('mean')
data['Genre_mean_rating']=genre_mean_rating

from sklearn.linear_model import LinearRegression

data["Director_encoded"]=data.groupby('Director')['Rating'].transform('mean')
data['Actor_encoded']=data.groupby('Actor 1')['Rating'].transform('mean')

features=['Year','Votes','Duration','Genre_mean_rating','Director_encoded','Actor_encoded']
X=data[features]
y=data['Rating']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
table=[["Mean Squared Error",mse],
       ["Mean Absolute Error",mae],
       ["R2 Score",r2]]
print(tabulate(table,headers=['Metric','Value'],tablefmt='pretty'))

plt.figure(figsize=(8,8))
plt.scatter(y_test,y_pred,alpha=0.5)
plt.xlabel("Actual ratings")
plt.ylabel('Predicted ratings')
plt.title("Actual vs. Predicted Ratings (Linear Regression)")
plt.show()