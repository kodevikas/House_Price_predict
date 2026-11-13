import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt
import seaborn as sns 


import warnings
warnings.filterwarnings("ignore")


df=pd.read_csv("lt_reality.csv")

print("Reached the end of the file")


df.shape


df.head()


df.columns


df.describe()


df.info()


## checking missing values


df.isnull().sum()


### checking duplicate 

df.duplicated().sum()


df['date_listed'] = pd.to_datetime(df['date_listed'], dayfirst=True)


print(df['date_listed'].dtype)
print(df['date_listed'].head())


df['year'] = df['date_listed'].dt.year
df['month'] = df['date_listed'].dt.month
df['day'] = df['date_listed'].dt.day


df[['date_listed', 'year', 'month', 'day']].head()


cal_num = df.select_dtypes(include=["object"]).columns.tolist()
print(cal_num)

num_cal= df.select_dtypes(exclude=["object"]).columns.tolist()
print(num_cal)



corr = df[num_cal].corr()['price'].sort_values(ascending=False)

print(corr)



# correlation heatmap

correlation_col=['size_sqft',
                 'num_bedrooms', 
                 'num_bathrooms', 
                 'floor_number', 
                 'total_floors',
                 'age_years',
                  'distance_to_metro_km', 
                  'view_score'
]

plt.figure(figsize=(8, 6))

corr = df[correlation_col].corr()

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
#plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

target="price"
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(target, errors='ignore')

plt.figure(figsize=(15, 8))

for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols), 1, i)
    sns.boxplot(x=df[col])
    plt.title(col)


#plt.show()




from sklearn.model_selection import train_test_split

# Separating independent and dependent variables
X = df.drop(columns=['price', 'property_id', 'date_listed', 'amenities'])
y = df['price']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
print("Reached the end of the file")


numerical_cols=X_train.select_dtypes(include=["number"]).columns.tolist()

categorical_cols=X_train.select_dtypes(exclude=["number"]).columns.tolist()




from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


X_train_processed=preprocessor.fit_transform(X_train)

X_test_processed=preprocessor.transform(X_test)


from sklearn.linear_model import LinearRegression

model_LR=LinearRegression()


model_LR.fit(X_train_processed,y_train)


y_pred=model_LR.predict(X_test_processed)



from sklearn.linear_model import Lasso 

lasso=Lasso(alpha=1.0)

lasso.fit(X_train_processed,y_train)
lasso_ypred=lasso.predict(X_test_processed)


from sklearn.linear_model import Ridge

ridge=Ridge(alpha=1.0)

ridge.fit(X_train_processed,y_train)
ridge_ypred=ridge.predict(X_test_processed)


from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor(criterion='squared_error',random_state=42)

model.fit(X_train_processed,y_train)


DT_pred=model.predict((X_test_processed))


from sklearn.ensemble import RandomForestRegressor

model_RF=RandomForestRegressor(n_estimators=100,
                                random_state=42)
model_RF.fit(X_train_processed,y_train)


RF_pred=model_RF.predict(X_test_processed)


from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

def evaluate(model_name, y_tests, y_predict):
    mse = mean_squared_error(y_tests, y_predict)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_tests, y_predict)

    print(f"{model_name} Results:")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}\n")
evaluate("Linear Regression",y_test,y_pred)
evaluate("Lasso Regression",y_test,lasso_ypred)
evaluate("Ridge Regression",y_test,ridge_ypred)
evaluate("Decision tree",y_test,DT_pred)
evaluate("Random Forest",y_test,RF_pred)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


model_LR=Pipeline([
    ("preprocessor",preprocessor),
    ("model",LinearRegression())
])
model_LR.fit(X_train,y_train)


LR_pred_pipe=model_LR.predict(X_test)


preprocessor_without_scaling = ColumnTransformer([
    ('num', 'passthrough', numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])


model_DT=Pipeline([
    ("preprocessor",preprocessor_without_scaling),
    ("model",DecisionTreeRegressor())
])

model_DT.fit(X_train,y_train)


DT_pred_pipe=model_DT.predict(X_test)


model_RF=Pipeline([
    ("preprocessor",preprocessor),
    ("model",RandomForestRegressor(n_estimators=100,
                                    random_state=42,
                                    n_jobs=-1
                                   ))
])

model_RF.fit(X_train,y_train)


RF_pred_pipe=model_RF.predict(X_test)


def evalute(model_name,y_tests,y_predict):
    mse=mean_squared_error(y_tests,y_pred)
    rmse=np.sqrt(mse)
    r2=r2_score(y_tests,y_pred)

    print(f"{model_name} Results:")
    print(f" RMSE: {rmse:.3f}")
    print(f" r2: {r2:.3f}\n")

evalute("Linear Regression",y_test,LR_pred_pipe)
evalute("Decision tree",y_test,DT_pred_pipe)
evalute("Random Forest",y_test,RF_pred_pipe)


from sklearn.model_selection import  KFold


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': [0.7, 1.0]
}


random_search = RandomizedSearchCV(
    estimator=model_RF,
    param_distributions=param_grid,
    n_iter=20,
    cv=kf,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)


print("Best Parameters:", random_search.best_params_)
print("Best R2 Score:", random_search.best_score_)

RF_best_model = random_search.best_estimator_

RF_y_pred = RF_best_model.predict(X_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

RF_y_pred = RF_best_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, RF_y_pred))
print("MSE:", mean_squared_error(y_test, RF_y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, RF_y_pred)))
print("R2 Score:", r2_score(y_test, RF_y_pred))


from sklearn.ensemble import VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


voting_model = VotingRegressor(
    estimators=[
        ('lr', model_LR),
        ('dt', model_DT),
        ('rf', RF_best_model)
    ],
    n_jobs=-1
)
voting_model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

voting_pred = voting_model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, voting_pred))
print("MSE:", mean_squared_error(y_test, voting_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, voting_pred)))
print("R2 Score:", r2_score(y_test, voting_pred))


# Training predictions
y_train_pred = RF_best_model.predict(X_train)

# Testing predictions
y_test_pred = RF_best_model.predict(X_test)

print("Training R2:", r2_score(y_train, y_train_pred))
print("Testing R2:", r2_score(y_test, y_test_pred))


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, RF_y_pred, alpha=0.5)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

#plt.show()


residuals = y_test - RF_y_pred

plt.figure(figsize=(8, 6))
plt.scatter(RF_y_pred, residuals, alpha=0.5)

plt.axhline(y=0, linestyle='--')

plt.xlabel("Predicted House Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")

#plt.show()


import joblib

joblib.dump(RF_best_model, "house_price_prediction_model.joblib")

print("Model saved successfully!")
