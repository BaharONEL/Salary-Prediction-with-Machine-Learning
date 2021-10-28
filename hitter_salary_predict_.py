### DEĞİŞKENlER ####
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör


###################################### Libraries ###################################################################
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn import preprocessing
from sklearn.neighbors import LocalOutlierFactor


pd.set_option("display.float_format",lambda x: "%.5f" % x)
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
import warnings
warnings.filterwarnings("ignore")
df_= pd.read_csv(r"C:\Users\Oguz\Desktop\DCMLBC06\HAFTA08\dataset\hitters.csv")
df = df_.copy()
df.head()
############################################### Check Data ############################################################
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)
############################################# Missing Value ###########################################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df, True)
df[df.Salary.isnull()==True].head()

df = df.dropna()
df.isnull().sum()
df.shape #263 gözlerm birimi ve 20 değişken kaldı

################################################## Outlier ###########################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def outlier_thresholds(dataframe, col_name):
    up_limit = dataframe[col_name].mean() + 3*dataframe[col_name].std()
    low_limit = dataframe[col_name].mean() - 3*dataframe[col_name].std()
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
for col in num_cols:
    df = remove_outlier(df, col)
df.shape

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df,"CHits",index=True) #Int64Index([16, 24], dtype='int64')
grab_outliers(df,"CAtBat",index=True) #Int64Index([16, 24, 70], dtype='int64')
grab_outliers(df,"CRuns",index=True)
grab_outliers(df,"CRBI",index=True) #Int64Index([24, 86, 117, 132, 143], dtype='int64')

df.drop([16,24,70,86,117,132,143],axis=0,inplace=True)

import seaborn as sns
sns.distplot(df.CRBI);
############################################# Local Outlier Factor ####################################################
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.columns = [x.upper() for x in df.columns]
for col in df.columns:
    print(col, check_outlier(df, col))

clf = LocalOutlierFactor(n_neighbors=10)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[2]

df[df_scores < th]
df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
df.shape

############################################ Feature Engineering ######################################################
df.describe().T
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = df[(df['SALARY']<1300) | (df['SALARY'].isnull())]

import seaborn as sns
sns.distplot(df.CHITS);


df.shape
df.head()
df.replace(0, 0.001, inplace=True)


df["NEW_C_RUNS_RATIO"] = df["RUNS"] / df["CRUNS"]
df["NEW_C_ATBAT_RATIO"] = df["ATBAT"] / df["CATBAT"]
df["NEW_C_HITS_RATIO"] = df["HITS"] / df["CHITS"]
df["NEW_C_HMRUN_RATIO"] = df["HMRUN"] / df["CHMRUN"]
df["NEW_C_RBI_RATIO"] = df["RBI"] / df["CRBI"]
df["NEW_C_WALKS_RATIO"] = df["WALKS"] / df["CWALKS"]
df["New_BattingAverage"] = df["CHITS"] / df["CATBAT"]
df["NEW_C_RUNNER"] = df["CRBI"] / df["CHITS"]
df["NEW_C_HIT-AND-RUN"] = df["CRUNS"] / df["CHITS"]
df["NEW_C_HMHITS_RATIO"] = df["CHMRUN"] / df["CHITS"]
df["NEW_C_HMATBAT_RATIO"] = df["CATBAT"] / df["CHMRUN"]
df["NEW_CATBAT_MEAN"] = df["CATBAT"] / df["YEARS"]
df["NEW_CHITS_MEAN"] = df["CHITS"] / df["YEARS"]
df["NEW_CHMRUN_MEAN"] = df["CHMRUN"] / df["YEARS"]
df["NEW_CRUNS_MEAN"] = df["CRUNS"] / df["YEARS"]
df["NEW_CRBI_MEAN"] = df["CRBI"] / df["YEARS"]
df["NEW_CWALKS_MEAN"] = df["CWALKS"] / df["YEARS"]

df.loc[(df["YEARS"] <= 2), "NEW_YEARS_LEVEL"] = "Junior"
df.loc[(df["YEARS"] > 2) & (df['YEARS'] <= 5), "NEW_YEARS_LEVEL"] = "Mid"
df.loc[(df["YEARS"] > 5) & (df['YEARS'] <= 10), "NEW_YEARS_LEVEL"] = "Senior"
df.loc[(df["YEARS"] > 10), "NEW_YEARS_LEVEL"] = "Expert"

df["NEW_ASSISTS_RATIO"] = df["ASSISTS"] / df["ATBAT"]
df["NEW_HITS_RECALL"] = df["HITS"] / (df["HITS"] + df["ERRORS"])
df["NEW_NET_HELPFUL_ERROR"] = (df["WALKS"] - df["ERRORS"]) / df["WALKS"]
df["NEW_TOTAL_SCORE"] = (df["RBI"] + df["ASSISTS"] + df["WALKS"] - df["ERRORS"]) / df["ATBAT"]
df["NEW_HIT_RATE"] = df["HITS"] / df["ATBAT"]
df["NEW_TOUCHER"] = df["ASSISTS"] / df["PUTOUTS"]
df["NEW_RUNNER"] = df["RBI"] / df["HITS"]
df["NEW_HIT-AND-RUN"] = df["RUNS"] / (df["HITS"])
df["NEW_HMHITS_RATIO"] = df["HMRUN"] / df["HITS"]
df["NEW_HMATBAT_RATIO"] = df["ATBAT"] / df["HMRUN"]
df["NEW_TOTAL_CHANCES"] = df["ERRORS"] + df["PUTOUTS"] +df["ASSISTS"]

df["New_TotalBases"] = ((df["CHITS"] * 2) + (4 * df["CHMRUN"]))
    df["New_SluggingPercentage"] = df["New_TotalBases"] / df["CATBAT"]
    df["New_IsolatedPower"] = df["New_SluggingPercentage"] - df["New_BattingAverage"]
    df["New_TripleCrown"] = (df["CHMRUN"] * 0.4) + (df["CRBI"] * 0.25) + (df["New_BattingAverage"] * 0.35)
    df["New_BattingAverageOnBalls"] = (df["CHITS"] - df["CHMRUN"]) / (df["CATBAT"] - df["CHMRUN"])
    df["New_RunsCreated"] = df["New_TotalBases"] * (df["CHITS"] + df["CWALKS"]) / (df["CATBAT"] + df["CWALKS"])
    df["New_FieldingPercentage"] = 1 - (
                (df["PUTOUTS"] + df["ASSISTS"]) / (df["PUTOUTS"] + df["ASSISTS"] + df["ERRORS"] + 1))


################################################## Correlations #######################################################
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True ,cmap="twilight_shifted_r")


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=False)


############################################# ENCODING ################################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols= [col for col in df.columns if len(df[col].unique())==2 and df[col].dtypes=="O"]

def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    label_encoder(df, col)

df.head()

onehot_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, onehot_cols)
df.head()
missing_values_table(df)
#df.to_csv("df_exp1", index=False)


######################################## Modeling ######################################################################
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
y=df["SALARY"]
X=df.drop("SALARY", axis=1)


dict_corr ={col:X[col].corr(y) for col in X.columns}
dict_corr=dict(sorted(dict_corr.items(), key=lambda x: abs(x[1]), reverse=True))
High_Corr_With_Y=[]
for k,v in dict_corr.items():
    print(k,"- Salary corr_value =",v)
    if abs(v)>=0.65:
        High_Corr_With_Y.append(k)
High_Corr_With_Y



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=85)


print("***************  Base Models  ***************")
    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              # ("CatBoost", CatBoostRegressor(verbose=False))
              ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
        print(f"Base_Model_RMSE: {round(rmse, 4)} ({name}) ")


############################################  Feature Importance ######################################################
def plot_importance(model, features, num=len("X"), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title(f" Features for {type(model).__name__}")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
###################################### Automated Hyperparameter Optimization ##########################################
    print("**************  Automated Hyperparameter Optimization  **************")


    rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500, 1000]}

    xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                      "max_depth": [5, 8, 12, 20],
                      "n_estimators": [100, 200, 300, 500],
                      "colsample_bytree": [0.5, 0.8, 1]}

    lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                       "n_estimators": [300, 500, 1500],
                       "colsample_bytree": [0.5, 0.7, 1]}

    gbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                  "n_estimators": [100, 300, 500, 1500],
                  "min_samples_split": [8, 15, 20]}

    regressors = [("RF", RandomForestRegressor(), rf_params),
                  ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
                  ('LightGBM', LGBMRegressor(), lightgbm_params),
                  ('GBM', GradientBoostingRegressor(), gbm_params)]

    best_models = {}

    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X_train, y_train)

        final_model = regressor.set_params(**gs_best.best_params_).fit(X_train, y_train)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")
        print(np.sqrt(-cross_val_score(final_model,X_train,y_train,cv=5,scoring="neg_mean_squared_error")))
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model


#plot_importance(best_models["GBM"].fit(X, y), X, 30)
#plot_importance(best_models["XGBoost"].fit(X, y), X, 30)
plot_importance(best_models["RF"].fit(X, y), X, 30)
#plot_importance(best_models["LightGBM"].fit(X, y), X, 30)

###################################### Stacking & Ensemble Learning ###################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"RMSE for Test: {round(rmse,4)} ({'Ensemble of RF and GBM '})")


########## RF ##########
# RMSE: 166.7414 (RF)
# RMSE (After): 166.2756 (RF)
# [193.69211755 144.78767343 134.05994937 160.47610687 196.59965404]