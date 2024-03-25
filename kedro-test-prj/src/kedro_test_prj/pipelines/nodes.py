import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib


# read dataset
def read_csv_file(df_path):
    df = pd.read_csv(df_path)
    return df

# drop features 
def drop_features(df):
    df = df.drop("Accounts Delinquent", axis = "columns")
    return df

def get_type_features(df):
    cate_features = [i for i in df.select_dtypes(include='object').columns] # Lấy danh sách biến rời rạc
    num_features = [i for i in df.select_dtypes(exclude='object').columns] # Lấy danh sách biến liên tục
    return cate_features, num_features

def process_outlier_fts(df):
    cate_features, num_features = get_type_features(df)
    num_features_1 = [i for i in num_features if df[i].std() > 5]
    
    for feature in num_features_1:
        g = dict(df.groupby(feature)['Loan Status'].mean())
        df[feature] = df[feature].replace(g)
        
    return df
    
def get_dummies(df):
    df = pd.get_dummies(df)
    return df

def get_train_test_dataset(df, test_size):
    X = df.drop('Loan Status', axis = "columns") # Lấy tất cả các biến trừ biến y
    y = df['Loan Status'] # Lấy biến y

    X = np.array(X) # Chuyển df sang numpy
    y = np.array(y) # Chuyển df sang numpy
    
    # Tái chọn mẫu (resampling) dữ liệu để xử lý vấn đề mất cân bằng giữa các lớp trong biến phụ thuộc
    sampler = RandomOverSampler()
    X_res, y_res = sampler.fit_resample(X, y)
    
    # Chia dữ liệu đã được tái chọn mẫu thành tập huấn luyện và tập kiểm thử
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = test_size, random_state=0)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    logreg_model = LogisticRegression()
    logreg_model.fit(X_train, y_train)
    return logreg_model

def train_knn(X_train, y_train):
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    return knn_model

def load_model(path_to_model):
    model = joblib.load(path_to_model)
    return model


def evalues_model(model, X_test, y_test):
    results = []
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    roc_score = roc_auc_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        'Accuracy': accuracy,
        'ROC Score': roc_score,
        'F1 Score': f1
    })
    
    # Hiển thị kết quả
    results_df = pd.DataFrame(results)
    return results_df
    