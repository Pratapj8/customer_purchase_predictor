import numpy as np

def load_model():
    theta = np.load(r"F:\Data_Science_Projects\customer_purchase_predictor\model\theta.npy")
    X_mean, X_std = np.load(r"F:\Data_Science_Projects\customer_purchase_predictor\model\X_mean_std.npy", allow_pickle=True)
    return theta, X_mean, X_std

def predict_purchase(user_input, theta, X_mean, X_std):
    user_input = np.array(user_input).reshape(1, -1)
    user_input_norm = (user_input - X_mean) / X_std
    user_input_b = np.c_[np.ones((1, 1)), user_input_norm]
    prediction = user_input_b @ theta
    return prediction[0]


