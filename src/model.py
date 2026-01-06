from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM,Dropout


def build_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(
        units=50,
        return_sequences=True,
        input_shape=input_shape
    ))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    return model