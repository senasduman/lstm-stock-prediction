import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from data_loader import (
    load_stock_data,
    scale_data,
    create_sequences
)
from model import build_lstm_model


def main():
    # 1. Load data
    df = load_stock_data(
        symbol="AAPL",
        start="2012-01-01",
        end="2024-01-01"
    )

    # 2. Scale data
    scaled_data, scaler = scale_data(df)

    # 3. Create sequences
    X, y = create_sequences(scaled_data, time_step=60)

    # 4. Train-test split (NO shuffle for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 5. Build model
    model = build_lstm_model((X_train.shape[1], 1))

    # 6. Train
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # 7. Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
