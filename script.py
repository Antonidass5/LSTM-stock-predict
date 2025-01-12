import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
from tkinter import messagebox
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose



nazwa_indeksu = "^TWII"

def getData(ticker, period, interval):
    try:
        data = yf.download(ticker, start = '2023-03-11',end = '2024-11-24', period=period, interval=interval)
        data['Difference'] = data['Close'].diff()
        next = []
        for i in range(len(data) - 1):
            next.append(data['Close'].values[i + 1])
        next.append(None)
        data['Next'] = next
        data.dropna(inplace=True)
        return data
    except Exception as e:
        messagebox.showerror("Błąd", f"Nie udało się pobrać danych: {e}")
        return None


data_set = getData(nazwa_indeksu, "1y", "1d")

data = data_set.asfreq('D')
data = data.fillna(method='ffill')
ts = data[['Close']]
decomposition = seasonal_decompose(ts, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residuals = decomposition.resid
data["decomposition"] = decomposition
data["trend"] = trend
data["seasonal"] = seasonal
data["residuals"] = residuals

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8))


axes[0].plot(ts, label='Original')
axes[0].set_ylabel('Original')

axes[1].plot(trend, label='Trend')
axes[1].set_ylabel('Trend')

axes[2].plot(seasonal, label='Seasonality')
axes[2].set_ylabel('Seasonality')

axes[3].plot(residuals, label='Residuals')
axes[3].set_ylabel('Residuals')

axes[0].set_title('Time Series Decomposition')
plt.tight_layout()
plt.show()

sc = MinMaxScaler(feature_range=(0, 1))
data_set_scaled = sc.fit_transform(data_set)

X = []
backcandles = 30
for j in range(data_set_scaled[0].size):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i - backcandles:i, j])

X = np.moveaxis(X, [0], [2])
X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
y = np.reshape(yi, (len(yi), 1))

splitlimit = int(len(X) * 0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

lstm_input = Input(shape=(backcandles, 7), name='lstm_input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dropout(0.2, name='dropout_layer')(inputs)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=16, epochs=100, shuffle=True, validation_split=0.1)

y_pred = model.predict(X_test)
test_dates = data_set.index[splitlimit + backcandles:]

# Odtwarzanie rzeczywistych i przewidywanych wartości w oryginalnej skali
y_test_diff = sc.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test), axis=1))[:, -1]
y_test_original = X_test[:, -1, 3] + y_test_diff
y_pred_diff = sc.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
y_pred_original = X_test[:, -1, 3] + y_pred_diff

# Obliczanie błędów punkt po punkcie
mse_values = []
mae_values = []
smape_values = []

for i in range(len(y_test_original)):
    mse = mean_squared_error([y_test_original[i]], [y_pred_original[i]])
    mae = mean_absolute_error([y_test_original[i]], [y_pred_original[i]])
    smape = 100 * (abs(y_test_original[i] - y_pred_original[i]) / (
                (abs(y_test_original[i]) + abs(y_pred_original[i])) / 2))

    mse_values.append(mse)
    mae_values.append(mae)
    smape_values.append(smape)

window_size = 10
mse_rolling = np.convolve(mse_values, np.ones(window_size) / window_size, mode='valid')
mae_rolling = np.convolve(mae_values, np.ones(window_size) / window_size, mode='valid')
smape_rolling = np.convolve(smape_values, np.ones(window_size) / window_size, mode='valid')

last_mse_rolling = mse_rolling[-1]
last_mae_rolling = mae_rolling[-1]
last_smape_rolling = smape_rolling[-1]

print(last_mse_rolling)
print(last_mae_rolling)
print(last_smape_rolling)

plt.figure(figsize=(16, 8))

# Wykres kroczącej średniej MSE
plt.subplot(3, 1, 1)
plt.plot(test_dates[window_size - 1:], mse_rolling, color='blue', label='Krocząca średnia MSE')
plt.xlabel('Data')
plt.ylabel('Błąd średniokwadratowy (MSE)')
plt.title('Krocząca średnia błędu MSE')
plt.legend()

# Wykres kroczącej średniej MAE
plt.subplot(3, 1, 2)
plt.plot(test_dates[window_size - 1:], mae_rolling, color='orange', label='Krocząca średnia MAE')
plt.xlabel('Data')
plt.ylabel('Błąd średni absolutny (MAE)')
plt.title('Krocząca średnia błędu MAE')
plt.legend()

# Wykres kroczącej średniej sMAPE
plt.subplot(3, 1, 3)
plt.plot(test_dates[window_size - 1:], smape_rolling, color='green', label='Krocząca średnia sMAPE')
plt.xlabel('Data')
plt.ylabel('Symetryczny błąd procentowy (sMAPE)')
plt.title('Krocząca średnia błędu sMAPE')
plt.legend()

plt.tight_layout()
plt.show()

# DataFrame z testowymi datami (przyszłe daty w stosunku do X_train)
test_dates = data_set.index[splitlimit + backcandles:]

# Odtwarzanie predykowanej ceny zamknięcia na podstawie y_pred
close_predicted = []
for i in range(len(y_pred)):
    last_close = X_test[i, -1, 3]
    predicted_close = last_close + y_pred[i]
    close_predicted.append(predicted_close)

# Przekształcanie danych testowych (y_test) do oryginalnej skali
y_test_diff = sc.inverse_transform(np.concatenate(
    (X_test[:, -1, :-1], y_test), axis=1))[:, -1]
y_test_original = X_test[:, -1, 3] + y_test_diff

# Przekształcanie predykcji do oryginalnej skali
y_pred_diff = sc.inverse_transform(np.concatenate(
    (X_test[:, -1, :-1], y_pred), axis=1))[:, -1]
y_pred_original = X_test[:, -1, 3] + y_pred_diff

# Pobieranie pełnych danych historycznych
historical_close = data_set['Close'].values

# Przewidywania na przyszłość (30 dni)
future_steps = 30
last_sequence = X_test[-1]  # ostatnia sekwencja danych
future_predictions = []

for _ in range(future_steps):
    next_pred = model.predict(np.expand_dims(last_sequence, axis=0))
    future_predictions.append(next_pred[0, 0])

    new_sequence = np.roll(last_sequence, -1, axis=0)
    new_sequence[-1, -1] = next_pred[0, 0]  # Zakładamy, że ostatnia kolumna to przewidywana różnica
    last_sequence = new_sequence

# Przekształcanie przyszłych predykcji do oryginalnej skali
future_predictions_scaled = sc.inverse_transform(
    np.concatenate((np.tile(X_test[-1, -1, :-1], (future_steps, 1)), np.array(future_predictions).reshape(-1, 1)),
                   axis=1)
)[:, -1]

future_dates = pd.date_range(start=data_set.index[-1], periods=future_steps + 1, inclusive='right')

# Tworzenie wykresu dla wszystkich danych
plt.figure(figsize=(16, 8))

data_hist = yf.download(nazwa_indeksu, start = '2023-03-11',end = '2024-12-24', period="1y", interval="1d")


plt.plot(data_hist.index, data_hist['Close'].values, color='blue', label='Dane historyczne (Close)')
plt.plot(test_dates, y_pred_original, color='green', label='Predykcja (Close)')
plt.plot(future_dates, future_predictions_scaled, color='red', label='Przyszłe przewidywania (Close)')

plt.xlabel('Data')
plt.ylabel('Cena zamknięcia')
plt.title('Predykcja ceny zamknięcia z LSTM (dane historyczne, predykcje i przyszłe przewidywania)')
plt.legend()
plt.show()
