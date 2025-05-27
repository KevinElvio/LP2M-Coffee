import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, GaussianNoise
from tensorflow.keras import backend as K

df = pd.read_csv('dataSetKopi_Bersih.csv')
df = df[['Species','Variety','Aroma','Flavor','Aftertaste','Acidity', 'Body', 'Balance', 'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points', 'Total.Cup.Points', 'Moisture']]
df

features = ['Aroma','Flavor','Aftertaste','Acidity', 'Body', 'Balance',
            'Uniformity', 'Clean.Cup', 'Sweetness', 'Cupper.Points',
            'Total.Cup.Points', 'Moisture']
X = df[features].values
y = df['Species']

# 3. Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 4. Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 5. Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# 6. Oversampling dengan RandomOverSampler
from imblearn.under_sampling import ClusterCentroids

ros = ClusterCentroids(random_state=42)
# ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# 7. Reshape untuk LSTM
X_res_lstm = np.expand_dims(X_resampled, axis=1)
X_test_lstm = np.expand_dims(X_test, axis=1)

# 8. One-hot encode label
y_res_cat = to_categorical(y_resampled)
y_test_cat = to_categorical(y_test)


# 9. Clear session & Build Model
K.clear_session()

model = Sequential()
model.add(GaussianNoise(0.05, input_shape=(1, X_res_lstm.shape[2])))  # Sedikit noise
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(y_res_cat.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# --- Train model ---
history = model.fit(
    X_res_lstm, y_res_cat,
    validation_data=(X_test_lstm, y_test_cat),
    epochs=30,
    batch_size=32,
    verbose=1
)


score = model.evaluate(X_test_lstm, y_test_cat,verbose=0)
print("Accuracy: {}".format(score[1] * 100))


yhat = model.predict(X_test_lstm, verbose=0)
classes_x=np.argmax(yhat,axis=1)

y_asdadasd = []
for data in y_test_cat:
  for i,item in enumerate(data):
    if item == 1:
      y_asdadasd.append(i)

npa = np.asarray(y_asdadasd)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support as score

precision,recall,fscore,support=score(npa,classes_x,average='macro')
print('Precision : ',format(precision))
print('Recall    : ',format(recall))
print('F-score   : ',format(fscore))
print('Accuracy : ',accuracy_score(npa, classes_x))


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

labels = ["Arabica", "Robusta"]

cm = confusion_matrix(npa, classes_x)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()




plt.show()


