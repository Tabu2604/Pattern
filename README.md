# Pattern
# Pattern Sense is a deep learning-based project designed to automate the classification of fabric patterns. The system utilizes advanced deep learning techniques to identify and categorize various fabric patterns, making it a valuable tool for industries such as fashion, textiles, and interior design.
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

Normalize pixel values
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

Define your deep learning model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

Train your model
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

Evaluate your model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')
