import numpy as np
import tensorflow as tf


training_data = [
    ("What is your name?", "My name is AI."),
    ("How are you?", "I'm fine, thank you."),
    ("Who created you?", "I was created by you."),
    ("What can you do?", "I can have basic conversations."),
    ("Goodbye", "Goodbye!"),
  
]


tokenized_data = []
for pair in training_data:
    tokenized_pair = ([char for char in pair[0]], [char for char in pair[1]])
    tokenized_data.append(tokenized_pair)


vocab = set()
for pair in tokenized_data:
    vocab.update(pair[0])
    vocab.update(pair[1])
vocab = sorted(vocab)
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
1

input_size = len(vocab)
output_size = len(vocab)
hidden_size = 128
learning_rate = 0.001
epochs = 1000


model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, input_shape=(input_size,), activation='relu'),
    tf.keras.layers.Dense(hidden_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X = []
y = []
for pair in tokenized_data:
    X.append([char_to_idx[char] for char in pair[0]])
    y.append([char_to_idx[char] for char in pair[1]])

model.fit(np.array(X), np.array(y), epochs=epochs)

def generate_response(input_text):
    input_seq = [char_to_idx[char] for char in input_text]
    input_seq = tf.expand_dims(input_seq, 0)
    predicted_output = model.predict(input_seq)
    predicted_idx = np.argmax(predicted_output)
    return vocab[predicted_idx]

print("Welcome to OmegaBeta! Type 'exit' to end the conversation.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("OmegaBeta: Goodbye!")
        break
    response = generate_response(user_input)
    print("OmegaBeta:", response)
