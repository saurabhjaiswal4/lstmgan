# -*- coding: utf-8 -*-
"""
Created on Mon May 27 03:25:06 2024

@author: Asus
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential

# Sample English-French sentence pairs
input_texts = ['I am an engineer.', 'He likes to code.', 'She is listening to music.']
target_texts = ['Je suis ingénieur.', 'Il aime coder.', 'Elle écoute de la musique.']

# Add start and end tokens to target sequences
target_texts = ['<start> ' + text + ' <end>' for text in target_texts]

# Define tokenizer for target language
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)
target_tokenizer.fit_on_texts(target_texts)

# Calculate timesteps and features
timesteps = len(max(input_texts, key=len))  # Length of the longest sequence
features = 1  # Assuming each element in the sequence is a single feature

# Reshape input_texts to add a timestep dimension
input_texts = np.array([[[char] for char in seq] for seq in input_texts])

# Define vocabulary sizes
input_vocab_size = 1000
embedding_dim = 100
output_vocab_size = len(target_tokenizer.word_index) + 1  # Add 1 for padding token

# Define generator (Seq2Seq) model
print("Shape of input_texts:", input_texts.shape)
# Reshape input_texts to remove the extra dimension
input_texts = np.array(input_texts) # Adjust axis as needed
generator = Sequential([
    Embedding(input_vocab_size, embedding_dim), 
    LSTM(256, return_sequences=True),  # Adjust hidden_units as needed
    Dense(output_vocab_size, activation='softmax')
])

discriminator = Sequential([
    Embedding(input_vocab_size, embedding_dim), 
    LSTM(256),  # Adjust hidden_units as needed
    Dense(1, activation='sigmoid')
])

#generator.build(input_shape=(None, timesteps, features))
timesteps = 10
features = 100 

# Define the input layer
input_layer = Input(shape=(timesteps, features))

# Pass the input layer through the layers of the generator model
embedding_layer = Embedding(input_vocab_size, embedding_dim)(input_layer)

#(embedding_layer)
from tensorflow.keras.layers import Reshape, Flatten

# Flatten the input tensor
flattened_embedding = Flatten()(embedding_layer)

# Reshape the flattened tensor to the target shape
reshaped_embedding = Reshape((100, 1000))(flattened_embedding)

# Now use the reshaped embedding as input to the LSTM layer
lstm_layer = LSTM(256, return_sequences=True)(reshaped_embedding)
output_layer = Dense(output_vocab_size, activation='softmax')(lstm_layer)

# Create the generator model
generator = Model(inputs=input_layer, outputs=output_layer)

# Compile the generator model
generator.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

#discriminator.build(input_shape=(None, timesteps, features))

from tensorflow.keras.models import Sequential

# Create a sequential model for the discriminator
discriminator = Sequential()

# Add layers to the discriminator
discriminator.add(LSTM(256, input_shape=(timesteps, features)))
discriminator.add(Dense(512, activation='relu'))
discriminator.add(Dense(1, activation='sigmoid'))

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam')


# Print model summaries
print("Generator Model Summary:")
generator.summary()
print("\nDiscriminator Model Summary:")
discriminator.summary()


