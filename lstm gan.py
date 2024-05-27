# -*- coding: utf-8 -*-
"""
Created on Mon May 27 03:05:29 2024

@author: Asus
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

'''# Sample English-French sentence pairs
input_texts = ['I am an engineer.', 'He likes to code.', 'She is listening to music.']
target_texts = ['Je suis ingénieur.', 'Il aime coder.', 'Elle écoute de la musique.']

# Define tokenizers for input and target texts
input_tokenizer = tf.keras.preprocessing.text.Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer = tf.keras.preprocessing.text.Tokenizer()
target_tokenizer.fit_on_texts(target_texts)'''

# Sample English-French sentence pairs
input_texts = ['I am an engineer.', 'He likes to code.', 'She is listening to music.']
target_texts = ['Je suis ingénieur.', 'Il aime coder.', 'Elle écoute de la musique.']

# Add start and end tokens to target sequences
target_texts = ['<start> ' + text + ' <end>' for text in target_texts]

# Define tokenizers for input and target texts
input_tokenizer = tf.keras.preprocessing.text.Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', lower=False)
target_tokenizer.fit_on_texts(target_texts)


# Convert text sequences to integer sequences
encoder_input_data = input_tokenizer.texts_to_sequences(input_texts)
decoder_input_data = target_tokenizer.texts_to_sequences(target_texts)

# Pad sequences to uniform length
max_seq_length = max(len(seq) for seq in encoder_input_data + decoder_input_data)
encoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(encoder_input_data, maxlen=max_seq_length, padding='post')
decoder_input_data = tf.keras.preprocessing.sequence.pad_sequences(decoder_input_data, maxlen=max_seq_length, padding='post')

# Define vocabulary sizes
num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

# Define model architecture
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train model
model.fit([encoder_input_data, decoder_input_data], decoder_input_data, batch_size=1, epochs=50)

# Define inference models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

# Define reverse target word index
reverse_target_word_index = {index: word for word, index in target_tokenizer.word_index.items()}

# Define maximum decoder sequence length
max_decoder_seq_length = max(len(seq) for seq in decoder_input_data)

# Function to decode sequence
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index['<start>']
    
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')
        if sampled_word:
            decoded_sentence += sampled_word + ' '

        if sampled_word == '<end>' or len(decoded_sentence.split()) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()


# Inference
import numpy as np

for input_seq, input_text in zip(encoder_input_data, input_texts):
    input_seq = np.array(input_seq).reshape(1, -1)
    decoded_sentence = decode_sequence(input_seq)
    print('Input sentence:', input_text)
    print('Decoded sentence:', decoded_sentence)


