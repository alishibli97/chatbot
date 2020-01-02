import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers , activations , models , preprocessing, utils
import os
import yaml
import pandas as pd

df = pd.read_csv("data.csv", encoding = "ISO-8859-1")
questions = list(df['user1'])
answers_before = list(df['user2'])

answers = list()
for i in range(len(answers_before)):
    answers.append('<START> ' + answers_before[i] + ' <END>')

tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)
VOCAB_SIZE = len(tokenizer.word_index)+1
print('VOCAB SIZE : {}'.format(VOCAB_SIZE))


# encoder_input_data
tokenized_questions = tokenizer.texts_to_sequences( questions )
maxlen_questions = max( [ len(x) for x in tokenized_questions ] )
padded_questions = preprocessing.sequence.pad_sequences( tokenized_questions , maxlen=maxlen_questions , padding='post' )
encoder_input_data = np.array( padded_questions )
print( encoder_input_data.shape , maxlen_questions )

# decoder_input_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
maxlen_answers = max( [ len(x) for x in tokenized_answers ] )
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
decoder_input_data = np.array( padded_answers )
print( decoder_input_data.shape , maxlen_answers )

# decoder_output_data
tokenized_answers = tokenizer.texts_to_sequences( answers )
for i in range(len(tokenized_answers)) :
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences( tokenized_answers , maxlen=maxlen_answers , padding='post' )
onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
decoder_output_data = np.array( onehot_answers )
print( decoder_output_data.shape )

# Saving all the arrays to storage
np.save( 'enc_in_data.npy' , encoder_input_data)
np.save( 'dec_in_data.npy' , decoder_input_data)
np.save( 'dec_tar_data.npy' , decoder_output_data)


encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True ) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax )
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

print(model.summary())


# Training
model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=75 )
model.save( 'model.h5' )


def make_inference_models():
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] )
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')


enc_model, dec_model = make_inference_models()

correct = 0
unanswered = 0

for i, qs in enumerate(questions):
    try:
        states_values = enc_model.predict(str_to_tokens(qs.replace('?', '')))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        if decoded_translation.replace(' end', '') == answers_before[i]:
            correct += 1
            print(correct)

    except:
        decoded_translation = 'Sorry not able to answer'
        unanswered += 1
        print(decoded_translation)
        continue

    print(decoded_translation.replace(' end', ''))

print(unanswered)
print(correct)
print(len(questions))
print('Accuracy: ' + str(correct / len(questions)))
