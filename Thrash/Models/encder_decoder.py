from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

enc_ip = Input(shape=(13, ))
dec_ip = Input(shape=(13, ))

embed = Embedding(len(vocab), output_dim=50, input_length=13, trainable=True)

enc_embed = embed(enc_ip)
enc_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h,c]

dec_embed = embed(dec_ip)
dec_lstm = LSTM(400, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)

dense = Dense(len(vocab), activation = 'softmax')

dense_op = dense(dec_op)

model = Model([enc_ip, dec_ip], dense_op)

model.compile(loss='categorical_crossentropy', metric=['acc'], optimizer='adam')

model.fit([encoder_ip, decoder_ip], decoder_final, epochs = 40)