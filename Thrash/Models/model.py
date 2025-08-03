import regex as re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
lines = open('C:/Users/Shaunak/Documents/Barbara/Data/movie_lines.txt', encoding='utf8', errors='ignore').read().split('\n')
conversations = open('C:/Users/Shaunak/Documents/Barbara/Data/movie_conversations.txt', encoding='utf8', errors='ignore').read().split('\n')

# Extract conversations
exchn = [conver.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(",", "").split() for conver in conversations]

# Create a dictionary to hold the dialogue lines
dialog = {line.split(' +++$+++ ')[0]: line.split(' +++$+++ ')[-1] for line in lines}

# Separate questions and answers
questions, answers = [], []
for conver in exchn:
    for i in range(len(conver) - 1):
        questions.append(dialog[conver[i]])
        answers.append(dialog[conver[i + 1]])

# Filter questions and answers based on length
sorted_ques = [questions[i] for i in range(len(questions)) if len(questions[i]) < 13]
sorted_ans = [answers[i] for i in range(len(answers)) if len(answers[i]) < 13]
questions, answers = sorted_ques, sorted_ans

# Text cleaning function
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"can't", "can not", txt)
    txt = re.sub(r"[^\w\s]", "", txt)
    return txt

# Clean questions and answers
clean_ques = [clean_text(line) for line in questions]
clean_ans = [clean_text(line) for line in answers]

# Limit the length of answers to 11 words
clean_ans = [' '.join(line.split()[:11]) for line in clean_ans]

# Limit the dataset to 30,000 pairs
clean_ques, clean_ans = clean_ques[:30000], clean_ans[:30000]

# Create a word count dictionary
word2count = {}
for line in clean_ques + clean_ans:
    for word in line.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Create a vocabulary dictionary
vocab = {word: idx for idx, (word, _) in enumerate(word2count.items())}

# Add special tokens to vocabulary
tokens = ['<PAD>', '<SOS>', '<EOS>', '<OUT>']
x = len(vocab)
for token in tokens:
    vocab[token] = x
    x += 1

# Adjust PAD token
vocab['cameron'] = vocab['<PAD>']
vocab['<PAD>'] = 0

# Create inverse vocabulary
inv_vocab = {v: k for k, v in vocab.items()}

# Append SOS and EOS tokens to answers
clean_ans = ['<SOS> ' + line + ' <EOS>' for line in clean_ans]

# Convert text to sequences of integers
def text_to_sequences(texts, vocab, maxlen):
    sequences = []
    for line in texts:
        seq = [vocab.get(word, vocab['<OUT>']) for word in line.split()]
        sequences.append(seq)
    return pad_sequences(sequences, maxlen, padding='post', truncating='post')

encoder_ip = text_to_sequences(clean_ques, vocab, 13)
decoder_ip = text_to_sequences(clean_ans, vocab, 13)

# Prepare decoder final output for training
decoder_final = np.array([seq[1:] + [vocab['<PAD>']] * (13 - len(seq[1:])) for seq in decoder_ip])
decoder_final = to_categorical(decoder_final, len(vocab))

# Define model architecture
VOCAB_SIZE = len(vocab)
embedding_dim = 50
latent_dim = 400
max_len = 13

enc_inp = Input(shape=(max_len,))
dec_inp = Input(shape=(max_len,))

embed = Embedding(VOCAB_SIZE + 1, embedding_dim, trainable=True)
enc_embed = embed(enc_inp)
enc_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
enc_op, h, c = enc_lstm(enc_embed)
enc_states = [h, c]

dec_embed = embed(dec_inp)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_op, _, _ = dec_lstm(dec_embed, initial_state=enc_states)

dense = Dense(VOCAB_SIZE, activation='softmax')
dense_op = dense(dec_op)

model = Model([enc_inp, dec_inp], dense_op)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

print(len(encoder_ip), len(decoder_final))
# Train the model
model.fit([encoder_ip, decoder_ip], decoder_final, epochs=15)

# Save the model
model.save('chatbot_model.h5')

# Define inference models
enc_model = Model(enc_inp, enc_states)

dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))
dec_states_inputs = [dec_state_input_h, dec_state_input_c]

dec_embed2 = embed(dec_inp)
dec_op2, h2, c2 = dec_lstm(dec_embed2, initial_state=dec_states_inputs)
dec_states2 = [h2, c2]
dec_dense2 = dense(dec_op2)

dec_model = Model([dec_inp] + dec_states_inputs, [dec_dense2] + dec_states2)

# Chatting function
def chatbot_response(input_text):
    input_text = clean_text(input_text)
    input_seq = text_to_sequences([input_text], vocab, max_len)
    states = enc_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = vocab['<SOS>']
    
    stop_condition = False
    decoded_translation = ''
    
    while not stop_condition:
        output_tokens, h, c = dec_model.predict([target_seq] + states)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = inv_vocab[sampled_token_index]
        
        if sampled_token != '<EOS>':
            decoded_translation += sampled_token + ' '
        
        if sampled_token == '<EOS>' or len(decoded_translation.split()) > max_len:
            stop_condition = True
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states = [h, c]
    
    return decoded_translation.strip()

# Start chatting
print("##########################################")
print("#       start chatting ver. 1.0          #")
print("##########################################")

while True:
    user_input = input("you : ")
    if user_input.lower() == 'q':
        break
    response = chatbot_response(user_input)
    print("chatbot attention : ", response)
    print("==============================================")
