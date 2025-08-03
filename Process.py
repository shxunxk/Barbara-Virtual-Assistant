# import spacy
# from Models.Load import load_and_predict
from Functions.Speak import Speak
from transformers import pipeline, Conversation

# Initialize a conversational pipeline
chatbot = pipeline('conversational', model='facebook/blenderbot-400M-distill')

# Create a Conversation object with the user input
conversation = Conversation("How are you?")

# Get the response
response = chatbot([conversation])

# Print the response text
print(type(response))


def Process(sentence):
    print()

    # ????

# import json
# from Train.TrainPatterns import TrainPattern
# import Levenshtein
# from langchain import LLMChain 
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# def Process(sentence):
#     print()




    # lc = LangChain()
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2LMHeadModel.from_pretrained('gpt2')

    # def chatbot_response(user_input):
    #     input_ids = tokenizer.encode(user_input, return_tensors='pt')
    #     outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
    #     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return generated_text
    
    # lc.add_chain('chatbot', chatbot_response)
    # response = lc.run_chain('chatbot', sentence)
    # print(response)
    # Speak(response)



# print(tokens)

    # lemmatized = {}
    # grammatics = {}
    # entity = {}
    # dependency = {}
    # res = nlp(doc)
    # for idx,token in enumerate(res):
    #     tokens.append(token)
    #     lemmatized[token.text] = token.lemma_
    #     tokens[idx] = lemmatized[token.text]
    #     grammatics[token.text] = token.pos_
    #     dependency[token.text] = token.dep_

    # for ent in res.ents:
    #     entity[ent.text] = ent.label_

    # print(lemmatized, grammatics)

    # doc = nlp(sentence)
    # tokens = [token.text for token in doc]
    # tokens = [token.lower() for token in tokens]
    # sentence = ' '.join(tokens)
    

    # intent = load_and_predict(sentence)
    # print(intent, 1)
    # idx = similar_tag(intent)
    # print(intents['intents'][idx]['tag'])
    # pattern = TrainPattern(sentence, intents['intents'][idx]['patterns'])
    # response = TrainPattern(pattern, intents['intents'][idx]['responses'])

    # with open('Data/intents.json') as f:
#     intents = json.load(f)

# nlp = spacy.load("en_core_web_sm")

# def similar_tag(intent, intents_data = intents):
#     max_similarity = 0
#     index = 0
#     for idx,intent_data in enumerate(intents_data["intents"]):
#         similarity = Levenshtein.ratio(intent, intent_data["tag"])
#         if similarity > max_similarity:
#             max_similarity = similarity
#             index = idx
#     return index
