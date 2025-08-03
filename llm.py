import ollama

messages = []

def llmModel(sentence):
    messages.append({"role": "user", "content": sentence})
    response = ollama.chat(
        model="deepseek-r1:1.5b",
        messages=messages
    )
    print(f"Model: {response['message']['content']}")
    messages.append({"role": "assistant", "content": response['message']['content']})
    return(response['message']['content'])
