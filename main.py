from crewai import Task, Crew
from crewai.llm import LLM
from crewai.tools import tool
from Functions.Speak import Speak
from Functions.Listen import Listen
import re
from Agents.crewai_agents import create_agents 
from Agents.rag_chroma import get_memory
from Agents.Tools import (
    web_search as _web_search,
    send_email as _send_email,
    set_reminder as _set_reminder,
)

conversation_state = {
    "last_user_input": None,
    "current_intent": None,
    "stop": False,
    "last_research_result": None,
    "prev_response": None
}

ollama_llm = LLM(
    model="ollama/Barbara:latest",
    base_url="http://localhost:11434",
    api_key="ollama", #present cause forced by crewAI
)


def strip_think_tags(text: str) -> str:
    """
    Remove all content between <think> and </think> tags from the text.
    Params:
    - text: The text to clean
    Returns:
    - Cleaned text without think tags and their content
    """

    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    cleaned_text = re.sub(r'</?think>', '', cleaned_text)
    
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


@tool("Web Search Tool")
def web_search(query: str) -> str:
    """Search the web and return a short formatted summary."""
    return _web_search(query)


@tool("Send Email Tool")
def send_email(to_address: str, from_address: str, subject: str, body: str, login: str, password: str) -> str:
    """Send an email via SMTP using the provided credentials."""
    return _send_email(
        to_address=to_address,
        from_address=from_address,
        subject=subject,
        body=body,
        login=login,
        password=password,
    )

@tool("Set Reminder Tool")
def set_reminder(reminder_text: str, remind_time: str) -> str:
    """Set a reminder with the given text and time."""
    return _set_reminder(reminder_text=reminder_text, remind_time=remind_time)


@tool("Retrieve Memory Tool")
def retrieve_memory(query: str) -> str:
    """Retrieve relevant snippets from prior conversation history."""
    return get_memory().format_retrieval(query=query, k=5)


_agents = create_agents(
    llm=ollama_llm,
    tools={
        "web_search": web_search,
        "send_email": send_email,
        "set_reminder": set_reminder,
        "retrieve_memory": retrieve_memory,
    },
)
chat_agent = _agents["chat_agent"]
research_agent = _agents["research_agent"]
email_agent = _agents["email_agent"]
reminder_agent = _agents["reminder_agent"]

# Intent classification function using chat_agent's LLM
def classify_intent(user_input):
    prompt = f"""Given the user message: '{user_input}', classify the intent as one of: 
    1. Research - if there is need of fetching information from the internet,
    2. Email - if user asks to send a mail/email to a recipient (look for words like 'send mail', 'send email', 'mail to', 'email to'),
    3. Reminder - if the user wants to set a reminder,
    4. Stop - if you feel through the users response that user is done with all the work 
    5. Chat - if its none of this.
    
    Respond with only the intent name (research, email, reminder, stop, or chat)."""
    response = chat_agent.llm.call([{"role": "user", "content": prompt}])
    raw = (response or "").strip().lower()

    # Be robust to outputs like "2. email" / "email." / "Intent: email"
    for key in ("research", "email", "reminder", "stop", "chat"):
        if key in raw:
            print(key)
            return key

    # Fallback: strip punctuation and take last token
    cleaned = re.sub(r"[^a-z]+", " ", raw).strip()
    intent = (cleaned.split()[-1] if cleaned else "chat")
    print(intent)
    return intent

# Helper function to route to appropriate agent based on intent
def route_to_agent(intent, user_input):
    if intent == "research":
        return research_agent.execute_task(
            Task(
                description=f"Research the following topic: {user_input}",
                expected_output="A comprehensive summary of the research findings",
                agent=research_agent
            )
        )
    elif intent == "email":
        return email_agent.execute_task(
            Task(
                description=f"""Extract email details from the user input and send the email using the send_email tool.
                User input: {user_input}
                
                Instructions:
                1. Extract the recipient email address.
                2. Extract the sender email address and password only as it is given by the user.
                3. Create an appropriate subject and body based on the email purpose in English.
                4. Use the send_email tool with the extracted credentials to actually send the email.
                5. Return a confirmation message that the email was sent successfully.
                
                Make sure to use the actual send_email tool, not just compose the email content.""",
                expected_output="Email sent successfully using the provided credentials",
                agent=email_agent
            )
        )
    elif intent == "reminder":
        return reminder_agent.execute_task(
            Task(
                description=f"Set a reminder based on: {user_input}",
                expected_output="Reminder setting confirmation",
                agent=reminder_agent
            )
        )
    else:  # chat intent
        memory = get_memory().format_retrieval(query=user_input, k=5)
        return chat_agent.execute_task(
            Task(
                description=f"""You are chatting casually with the user.

Use the retrieved memory below to stay consistent with prior conversation (preferences, facts, names). If memory is irrelevant, ignore it.

{memory}

User message: {user_input}""",
                expected_output="A helpful and contextually relevant response. If the user wants to casually communicate come up with a reply",
                agent=chat_agent
            )
        )

# Create a single task that handles all intents
main_task = Task(
    description="""Analyze the user's input and respond appropriately based on their intent:
    - If the intent is 'research': Search for information and provide a summary
    - If the intent is 'email': Help compose and send an email
    - If the intent is 'reminder': Set a calendar reminder
    - If the intent is 'chat': Provide a conversational response
    - If the intent is 'stop': End the conversation
    Also check if the message has something to do with the previous task and manipulate
    the prev_response according to requirement only if it related to previous response 
    else produce new output.
    """,
    
    expected_output="An appropriate response based on the user's intent",
    agent=chat_agent,
)

# Assemble crew
crew = Crew(
    agents=[chat_agent, research_agent, email_agent, reminder_agent],
    tasks=[main_task]
)

def run_conversation():
    while not conversation_state.get("stop"):
        user_input = Listen("Listening...")
        conversation_state["last_user_input"] = user_input
        conversation_state["current_intent"] = classify_intent(user_input)

        if conversation_state["current_intent"] == "stop":
            Speak("Agent: Conversation ended. Goodbye!")
            conversation_state["stop"] = True
            break

        response = route_to_agent(conversation_state["current_intent"], user_input)
        cleaned_response = strip_think_tags(response)
        conversation_state["prev_response"] = cleaned_response
        # Only store normal chat turns to keep memory relevant.
        if conversation_state["current_intent"] == "chat":
            get_memory().add_turn(user_text=user_input, assistant_text=cleaned_response)
        Speak(cleaned_response)

if __name__ == "__main__":
    Speak("Hey Shaunak, what can I help you with?")
    run_conversation()