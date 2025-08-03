from crewai import Agent, Task, Crew
from crewai.llm import LLM
import smtplib
from email.mime.text import MIMEText
from crewai.tools import tool
import smtplib
from email.mime.text import MIMEText
from Functions.Speak import Speak
import requests
import json
from urllib.parse import quote_plus
import re

conversation_state = {
    "last_user_input": None,
    "current_intent": None,
    "stop": False,
    "last_research_result": None,
}

ollama_llm = LLM(
    model="ollama/Barbara:latest",
    base_url="http://localhost:11434",
    api_key="ollama",
)


@tool("Web Search Tool")
def web_search(query: str) -> str:
    """
    Search the web for information using DuckDuckGo Instant Answer API.
    Params:
    - query: The search query to look up
    """
    try:
        # Use DuckDuckGo Instant Answer API (free, no API key required)
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract relevant information
        result = {
            "query": query,
            "abstract": data.get("Abstract", "No abstract available"),
            "abstract_source": data.get("AbstractSource", ""),
            "abstract_url": data.get("AbstractURL", ""),
            "related_topics": []
        }
        
        # Add related topics if available
        if "RelatedTopics" in data and data["RelatedTopics"]:
            for topic in data["RelatedTopics"][:3]:  # Limit to 3 topics
                if isinstance(topic, dict) and "Text" in topic:
                    result["related_topics"].append(topic["Text"])
        
        # Format the response
        formatted_result = f"Search Results for '{query}':\n\n"
        formatted_result += f"Summary: {result['abstract']}\n"
        if result['abstract_source']:
            formatted_result += f"Source: {result['abstract_source']}\n"
        if result['abstract_url']:
            formatted_result += f"URL: {result['abstract_url']}\n"
        
        if result['related_topics']:
            formatted_result += "\nRelated Topics:\n"
            for i, topic in enumerate(result['related_topics'], 1):
                formatted_result += f"{i}. {topic}\n"
        
        return formatted_result
        
    except Exception as e:
        return f"ERROR: Failed to search for '{query}'. Error: {str(e)}"


def strip_think_tags(text: str) -> str:
    """
    Remove all content between <think> and </think> tags from the text.
    Params:
    - text: The text to clean
    Returns:
    - Cleaned text without think tags and their content
    """
    # Remove content between <think> and </think> tags
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Also remove any remaining <think> or </think> tags that might be unmatched
    cleaned_text = re.sub(r'</?think>', '', cleaned_text)
    
    # Clean up extra whitespace and newlines
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # Remove multiple empty lines
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing whitespace
    
    return cleaned_text


@tool("Send Email Tool")
def send_email(to_address: str, from_address: str, subject: str, body: str, login: str, password: str) -> str:
    """
    Sends an email using SMTP protocol.
    Params:
    - to_address: recipient email
    - subject: subject of email
    - body: email content
    - from_address: sender email
    - 
    """
    smtp_server = "smtp.gmail.com"
    port = 587
    print('Hi')
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_address
    msg["To"] = to_address

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(login, password)
            server.sendmail(from_address, [to_address], msg.as_string())
        return f"SUCCESS: Email sent to {to_address}"
    except Exception as e:
        return f"ERROR: {str(e)}"

@tool("Set Reminder Tool")
def set_reminder(reminder_text: str, remind_time: str) -> str:
    """
    Sets a calendar reminder.
    Params:
    - reminder_text: description of the reminder
    - remind_time: time for the reminder
    """
    print(f"Setting reminder: '{reminder_text}' at {remind_time}")
    return f"SUCCESS: Reminder set for {remind_time}: {reminder_text}"


# Define agents
chat_agent = Agent(
    role='Communicate',
    goal="Execute regular and daily communication with the user.",
    backstory="Expert in conversational AI.",
    llm=ollama_llm
)

research_agent = Agent(
    role="Researcher",
    goal="Search the internet for information.",
    backstory="Expert research assistant.",
    llm=ollama_llm,
    tools=[web_search]
)

email_agent = Agent(
    role="Email Sender",
    goal="Extract email details from user input and actually send emails using the provided credentials and the send_email tool.",
    backstory="Expert at extracting email information and using SMTP tools to send actual emails, not just composing them.",
    # llm=ollama_llm,
    tools=[send_email]
)

reminder_agent = Agent(
    role="Reminder Setter",
    goal="Create calendar reminders based on input parameters.",
    backstory="Sets reminders by interfacing directly with the calendar API.",
    tools=[set_reminder],
    llm=ollama_llm
)

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
    intent = response.strip().lower()
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
        return chat_agent.execute_task(
            Task(
                description=f"Provide a conversational response to: {user_input}",
                expected_output="A helpful and contextually relevant response",
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
    - If the intent is 'stop': End the conversation""",
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
        user_input = input("Input:")
        conversation_state["last_user_input"] = user_input
        conversation_state["current_intent"] = classify_intent(user_input)

        if conversation_state["current_intent"] == "stop":
            Speak("Agent: Conversation ended. Goodbye!")
            conversation_state["stop"] = True
            break

        response = route_to_agent(conversation_state["current_intent"], user_input)
        cleaned_response = strip_think_tags(response)
        
        Speak(cleaned_response)

if __name__ == "__main__":
    Speak("Hey Shaunak, what can I help you with?")
    run_conversation()