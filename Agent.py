from crewai import Agent, Task, Crew
from crewai.tasks.conditional_task import ConditionalTask
from crewai.llm import LLM
import smtplib
from email.mime.text import MIMEText
from crewai_tools import SerperDevTool
from crewai.tools import tool
import smtplib
from email.mime.text import MIMEText
from Functions.Speak import Speak

conversation_state = {
    "last_user_input": None,
    "current_intent": None,
    "stop": False,
    "last_research_result": None,
}

ollama_llm = LLM(
    model="ollama/Barbara:latest",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


@tool("Send Email Tool")
def send_email(to_address: str, subject: str, body: str, login: str, password: str) -> str:
    """
    Sends an email using SMTP protocol.
    Params:
    - to_address: recipient email
    - subject: subject of email
    - body: email content
    """
    from_address = "ssnagvenkar@gmail.com"
    smtp_server = "smtp.gmail.com"
    port = 587

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
    tools=[SerperDevTool()]
)

email_agent = Agent(
    role="Email Sender",
    goal="Send emails with given content to recipients.",
    backstory="Drafts and sends emails professionally.",
    llm=ollama_llm,
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
    prompt = f"""Given the user message: '{user_input}', classify the intent as one of: research,
    email, reminder, stop and if its none of this than it is chat."""
    response = chat_agent.llm.call([{"role": "user", "content": prompt}])
    intent = response.strip().lower()
    return intent

# Conditional functions for each task
def condition_research(_):
    return conversation_state.get("current_intent") == "research"

def condition_email(_):
    return conversation_state.get("current_intent") == "email"

def condition_reminder(_):
    return conversation_state.get("current_intent") == "reminder"

def condition_chat(_):
    intent = conversation_state.get("current_intent")
    return intent not in {"research", "email", "reminder", "stop"}

# Wrap tasks with ConditionalTask
research_task = ConditionalTask(
    description="Search and summarize the latest in AI.",
    expected_output="A concise summary of the latest AI developments.",
    agent=research_agent,
    condition=condition_research,
)

email_task = ConditionalTask(
    description="Send summary via email.",
    expected_output="A confirmation message stating that the summary email was sent successfully.",
    agent=email_agent,
    condition=condition_email,
)

reminder_task = ConditionalTask(
    description="Set reminders.",
    expected_output="A confirmation message that the reminder was set with the given text and time.",
    agent=reminder_agent,
    condition=condition_reminder,
)

chat_task = Task(
    description="Reply to user messages.",
    expected_output="A helpful and contextually relevant conversational response.",
    agent=chat_agent,
    condition=condition_chat,
)

# Assemble crew
crew = Crew(
    agents=[chat_agent, research_agent, email_agent, reminder_agent],
    tasks=[chat_task, research_task, email_task, reminder_task]
)

# Interactive loop for continuous conversation
def run_conversation():
    while not conversation_state.get("stop"):
        user_input = input("Input:")
        conversation_state["last_user_input"] = user_input
        conversation_state["current_intent"] = classify_intent(user_input)

        if conversation_state["current_intent"] == "stop":
            Speak("Agent: Conversation ended. Goodbye!")
            conversation_state["stop"] = True
            break

        outputs = crew.kickoff()

        chat_output = outputs.get(chat_task.description)
        if chat_output:
            print(f"Agent: {chat_output}")

if __name__ == "__main__":
    Speak("Hello! How can I assist you today?")
    run_conversation()