from crewai import Agent


def create_agents(*, llm, tools):
    """
    Factory for all CrewAI agents used by the app.

    Args:
        llm: crewai.llm.LLM instance
        tools: dict with keys: web_search, send_email, set_reminder, (optional) retrieve_memory
    """

    chat_agent_kwargs = {
        "role": "Communicate",
        "goal": "Execute regular and daily communication with the user.",
        "backstory": "Expert in conversational AI.",
        "llm": llm,
    }
    if "retrieve_memory" in tools:
        chat_agent_kwargs["tools"] = [tools["retrieve_memory"]]

    chat_agent = Agent(**chat_agent_kwargs)

    research_agent = Agent(
        role="Researcher",
        goal="Search the internet for information.",
        backstory="Expert research assistant.",
        llm=llm,
        tools=[tools["web_search"]],
    )

    email_agent = Agent(
        role="Email Sender",
        goal="Extract email details from user input and actually send emails using the provided credentials and the send_email tool.",
        backstory="Expert at extracting email information and using SMTP tools to send actual emails, not just composing them.",
        llm=llm,
        tools=[tools["send_email"]],
    )

    reminder_agent = Agent(
        role="Reminder Setter",
        goal="Create calendar reminders based on input parameters.",
        backstory="Sets reminders by interfacing directly with the calendar API.",
        llm=llm,
        tools=[tools["set_reminder"]],
    )

    return {
        "chat_agent": chat_agent,
        "research_agent": research_agent,
        "email_agent": email_agent,
        "reminder_agent": reminder_agent,
    }

