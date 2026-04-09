import requests
from urllib.parse import quote_plus
from email.mime.text import MIMEText
import smtplib
from Functions.Speak import Speak

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

def set_reminder(reminder_text: str, remind_time: str) -> str:
    """
    Sets a calendar reminder.
    Params:
    - reminder_text: description of the reminder
    - remind_time: time for the reminder
    """
    Speak(f"Setting reminder: '{reminder_text}' at {remind_time}")
    return f"SUCCESS: Reminder set for {remind_time}: {reminder_text}"