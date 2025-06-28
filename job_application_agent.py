import asyncio
import os
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from browser_use.browser import BrowserSession, BrowserProfile

async def main():
    # Ensure the resume file exists for demonstration purposes
    # In a real scenario, 'user/resume.pdf' would be an actual path
    resume_path = Path("user/resume.pdf")
    if not resume_path.parent.exists():
        resume_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume_path.exists():
        # Create a dummy resume file if it doesn't exist
        resume_path.write_text("This is a dummy resume content.")
        print(f"Created dummy resume file at: {resume_path.resolve()}")

    # Configure the browser profile to mimic web UI behavior
    # headless=False for GUI, user_data_dir for memory/persistent profile
    browser_profile = BrowserProfile(
        headless=False,  # Run browser with GUI
        user_data_dir="./browser_profile_data",  # Use a persistent user data directory for memory
    # No executable_path or cdp_url means browser-use will launch its own browser
    )

    # Initialize the browser session with the configured profile
    # keep_alive=True ensures the browser stays open after the agent finishes,
    # allowing for manual inspection or subsequent tasks.
    browser_session = BrowserSession(
        browser_profile=browser_profile,
        keep_alive=True, # Keep browser open between tasks (or after this single task)
    )

    # Initialize the LLM (Gemini-2.5-flash)
    # Ensure GOOGLE_API_KEY is set in your environment variables or .env file
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    # Define the task, including the file upload
    # The agent will interpret "uploading resume from path" and interact with file input elements
    task = f"Go to https://www.example.com/job-application and complete the job application including uploading resume from path '{resume_path.resolve()}'"

    # Initialize the agent
    agent = Agent(
        task=task,
        llm=llm,
        browser_session=browser_session, # Pass the browser_session here
        # use_vision=True, # Enable vision if the LLM supports it and it's beneficial for the task
    )

    print("Starting agent task...")
    try:
        result = await agent.run()
        print("\nAgent task completed.")
        print("Final result:", result)
    except Exception as e:
        print(f"An error occurred during agent execution: {e}")
    finally:
        # Explicitly close the browser session if keep_alive is True and you're done
        # If keep_alive is False, the session closes automatically after agent.run()
        await browser_session.close()
        print("Browser session closed.")

if __name__ == "__main__":
    asyncio.run(main())
