from __future__ import annotations

import os
import sys
from typing import List, Optional
from pathlib import Path
import webbrowser
import argparse

from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import (
    SQLDatabaseToolkit,
    create_sql_agent,
)
from langchain.callbacks import StreamingStdOutCallbackHandler
import markdown

# ---------------------------------------------------------------------------
# Helper — build the SQLDatabase instance
# ---------------------------------------------------------------------------



PREVIEW_FILE = Path("preview.html")

# def markdown_to_html(md: str) -> str:
#     """Convert markdown text to a full HTML document string."""
#     body = markdown.markdown(md, extensions=["tables", "fenced_code"])
#     return HTML_TEMPLATE.format(body=body)


def write_preview(full_html: str) -> None:
    """Dump *full_html* to *preview.html* and open in the default browser."""
    PREVIEW_FILE.write_text(full_html, encoding="utf-8")
    webbrowser.open(PREVIEW_FILE.resolve().as_uri())

def get_database() -> SQLDatabase:
    """Create a :class:`SQLDatabase` from the DB_URL environment variable."""
    db_url: Optional[str] = os.getenv("DB_URL")
    if not db_url:
        raise RuntimeError(
            "Environment variable DB_URL is not set.\n"
            "Example for PostgreSQL:\n"
            "  export DB_URL='postgresql+psycopg2://user:password@host:5432/db'"
        )

    try:
        db = SQLDatabase.from_uri(db_url)
    except SQLAlchemyError as ex:
        print("❌  Failed to connect to database:", ex, file=sys.stderr)
        raise

    return db


# ---------------------------------------------------------------------------
# Helper — build the LangChain SQL agent
# ---------------------------------------------------------------------------

def build_agent(streaming: bool = True):
    """Return a fully configured SQL agent ready for conversation."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # switch to "gpt-4o" when GA if you have access
        temperature=0,
        streaming=streaming,
        callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
    )

    # You can swap ChatOpenAI for any BaseChatModel (Groq, Anthropic, etc.)
    db = get_database()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Custom system prompt ensures consistent formatting and safe behaviour.
    prefix = (
        "You are **ChatDB**, an expert data analyst and front‑end developer. "
        "You have access to a SQL database and can write SELECT statements to "
        "answer the user's questions. **NEVER** modify data. **Return the "
        "answer as a COMPLETE, standalone HTML5 document**, beginning with "
        "'<!doctype html>'. Your page must: \n"
        "• Include a <head> with <meta charset='utf-8'> and <title>ChatDB "
        "Answer</title>.\n"
        "• Bring in Tailwind CSS via CDN ('https://cdn.tailwindcss.com').\n"
        "• Respect prefers‑color‑scheme dark.\n"
        "• In <body>, present data beautifully. For tabular results, create a "
        "<table> with Tailwind classes. Add a search input above the table and "
        "client‑side JS to filter rows and allow header‑click sorting.\n"
        "• For single scalar answers, display them inside a centered <div> with "
        "text‑4xl font‑bold.\n"
        "• If no data, show a <p class='text-red-600'>No records found.</p>.\n"
        "• Place all scripts at the end of <body>. Use vanilla JS (no external "
        "libs other than Tailwind).\n"
        "Return **only** the HTML file content, no extra commentary or code "
        "fences."
    )

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",  # enables tool calling (GPT‑4o‑style)
        prefix=prefix,
        verbose=True,
    )
    return agent


# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def interactive_chat(enable_html: bool = True) -> None:
    agent = build_agent()
    print("\n📊  ChatDB v0.4 ready! (type 'exit' to quit, 'nohtml' to disable preview)\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye 👋")
            break

        cmd = query.lower()
        if cmd in {"exit", "quit"}:
            print("Bye 👋")
            break
        if cmd == "nohtml":
            enable_html = False
            print("🔕  Browser preview disabled.")
            continue
        if cmd == "html":
            enable_html = True
            print("🌐  Browser preview enabled.")
            continue
        if not query:
            continue

        try:
            response: str = agent.run(query)
            if enable_html:
                write_preview(response)
                print("(Opened preview.html)\n")
            else:
                print("\nAssistant (HTML):\n", response[:500], "…\n")
        except Exception as ex:
            print("❌  Error:", ex, file=sys.stderr)


# ---------------------------------------------------------------------------
# Kick things off
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Natural‑language SQL agent → HTML page")
    parser.add_argument("--no-html", action="store_true", help="Disable browser preview")
    args = parser.parse_args()
    interactive_chat(enable_html=not args.no_html)

if __name__ == "__main__":
    main()
