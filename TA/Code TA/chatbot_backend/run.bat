@echo off
REM Convenience launcher for Windows. Use from chatbot_backend/ directory.

if not exist .env (
  echo .env not found — copy .env.example to .env and add your OPENAI_API_KEY first.
  exit /b 1
)

if not exist .venv (
  echo Creating virtualenv...
  python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -r requirements.txt
python main.py
