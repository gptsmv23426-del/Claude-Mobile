@echo off
cd /d "C:\Users\gvsmv\CLAUDE\Github\Trading-Bot"
call venv\Scripts\activate.bat
python main.py >> logs\startup.log 2>&1
