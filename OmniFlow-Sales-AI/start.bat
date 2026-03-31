@echo off
echo.
echo  ╔════════════════════════════════════════════╗
echo  ║     OmniFlow Sales AI — Startup Script     ║
echo  ╚════════════════════════════════════════════╝
echo.

set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%

echo [1/2] Starting FastAPI backend on http://127.0.0.1:8000 ...
start "FastAPI Backend" cmd /k "cd /d %PROJECT_DIR% && venv\Scripts\activate && uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Django frontend on http://127.0.0.1:8080 ...
start "Django Frontend" cmd /k "cd /d %PROJECT_DIR%\django_app && ..\venv\Scripts\activate && python manage.py runserver 8080"

echo.
echo  Both servers starting...
echo  FastAPI  →  http://127.0.0.1:8000
echo  Django   →  http://127.0.0.1:8080
echo  API Docs →  http://127.0.0.1:8000/docs
echo.
pause
