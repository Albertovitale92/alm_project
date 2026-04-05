@echo off
echo Avvio di Visual Studio Code...
code .

echo Attivazione ambiente virtuale...
call venv\Scripts\activate.bat

echo Avvio del programma ALM...
python main.py

echo.
echo Programma terminato.
pause