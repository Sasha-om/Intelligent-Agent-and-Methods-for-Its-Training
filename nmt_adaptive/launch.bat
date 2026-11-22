@echo off
rem Спільний запуск nmt_adaptive. Режими: student (тест на цьому ПК), teacher (сторінка вчителя),
rem class (тест для всього класу з цього ПК + сторінка вчителя). Порти — як CLASS_PORT і TEACHER_PORT у config.py.
rem Перший запуск: створює .venv і встановлює бібліотеки з requirements.txt (потрібен інтернет).
chcp 65001 >nul
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if exist ".venv\installed.ok" goto ready
echo Перший запуск: готую Python і бібліотеки. Це займе 1-3 хвилини...
if not exist "%PY%" py -3 -m venv .venv 2>nul
if not exist "%PY%" python -m venv .venv 2>nul
if not exist "%PY%" goto nopython
"%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto nopip
echo ok> ".venv\installed.ok"
:ready
echo.
if "%~1"=="teacher" goto teacher
if "%~1"=="class" goto class
echo Адаптивний тест НМТ відкривається в браузері...
echo Якщо браузер не відкрився сам, відкрий http://localhost:8501
echo Щоб зупинити — просто закрий це вікно.
echo.
"%PY%" -m streamlit run app\app_student.py --server.port 8501
goto end
:teacher
echo Сторінка вчителя відкривається в браузері: http://localhost:8502
echo Щоб зупинити — просто закрий це вікно.
echo.
"%PY%" -m streamlit run app\app_teacher.py --server.port 8502
goto end
:class
echo ================ РЕЖИМ КЛАСУ ================
"%PY%" -m app.netinfo
echo.
echo Учні відкривають цю адресу в браузері на своїх комп'ютерах (усі мають бути в одній мережі школи).
echo Результати одразу потрапляють на сторінку вчителя на цьому комп'ютері.
echo Якщо Windows спитає про брандмауер — натисни «Дозволити доступ» (приватні мережі).
echo Не закривай це вікно до кінця уроку.
echo.
start "Сторінка вчителя" "%PY%" -m streamlit run app\app_teacher.py --server.port 8502
"%PY%" -m streamlit run app\app_student.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
goto end
:nopython
echo Не знайдено Python 3.11 або новіший. Встанови його з https://www.python.org
echo (під час встановлення познач «Add python.exe to PATH») і запусти ще раз.
:nopip
if exist "%PY%" echo Не вдалося встановити бібліотеки. Перевір інтернет і запусти ще раз.
pause
exit /b 1
:end
pause
