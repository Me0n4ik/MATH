set env_name=.venv

python -m venv %env_name%

"%root%\%env_name%\Scripts\python.exe" -m pip install --upgrade pip

"%root%\%env_name%\Scripts\pip.exe" install -r "requirements.txt"