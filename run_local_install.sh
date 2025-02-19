#uninstall everything
pip freeze | xargs pip uninstall -y

#reinstall required packages
pip install -r requirements.txt

#fastapi-cli
pip install fastapi-cli
