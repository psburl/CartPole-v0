echo 'checking for dependencies...'
pip install -U -r dependencies/requirements.txt --disable-pip-version-check --exists-action w | grep -v 'Requirement already up-to-date' 