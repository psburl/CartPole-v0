sh ./installDependencies.sh
python -W ignore src/cartpole.py | grep -v 'not supported' | grep -v 'WARN' | grep -v 'tensorflow' 