#!/bin/bash

#Install dependancies
sudo apt update && sudo apt-get install -y lsb-release
echo "LSBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
#sudo apt-get install -y build-essential

#GSF npm packages
npm install -g @grnsft/if@0.6.0
npm install -g @grnsft/if-plugins
npm install -g @grnsft/if-unofficial-plugins
echo "GSF doneeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"

#Clears cached packages
#sudo rm -rf /var/lib/apt/lists/*

#Windows to Unix
sed -i 's/\r$//' app/start_server.sh

#Ollama install
curl -fsSL https://ollama.com/install.sh | sh

#Requirements
pip install --no-cache-dir -r requirements.txt
export OLLAMA_MODELS=~/.ollama/models

#Ensure ollama is executable
chmod +x app/ollama
chmod +x app/start_server.sh

ollama pull llama3
echo "llama3 model pulled"