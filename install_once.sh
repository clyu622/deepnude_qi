#!/bin/bash

# Update and Upgrade the Pi, just in case
sudo apt-get update
sudo apt-get upgrade -y

# Install Python3 and Pip if they are not already installed
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install libgl1 

# Install necessary Python libraries
pip install opencv-python 
pip install ultralytics
pip install lapx
pip install flickr_api
pip install Pillow
pip install webuiapi
pip install mediapipe
pip install typing-extensions --upgrade
echo "Installation complete!"
