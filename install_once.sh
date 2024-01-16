#!/bin/bash

# Update and Upgrade the Pi, just in case
sudo apt-get update
sudo apt-get upgrade -y

# Install Python3 and Pip if they are not already installed
sudo apt-get install python3
sudo apt-get install python3-pip
sudo apt-get install libgl1 

# Install necessary Python libraries
pip3 install opencv-python ultralytics flickr_api Pillow mediapipe webuiapi

echo "Installation complete!"
