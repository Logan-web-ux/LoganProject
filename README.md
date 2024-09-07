# SecurityAI

 This program is meant to act as the base for security programs for houses and other rooms. Its purpose is to scan in front of the door and identify if it is the owner/user or an unindentified person at the door.


## The Algorithm

This program uses a retrained version of resnet18 to recognize the user's face. The video feed gives the program a picture, and that picture is put through the model to be classified. It is then shown on the output screen with colored feed. Green means it recognizes the user, red meaning its an unknown person at the door, and no color meaning its an empty room. The aforementioned python program is a loop so it will continuously process new pictures.

## Running this project

1. git clone https://github.com/Logan-web-ux/LoganProject.git
2. git clone --recursive https://github.com/dusty-nv/jetson-inference
3. Make sure to have python packages installed using "sudo apt-get install libpython3-dev python3-numpy"
4. Make a build directory and run "cmake ../" in build directory
5. Run "python3 my-recognition.py --network=Models/resnet18.onnx --labels=Models/labels.txt --input_blob=input_0 --output_blob=output_0 --ssl-key=key.pem --ssl-cert=cert.pem /dev/video0 webrtc://@:8554/my_output"
6. Go to this link https://<LOCAL_IP>:8554/

Link to Demo:
https://drive.google.com/file/d/1mY38JOMm28Wnonwnv0JH9quxDhW-XH8C/view?usp=sharing
