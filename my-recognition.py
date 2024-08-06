




#Template checking for labels to match camera view
#!/usr/bin/python3
import jetson_inference
import jetson_utils

import argparse
import sys
from jetson_inference import imageNet
from jetson_utils import videoSource, videoOutput, cudaFont, Log, cudaDrawRect
import time
# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=imageNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="Models/resnet-18.onnx", help="model to use, can be:  googlenet, resnet-18, etc. (see --help for others)")
parser.add_argument("--ssl-key", type=str, help="path to SSL key file")
parser.add_argument("--ssl-cert", type=str, help="path to SSL certificate file")
parser.add_argument("--labels", type=str, default="model/labels.txt", help="path to the labels file")
parser.add_argument("--input_blob", type=str, default="input_0", help="name of the input blob")
parser.add_argument("--output_blob", type=str, default="output_0", help="name of the output blob")
opt = parser.parse_args()

# Load the network
net = imageNet(opt.network, sys.argv)

input = videoSource(opt.input, argv=sys.argv)
output = videoOutput(opt.output, argv=sys.argv)
font = cudaFont()
overlay_color = (0, 0, 255, 128)
count = 0
# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  



    classID, confidence = net.Classify(img)

    # draw predicted class labels
    classLabel = net.GetClassLabel(classID)
    confidence *= 100.0
    if count%30 == 0:
        if "Logan" in classLabel:
            overlay_color = (0, 255, 0, 64)
        elif "Others" in classLabel:
            overlay_color = (255, 0, 0, 64)
        else:
            overlay_color = (0, 0, 0, 0)
    count += 1
    cudaDrawRect(img, rect=(0, 0, img.width, img.height), color=overlay_color)
    print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")
    

    font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", 
                        x=5, y=5 + 1 * (font.GetSize() + 5),
                        color=font.White, background=font.Gray40)
    
    output.Render(img)

    # update the title bar


    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break


