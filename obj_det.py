import cv2
import numpy as np
from gtts import gTTS
import pygame
import os
import time
import speech_recognition as sr

# Ensure YOLO files are in place
if not os.path.exists("yolov4.weights"):
    print("Downloading yolov4.weights...")
    os.system("curl -L -o yolov4.weights https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights")

if not os.path.exists("yolov4.cfg"):
    print("Downloading yolov4.cfg...")
    os.system("curl -L -o yolov4.cfg https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg")

if not os.path.exists("coco.names"):
    print("Downloading coco.names...")
    os.system("curl -L -o coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")

# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture for the webcam
video = cv2.VideoCapture(0)
labels = []

# Initialize Pygame for playing the TTS audio
pygame.mixer.init()

# Alert object and language map
ALERT_OBJECT = "person"
SNAPSHOT_OBJECT = "car"
language_map = {
    "person": "en",
    "cat": "fr",
    "dog": "es"
}

# Function to speak text using TTS
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("tts.mp3")
    pygame.mixer.music.load("tts.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    os.remove("tts.mp3")

# Function to listen for voice commands
def listen_for_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = r.listen(source)
        try:
            command = r.recognize_google(audio).lower()
            return command
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
            return None

# Function to handle multilingual text-to-speech
def speak_multilingual(labels):
    for label in labels:
        lang = language_map.get(label, 'en')  # Default to English if label not found in language_map
        speak(f"Detected: {label}", lang)

# Night vision mode
night_mode = False
detection_active = True

# Record the start time
start_time = time.time()

# Heatmap initialization
heatmap = None

# Run loop for at least 2 minutes
while True:
    command = listen_for_command()
    if command:
        if "start detection" in command:
            detection_active = True
            speak("Detection started.")
        elif "stop detection" in command:
            detection_active = False
            speak("Detection stopped.")
        elif "night vision on" in command:
            night_mode = True
            speak("Night vision mode activated.")
        elif "night vision off" in command:
            night_mode = False
            speak("Night vision mode deactivated.")

    if not detection_active:
        continue

    ret, frame = video.read()
    if not ret:
        break

    height, width, channels = frame.shape
    if heatmap is None:
        heatmap = np.zeros((height, width), np.float32)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    detected_labels = []
    detections = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detections.append([x, y, x + w, y + h, confidence])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Ensure indexes is in the right format
    if isinstance(indexes, tuple):
        indexes = indexes[0]

    # Flatten and ensure it's a list of indices
    if indexes is not None and len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []  # No detections, set to empty list

    for i in indexes:
        i = int(i)  # Convert to integer
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        detected_labels.append(label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Update heatmap
        heatmap[y:y + h, x:x + w] += 1

    # Check for new labels to speak
    if labels != detected_labels:
        labels = detected_labels
        if labels:
            speak_multilingual(labels)

    # Alert for specific objects
    if ALERT_OBJECT in labels:
        speak(f"Alert: {ALERT_OBJECT} detected!")

    # Save snapshot for specific objects
    if SNAPSHOT_OBJECT in labels:
        snapshot_filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snapshot_filename, frame)
        speak(f"Snapshot saved as {snapshot_filename}")

    # Write detected objects to a log file
    with open("detections_log.txt", "a") as log_file:
        for label in labels:
            log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {label}\n")

    # Apply night vision mode
    if night_mode:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        night_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_HOT)
        cv2.imshow("Night Vision", night_frame)
    else:
        cv2.imshow("Image", frame)

    # Display heatmap
    if heatmap.max() > 0:
        heatmap_display = cv2.applyColorMap(np.uint8(heatmap * 255 / heatmap.max()), cv2.COLORMAP_JET)
    else:
        heatmap_display = np.zeros_like(frame)  # Create a blank image if no heatmap data

    overlay = cv2.addWeighted(frame, 0.7, heatmap_display, 0.3, 0)
    cv2.imshow("Heatmap", overlay)

    # Check if 2 minutes have passed
    if time.time() - start_time >= 120:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
pygame.quit()
