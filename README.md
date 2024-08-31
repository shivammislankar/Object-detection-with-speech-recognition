<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
   
</head>
<body>
    <h1>Object Detection with YOLOv4 and Voice Alerts</h1>
    <p>This project is a real-time object detection system using YOLOv4, with integrated voice commands and text-to-speech (TTS) alerts. The system is capable of detecting objects through a webcam, applying night vision mode, generating heatmaps, and issuing voice alerts in multiple languages for detected objects.</p>
    <h2>Features</h2>
    <ul>
        <li><strong>Real-Time Object Detection:</strong> Utilizes YOLOv4 to detect objects in real-time using your webcam.</li>
        <li><strong>Voice Commands:</strong> Control the detection system using voice commands such as starting/stopping detection and toggling night vision mode.</li>
        <li><strong>Text-to-Speech (TTS) Alerts:</strong> Provides verbal notifications of detected objects in multiple languages.</li>
        <li><strong>Snapshot Saving:</strong> Automatically saves snapshots when specific objects are detected.</li>
        <li><strong>Heatmap Generation:</strong> Generates a heatmap overlay showing areas where objects have been detected most frequently.</li>
        <li><strong>Night Vision Mode:</strong> Apply a night vision filter to the video feed.</li>
    </ul>
    <h2>Requirements</h2>
    <p>Before running the project, ensure that the following dependencies are installed:</p>
    <ul>
        <li>Python 3.x</li>
        <li>OpenCV (<code>cv2</code>)</li>
        <li>NumPy</li>
        <li><code>gTTS</code> (Google Text-to-Speech)</li>
        <li>Pygame</li>
        <li><code>speech_recognition</code></li>
    </ul>
    <p>You can install the required Python packages using pip:</p>
    <pre><code>pip install opencv-python numpy gtts pygame SpeechRecognition</code></pre>
    <h2>Installation</h2>
    <ol>
        <li><strong>Clone the repository:</strong>
            <pre><code>git clone https://github.com/yourusername/object-detection-yolov4.git
cd object-detection-yolov4</code></pre>
        </li>
        <li><strong>Download YOLOv4 Weights and Configuration Files:</strong>
            <p>The script will automatically download <code>yolov4.weights</code>, <code>yolov4.cfg</code>, and <code>coco.names</code> if they do not exist in the working directory. You can also download them manually:</p>
            <ul>
                <li><a href="https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" target="_blank">yolov4.weights</a></li>
                <li><a href="https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" target="_blank">yolov4.cfg</a></li>
                <li><a href="https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" target="_blank">coco.names</a></li>
            </ul>
        </li>
        <li><strong>Run the Object Detection Script:</strong>
            <pre><code>python obj_det.py</code></pre>
        </li>
    </ol>
    <h2>Usage</h2>
    <h3>Voice Commands</h3>
    <p>The system listens for the following voice commands:</p>
    <ul>
        <li><strong>"start detection"</strong> - Starts the object detection process.</li>
        <li><strong>"stop detection"</strong> - Stops the object detection process.</li>
        <li><strong>"night vision on"</strong> - Activates night vision mode.</li>
        <li><strong>"night vision off"</strong> - Deactivates night vision mode.</li>
    </ul>
    <h3>Detected Object Alerts</h3>
    <ul>
        <li><strong>TTS Alerts:</strong> The system will announce the names of detected objects using Google TTS, in the specified language based on the object class (e.g., "person" in English, "cat" in French, "dog" in Spanish).</li>
        <li><strong>Snapshot Saving:</strong> When a "car" is detected, a snapshot will be saved in the working directory.</li>
    </ul>
    <h3>Heatmap</h3>
    <p>A heatmap showing the regions of the frame where objects have been detected most frequently is displayed alongside the regular detection feed.</p>
    <h3>Exiting</h3>
    <p>Press the <code>q</code> key to exit the application manually or wait for 2 minutes for the system to exit automatically.</p>
    <h2>Customization</h2>
    <h3>Adjusting Confidence Threshold</h3>
    <p>You can adjust the confidence threshold for object detection by modifying the following line in the script:</p>
    <pre><code>if confidence &gt; 0.5:  # Change 0.5 to your desired threshold</code></pre>
    <h3>Adding More Languages</h3>
    <p>To add more languages for TTS, update the <code>language_map</code> dictionary in the script:</p>
    <pre><code>language_map = {
    "person": "en",
    "cat": "fr",
    "dog": "es",
    "bicycle": "de"  # Example for German
}</code></pre>
    <h3>Changing Alerted Objects</h3>
    <p>To change the object classes that trigger TTS alerts or snapshots, modify the <code>ALERT_OBJECT</code> and <code>SNAPSHOT_OBJECT</code> variables:</p>
    <pre><code>ALERT_OBJECT = "person"
SNAPSHOT_OBJECT = "car"</code></pre>
    <h2>Troubleshooting</h2>
    <ul>
        <li><strong>IndexError or RuntimeWarnings:</strong> Ensure that your webcam is properly connected, and that the script has permissions to access the microphone for voice commands.</li>
        <li><strong>No Detected Objects:</strong> Check that your camera is functioning correctly, and adjust the confidence threshold if necessary.</li>
    </ul>
    <h2>License</h2>
    <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for more details.</p>
</body>
</html>
