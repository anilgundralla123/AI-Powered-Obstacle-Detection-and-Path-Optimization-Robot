import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import RPi.GPIO as GPIO
import time

# Dataset file name
dataset_file = "movement_with_reverse_data.csv"

# Check if the dataset file exists
if not os.path.exists(dataset_file):
    print(f"Error: Dataset file '{dataset_file}' not found.")
    print("Please provide the dataset file and try again.")
    exit(1)  # Exit the program if the dataset is missing

# Load the dataset
print(f"Loading dataset from {dataset_file}...")
dataset = pd.read_csv(dataset_file)
print("Dataset loaded successfully.")

# Prepare data for training
X = dataset[["Distance (cm)"]]  # Features
y = dataset["Action (0=stop, 1=forward, 2=left, 3=right, 4=reverse)"]  # Target labels

# Encode target labels
label_map = {"Stop": 0, "Turn Left": 2, "Turn Right": 3, "Go Forward": 1,"Go reverse":4}
y_encoded = y.map(label_map)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train Logistic Regression model
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=200, random_state=42)
lr_model.fit(X_train, y_train)

# Validate Logistic Regression model
y_pred = lr_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_map.keys()))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# GPIO setup for ultrasonic sensor and motor control
TRIG = 5
ECHO = 13
MOTOR_PINS = {"IN1": 17, "IN2": 18, "IN3": 22, "IN4": 23, "ENA": 24, "ENB": 25}

# GPIO setup
GPIO.setwarnings(False)  # Disable GPIO warnings
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
for pin in MOTOR_PINS.values():
    GPIO.setup(pin, GPIO.OUT)

# Initialize PWM on ENA and ENB with 40% duty cycle
pwmA = GPIO.PWM(MOTOR_PINS["ENA"], 1000)  # Frequency of 1000 Hz
pwmB = GPIO.PWM(MOTOR_PINS["ENB"], 1000)
pwmA.start(30)
pwmB.start(30)

# Ultrasonic sensor function
def measure_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.2)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 34300 / 2  # Speed of sound = 34300 cm/s
    return round(distance, 2)

# Motor control function
def control_robot(action):
    if action == "Stop":
        GPIO.output(MOTOR_PINS["IN1"], GPIO.LOW)
        GPIO.output(MOTOR_PINS["IN2"], GPIO.LOW)
        GPIO.output(MOTOR_PINS["IN3"], GPIO.LOW)
        GPIO.output(MOTOR_PINS["IN4"], GPIO.LOW)
        pwmA.ChangeDutyCycle(0)
        pwmB.ChangeDutyCycle(0)
    else:
        pwmA.ChangeDutyCycle(25)
        pwmB.ChangeDutyCycle(25)
        if action == "Turn Left":
            GPIO.output(MOTOR_PINS["IN1"], GPIO.LOW)
            GPIO.output(MOTOR_PINS["IN2"], GPIO.HIGH)
            GPIO.output(MOTOR_PINS["IN3"], GPIO.HIGH)
            GPIO.output(MOTOR_PINS["IN4"], GPIO.LOW)
        elif action == "Turn Right":
            GPIO.output(MOTOR_PINS["IN1"], GPIO.HIGH)
            GPIO.output(MOTOR_PINS["IN2"], GPIO.LOW)
            GPIO.output(MOTOR_PINS["IN3"], GPIO.LOW)
            GPIO.output(MOTOR_PINS["IN4"], GPIO.HIGH)
        elif action == "Go Forward":
            GPIO.output(MOTOR_PINS["IN1"], GPIO.HIGH)
            GPIO.output(MOTOR_PINS["IN2"], GPIO.LOW)
            GPIO.output(MOTOR_PINS["IN3"], GPIO.HIGH)
            GPIO.output(MOTOR_PINS["IN4"], GPIO.LOW)
        elif action == "Go reverse":
            GPIO.output(MOTOR_PINS["IN1"], GPIO.LOW)
            GPIO.output(MOTOR_PINS["IN2"], GPIO.HIGH)
            GPIO.output(MOTOR_PINS["IN3"], GPIO.LOW)
            GPIO.output(MOTOR_PINS["IN4"], GPIO.HIGH)


# Real-time prediction loop
try:
    while True:
        distance = measure_distance()
        example_input = pd.DataFrame({"Distance (cm)": [distance]})

        predicted_action = lr_model.predict(example_input)[0]
        reverse_label_map = {v: k for k, v in label_map.items()}
        action = reverse_label_map[predicted_action]

        print(f"Measured Distance: {distance:.2f} cm")
        print(f"Predicted Action: {action}")

        control_robot(action)
        time.sleep(1)

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    GPIO.cleanup()  # Cleanup GPIO resources
