import os
import csv
from datetime import datetime, timedelta
from ultralytics import YOLO
import cv2

video_path = 'C:/Users/hp/Music/wild_life_conservation/gargi'             
video_path_out = '{}_out.mp4'.format(video_path)
output_folder = 'C:/Users/hp/Music/wild_life_conservation/gargi/detected_images'  # Change this to the desired folder path

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('C:/Users/hp/Music/wild_life_conservation/gargi/poacher.mp4')

ret, frame = cap.read()
H, W, _ = frame.shape

out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

display_out = cv2.VideoWriter('output_display.mp4', cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = 'C:/Users/hp/Music/wild_life_conservation/gargi/yolo1800.pt'

model = YOLO(model_path)  
threshold = 0.6

last_poacher_time = datetime.now() - timedelta(seconds=11)  # Initial value to ensure the first poacher is logged

def log_to_csv(class_name, camera, frame):
    global last_poacher_time
    current_time = datetime.now()
    
    if class_name == 'Poacher' and (current_time - last_poacher_time).total_seconds() > 5:
        date_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file_path = 'C:/Users/hp/Music/wild_life_conservation/gargi/logs.csv'
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([class_name, date_time, camera])

        # Save the detected frame as an image
        image_file_path = os.path.join(output_folder, f'detected_{date_time}.png')
        cv2.imwrite(image_file_path, frame)


        last_poacher_time = current_time

while ret:
    results = model(frame)[0]
    
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1 + 5), int(y1 + 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
            
        if results.names[int(class_id)] == 'Poacher':
            print('........................POACHER.........................DETECTED.....................') 
            log_to_csv('Poacher', 'CAM1', frame)

    out.write(frame)
    display_out.write(frame)
    cv2.imshow('Poacher Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
out.release()
display_out.release()
cv2.destroyAllWindows()
