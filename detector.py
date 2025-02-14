from ultralytics import YOLO
import cv2

# Load pre-trained PPE model
model = YOLO("best.pt")# PPE-specific model

# Load the video
video_path = "worker.mp4"
cap = cv2.VideoCapture(video_path)

# Resize factor for display
scale = 0.3  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for display
    frame_resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    # Perform detection
    results = model(frame_resized)

    # Process detections
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                if box.xyxy is not None and box.conf is not None:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    conf = float(box.conf[0])  # Confidence score
                    cls = int(box.cls[0])  # Class index

                    # Draw bounding box
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label detected object
                    label = f"{model.names[cls]}: {conf:.2f}"
                    cv2.putText(frame_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow("PPE Detection", frame_resized)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
