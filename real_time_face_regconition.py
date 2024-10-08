import cv2

haar_cascade = cv2.CascadeClassifier('haar_face.xml')
def real_time_face_recognition(model):
    # Open webcam for capturing video
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Convert to grayscale (required for face detection)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame using the Haar Cascade
        faces_rect = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

        # Loop through all the detected faces
        for (x, y, w, h) in faces_rect:
            faces_roi = gray_frame[y:y + h, x:x + w]  # Extract the region of interest (the face)

            # Predict the label of the face
            label, confidence = model.predict(faces_roi)

            # Display the predicted label and confidence on the frame
            cv2.putText(frame, f'{labels[label]} - {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting frame with face detection and recognition
        cv2.imshow('Face Recognition', frame)

        # Press 'q' to exit the webcam feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()