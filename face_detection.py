import cv2

# Inicializáljuk a CascadeClassifier-t
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Inicializáljuk a videófelvevőt
video_capture = cv2.VideoCapture(0)


def detect_bounding_box(frame):
    # Átalakítjuk a képet szürkeárnyalatúvá
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Felismerjük az arcokat a szürkeárnyalatos képen
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        # Kirajzoljuk a körvonalakat az arcok körül
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    return faces


while True:
    # Olvassuk be a videó képkockáit
    ret, frame = video_capture.read()

    if not ret:
        break  # Kilépés, ha nem sikerül a képkocka beolvasása

    detected_faces = detect_bounding_box(frame)

    # Kirajzoljuk a feldolgozott képet egy ablakban
    cv2.imshow("My Face Detection Project", frame)

    # Kilépés a 'q' billentyű lenyomására
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# A videófelvevő felszabadítása és ablakok bezárása
video_capture.release()
cv2.destroyAllWindows()
