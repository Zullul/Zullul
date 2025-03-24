import cv2
import numpy

# URL streaming dari ESP32
esp32_url = "http://192.168.1.20:81/stream"

print(f"Mencoba menghubungkan ke: {esp32_url}")

# Inisialisasi deteksi wajah dan mata dan senyum
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Membuka stream video
cap = cv2.VideoCapture(esp32_url)

if not cap.isOpened():
    print(f"Gagal membuka stream video dari URL: {esp32_url}")
else:
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Gagal menerima frame (stream mungkin terputus).")
                break

            # Konversi frame ke skala abu-abu
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Deteksi wajah
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Deteksi mata di dalam area wajah
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ax, ay, aw, ah) in eyes:
                    cv2.rectangle(roi_color, (ax, ay), (ax + aw, ay + ah), (0, 127, 255), 2)

                # Deteksi Senyum di dalam area wajah
                smiles = smile_cascade.detectMultiScale(roi_gray)
                for (ix, iy, iw, ih) in smiles:
                    cv2.rectangle(roi_color, (ix, iy), (ix + iw, iy + ih), (0, 255, 0), 2)

            # Tampilkan hasil stream video
            cv2.imshow("ESP32-CAM Stream", frame)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Menghentikan stream.")
                break

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

print("Stream video ditutup.")
