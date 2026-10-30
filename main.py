
import cv2
import mediapipe as mp
import numpy as np
import math

# ==========================================
# 1. PENGATURAN KAMERA DAN MEDIAPIPE
# ==========================================

kamera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

deteksi_tangan = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ==========================================
# 2. PENGATURAN KUBUS 3D
# ==========================================

cube_x = 320
cube_y = 240
cube_size = 65

rotasi_x = 0
rotasi_y = 0

sedang_dipegang = False
offset_x = 0
offset_y = 0

# ==========================================
# 3. FUNGSI MENGGAMBAR KUBUS 3D
# ==========================================

def gambar_kubus(frame, cx, cy, size, rx, ry):

    titik = np.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ], dtype=float) * size

    # Matriks rotasi X
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])

    # Matriks rotasi Y
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])

    # Menggabungkan rotasi
    titik = titik @ (Ry @ Rx).T

    # Proyeksi 3D ke 2D
    hasil = []

    for x, y, z in titik:
        jarak = 350 + z
        px = int(cx + (x * 500 / jarak))
        py = int(cy + (y * 500 / jarak))
        hasil.append((px, py))

    # Garis penghubung kubus
    garis = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    for a, b in garis:
        cv2.line(
            frame,
            hasil[a],
            hasil[b],
            (0, 255, 0),
            3
        )

    # Titik sudut kubus
    for titik_layar in hasil:
        cv2.circle(
            frame,
            titik_layar,
            4,
            (0, 255, 255),
            -1
        )


# ==========================================
# 4. PROGRAM UTAMA
# ==========================================

while kamera.isOpened():

    berhasil, frame = kamera.read()

    if not berhasil:
        print("Kamera tidak ditemukan!")
        break

    # Membalik kamera agar gerakan terasa alami
    frame = cv2.flip(frame, 1)

    tinggi, lebar, _ = frame.shape

    # Mengubah BGR menjadi RGB untuk MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hasil = deteksi_tangan.process(rgb)

    status = "TUNJUKKAN TANGAN"

    # ======================================
    # 5. DETEKSI TANGAN
    # ======================================

    if hasil.multi_hand_landmarks:

        for tangan in hasil.multi_hand_landmarks:

            # Menggambar 21 titik tangan
            mp_drawing.draw_landmarks(
                frame,
                tangan,
                mp_hands.HAND_CONNECTIONS
            )

            landmark = tangan.landmark

            # Mengambil koordinat ibu jari
            ibu_jari = landmark[4]

            # Mengambil koordinat telunjuk
            telunjuk = landmark[8]

            # Mengubah koordinat menjadi piksel
            ibu_x = int(ibu_jari.x * lebar)
            ibu_y = int(ibu_jari.y * tinggi)

            tel_x = int(telunjuk.x * lebar)
            tel_y = int(telunjuk.y * tinggi)

            # ==================================
            # 6. DETEKSI GESTUR CUBIT
            # ==================================

            jarak = math.hypot(
                ibu_x - tel_x,
                ibu_y - tel_y
            )

            mencubit = jarak < 45

            # Titik tengah cubitan
            tengah_x = (ibu_x + tel_x) // 2
            tengah_y = (ibu_y + tel_y) // 2

            # ==================================
            # 7. MENGAMBIL KUBUS
            # ==================================

            jarak_kubus = math.hypot(
                tel_x - cube_x,
                tel_y - cube_y
            )

            if mencubit and not sedang_dipegang:

                if jarak_kubus < cube_size * 1.8:

                    sedang_dipegang = True

                    offset_x = cube_x - tengah_x
                    offset_y = cube_y - tengah_y

            # ==================================
            # 8. MENGGESER KUBUS
            # ==================================

            if sedang_dipegang:

                if mencubit:

                    cube_x = tengah_x + offset_x
                    cube_y = tengah_y + offset_y

                    status = "KUBUS SEDANG DIPEGANG"

                    cv2.circle(
                        frame,
                        (tengah_x, tengah_y),
                        12,
                        (0, 255, 255),
                        2
                    )

                else:

                    sedang_dipegang = False
                    status = "KUBUS DILEPASKAN"

            elif mencubit:

                status = "CUBIT DEKAT KUBUS"

            else:

                status = "GERAKKAN TELUNJUK UNTUK ROTASI"

                # Mengatur rotasi berdasarkan posisi telunjuk
                rotasi_y = (telunjuk.x - 0.5) * 3.0
                rotasi_x = (telunjuk.y - 0.5) * 3.0

            # Menandai ujung telunjuk
            cv2.circle(
                frame,
                (tel_x, tel_y),
                10,
                (255, 0, 255),
                -1
            )

    else:

        sedang_dipegang = False

    # ======================================
    # 9. MENGGAMBAR KUBUS
    # ======================================

    gambar_kubus(
        frame,
        cube_x,
        cube_y,
        cube_size,
        rotasi_x,
        rotasi_y
    )

    # ======================================
    # 10. INFORMASI LAYAR
    # ======================================

    cv2.putText(
        frame,
        "3D GESTURE CONTROL",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        status,
        (15, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Pinch: Grab | Release: Drop | Index: Rotate",
        (15, tinggi - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # Menampilkan hasil
    cv2.imshow(
        "JARVIS - 3D Gesture Control",
        frame
    )

    # Keluar dengan tombol Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================================
# 11. MEMBERSIHKAN PROGRAM
# ==========================================

kamera.release()
deteksi_tangan.close()
cv2.destroyAllWindows()
