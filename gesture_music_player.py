import cv2
import mediapipe as mp
import pygame
import os
from mutagen.mp3 import MP3
import math
import asyncio
import platform

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Gesture-Controlled Music Player")
font = pygame.font.SysFont("arial", 24)

music_folder = "music"
songs = [f for f in os.listdir(music_folder) if f.endswith(".mp3")]
if not songs:
    raise FileNotFoundError("No MP3 files found in the music folder")
current_song_index = 0
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(music_folder, songs[current_song_index]))

def get_song_title(file_path):
    audio = MP3(file_path)
    return audio.get("TIT2", os.path.basename(file_path))

is_playing = False
volume = 0.5
pygame.mixer.music.set_volume(volume)
last_gesture = None
last_swipe_x = None
swipe_threshold = 0.2  
volume_change_cooldown = 0
gesture_cooldown = 0

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark2.x - landmark1.x) ** 2 + (landmark2.y - landmark1.y) ** 2)

def is_hand_closed(hand_landmarks):
    finger_tips = [8, 12, 16, 20] 
    finger_bases = [5, 9, 13, 17]  
    for tip, base in zip(finger_tips, finger_bases):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
            return False
    return True

async def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    global is_playing, current_song_index, volume, last_gesture, last_swipe_x, volume_change_cooldown, gesture_cooldown

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                return

        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)  
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        gesture_status = "No gesture detected"
        hand_detected = False

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_detected = True
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label = handedness.classification[0].label

                if label == "Left" and gesture_cooldown == 0:
                    if is_hand_closed(hand_landmarks):
                        if last_gesture != "play_pause":
                            if is_playing:
                                pygame.mixer.music.pause()
                                is_playing = False
                                gesture_status = "Paused"
                            else:
                                pygame.mixer.music.unpause()
                                is_playing = True
                                gesture_status = "Playing"
                            last_gesture = "play_pause"
                            gesture_cooldown = 30  

                elif label == "Right" and gesture_cooldown == 0:
                    wrist_x = hand_landmarks.landmark[0].x
                    if last_swipe_x is not None:
                        swipe_distance = wrist_x - last_swipe_x
                        if swipe_distance > swipe_threshold:
                            if current_song_index < len(songs) - 1:
                                current_song_index += 1
                                pygame.mixer.music.load(os.path.join(music_folder, songs[current_song_index]))
                                if is_playing:
                                    pygame.mixer.music.play()
                                gesture_status = f"Next: {get_song_title(os.path.join(music_folder, songs[current_song_index]))}"
                                last_gesture = "next"
                                gesture_cooldown = 30
                        elif swipe_distance < -swipe_threshold:
                            if current_song_index > 0:
                                current_song_index -= 1
                                pygame.mixer.music.load(os.path.join(music_folder, songs[current_song_index]))
                                if is_playing:
                                    pygame.mixer.music.play()
                                gesture_status = f"Previous: {get_song_title(os.path.join(music_folder, songs[current_song_index]))}"
                                last_gesture = "prev"
                                gesture_cooldown = 30
                    last_swipe_x = wrist_x

                    if volume_change_cooldown == 0:
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        distance = calculate_distance(thumb_tip, index_tip)
                        new_volume = min(max(distance * 2, 0.0), 1.0)  
                        if abs(new_volume - volume) > 0.05: 
                            volume = new_volume
                            pygame.mixer.music.set_volume(volume)
                            gesture_status = f"Volume: {int(volume * 100)}%"
                            volume_change_cooldown = 10

        if gesture_cooldown > 0:
            gesture_cooldown -= 1
        if volume_change_cooldown > 0:
            volume_change_cooldown -= 1
        if gesture_cooldown == 0:
            last_gesture = None

        screen.fill((255, 255, 255))  
        song_title = get_song_title(os.path.join(music_folder, songs[current_song_index]))
        song_text = font.render(f"Song: {song_title}", True, (0, 0, 0))
        status_text = font.render(f"Status: {gesture_status}", True, (0, 0, 0))
        volume_text = font.render(f"Volume: {int(volume * 100)}%", True, (0, 0, 0))
        screen.blit(song_text, (20, 20))
        screen.blit(status_text, (20, 60))
        screen.blit(volume_text, (20, 100))
        pygame.display.flip()

        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        await asyncio.sleep(1.0 / 30)  

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())