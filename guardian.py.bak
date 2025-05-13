import cv2
import numpy as np
import datetime
import time
import os
import logging
import threading
import requests
import telebot
import io
import colorama
from colorama import Fore, Back, Style
import pyaudio
from audio_compat import rms, HAVE_AUDIOOP
import wave
import math

colorama.init()

class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.ERROR: Fore.RED + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + Style.RESET_ALL,
        logging.INFO: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        logging.DEBUG: "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        'SUCCESS': Fore.GREEN + "%(asctime)s - %(name)s - SUCCESS - %(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        if hasattr(record, 'success') and record.success:
            log_fmt = self.FORMATS['SUCCESS']
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, 'SUCCESS')

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.SUCCESS):
        record = self.makeRecord(self.name, logging.SUCCESS, "", 0, message, args, None, "", None, **kwargs)
        record.success = True
        self.handle(record)

logging.Logger.success = success

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("guardian.log"),
        handler
    ]
)
logger = logging.getLogger("Guardian")

class Guardian:
    def __init__(self, config=None):
        self.config = {
            "motion_threshold": 15000,
            "face_detection_enabled": True,
            "telegram_token": "",
            "telegram_chat_id": "",
            "alarm_sound_path": "alarm.wav",
            "recording_path": "recordings/",
            "night_mode_start": 22,
            "night_mode_end": 6,
            "panic_color": [0, 0, 255],
            "camera_resolution": (640, 480),
            "show_preview": True,
            "stabilization_time": 5,
            "red_detection_threshold": 500000,
            "sound_detection_enabled": True,
            "sound_threshold": 5000,
            "consecutive_motion_frames": 5,
            "motion_consistency_threshold": 0.6,
            "min_area_percent": 0.01,
            "max_area_percent": 0.5
        }

        if config:
            self.config.update(config)

        os.makedirs(self.config["recording_path"], exist_ok=True)

        self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_resolution"][0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_resolution"][1])

        if not self.cap.isOpened():
            logger.error("Error: Could not open camera.")
            exit()

        self.is_recording = False
        self.panic_mode = False
        self.alert_visual = False
        self.alert_timer = 0
        self.motion_detected_time = 0
        self.motion_frames_count = 0
        self.last_motion_contours = None
        self.previous_flow = None

        self.audio_thread = None
        self.sound_detected = False
        self.last_sound_level = 0
        self.baseline_sound = 0
        self.sound_detection_active = False

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if self.config["telegram_token"]:
            try:
                self.bot = telebot.TeleBot(self.config["telegram_token"])
                logger.success("Telegram bot initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.bot = None
        else:
            self.bot = None
            logger.warning("No Telegram bot token provided")

        if self.config["sound_detection_enabled"]:
            try:
                self.init_audio()
                logger.success("Audio detection initialized")
            except Exception as e:
                logger.error(f"Failed to initialize audio: {e}")
                self.config["sound_detection_enabled"] = False

        logger.info(f"Learning normal scene patterns for {self.config['stabilization_time']} seconds...")
        start_frames = []
        flow_patterns = []
        stabilization_start = time.time()

        while time.time() - stabilization_start < self.config["stabilization_time"]:
            ret, frame = self.cap.read()
            if ret:
                if self.config["show_preview"]:
                    status_text = f"Learning scene: {int(time.time() - stabilization_start)}/{self.config['stabilization_time']}s"
                    cv2.putText(frame, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Guardian - Learning Scene", frame)
                    cv2.waitKey(1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                start_frames.append(gray)

                if len(start_frames) > 1:
                    flow = cv2.calcOpticalFlowFarneback(
                        start_frames[-2], start_frames[-1],
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    flow_patterns.append(flow)

            time.sleep(0.1)

        if start_frames:
            self.prev_frame = sum(start_frames) // len(start_frames)
            self.prev_frame = cv2.GaussianBlur(self.prev_frame, (21, 21), 0)

            if flow_patterns:
                self.avg_flow = sum(flow_patterns) / len(flow_patterns)
            else:
                self.avg_flow = None

            logger.success("Camera stabilized and scene learned successfully")
        else:
            ret, frame = self.cap.read()
            if ret:
                self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.prev_frame = cv2.GaussianBlur(self.prev_frame, (21, 21), 0)
                self.avg_flow = None
                logger.warning("Camera stabilization limited - using single frame")
            else:
                logger.error("Failed to initialize camera frames")
                exit()

        if self.config["sound_detection_enabled"]:
            self.learn_sound_baseline()

        if self.config["sound_detection_enabled"]:
            self.sound_detection_active = True
            self.audio_thread = threading.Thread(target=self.monitor_sound)
            self.audio_thread.daemon = True
            self.audio_thread.start()

        logger.success("Guardian initialized successfully")

    def init_audio(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )

    def learn_sound_baseline(self):
        if not self.config["sound_detection_enabled"]:
            return

        logger.info("Learning ambient sound baseline...")
        sound_samples = []

        start_time = time.time()
        while time.time() - start_time < 3:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                rms = audioop.rms(data, 2)
                sound_samples.append(rms)
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error sampling sound: {e}")
                break

        if sound_samples:
            self.baseline_sound = sum(sound_samples) / len(sound_samples)
            std_dev = math.sqrt(sum((x - self.baseline_sound) ** 2 for x in sound_samples) / len(sound_samples))
            self.baseline_sound = self.baseline_sound + std_dev

            self.config["sound_threshold"] = max(self.baseline_sound * 2, 1000)

            logger.success(f"Sound baseline: {self.baseline_sound:.2f}, threshold: {self.config['sound_threshold']:.2f}")
        else:
            logger.warning("Failed to learn sound baseline")

    def monitor_sound(self):
        logger.info("Sound monitoring thread started")

        while self.sound_detection_active:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                rms_val = audioop.rms(data, 2)
                self.last_sound_level = rms_val

                if rms_val > self.config["sound_threshold"]:
                    if not self.sound_detected:
                        logger.info(f"Unusual sound detected! Level: {rms_val} (threshold: {self.config['sound_threshold']})")
                        self.sound_detected = True

                        self.record_audio_clip()
                else:
                    if self.sound_detected and time.time() - self.sound_detected_time > 3:
                        self.sound_detected = False

            except Exception as e:
                logger.error(f"Error in sound monitoring: {e}")
                time.sleep(1)

            time.sleep(0.05)

    def record_audio_clip(self):
        try:
            self.sound_detected_time = time.time()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_path = os.path.join(self.config["recording_path"], f"sound_{timestamp}.wav")

            wf = wave.open(audio_path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)

            logger.info("Recording audio clip...")
            frames = []
            start_time = time.time()

            while time.time() - start_time < 5:
                data = self.stream.read(1024, exception_on_overflow=False)
                frames.append(data)

            wf.writeframes(b''.join(frames))
            wf.close()

            logger.success(f"Audio clip saved: {audio_path}")

            image_path = self.capture_image()

            if image_path:
                success = self.send_telegram_alert(image_path, audio_path=audio_path)

                if not success:
                    self.sound_alarm()

        except Exception as e:
            logger.error(f"Error recording audio clip: {e}")

    def is_night_time(self):
        current_hour = datetime.datetime.now().hour
        start = self.config["night_mode_start"]
        end = self.config["night_mode_end"]

        if start < end:
            return start <= current_hour < end
        else:
            return current_hour >= start or current_hour < end

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        threshold = self.config["motion_threshold"]
        if self.is_night_time():
            threshold = threshold * 0.8

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * self.config["min_area_percent"]
        max_area = frame_area * self.config["max_area_percent"]

        valid_motion = False
        motion_contours = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)

            if 0.5 < circularity < 0.95:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h

                if 0.75 < aspect_ratio < 1.25 and circularity > 0.8:
                    if area < threshold * 2:
                        continue

            if area > threshold:
                motion_contours.append(contour)
                valid_motion = True

        current_time = time.time()

        if not hasattr(self, 'slow_reference_frame') or current_time - self.last_slow_reference_update > 2.0:
            self.slow_reference_frame = self.prev_frame.copy()
            self.last_slow_reference_update = current_time

        slow_frame_delta = cv2.absdiff(self.slow_reference_frame, gray)
        slow_thresh = cv2.threshold(slow_frame_delta, 20, 255, cv2.THRESH_BINARY)[1]
        slow_thresh = cv2.dilate(slow_thresh, None, iterations=2)
        slow_contours, _ = cv2.findContours(slow_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in slow_contours:
            area = cv2.contourArea(contour)
            if area > threshold * 0.7:
                motion_contours.append(contour)
                valid_motion = True

        if valid_motion:
            self.motion_frames_count += 1
            self.last_motion_contours = motion_contours
            self.last_motion_time = current_time
        elif current_time - getattr(self, 'last_motion_time', 0) > 1.0:
            self.motion_frames_count = 0

        alpha = 0.05
        self.prev_frame = cv2.addWeighted(self.prev_frame, 1-alpha, gray, alpha, 0)

        motion_detected = self.motion_frames_count >= self.config["consecutive_motion_frames"]

        if motion_detected:
            self.motion_detected_time = current_time
            if not hasattr(self, 'last_motion_log_time') or current_time - self.last_motion_log_time > 3.0:
                logger.info("Motion detected")
                self.last_motion_log_time = current_time

        return motion_detected, thresh, motion_contours

    def calculate_flow_similarity(self, flow1, flow2):
        try:
            mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
            mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

            threshold = 0.5
            significant_mask = (mag1 > threshold) & (mag2 > threshold)

            if np.sum(significant_mask) == 0:
                return 0

            angle_diff = np.minimum(np.abs(ang1 - ang2), 2*np.pi - np.abs(ang1 - ang2))
            angle_similarity = 1 - (angle_diff[significant_mask] / np.pi)

            return np.mean(angle_similarity)
        except Exception as e:
            logger.error(f"Error calculating flow similarity: {e}")
            return 0

    def is_directional_motion(self, flow):
        try:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            significant_mask = mag > 0.5
            if np.sum(significant_mask) < 50:
                return False

            angles = ang[significant_mask]

            angles_deg = np.degrees(angles) % 360
            hist, _ = np.histogram(angles_deg, bins=8, range=(0, 360))

            main_direction_count = np.max(hist)
            total_vectors = np.sum(hist)

            return (main_direction_count / total_vectors) > 0.6

        except Exception as e:
            logger.error(f"Error analyzing motion direction: {e}")
            return False

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return len(faces) > 0, faces

    def detect_panic_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        return np.sum(mask) > self.config["red_detection_threshold"]

    def send_telegram_alert(self, image_path, video_path=None, audio_path=None):

        if not self.bot:
            logger.warning("Telegram bot not initialized, can't send alert")
            return False

        try:

            with open(image_path, 'rb') as photo:
                self.bot.send_photo(
                    self.config["telegram_chat_id"],
                    photo,
                    caption=f"⚠️ INTRUSION DETECTED! {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )

            if video_path and os.path.exists(video_path):
                with open(video_path, 'rb') as video:
                    self.bot.send_video(
                        self.config["telegram_chat_id"],
                        video,
                        caption=f"🎥 Intrusion footage {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

            if audio_path and os.path.exists(audio_path):
                with open(audio_path, 'rb') as audio:
                    self.bot.send_audio(
                        self.config["telegram_chat_id"],
                        audio,
                        caption=f"🔊 Unusual sound detected {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )

            logger.success("Telegram alert sent successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
            return False

    def sound_alarm(self):

        try:
            import pygame
            pygame.mixer.init()

            if not os.path.exists(self.config["alarm_sound_path"]):
                logger.warning(f"Alarm file {self.config['alarm_sound_path']} not found. Using system beep instead.")
                import winsound
                winsound.Beep(1000, 1000)
                return

            pygame.mixer.music.load(self.config["alarm_sound_path"])
            pygame.mixer.music.play()

            logger.success("Alarm sounded")

        except Exception as e:
            logger.error(f"Failed to play alarm sound: {e}")

    def capture_image(self):

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.config["recording_path"], f"intrusion_{timestamp}.jpg")

        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to capture image")
            return None

        cv2.imwrite(image_path, frame)
        logger.success(f"Image captured: {image_path}")

        return image_path

    def record_video(self, duration=10):

        if self.is_recording:
            logger.warning("Already recording, ignoring request")
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.config["recording_path"], f"intrusion_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))
        out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        self.is_recording = True
        start_time = time.time()

        while self.is_recording and (time.time() - start_time) < duration:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture video frame")
                break

            out.write(frame)

            if self.config["show_preview"]:

                cv2.putText(frame, "RECORDING", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Time: {time.time() - start_time:.1f}s",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Guardian - Live Feed", frame)
                cv2.waitKey(1)

            time.sleep(0.01)

        out.release()
        self.is_recording = False
        logger.success(f"Video recorded: {video_path}, duration: {time.time() - start_time:.2f}s")

        return video_path

    def panic_record(self, duration=60):

        logger.warning("PANIC MODE ACTIVATED - Recording emergency footage")
        self.panic_mode = True
        video_path = self.record_video(duration)
        self.panic_mode = False

        image_path = self.capture_image()
        if image_path:
            success = self.send_telegram_alert(
                image_path,
                video_path,
            )
            if not success:
                self.sound_alarm()

        return video_path

    def ghost_mode(self):

        pass

    def mock_intruder_alert(self):

        logger.info("Triggering mock intrusion alert")

        sample_image = os.path.join(self.config["recording_path"], "sample_intrusion.jpg")
        sample_video = os.path.join(self.config["recording_path"], "sample_intrusion.mp4")

        if not os.path.exists(sample_image):
            sample_image = self.capture_image()

        if not os.path.exists(sample_video):
            sample_video = self.record_video(5)

        success = self.send_telegram_alert(sample_image, sample_video)
        if not success:
            self.sound_alarm()

        logger.success("Mock intrusion alert completed")

    def run(self):

        logger.success("Guardian monitoring started")

        try:
            while True:

                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame from camera")
                    time.sleep(1)
                    continue

                display_frame = frame.copy()

                if self.config["show_preview"]:

                    cv2.putText(display_frame, f"Motion frames: {self.motion_frames_count}/{self.config['consecutive_motion_frames']}",
                                (10, display_frame.shape[0] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if self.config["sound_detection_enabled"]:
                        sound_text = f"Sound: {self.last_sound_level:.0f}/{self.config['sound_threshold']:.0f}"
                        cv2.putText(display_frame, sound_text,
                                    (10, display_frame.shape[0] - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                if self.detect_panic_color(frame):

                    cv2.putText(display_frame, "PANIC COLOR DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if self.config["show_preview"]:
                        cv2.imshow("Guardian - Live Feed", display_frame)
                        cv2.waitKey(1)

                    self.panic_record()
                    continue

                if self.sound_detected and not self.is_recording:

                    cv2.putText(display_frame, "SOUND DETECTED", (display_frame.shape[1] - 200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                motion_detected, motion_frame, motion_contours = self.detect_motion(frame)

                if motion_detected:
                    logger.info("Motion detected")
                    self.alert_visual = True
                    self.alert_timer = time.time()

                    cv2.drawContours(display_frame, motion_contours, -1, (0, 255, 0), 2)

                    faces_detected = False
                    if self.config["face_detection_enabled"]:
                        faces_detected, faces = self.detect_faces(frame)

                        if faces_detected:
                            logger.info(f"Face detected! Found {len(faces)} faces")

                            for (x, y, w, h) in faces:
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    if faces_detected or not self.config["face_detection_enabled"]:

                        cv2.putText(display_frame, "! INTRUSION DETECTED !",
                                    (int(display_frame.shape[1]/2)-120, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        image_path = self.capture_image()

                        video_path = self.record_video(10)

                        if image_path:
                            success = self.send_telegram_alert(image_path, video_path)

                            if not success:

                                red_overlay = np.zeros_like(display_frame)
                                red_overlay[:] = (0, 0, 255)
                                display_frame = cv2.addWeighted(display_frame, 0.7, red_overlay, 0.3, 0)

                current_time = time.time()
                if self.alert_visual:

                    if current_time - self.alert_timer > 5.0:

                        self.alert_visual = False
                        logger.info("Motion alert ended")
                    else:

                        if int(current_time * 2) % 2 == 0:
                            cv2.putText(display_frame, "MOTION DETECTED", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            display_frame = cv2.copyMakeBorder(
                                display_frame, 10, 10, 10, 10,
                                cv2.BORDER_CONSTANT, value=(0, 0, 255)
                            )

                if self.config["show_preview"]:

                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(display_frame, f"Time: {current_time}",
                                (10, display_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    mode = "Night Mode" if self.is_night_time() else "Day Mode"
                    cv2.putText(display_frame, f"Mode: {mode}",
                                (10, display_frame.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow("Guardian - Live Feed", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                time.sleep(0.03)

        except KeyboardInterrupt:
            logger.info("Guardian monitoring stopped by user")
        except Exception as e:
            logger.error(f"Guardian encountered an error: {e}")
        finally:

            if self.config["sound_detection_enabled"]:
                self.sound_detection_active = False
                if self.audio_thread:
                    self.audio_thread.join(timeout=1)
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()

            self.cap.release()
            cv2.destroyAllWindows()
            logger.success("Guardian resources released")

if __name__ == "__main__":

    try:
        from config import config
        guardian = Guardian(config)
    except ImportError:
        logger.warning("Configuration file not found, using defaults")
        guardian = Guardian()