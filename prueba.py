import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import face_recognition
import os
import sqlite3
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
import pygame
import threading
import time
import logging
import json
import queue
import shutil
from collections import deque
from math import hypot

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ebi_log.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Configuración global
DB_FILE = './ebi_database.db'
DETECTIONS_DB_FILE = './detections_database.db'
ALARM_SOUND = 'alarm.wav'  # Archivo de sonido de alarma
APP_TITLE = "EBI - Escáner Biométrico Inteligente"
TEMP_IMAGE_DIR = './temp_images'

# Configuración para enviar correos (modificar con tus datos)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'brianbagnato2023@gmail.com',
    'password': 'opmt nees umbw tfgd',
    'recipient': 'brianbagnato2023@gmail.com'
}

# Crear directorios necesarios
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Crear la base de datos de intrusos
def create_database():
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS intrusos
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     nombre TEXT,
                     dni TEXT,
                     descripcion TEXT,
                     foto_blob BLOB NOT NULL,
                     encoding BLOB NOT NULL,
                     fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        logging.info("Base de datos de intrusos creada/verificada correctamente")
    except Exception as e:
        logging.error(f"Error al crear la base de datos de intrusos: {e}")

# Crear la base de datos de detecciones
def create_detections_database():
    try:
        conn = sqlite3.connect(DETECTIONS_DB_FILE)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS detecciones
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     intruso_id INTEGER,
                     nombre TEXT,
                     dni TEXT,
                     fecha_deteccion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     foto_blob BLOB,
                     ubicacion TEXT DEFAULT 'Desconocida',
                     FOREIGN KEY (intruso_id) REFERENCES intrusos (id))''')
        conn.commit()
        conn.close()
        logging.info("Base de datos de detecciones creada/verificada correctamente")
    except Exception as e:
        logging.error(f"Error al crear la base de datos de detecciones: {e}")

# Limpiar directorio temporal
def clean_temp_directory():
    try:
        for filename in os.listdir(TEMP_IMAGE_DIR):
            file_path = os.path.join(TEMP_IMAGE_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Error al eliminar {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error al limpiar directorio temporal: {e}")

# ----------------- Funciones para EAR (detectar parpadeo) -----------------
def eye_aspect_ratio(eye):
    """
    eye: lista de puntos (x,y) del ojo en el orden que devuelve face_recognition
    Calcula la relación A+B / (2*C)
    """
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def detect_blink_in_frames(frames, ear_threshold=0.23, consec_frames_for_blink=2):
    """
    frames: lista de frames BGR (numpy arrays), se analiza cada frame buscando eyes landmarks
    Devuelve True si encuentra al menos un parpadeo.
    """
    blink_count = 0
    consec = 0

    for f in frames:
        try:
            small = cv2.resize(f, (0, 0), fx=0.5, fy=0.5)  # reducir para velocidad
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(rgb)
            if not landmarks_list:
                consec = 0
                continue
            lm = landmarks_list[0]  # tomamos la primera cara
            left = lm.get('left_eye')
            right = lm.get('right_eye')
            if left and right and len(left) >= 6 and len(right) >= 6:
                leftEAR = eye_aspect_ratio(left)
                rightEAR = eye_aspect_ratio(right)
                ear = (leftEAR + rightEAR) / 2.0
                if ear < ear_threshold:
                    consec += 1
                else:
                    if consec >= consec_frames_for_blink:
                        blink_count += 1
                    consec = 0
        except Exception:
            # si falla un frame, lo ignoramos
            consec = 0
            continue

    # Si quedó un parpadeo al final del buffer:
    if consec >= consec_frames_for_blink:
        blink_count += 1

    return blink_count >= 1
# -----------------------------------------------------------------------

# Clase principal de la aplicación
class EBIApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')
        self.root.minsize(800, 600)  # Tamaño mínimo para responsividad
        
        # Limpiar directorio temporal al iniciar
        clean_temp_directory()
        
        # Variables de estado
        self.camera_active = False
        self.detection_active = False
        self.cap = None
        self.current_frame = None
        self.known_face_encodings = []
        self.known_face_data = []
        self.detection_thread = None
        self.stop_detection_flag = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.last_detection_time = {}
        self.detection_cooldown = 3  # Segundos entre detecciones del mismo intruso
        
        # Inicializar pygame para sonido
        try:
            pygame.mixer.init()
            logging.info("Mixer de pygame inicializado correctamente")
        except Exception as e:
            logging.error(f"Error al inicializar pygame mixer: {e}")
        
        # Crear bases de datos si no existen
        create_database()
        create_detections_database()
        
        # Cargar intrusos existentes
        self.load_intrusos()
        
        # Crear el contenedor principal con mejor responsividad
        self.container = tk.Frame(root, bg='#2c3e50')
        self.container.pack(fill="both", expand=True, padx=10, pady=10)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Crear los frames (pantallas)
        self.frames = {}
        for F in (StartFrame, CargarIntrusoFrame, BuscarIntrusoFrame, HistorialFrame):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Mostrar el frame inicial
        self.show_frame(StartFrame)
        
        # Configurar cierre limpio
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configurar evento para redimensionamiento
        self.root.bind('<Configure>', self.on_resize)
    
    def on_resize(self, event):
        """Manejar redimensionamiento de la ventana para responsividad"""
        if hasattr(self, 'current_frame') and self.current_frame:
            if hasattr(self.current_frame, 'on_resize'):
                self.current_frame.on_resize(event)
    
    def on_closing(self):
        """Manejar el cierre de la aplicación de forma limpia"""
        self.stop_detection()
        self.stop_camera()
        clean_temp_directory()
        self.root.destroy()
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        self.current_frame = frame
        
        # Detener la cámara si no estamos en un frame que la use
        if cont not in [BuscarIntrusoFrame, CargarIntrusoFrame] and self.camera_active:
            self.stop_camera()
        # Si vamos a BuscarIntrusoFrame o CargarIntrusoFrame y la cámara no está activa, iniciarla
        elif cont in [BuscarIntrusoFrame, CargarIntrusoFrame] and not self.camera_active:
            self.start_camera()
    
    def start_camera(self):
        if not self.camera_active:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "No se pudo abrir la cámara")
                    return False
                
                # Configurar resolución para mejor rendimiento
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.camera_active = True
                self.update_camera()
                logging.info("Cámara iniciada correctamente")
                return True
            except Exception as e:
                logging.error(f"Error al iniciar la cámara: {e}")
                messagebox.showerror("Error", f"No se pudo abrir la cámara: {e}")
                return False
        return True
    
    def stop_camera(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            logging.info("Cámara detenida")
    
    def update_camera(self):
        # Solo actualizar si estamos en el frame correcto y la cámara está activa
        if self.camera_active and self.cap is not None and isinstance(self.current_frame, (BuscarIntrusoFrame, CargarIntrusoFrame)):
            try:
                ret, frame = self.cap.read()
                if ret:
                    # Si el frame actual tiene buffer de frames, guardamos una copia (para detectar parpadeo)
                    try:
                        if hasattr(self.current_frame, 'collect_frames') and self.current_frame.collect_frames:
                            # guardamos una copia reducida para ahorrar memoria
                            self.current_frame.recent_frames.append(frame.copy())
                    except Exception as e:
                        logging.error(f"Error al guardar frame: {e}")

                    # Reducir la resolución para mostrar (preview)
                    frame_disp = cv2.resize(frame, (320, 240))
                    cv2image = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)

                    if hasattr(self.current_frame, 'camera_label'):
                        self.current_frame.camera_label.imgtk = imgtk
                        self.current_frame.camera_label.configure(image=imgtk)

                    self.root.after(30, self.update_camera)  # Reducir a 30ms
            except Exception as e:
                logging.error(f"Error en update_camera: {e}")
                self.root.after(100, self.update_camera)
    
    def load_intrusos(self):
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT id, nombre, dni, descripcion, encoding FROM intrusos")
            rows = c.fetchall()
            
            self.known_face_encodings = []
            self.known_face_data = []
            
            for row in rows:
                id_val, nombre, dni, desc, encoding_blob = row
                encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                self.known_face_encodings.append(encoding)
                self.known_face_data.append({
                    'id': id_val,
                    'nombre': nombre,
                    'dni': dni,
                    'desc': desc
                })
            
            conn.close()
            logging.info(f"Intrusos cargados: {len(self.known_face_data)}")
        except Exception as e:
            logging.error(f"Error al cargar intrusos: {e}")
    
    def save_intruso(self, nombre, dni, desc, foto_path):
        try:
            # Cargar la imagen y obtener el encoding facial
            image = face_recognition.load_image_file(foto_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                messagebox.showerror("Error", "No se detectó un rostro en la imagen")
                return False
            
            face_encoding = face_encodings[0]
            
            # Leer la imagen como bytes para almacenar como BLOB
            with open(foto_path, 'rb') as f:
                foto_blob = f.read()
            
            # Guardar en la base de datos
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO intrusos (nombre, dni, descripcion, foto_blob, encoding) VALUES (?, ?, ?, ?, ?)",
                      (nombre, dni, desc, foto_blob, face_encoding.tobytes()))
            conn.commit()
            conn.close()
            
            # Actualizar la lista de intrusos en memoria
            self.load_intrusos()
            logging.info(f"Intruso guardado: {nombre}")
            return True
        except Exception as e:
            logging.error(f"Error al guardar intruso: {e}")
            messagebox.showerror("Error", f"No se pudo guardar el intruso: {e}")
            return False
    
    def start_detection(self):
        if not self.camera_active:
            if not self.start_camera():
                return False
        
        self.detection_active = True
        self.stop_detection_flag.clear()
        
        # Iniciar hilo de detección
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        logging.info("Detección iniciada")
        return True
    
    def stop_detection(self):
        self.detection_active = False
        self.stop_detection_flag.set()
        logging.info("Detección detenida")
    
    def detection_loop(self):
        detection_interval = 0.5  # Segundos entre detecciones
        last_detection_time = 0
        
        while self.detection_active and not self.stop_detection_flag.is_set():
            current_time = time.time()
            
            # Realizar detección solo si ha pasado el intervalo
            if current_time - last_detection_time >= detection_interval:
                self.detect_faces()
                last_detection_time = current_time
            
            time.sleep(0.1)  # Pequeña pausa para no saturar la CPU
    
    def detect_faces(self):
        if not self.camera_active or self.cap is None:
            return
        
        try:
            ret, frame = self.cap.read()
            if ret:
                # Reducir tamaño para mejor rendimiento
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detectar rostros
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                # Comparar con rostros conocidos
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)  # Reducido para mayor precisión
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        face_data = self.known_face_data[first_match_index]
                        
                        # Verificar cooldown para evitar detecciones repetidas
                        current_time = time.time()
                        last_detection = self.last_detection_time.get(face_data['id'], 0)
                        
                        if current_time - last_detection < self.detection_cooldown:
                            continue  # Saltar detección si está en cooldown
                        
                        # Actualizar tiempo de última detección
                        self.last_detection_time[face_data['id']] = current_time
                        
                        # Guardar detección en base de datos
                        self.save_detection(face_data, frame)
                        
                        # Activar alarma y enviar alerta (sin cambiar de frame)
                        threading.Thread(target=self.trigger_alarm, args=(face_data, frame.copy()), daemon=True).start()
        except Exception as e:
            logging.error(f"Error en detección de rostros: {e}")
    
    def save_detection(self, face_data, frame):
        try:
            # Convertir frame a bytes para almacenar como BLOB
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                logging.error("Error al codificar la imagen para la detección")
                return
            
            image_bytes = encoded_image.tobytes()
            
            # Guardar en base de datos
            conn = sqlite3.connect(DETECTIONS_DB_FILE)
            c = conn.cursor()
            c.execute("INSERT INTO detecciones (intruso_id, nombre, dni, foto_blob) VALUES (?, ?, ?, ?)",
                      (face_data['id'], face_data['nombre'], face_data['dni'], image_bytes))
            conn.commit()
            conn.close()
            
            logging.info(f"Detección guardada: {face_data['nombre']}")
        except Exception as e:
            logging.error(f"Error al guardar detección: {e}")
    
    def trigger_alarm(self, face_data, frame):
        try:
            # Reproducir sonido de alarma
            if os.path.exists(ALARM_SOUND):
                try:
                    sound = pygame.mixer.Sound(ALARM_SOUND)
                    sound.play()
                except: 
                    pass  # Silenciar errores de sonido para no bloquear la app
        except:
            pass
        
        # Enviar alerta por correo en segundo plano
        threading.Thread(target=self.send_alert, args=(face_data, frame.copy()), daemon=True).start()
    
    def send_alert(self, face_data, frame):
        # Guardar imagen temporal para el email
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_img_path = os.path.join(TEMP_IMAGE_DIR, f"temp_detection_{timestamp}.jpg")
        cv2.imwrite(temp_img_path, frame)
        
        # Enviar por email
        self.send_email_alert(face_data, temp_img_path)
        
        # Programar eliminación de imagen temporal después de un tiempo
        self.root.after(10000, lambda: self.cleanup_temp_image(temp_img_path))
    
    def cleanup_temp_image(self, img_path):
        """Eliminar imagen temporal de forma segura"""
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
                logging.info(f"Imagen temporal eliminada: {img_path}")
        except Exception as e:
            logging.error(f"Error al eliminar imagen temporal: {e}")
    
    def send_email_alert(self, face_data, img_path):
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_CONFIG['email']
            msg['To'] = EMAIL_CONFIG['recipient']
            
            # Codificación robusta para caracteres especiales
            subject = "Alerta de intruso detectado!"
            msg['Subject'] = Header(subject, 'utf-8')
            
            # Crear cuerpo del mensaje con encoding seguro
            body_content = {
                'nombre': face_data['nombre'] or 'Desconocido',
                'dni': face_data['dni'] or 'No disponible',
                'desc': face_data['desc'] or 'Sin descripción',
                'fecha': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'ubicacion': 'Ubicación no especificada'
            }
            
            # Usar una plantilla detallada
            body = f"""
            ALERTA DE INTRUSO DETECTADO - SISTEMA EBI
            
            INFORMACIÓN DEL INTRUSO:
            • Nombre: {body_content['nombre']}
            • DNI: {body_content['dni']}
            • Descripción: {body_content['desc']}
            
            INFORMACIÓN DE LA DETECCIÓN:
            • Fecha y Hora: {body_content['fecha']}
            • Ubicación: {body_content['ubicacion']}
            • Sistema: EBI - Escáner Biométrico Inteligente
            
            Se ha detectado a esta persona en las inmediaciones. Por favor, verificar.
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # Adjuntar imagen
            with open(img_path, 'rb') as f:
                img_data = f.read()
            
            img = MIMEImage(img_data)
            img.add_header('Content-Disposition', 'attachment', filename='detection.jpg')
            msg.attach(img)
            
            # Enviar correo
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email de alerta enviado para: {face_data['nombre']}")
            
        except Exception as e:
            logging.error(f"Error al enviar correo: {str(e)}")

# Frame de inicio
class StartFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#2c3e50')
        
        # Hacer que el frame se expanda
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Frame interno para centrar los botones
        inner_frame = tk.Frame(self, bg='#2c3e50')
        inner_frame.place(relx=0.5, rely=0.5, anchor='center')
        
        title = tk.Label(inner_frame, text="EBI - Escáner Biométrico Inteligente", 
                        font=("Arial", 24, "bold"), fg='white', bg='#2c3e50')
        title.pack(pady=30)
        
        subtitle = tk.Label(inner_frame, text="SELECCIONE UN MODO", 
                           font=("Arial", 18), fg='white', bg='#2c3e50')
        subtitle.pack(pady=10)
        
        # Frame para botones con mejor responsividad
        btn_frame = tk.Frame(inner_frame, bg='#2c3e50')
        btn_frame.pack(expand=True, pady=20)
        
        btn_buscar = tk.Button(btn_frame, text="Buscar Intrusos", font=("Arial", 16), 
                              command=lambda: controller.show_frame(BuscarIntrusoFrame),
                              width=20, height=2, bg='#27ae60', fg='white')
        btn_buscar.pack(pady=20, fill='x')
        
        btn_cargar = tk.Button(btn_frame, text="Cargar Intruso", font=("Arial", 16), 
                              command=lambda: controller.show_frame(CargarIntrusoFrame),
                              width=20, height=2, bg='#3498db', fg='white')
        btn_cargar.pack(pady=20, fill='x')
        
        btn_historial = tk.Button(btn_frame, text="Ver Historial", font=("Arial", 16), 
                                 command=lambda: controller.show_frame(HistorialFrame),
                                 width=20, height=2, bg='#e67e22', fg='white')
        btn_historial.pack(pady=20, fill='x')
    
    def on_resize(self, event):
        """Manejar redimensionamiento para responsividad"""
        pass

# Frame para cargar intruso
class CargarIntrusoFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#2c3e50')
        self.photo_path = None
        self.preview_img = None
        # BUFFER para los últimos frames (se usa para detectar parpadeo)
        self.collect_frames = False
        self.recent_frames = deque(maxlen=60)  # almacena ~60 frames (2 sec aprox a 30fps)
        self.space_enabled = False  # solo permitir tomar foto si se presionó el botón

        # Configurar grid para que sea responsivo
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Botón volver
        btn_volver = tk.Button(self, text="← Volver", font=("Arial", 12),
                               command=self.go_back, bg='#7f8c8d', fg='white')
        btn_volver.grid(row=0, column=0, padx=10, pady=10, sticky='nw')

        # Título
        title = tk.Label(self, text="CARGAR INTRUSO", font=("Arial", 20, "bold"), fg='white', bg='#2c3e50')
        title.grid(row=0, column=1, columnspan=2, pady=20)

        # Formulario
        form_frame = tk.Frame(self, bg='#2c3e50')
        form_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=10, sticky='ew')
        form_frame.grid_columnconfigure(1, weight=1)

        tk.Label(form_frame, text="NOMBRE:", font=("Arial", 12), fg='white', bg='#2c3e50').grid(row=0, column=0, padx=10, pady=8, sticky='e')
        self.entry_nombre = tk.Entry(form_frame, font=("Arial", 12))
        self.entry_nombre.grid(row=0, column=1, padx=10, pady=8, sticky='ew')

        tk.Label(form_frame, text="DNI:", font=("Arial", 12), fg='white', bg='#2c3e50').grid(row=1, column=0, padx=10, pady=8, sticky='e')
        self.entry_dni = tk.Entry(form_frame, font=("Arial", 12))
        self.entry_dni.grid(row=1, column=1, padx=10, pady=8, sticky='ew')

        tk.Label(form_frame, text="DESCRIPCIÓN:", font=("Arial", 12), fg='white', bg='#2c3e50').grid(row=2, column=0, padx=10, pady=8, sticky='ne')
        self.entry_desc = tk.Text(form_frame, font=("Arial", 12), height=4)
        self.entry_desc.grid(row=2, column=1, padx=10, pady=8, sticky='ew')

        # --- BOTÓN TOMAR FOTO (debajo de descripción) ---
        self.btn_take = tk.Button(form_frame, text="Tomar foto", font=("Arial", 12),
                                  command=self.start_camera_for_capture, bg='#3498db', fg='white')
        self.btn_take.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        # Instrucciones (se actualizarán cuando presiones el botón)
        self.instrucciones_label = tk.Label(self, text="", font=("Arial", 12), fg='#f39c12', bg='#2c3e50')
        self.instrucciones_label.grid(row=2, column=0, columnspan=3, pady=4)

        # --- Paneles para cámara en vivo (izq) y preview de la foto tomada (der) ---
        preview_container = tk.Frame(self, bg='#2c3e50')
        preview_container.grid(row=4, column=0, columnspan=3, padx=20, pady=10, sticky='nsew')
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)
        preview_container.grid_columnconfigure(1, weight=1)

        camera_preview_frame = tk.Frame(preview_container, bg='#000000')
        camera_preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.camera_label = tk.Label(camera_preview_frame, bg='#000000')
        self.camera_label.pack(expand=True, fill='both')

        self.preview_frame = tk.Frame(preview_container, bg='#34495e')
        self.preview_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.preview_label = tk.Label(self.preview_frame, bg='#34495e', text="Vista previa")
        self.preview_label.pack(expand=True, fill='both')

        # Botón guardar (debajo)
        btn_guardar = tk.Button(self, text="Guardar Intruso", font=("Arial", 14, "bold"),
                                command=self.save_intruder, bg='#27ae60', fg='white')
        btn_guardar.grid(row=6, column=0, columnspan=3, pady=12, padx=20, sticky='ew')

        # Mensaje de estado
        self.message_label = tk.Label(self, text="", font=("Arial", 12), fg='#2ecc71', bg='#2c3e50')
        self.message_label.grid(row=7, column=0, columnspan=3, pady=6)

        # Bind para la tecla espacio; solo funcionará si self.space_enabled == True
        self.bind('<space>', self.take_photo)
        # el botón start_camera_for_capture llamará a self.focus_set() para que el bind funcione

    def on_resize(self, event):
        """Ajustar elementos al redimensionar la ventana"""
        if hasattr(self, 'preview_label') and self.preview_img:
            try:
                # Redimensionar la imagen de vista previa si existe
                img = Image.open(self.photo_path)
                # Calcular nuevo tamaño manteniendo aspect ratio
                width, height = img.size
                max_size = min(self.preview_frame.winfo_width(), self.preview_frame.winfo_height()) - 20
                ratio = min(max_size/width, max_size/height)
                new_size = (int(width*ratio), int(height*ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                self.preview_img = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=self.preview_img)
            except:
                pass

    def go_back(self):
        # Al volver, detener la cámara si está activa y limpiar flags
        self.space_enabled = False
        if self.controller.camera_active:
            self.controller.stop_camera()
        self.controller.show_frame(StartFrame)

    def start_camera_for_capture(self):
        """
        Se llama cuando se presiona 'Tomar foto'.
        Inicia la cámara (si corresponde), muestra la vista previa y habilita
        la recolección de frames para detectar parpadeo.
        """
        # Iniciar cámara si está apagada
        if not self.controller.camera_active:
            ok = self.controller.start_camera()
            if not ok:
                return

        # Limpiar buffer y habilitar recolección
        self.recent_frames.clear()
        self.collect_frames = True

        # Habilitar captura por espacio
        self.space_enabled = True
        self.instrucciones_label.config(text="Aprete ESPACIO para tomar la foto (asegurá parpadear)")
        self.message_label.config(text="Vista previa activada. Asegurate de parpadear una vez y presionar ESPACIO.")
        # Dar foco al frame para que capture la tecla ESPACIO
        self.focus_set()

    def take_photo(self, event=None):
        """
        Captura la foto solo si antes se presionó 'Tomar foto' (self.space_enabled True).
        """
        if not self.space_enabled:
            # ignorar si no se habilitó la captura
            return

        if not self.controller.camera_active:
            if not self.controller.start_camera():
                return

        # Pequeña pausa para que la cámara estabilice (si hace falta)
        self.controller.root.after(150, self._capture_photo)

    def _capture_photo(self):
        # Antes de guardar la foto, verificamos que haya parpadeo en el buffer
        # Usamos los frames recolectados en self.recent_frames
        try:
            # Hacemos una copia de la lista para procesar sin interferencia
            frames_to_check = list(self.recent_frames)
            # Si no tenemos frames suficientes, avisamos al usuario
            if len(frames_to_check) < 6:
                messagebox.showwarning("Advertencia", "No hubo suficientes frames para verificar parpadeo. Intentá de nuevo (mirá la cámara y parpadeá).")
                return

            blink_ok = detect_blink_in_frames(frames_to_check, ear_threshold=0.23, consec_frames_for_blink=2)
            if not blink_ok:
                messagebox.showerror("No se detectó parpadeo", "No se detectó un parpadeo en la vista previa. Intentá de nuevo: mirá la cámara y parpadeá antes de presionar ESPACIO.")
                # dejamos collect_frames True para que pueda intentarlo otra vez
                return
        except Exception as e:
            logging.error(f"Error al verificar parpadeo: {e}")
            # en caso de falla, preferimos no bloquear: avisamos y no guardamos
            messagebox.showerror("Error", "Error al verificar parpadeo. Intentá de nuevo.")
            return

        # Si pasamos la verificación, tomamos el frame actual y guardamos la foto
        ret, frame = self.controller.cap.read()
        if ret:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.photo_path = os.path.join(TEMP_IMAGE_DIR, f"temp_photo_{timestamp}.jpg")
            cv2.imwrite(self.photo_path, frame)

            # Mostrar la foto tomada en el preview derecho
            self.show_photo(self.photo_path)

            # Desactivar captura por espacio hasta que presiones 'Tomar foto' otra vez
            self.space_enabled = False
            self.collect_frames = False
            self.recent_frames.clear()
            self.instrucciones_label.config(text="")
            self.message_label.config(text="Foto guardada. Presione 'Guardar Intruso' para terminar.")
        else:
            messagebox.showerror("Error", "No se pudo leer la cámara. Asegúrese de que esté conectada.")

    def show_photo(self, path):
        try:
            img = Image.open(path)
            # Redimensionar manteniendo aspect ratio
            width, height = img.size
            max_size = 300
            ratio = min(max_size/width, max_size/height)
            new_size = (int(width*ratio), int(height*ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            self.preview_img = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self.preview_img, text="")
        except Exception as e:
            logging.error(f"Error al mostrar la foto: {e}")
            messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")

    def save_intruder(self):
        if not self.photo_path:
            messagebox.showerror("Error", "Debe tomar una foto primero (Presione 'Tomar foto' y luego ESPACIO)")
            return

        nombre = self.entry_nombre.get()
        dni = self.entry_dni.get()
        desc = self.entry_desc.get("1.0", tk.END).strip()

        if not nombre:
            messagebox.showerror("Error", "Debe ingresar un nombre")
            return

        if self.controller.save_intruso(nombre, dni, desc, self.photo_path):
            messagebox.showinfo("Éxito", "Intruso guardado correctamente")
            # Limpiar formulario
            self.entry_nombre.delete(0, tk.END)
            self.entry_dni.delete(0, tk.END)
            self.entry_desc.delete("1.0", tk.END)
            self.preview_label.configure(image='', text="Vista previa")
            self.photo_path = None
            self.message_label.config(text="")
            self.instrucciones_label.config(text="")
            self.space_enabled = False

            # Detener cámara y volver al inicio
            self.controller.stop_camera()
            self.controller.show_frame(StartFrame)

# Frame para buscar intruso (modo operativo)
class BuscarIntrusoFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.controller.current_frame = self
        self.configure(bg='#2c3e50')
        
        # Configurar grid para expansión
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Botón volver
        btn_volver = tk.Button(self, text="← Volver", font=("Arial", 12), 
                              command=self.stop_and_go_back,
                              bg='#7f8c8d', fg='white')
        btn_volver.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
        
        # Título
        title = tk.Label(self, text="MODO OPERATIVO - DETECCIÓN ACTIVA", font=("Arial", 20, "bold"), bg='#2c3e50', fg='white')
        title.grid(row=1, column=0, pady=20)
        
        # Etiqueta de estado
        self.status_label = tk.Label(self, text="Cámara inactiva", font=("Arial", 14), bg='#2c3e50', fg='white')
        self.status_label.grid(row=2, column=0, pady=10)
        
        # Frame para la cámara
        camera_frame = tk.Frame(self, bg='#000000')
        camera_frame.grid(row=3, column=0, pady=20, padx=20, sticky='nsew')
        camera_frame.grid_rowconfigure(0, weight=1)
        camera_frame.grid_columnconfigure(0, weight=1)
        
        self.camera_label = tk.Label(camera_frame, bg='#000000')
        self.camera_label.grid(row=0, column=0, sticky='nsew')
        
        # Botones de control
        btn_frame = tk.Frame(self, bg='#2c3e50')
        btn_frame.grid(row=4, column=0, pady=20)
        
        self.btn_toggle = tk.Button(btn_frame, text="Iniciar Detección", font=("Arial", 14), 
                                   command=self.toggle_detection, width=20, height=2, bg='#27ae60', fg='white')
        self.btn_toggle.pack(side='left', padx=10)
        
        # Actualizar estado inicial
        self.update_status()
    
    def on_resize(self, event):
        """Manejar redimensionamiento para responsividad"""
        pass
    
    def update_status(self):
        if self.controller.detection_active:
            self.status_label.config(text="Detección activa - Buscando intrusos...", fg='#2ecc71')
            self.btn_toggle.config(text="Detener Detección", bg='#e74c3c')
        else:
            self.status_label.config(text="Detección detenida", fg='#e74c3c')
            self.btn_toggle.config(text="Iniciar Detección", bg='#27ae60')
    
    def stop_and_go_back(self):
        self.controller.stop_detection()
        self.controller.stop_camera()
        self.controller.show_frame(StartFrame)
    
    def toggle_detection(self):
        if not self.controller.detection_active:
            if self.controller.start_detection():
                self.status_label.config(text="Detección activa - Buscando intrusos...", fg='#2ecc71')
                self.btn_toggle.config(text="Detener Detección", bg='#e74c3c')
        else:
            self.controller.stop_detection()
            self.status_label.config(text="Detección detenida", fg='#e74c3c')
            self.btn_toggle.config(text="Iniciar Detección", bg='#27ae60')

# Frame para ver historial de detecciones
class HistorialFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#2c3e50')
        
        # Configurar grid para expansión
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Botón volver
        btn_volver = tk.Button(self, text="← Volver", font=("Arial", 12), 
                              command=lambda: controller.show_frame(StartFrame),
                              bg='#7f8c8d', fg='white')
        btn_volver.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
        
        # Título
        title = tk.Label(self, text="HISTORIAL DE DETECCIONES", font=("Arial", 20, "bold"), bg='#2c3e50', fg='white')
        title.grid(row=1, column=0, pady=20)
        
        # Frame para controles
        controls_frame = tk.Frame(self, bg='#2c3e50')
        controls_frame.grid(row=2, column=0, pady=10)
        
        # Botón para cargar detecciones
        btn_cargar = tk.Button(controls_frame, text="Actualizar", font=("Arial", 12), 
                              command=self.load_detections, bg='#3498db', fg='white')
        btn_cargar.pack(side='left', padx=10)
        
        # Botón para limpiar historial
        btn_limpiar = tk.Button(controls_frame, text="Limpiar Historial", font=("Arial", 12), 
                               command=self.clear_history, bg='#e74c3c', fg='white')
        btn_limpiar.pack(side='left', padx=10)
        
        # Botón para exportar
        btn_exportar = tk.Button(controls_frame, text="Exportar CSV", font=("Arial", 12), 
                                command=self.export_csv, bg='#27ae60', fg='white')
        btn_exportar.pack(side='left', padx=10)
        
        # Crear Treeview para mostrar detecciones
        tree_frame = tk.Frame(self, bg='#2c3e50')
        tree_frame.grid(row=4, column=0, padx=20, pady=10, sticky='nsew')
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        columns = ("id", "nombre", "dni", "fecha")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15)
        
        # Definir encabezados
        self.tree.heading("id", text="ID")
        self.tree.heading("nombre", text="Nombre")
        self.tree.heading("dni", text="DNI")
        self.tree.heading("fecha", text="Fecha de Detección")
        
        # Definir anchos de columna
        self.tree.column("id", width=50, anchor='center')
        self.tree.column("nombre", width=150, anchor='center')
        self.tree.column("dni", width=100, anchor='center')
        self.tree.column("fecha", width=150, anchor='center')
        
        # Añadir scrollbar
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        # Bind double click para ver detalles
        self.tree.bind("<Double-1>", self.on_item_double_click)
        
        # Cargar detecciones al inicializar
        self.load_detections()
    
    def on_resize(self, event):
        """Manejar redimensionamiento para responsividad"""
        # Ajustar el ancho de las columnas según el tamaño disponible
        tree_width = self.tree.winfo_width()
        if tree_width > 500:  # Solo ajustar si hay suficiente espacio
            self.tree.column("id", width=int(tree_width * 0.1))
            self.tree.column("nombre", width=int(tree_width * 0.3))
            self.tree.column("dni", width=int(tree_width * 0.2))
            self.tree.column("fecha", width=int(tree_width * 0.4))
    
    def load_detections(self):
        # Limpiar treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        try:
            conn = sqlite3.connect(DETECTIONS_DB_FILE)
            c = conn.cursor()
            c.execute("SELECT id, nombre, dni, fecha_deteccion FROM detecciones ORDER BY fecha_deteccion DESC")
            rows = c.fetchall()
            
            for row in rows:
                self.tree.insert("", "end", values=row)
            
            conn.close()
            logging.info(f"Detecciones cargadas: {len(rows)}")
        except Exception as e:
            logging.error(f"Error al cargar detecciones: {e}")
    
    def on_item_double_click(self, event):
        item = self.tree.selection()[0]
        item_values = self.tree.item(item, "values")
        detection_id = item_values[0]
        
        # Mostrar detalles de la detección
        self.show_detection_details(detection_id)
    
    def show_detection_details(self, detection_id):
        try:
            conn = sqlite3.connect(DETECTIONS_DB_FILE)
            c = conn.cursor()
            c.execute("SELECT id, intruso_id, nombre, dni, fecha_deteccion, foto_blob FROM detecciones WHERE id = ?", (detection_id,))
            row = c.fetchone()
            conn.close()
            
            if row:
                # Crear ventana de detalles
                details_window = tk.Toplevel(self)
                details_window.title(f"Detalles de Detección #{detection_id}")
                details_window.geometry("500x400")
                details_window.configure(bg='#2c3e50')
                details_window.resizable(True, True)
                
                # Configurar grid para expansión
                details_window.grid_rowconfigure(1, weight=1)
                details_window.grid_columnconfigure(0, weight=1)
                
                # Mostrar información
                info_frame = tk.Frame(details_window, bg='#2c3e50')
                info_frame.grid(row=0, column=0, pady=10, padx=10, sticky='ew')
                info_frame.grid_columnconfigure(1, weight=1)
                
                tk.Label(info_frame, text=f"ID: {row[0]}", font=("Arial", 12), bg='#2c3e50', fg='white').grid(row=0, column=0, sticky='w', pady=5)
                tk.Label(info_frame, text=f"Nombre: {row[2]}", font=("Arial", 12), bg='#2c3e50', fg='white').grid(row=1, column=0, sticky='w', pady=5)
                tk.Label(info_frame, text=f"DNI: {row[3]}", font=("Arial", 12), bg='#2c3e50', fg='white').grid(row=2, column=0, sticky='w', pady=5)
                tk.Label(info_frame, text=f"Fecha: {row[4]}", font=("Arial", 12), bg='#2c3e50', fg='white').grid(row=3, column=0, sticky='w', pady=5)
                
                # Mostrar imagen desde BLOB
                img_frame = tk.Frame(details_window, bg='#34495e')
                img_frame.grid(row=1, column=0, pady=10, padx=10, sticky='nsew')
                img_frame.grid_rowconfigure(0, weight=1)
                img_frame.grid_columnconfigure(0, weight=1)
                
                if row[5]:
                    try:
                        # Convertir BLOB a imagen
                        nparr = np.frombuffer(row[5], np.uint8)
                        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img_np)
                        
                        # Redimensionar manteniendo aspect ratio
                        width, height = img.size
                        max_size = 300
                        ratio = min(max_size/width, max_size/height)
                        new_size = (int(width*ratio), int(height*ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                        
                        photo_img = ImageTk.PhotoImage(img)
                        
                        img_label = tk.Label(img_frame, image=photo_img, bg='#34495e')
                        img_label.image = photo_img  # Keep a reference
                        img_label.grid(row=0, column=0, sticky='nsew')
                    except Exception as e:
                        logging.error(f"Error al cargar imagen desde BLOB: {e}")
                        tk.Label(img_frame, text="Imagen no disponible", font=("Arial", 12), bg='#34495e', fg='white').grid(row=0, column=0, sticky='nsew')
                else:
                    tk.Label(img_frame, text="Imagen no disponible", font=("Arial", 12), bg='#34495e', fg='white').grid(row=0, column=0, sticky='nsew')
                
                # Botón para cerrar
                btn_cerrar = tk.Button(details_window, text="Cerrar", font=("Arial", 12), 
                                      command=details_window.destroy, bg='#3498db', fg='white')
                btn_cerrar.grid(row=2, column=0, pady=10)
        except Exception as e:
            logging.error(f"Error al mostrar detalles: {e}")
            messagebox.showerror("Error", "No se pudieron cargar los detalles")
    
    def clear_history(self):
        if messagebox.askyesno("Confirmar", "¿Está seguro de que desea eliminar todo el historial de detecciones?"):
            try:
                conn = sqlite3.connect(DETECTIONS_DB_FILE)
                c = conn.cursor()
                c.execute("DELETE FROM detecciones")
                conn.commit()
                conn.close()
                
                # Limpiar treeview
                for item in self.tree.get_children():
                    self.tree.delete(item)
                
                messagebox.showinfo("Éxito", "Historial eliminado correctamente")
                logging.info("Historial de detecciones eliminado")
            except Exception as e:
                logging.error(f"Error al eliminar historial: {e}")
                messagebox.showerror("Error", "No se pudo eliminar el historial")
    
    def export_csv(self):
        try:
            conn = sqlite3.connect(DETECTIONS_DB_FILE)
            c = conn.cursor()
            c.execute("SELECT * FROM detecciones ORDER BY fecha_deteccion DESC")
            rows = c.fetchall()
            conn.close()
            
            if not rows:
                messagebox.showinfo("Info", "No hay datos para exportar")
                return
            
            # Pedir ubicación para guardar
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("ID,Intruso_ID,Nombre,DNI,Fecha\n")
                    for row in rows:
                        # Escapar comas en los valores
                        escaped_row = []
                        for value in row[:5]:  # Solo las primeras 5 columnas (excluyendo el BLOB)
                            if value is not None and ',' in str(value):
                                escaped_value = f'"{value}"'
                                escaped_row.append(escaped_value)
                            else:
                                escaped_row.append(str(value) if value is not None else '')
                        
                        f.write(','.join(escaped_row) + '\n')
                
                messagebox.showinfo("Éxito", f"Datos exportados a {file_path}")
                logging.info(f"Datos exportados a {file_path}")
        except Exception as e:
            logging.error(f"Error al exportar CSV: {e}")
            messagebox.showerror("Error", "No se pudo exportar el historial")

# Iniciar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = EBIApp(root)
    root.mainloop()