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
import pygame
import threading
import time

# Configuración global
DB_FILE = './EBI para Municipio/ebi_database.db'
ALARM_SOUND = 'alarm.wav'  # Archivo de sonido de alarma (debe existir)
APP_TITLE = "EBI - Escáner Biométrico Inteligente"

# Configuración para enviar correos (modificar con tus datos)
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'email': 'tu_email@gmail.com',
    'password': 'tu_contraseña',
    'recipient': 'destinatario@email.com'
}

# Crear la base de datos
def create_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS intrusos
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 nombre TEXT,
                 dni TEXT,
                 descripcion TEXT,
                 foto_path TEXT NOT NULL,
                 encoding BLOB NOT NULL,
                 fecha_carga TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Clase principal de la aplicación
class EBIApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables de estado
        self.camera_active = False
        self.cap = None
        self.current_frame = None
        self.known_face_encodings = []
        self.known_face_data = []
        
        # Crear base de datos si no existe
        create_database()
        
        # Cargar intrusos existentes
        self.load_intrusos()
        
        # Crear el contenedor principal
        self.container = tk.Frame(root)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Crear los frames (pantallas)
        self.frames = {}
        for F in (StartFrame, CargarIntrusoFrame, BuscarIntrusoFrame, OperandoFrame):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Mostrar el frame inicial
        self.show_frame(StartFrame)
    
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
        # Detener la cámara si se cambia de frame
        if cont != BuscarIntrusoFrame and self.camera_active:
            self.stop_camera()
    
    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la cámara")
                return False
            
            self.camera_active = True
            self.update_camera()
        return True
    
    def stop_camera(self):
        if self.camera_active:
            self.camera_active = False
            if self.cap:
                self.cap.release()
    
    def update_camera(self):
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                # Convertir a RGB para Tkinter
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Actualizar el frame de la cámara
                if self.current_frame:
                    self.current_frame.camera_label.imgtk = imgtk
                    self.current_frame.camera_label.configure(image=imgtk)
                
                # Programar la próxima actualización
                self.root.after(10, self.update_camera)
    
    def load_intrusos(self):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT nombre, dni, descripcion, foto_path, encoding FROM intrusos")
        rows = c.fetchall()
        
        self.known_face_encodings = []
        self.known_face_data = []
        
        for row in rows:
            nombre, dni, desc, foto_path, encoding_blob = row
            encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            self.known_face_encodings.append(encoding)
            self.known_face_data.append({
                'nombre': nombre,
                'dni': dni,
                'desc': desc,
                'foto_path': foto_path
            })
        
        conn.close()
    
    def save_intruso(self, nombre, dni, desc, foto_path):
        # Cargar la imagen y obtener el encoding facial
        image = face_recognition.load_image_file(foto_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if not face_encodings:
            messagebox.showerror("Error", "No se detectó un rostro en la imagen")
            return False
        
        face_encoding = face_encodings[0]
        
        # Guardar en la base de datos
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("INSERT INTO intrusos (nombre, dni, descripcion, foto_path, encoding) VALUES (?, ?, ?, ?, ?)",
                  (nombre, dni, desc, foto_path, face_encoding.tobytes()))
        conn.commit()
        conn.close()
        
        # Actualizar la lista de intrusos en memoria
        self.load_intrusos()
        return True
    
    def detect_faces(self):
        if not self.camera_active:
            return
        
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
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                
                if True in matches:
                    first_match_index = matches.index(True)
                    face_data = self.known_face_data[first_match_index]
                    
                    # Activar alarma y enviar alerta
                    threading.Thread(target=self.trigger_alarm, args=(face_data, frame)).start()
    
    def trigger_alarm(self, face_data, frame):
        # Inicializar pygame mixer
        pygame.mixer.init()
        
        try:
            # Reproducir sonido de alarma
            sound = pygame.mixer.Sound(ALARM_SOUND)
            sound.play()
        except: 
            print("Error al reproducir la alarma")
        
        # El resto del código permanece igual...
        self.send_alert(face_data, frame)
        self.show_frame(OperandoFrame)
        self.frames[OperandoFrame].update_alert(face_data)
        
        # Volver a la detección después de 5 segundos
        time.sleep(5)
        self.show_frame(BuscarIntrusoFrame)
    
    def send_alert(self, face_data, frame):
        # Guardar imagen temporal
        temp_img_path = f"temp_detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(temp_img_path, frame)
        
        # Enviar por email
        self.send_email_alert(face_data, temp_img_path)
        
        # Aquí se podría agregar el envío por SMS o WhatsApp usando Twilio
        
        # Eliminar imagen temporal
        os.remove(temp_img_path)
    
    def send_email_alert(self, face_data, img_path):
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['email']
        msg['To'] = EMAIL_CONFIG['recipient']
        msg['Subject'] = "¡Alerta de intruso detectado!"
        
        body = f"""
        <h2>¡Intruso detectado!</h2>
        <p><b>Nombre:</b> {face_data['nombre'] or 'Desconocido'}</p>
        <p><b>DNI:</b> {face_data['dni'] or 'No disponible'}</p>
        <p><b>Descripción:</b> {face_data['desc'] or 'Sin descripción'}</p>
        <p><b>Fecha y hora:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Se adjunta la imagen capturada:</p>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Adjuntar imagen
        with open(img_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(img_path))
            msg.attach(img)
        
        try:
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['email'], EMAIL_CONFIG['password'])
            server.sendmail(EMAIL_CONFIG['email'], EMAIL_CONFIG['recipient'], msg.as_string())
            server.quit()
            print("Alerta enviada por correo")
        except Exception as e:
            print(f"Error al enviar correo: {e}")

# Frame de inicio
class StartFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#0B4C96')
        
        title = tk.Label(self, text="ELEGIR MODO", font=("Arial", 24, "bold"), bg='#0B4C96')
        title.pack(pady=40)
        
        frame = tk.Frame(self, bg='#0B4C96')
        frame.pack(expand=True)
        
        btn_buscar = tk.Button(frame, text="Buscar intruso", font=("Arial", 14), 
                              command=lambda: controller.show_frame(BuscarIntrusoFrame),
                              width=20, height=2, bg='#4CAF50', fg='white')
        btn_buscar.pack(pady=20)
        
        btn_cargar = tk.Button(frame, text="Cargar intruso", font=("Arial", 14), 
                              command=lambda: controller.show_frame(CargarIntrusoFrame),
                              width=20, height=2, bg='#2196F3', fg='white')
        btn_cargar.pack(pady=20)

# Frame para cargar intruso
class CargarIntrusoFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#0B4C96')
        self.photo_path = None
        self.preview_img = None
        
        # Botón volver
        btn_volver = tk.Button(self, text="← Volver", font=("Arial", 12), 
                              command=lambda: controller.show_frame(StartFrame),
                              bg='#9E9E9E', fg='white')
        btn_volver.grid(row=0, column=0, padx=10, pady=10, sticky='nw')
        
        # Título
        title = tk.Label(self, text="CARGAR INTRUSO", font=("Arial", 20, "bold"), bg='#0B4C96')
        title.grid(row=0, column=1, columnspan=2, pady=20)
        
        # Formulario
        form_frame = tk.Frame(self, bg='#0B4C96')
        form_frame.grid(row=1, column=0, columnspan=3, padx=20, pady=10)
        
        tk.Label(form_frame, text="NOMBRE:", font=("Arial", 12), bg='#0B4C96').grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.entry_nombre = tk.Entry(form_frame, font=("Arial", 12), width=30)
        self.entry_nombre.grid(row=0, column=1, padx=10, pady=10)
        
        tk.Label(form_frame, text="DNI:", font=("Arial", 12), bg='#0B4C96').grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.entry_dni = tk.Entry(form_frame, font=("Arial", 12), width=30)
        self.entry_dni.grid(row=1, column=1, padx=10, pady=10)
        
        tk.Label(form_frame, text="DESCRIPCIÓN:", font=("Arial", 12), bg='#0B4C96').grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.entry_desc = tk.Text(form_frame, font=("Arial", 12), width=30, height=4)
        self.entry_desc.grid(row=2, column=1, padx=10, pady=10)
        
        # Botones para la foto
        btn_frame = tk.Frame(self, bg='#0B4C96')
        btn_frame.grid(row=2, column=0, columnspan=3, pady=20)
        
        btn_tomar = tk.Button(btn_frame, text="Tomar foto", font=("Arial", 12), 
                            command=self.take_photo, width=15, bg='#FF9800', fg='white')
        btn_tomar.pack(side='left', padx=10)
        
        btn_subir = tk.Button(btn_frame, text="Subir foto", font=("Arial", 12), 
                            command=self.upload_photo, width=15, bg='#FF9800', fg='white')
        btn_subir.pack(side='left', padx=10)
        
        # Preview de la foto
        self.preview_frame = tk.Frame(self, bg='#e0e0e0', width=300, height=300)
        self.preview_frame.grid(row=3, column=0, columnspan=3, pady=20)
        
        self.preview_label = tk.Label(self.preview_frame, bg='#e0e0e0')
        self.preview_label.pack(padx=10, pady=10)
        
        # Botón guardar
        btn_guardar = tk.Button(self, text="Guardar Intruso", font=("Arial", 14, "bold"), 
                              command=self.save_intruder, width=20, height=2, bg='#4CAF50', fg='white')
        btn_guardar.grid(row=4, column=0, columnspan=3, pady=20)
    
    def take_photo(self):
        if not self.controller.start_camera():
            return
        
        ret, frame = self.controller.cap.read()
        if ret:
            # Guardar la foto temporal
            self.photo_path = f"temp_photo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(self.photo_path, frame)
            
            # Mostrar la foto
            self.show_photo(self.photo_path)
    
    def upload_photo(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )
        
        if file_path:
            self.photo_path = file_path
            self.show_photo(file_path)
    
    def show_photo(self, path):
        img = Image.open(path)
        img.thumbnail((300, 300))
        self.preview_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_img)
    
    def save_intruder(self):
        if not self.photo_path:
            messagebox.showerror("Error", "Debe tomar o subir una foto")
            return
        
        nombre = self.entry_nombre.get()
        dni = self.entry_dni.get()
        desc = self.entry_desc.get("1.0", tk.END).strip()
        
        if self.controller.save_intruso(nombre, dni, desc, self.photo_path):
            messagebox.showinfo("Éxito", "Intruso guardado correctamente")
            # Limpiar formulario
            self.entry_nombre.delete(0, tk.END)
            self.entry_dni.delete(0, tk.END)
            self.entry_desc.delete("1.0", tk.END)
            self.preview_label.configure(image='')
            self.photo_path = None
            
            # Volver al inicio
            self.controller.show_frame(StartFrame)

# Frame para buscar intruso (modo operativo)
class BuscarIntrusoFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.controller.current_frame = self
        self.configure(bg='#f0f0f0')
        
        # Botón volver
        btn_volver = tk.Button(self, text="← Volver", font=("Arial", 12), 
                              command=lambda: controller.show_frame(StartFrame),
                              bg='#9E9E9E', fg='white')
        btn_volver.pack(padx=10, pady=10, anchor='nw')
        
        # Título
        title = tk.Label(self, text="MODO OPERATIVO", font=("Arial", 20, "bold"), bg='#f0f0f0')
        title.pack(pady=20)
        
        # Etiqueta de estado
        self.status_label = tk.Label(self, text="Cámara inactiva", font=("Arial", 14), bg='#f0f0f0')
        self.status_label.pack(pady=10)
        
        # Frame para la cámara
        camera_frame = tk.Frame(self, bg='#000000')
        camera_frame.pack(pady=20)
        
        self.camera_label = tk.Label(camera_frame, bg='#000000')
        self.camera_label.pack()
        
        # Botones de control
        btn_frame = tk.Frame(self, bg='#f0f0f0')
        btn_frame.pack(pady=20)
        
        self.btn_start = tk.Button(btn_frame, text="Iniciar Detección", font=("Arial", 14), 
                                 command=self.toggle_detection, width=20, height=2, bg='#4CAF50', fg='white')
        self.btn_start.pack(side='left', padx=10)
        
        btn_stop = tk.Button(btn_frame, text="Detener", font=("Arial", 14), 
                           command=self.stop_detection, width=20, height=2, bg='#F44336', fg='white')
        btn_stop.pack(side='left', padx=10)
        
        # Configurar detección de tecla espaciadora
        self.root = parent.winfo_toplevel()
        self.root.bind("<space>", lambda event: self.controller.show_frame(StartFrame))
    
    def toggle_detection(self):
        if not self.controller.camera_active:
            if self.controller.start_camera():
                self.status_label.config(text="Detección activa")
                self.btn_start.config(text="Pausar Detección")
                # Iniciar detección de rostros
                threading.Thread(target=self.detect_faces_loop, daemon=True).start()
        else:
            self.controller.stop_camera()
            self.status_label.config(text="Cámara pausada")
            self.btn_start.config(text="Reanudar Detección")
    
    def stop_detection(self):
        self.controller.stop_camera()
        self.status_label.config(text="Cámara detenida")
        self.btn_start.config(text="Iniciar Detección")
        self.controller.show_frame(StartFrame)
    
    def detect_faces_loop(self):
        while self.controller.camera_active:
            self.controller.detect_faces()
            time.sleep(0.5)  # Revisar cada medio segundo

# Frame de alerta
class OperandoFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg='#f0f0f0')
        
        # Título
        title = tk.Label(self, text="¡ALERTA DE INTRUSO!", font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#F44336')
        title.pack(pady=40)
        
        # Mensaje
        self.alert_label = tk.Label(self, text="", font=("Arial", 16), bg='#f0f0f0')
        self.alert_label.pack(pady=20)
        
        # Imagen de la persona
        self.photo_frame = tk.Frame(self, bg='#e0e0e0', width=300, height=300)
        self.photo_frame.pack(pady=20)
        
        self.photo_label = tk.Label(self.photo_frame, bg='#e0e0e0')
        self.photo_label.pack(padx=10, pady=10)
        
        # Botón para continuar
        btn_continuar = tk.Button(self, text="Continuar Detección", font=("Arial", 14), 
                                command=lambda: controller.show_frame(BuscarIntrusoFrame),
                                width=20, height=2, bg='#4CAF50', fg='white')
        btn_continuar.pack(pady=30)
    
    def update_alert(self, face_data):
        # Actualizar texto
        alert_text = f"Se ha detectado a {face_data['nombre'] or 'una persona no identificada'}\n"
        alert_text += f"DNI: {face_data['dni'] or 'No disponible'}\n"
        alert_text += f"Descripción: {face_data['desc'] or 'Sin descripción'}\n"
        alert_text += f"Fecha y hora: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.alert_label.config(text=alert_text)
        
        # Mostrar foto si está disponible
        if face_data['foto_path'] and os.path.exists(face_data['foto_path']):
            try:
                img = Image.open(face_data['foto_path'])
                img.thumbnail((300, 300))
                photo_img = ImageTk.PhotoImage(img)
                self.photo_label.configure(image=photo_img)
                self.photo_label.image = photo_img
            except:
                self.photo_label.config(text="Imagen no disponible")
        else:
            self.photo_label.config(text="Imagen no disponible")

# Iniciar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = EBIApp(root)
    root.mainloop()