import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import os
from events import (
    on_mouse_down,
    on_mouse_drag,
    on_mouse_up,
    on_mouse_wheel,
    on_listbox_click,
    on_listbox_double_click,
    on_template_click
)


class ImageToolApp:
    def __init__(self, root):
        self.on_mouse_down = on_mouse_down.__get__(self)
        self.on_mouse_drag = on_mouse_drag.__get__(self)
        self.on_mouse_up = on_mouse_up.__get__(self)
        self.on_mouse_wheel = on_mouse_wheel.__get__(self)
        self.on_listbox_click = on_listbox_click.__get__(self)
        self.on_listbox_double_click = on_listbox_double_click.__get__(self)
        self.on_template_click = on_template_click.__get__(self)

        self.detections = []
        self.selected_index = None
        self.accepted_detections = []
        self.template_history = []
        self.template_thumbnails = []
        self.root = root
        self.root.title("Detector de Padrões")
        root.state('zoomed')

        self.zoom = 1.0
        self.zoom_step = 0.1
        self.min_zoom, self.max_zoom = 0.2, 3.0

        self.mode = 'circle'
        self.drawing = False
        self.center = None
        self.radius = 0
        self.start_point = None
        self.end_point = None

        self.img = None
        self.clone = None
        self.gray = None
        self.photo = None
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(self.main_frame)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        load_btn = ttk.Button(toolbar_frame, text="Carregar Imagem", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.mode_label = ttk.Label(toolbar_frame, text="Modo: Círculo")
        self.mode_label.pack(side=tk.LEFT, padx=5)

        circle_btn = ttk.Button(toolbar_frame, text="Modo Círculo", command=lambda: self.set_mode('circle'))
        circle_btn.pack(side=tk.LEFT, padx=5)

        rect_btn = ttk.Button(toolbar_frame, text="Modo Retângulo", command=lambda: self.set_mode('rectangle'))
        rect_btn.pack(side=tk.LEFT, padx=5)

        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg='black')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky='ns')

        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky='ew')

        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind_all("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind_all("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        middle_frame = ttk.Frame(self.main_frame, width=250)
        middle_frame.pack(side=tk.RIGHT, fill=tk.Y)
        middle_frame.pack_propagate(False)

        self.listbox = tk.Listbox(middle_frame)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.listbox.bind("<Button-1>", self.on_listbox_click)
        self.listbox.bind("<Double-Button-1>", self.on_listbox_double_click)

        preview_frame = ttk.LabelFrame(self.main_frame, text="Histórico de Recortes", width=120)
        preview_frame.pack(side=tk.RIGHT, fill=tk.Y)
        preview_frame.pack_propagate(False)

        self.template_panel = tk.Canvas(preview_frame, bg='white', width=120)
        self.template_panel.pack(fill=tk.BOTH, expand=True)
        self.template_panel.bind("<Button-1>", self.on_template_click)

    def set_mode(self, mode):
        self.mode = mode
        self.mode_label.config(text=f"Modo: {'Círculo' if mode == 'circle' else 'Retângulo'}")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.jpg *.png *.bmp")])
        if not path:
            return

        self.original_img = cv2.imread(path)
        if self.original_img is None:
            print("Erro ao carregar a imagem.")
            return

        self.zoom = 1.0
        self.update_image_with_zoom()
        self.apply_template_history()  # <- ADICIONE ESSA LINHA AQUI
        self.load_template_history_from_folder()  # <- ADICIONADO AQUI

    def update_image_with_zoom(self):
        if self.original_img is None:
            return
        h, w = self.original_img.shape[:2]
        new_w, new_h = int(w * self.zoom), int(h * self.zoom)
        resized = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.img = resized.copy()
        self.clone = self.img.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.draw_detections()
        self.show_image_on_canvas()

    def show_image_on_canvas(self):
        if self.img is None:
            return

        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        pil_img = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_detections(self):
        self.img[:] = self.clone.copy()
        for tipo, data in self.accepted_detections:
            if tipo == 'circle':
                cx, cy, r = [int(v * self.zoom) for v in data]
                cv2.circle(self.img, (cx, cy), r, (0, 255, 0), 2)
            elif tipo == 'rect':
                x1, y1, x2, y2 = [int(v * self.zoom) for v in data]
                cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for i, (tipo, data) in enumerate(self.detections):
            if (tipo, data) in self.accepted_detections:
                continue

            color = (0, 0, 255)
            if self.selected_index == i:
                color = (0, 255, 255)

            if tipo == 'circle':
                cx, cy, r = [int(v * self.zoom) for v in data]
                cv2.circle(self.img, (cx, cy), r, color, 2)
            elif tipo == 'rect':
                x1, y1, x2, y2 = [int(v * self.zoom) for v in data]
                cv2.rectangle(self.img, (x1, y1), (x2, y2), color, 2)

    def update_detection_list(self):
        self.listbox.delete(0, tk.END)
        if self.original_img is None:
            return
        img_h, img_w = self.original_img.shape[:2]
        for i, (tipo, data) in enumerate(self.detections):
            if tipo == 'circle':
                cx, cy, r = data
                x1, y1 = cx - r, cy - r
                x2, y2 = cx + r, cy + r
            else:
                x1, y1, x2, y2 = data

            x_center = (x1 + x2) / 2.0 / img_w
            y_center = (y1 + y2) / 2.0 / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h
            
            desc = f"{i}: {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            self.listbox.insert(tk.END, desc)

    def add_template_to_history(self, template_img):
        if len(self.template_history) >= 10:
            self.template_history.pop(0)

        self.template_history.append(template_img)

        # Criar pasta "recortes" se não existir
        os.makedirs("recortes", exist_ok=True)

        # Salvar imagem como historyN.png
        index = len(self.template_history)
        save_path = os.path.join("recortes", f"history{index}.png")

        # Converter para BGR se for grayscale (para evitar erro ao salvar)
        if len(template_img.shape) == 2:
            to_save = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
        else:
            to_save = template_img.copy()

        cv2.imwrite(save_path, to_save)

        self.refresh_template_gallery()

    def refresh_template_gallery(self):
        self.template_panel.delete("all")
        self.template_thumbnails.clear()
        for i, template in enumerate(self.template_history):
            # Garantir que a imagem tenha 3 canais (RGB)
            if len(template.shape) == 2:
                # Grayscale -> RGB
                template_rgb = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB)
            else:
                template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

            thumb = cv2.resize(template_rgb, (60, 60))
            img_pil = Image.fromarray(thumb)
            img_tk = ImageTk.PhotoImage(img_pil)
            self.template_panel.create_image(0, i * 60, anchor=tk.NW, image=img_tk)
            self.template_thumbnails.append(img_tk)  # precisa manter referência para não sumir



    def apply_template_history(self):
        if self.original_img is None or not self.template_history:
            return

        self.detections.clear()  # limpa detecções anteriores

        self.gray = cv2.cvtColor(self.clone.copy(), cv2.COLOR_BGR2GRAY)
        scale = 1.0 / self.zoom

        for template in self.template_history:
            result = cv2.matchTemplate(self.gray, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= 0.7)
            added_points = []

            for pt in zip(*loc[::-1]):
                skip = False
                for added in added_points:
                    if np.linalg.norm(np.array(pt) - np.array(added)) < 10:
                        skip = True
                        break
                if skip:
                    continue

                added_points.append(pt)

                x, y = pt
                w, h = template.shape[::-1]
                x1, y1 = int(x * scale), int(y * scale)
                x2, y2 = int((x + w) * scale), int((y + h) * scale)
                self.detections.append(('rect', (x1, y1, x2, y2)))

        self.update_image_with_zoom()
        self.update_detection_list()

    def load_template_history_from_folder(self):
        self.template_history.clear()

        if not os.path.exists("recortes"):
            return

        arquivos = sorted(os.listdir("recortes"))
        for nome in arquivos:
            if nome.lower().endswith((".png", ".jpg", ".bmp")):
                caminho = os.path.join("recortes", nome)
                img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.template_history.append(img)

        # ADICIONE ESTA LINHA:
        self.refresh_template_gallery()

    def apply_single_template(self, template):
        self.detections = [det for det in self.detections if det in self.accepted_detections]

        if self.original_img is None:
            return

        # converter imagem base para grayscale
        gray = cv2.cvtColor(self.clone.copy(), cv2.COLOR_BGR2GRAY)

        # garantir que template seja grayscale 2D
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # garantir que template e gray tenham o mesmo tipo
        if template.dtype != gray.dtype:
            template = template.astype(gray.dtype)

        # redimensionar o template para o zoom atual da imagem
        scale = self.zoom
        if scale != 1.0:
            new_w = int(template.shape[1] * scale)
            new_h = int(template.shape[0] * scale)
            if new_w > 0 and new_h > 0:
                template_resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                template_resized = template.copy()
        else:
            template_resized = template.copy()

        # executar a correspondência de template usando o template redimensionado
        result = cv2.matchTemplate(gray, template_resized, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.7)

        # continue o resto do seu código normalmente,
        # mas use template_resized para pegar as dimensões:
        added_points = []

        for pt in zip(*loc[::-1]):
            # ... seu código para filtro, adição e detecção

            x, y = pt
            h, w = template_resized.shape[:2]  # usar o redimensionado
            x1, y1 = int(x / scale), int(y / scale)
            x2, y2 = int((x + w) / scale), int((y + h) / scale)
            self.detections.append(('rect', (x1, y1, x2, y2)))

        self.update_image_with_zoom()
        self.update_detection_list()

        
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToolApp(root)
    root.mainloop()
