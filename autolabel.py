import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

class ImageToolApp:
    def __init__(self, root):
        self.detections = []  # (tipo, coord) => tipo: 'circle' ou 'rect', coord: tupla
        self.selected_index = None         # índice temporariamente destacado (1 clique)
        self.accepted_detections = []  
        self.root = root
        self.root.title("Detector de Padrões")
        root.geometry("1000x700")
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
        self.photo = None  # para manter referência da imagem PhotoImage

        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Frame do canvas com scrollbars
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas para imagem
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg='black')
        self.canvas.grid(row=0, column=0, sticky='nsew')

        # Scrollbars vertical e horizontal
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky='ns')

        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky='ew')

        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # Configurar grid do frame para expandir canvas
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        # Bind eventos
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)  # Windows/Mac
        self.canvas.bind_all("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind_all("<Button-5>", self.on_mouse_wheel)    # Linux scroll down

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # Barra lateral fixa (largura fixa)
        # Frame intermediário para lista de detecções
        middle_frame = ttk.Frame(self.main_frame, width=250)
        middle_frame.pack(side=tk.RIGHT, fill=tk.Y)
        middle_frame.pack_propagate(False)

        self.listbox = tk.Listbox(middle_frame)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame lateral para botões e modo
        side_frame = ttk.Frame(self.main_frame, width=220)
        side_frame.pack(side=tk.RIGHT, fill=tk.Y)
        side_frame.pack_propagate(False)

        load_btn = ttk.Button(side_frame, text="Carregar Imagem", command=self.load_image)
        load_btn.pack(pady=10)

        self.mode_label = ttk.Label(side_frame, text="Modo: Círculo")
        self.mode_label.pack(pady=10)

        circle_btn = ttk.Button(side_frame, text="Modo Círculo", command=lambda: self.set_mode('circle'))
        circle_btn.pack(pady=5)

        rect_btn = ttk.Button(side_frame, text="Modo Retângulo", command=lambda: self.set_mode('rectangle'))
        rect_btn.pack(pady=5)

        side_frame.pack_propagate(False)  # para manter largura fixa

        self.listbox.bind("<Button-1>", self.on_listbox_click)
        self.listbox.bind("<Double-Button-1>", self.on_listbox_double_click)
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
        # Redimensiona imagem com zoom = 1 (original)
        self.update_image_with_zoom()

    def update_image_with_zoom(self):
        if self.original_img is None:
            return
        h, w = self.original_img.shape[:2]
        new_w, new_h = int(w * self.zoom), int(h * self.zoom)
        resized = cv2.resize(self.original_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        self.img = resized.copy()
        self.clone = self.img.copy()
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.draw_detections()  # <- Adiciona isso
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

    def on_mouse_down(self, event):
        if self.img is None:
            return
        self.drawing = True
        # Considerar scroll offset para coordenadas reais no canvas
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        if self.mode == 'circle':
            self.center = (x, y)
        elif self.mode == 'rectangle':
            self.start_point = (x, y)

    def on_mouse_drag(self, event):
        self.detections = [det for det in self.detections if det in self.accepted_detections]
        # self.detections.clear()  # Limpa detecções ao arrastar
        if not self.drawing or self.img is None:
            return
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        self.img[:] = self.clone.copy()

        if self.mode == 'circle':
            self.radius = int(np.hypot(x - self.center[0], y - self.center[1]))
            cv2.circle(self.img, self.center, self.radius, (0, 255, 0), 1)
        elif self.mode == 'rectangle':
            self.end_point = (x, y)
            cv2.rectangle(self.img, self.start_point, self.end_point, (0, 255, 0), 1)

        self.show_image_on_canvas()

    def on_mouse_up(self, event):
        if not self.drawing or self.img is None:
            return
        self.drawing = False
        x = int(self.canvas.canvasx(event.x))
        y = int(self.canvas.canvasy(event.y))

        self.img[:] = self.clone.copy()

        if self.mode == 'circle':
            self.radius = int(np.hypot(x - self.center[0], y - self.center[1]))
            x1, y1 = self.center[0] - self.radius, self.center[1] - self.radius
            x2, y2 = self.center[0] + self.radius, self.center[1] + self.radius
        else:
            x1, y1 = self.start_point
            x2, y2 = x, y
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
            self.radius = max((x2 - x1) // 2, (y2 - y1) // 2)

        # Verifica se coordenadas estão dentro da imagem para evitar erro
        h, w = self.gray.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))

        if x2 <= x1 or y2 <= y1:
            print("Seleção inválida.")
            self.show_image_on_canvas()
            return

        template = self.gray[y1:y2, x1:x2]
        result = cv2.matchTemplate(self.gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where(result >= threshold)

        scale = 1.0 / self.zoom
        min_dist = 10  # distância mínima entre detecções similares
        added_points = []

        for pt in zip(*loc[::-1]):
            
            skip = False
            for added in added_points:
                if np.linalg.norm(np.array(pt) - np.array(added)) < min_dist:
                    skip = True
                    break
            if skip:
                continue

            # Verificar se já foi aceito algo próximo
            for tipo, data in self.accepted_detections:
                if self.mode == 'circle' and tipo == 'circle':
                    acx, acy, ar = [int(v * self.zoom) for v in data]
                    ax1, ay1 = acx - ar, acy - ar
                    ax2, ay2 = acx + ar, acy + ar

                    rx1, ry1 = pt
                    rx2, ry2 = pt[0] + template.shape[1], pt[1] + template.shape[0]

                    overlap_x = max(0, min(ax2, rx2) - max(ax1, rx1))
                    overlap_y = max(0, min(ay2, ry2) - max(ay1, ry1))
                    if overlap_x * overlap_y > 0:
                        skip = True
                        break
                elif self.mode == 'rectangle' and tipo == 'rect':
                    ax1, ay1, ax2, ay2 = [int(v * self.zoom) for v in data]
                    rx1, ry1 = pt
                    rx2, ry2 = pt[0] + template.shape[1], pt[1] + template.shape[0]
                    overlap_x = max(0, min(ax2, rx2) - max(ax1, rx1))
                    overlap_y = max(0, min(ay2, ry2) - max(ay1, ry1))
                    if overlap_x * overlap_y > 0:
                        skip = True
                        break
            if skip:
                continue
            added_points.append(pt)

            if self.mode == 'circle':
                cx = int((pt[0] + template.shape[1] // 2) * scale)
                cy = int((pt[1] + template.shape[0] // 2) * scale)
                r = int(max(template.shape[0], template.shape[1]) // 2 * scale)
                self.detections.append(('circle', (cx, cy, r)))
            else:
                x1, y1 = int(pt[0] * scale), int(pt[1] * scale)
                x2, y2 = int((pt[0] + template.shape[1]) * scale), int((pt[1] + template.shape[0]) * scale)
                self.detections.append(('rect', (x1, y1, x2, y2)))

        # Desenhar com base nas detecções salvas
        self.draw_detections()
        cv2.imwrite("padroes_detectados.png", self.img)
        print(f"[✔] Detecções marcadas com {self.mode} e salvas como 'padroes_detectados.png'")
        self.show_image_on_canvas()
        self.update_detection_list()
            
    def on_mouse_wheel(self, event):
        ctrl = (event.state & 0x0004) != 0   # Ctrl pressionado
        shift = (event.state & 0x0001) != 0  # Shift pressionado

        if ctrl:
            if event.num == 4 or event.delta > 0:
                self.zoom = min(self.zoom + self.zoom_step, self.max_zoom)
            elif event.num == 5 or event.delta < 0:
                self.zoom = max(self.zoom - self.zoom_step, self.min_zoom)
            self.update_image_with_zoom()
        elif shift:
            if event.num == 4 or event.delta > 0:
                self.canvas.xview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                self.canvas.xview_scroll(1, "units")
        else:
            if event.num == 4 or event.delta > 0:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                self.canvas.yview_scroll(1, "units")


    def draw_detections(self):
        self.img[:] = self.clone.copy()

        # Desenha primeiro os aceitos (verde)
        for tipo, data in self.accepted_detections:
            if tipo == 'circle':
                cx, cy, r = [int(v * self.zoom) for v in data]
                cv2.circle(self.img, (cx, cy), r, (0, 255, 0), 2)
            elif tipo == 'rect':
                x1, y1, x2, y2 = [int(v * self.zoom) for v in data]
                cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Agora os demais (vermelho), exceto os aceitos
        for i, (tipo, data) in enumerate(self.detections):
            if (tipo, data) in self.accepted_detections:
                continue  # já desenhado como verde

            color = (0, 0, 255)  # vermelho
            if self.selected_index == i:
                color = (0, 255, 255)  # amarelo

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
            else:  # 'rect'
                x1, y1, x2, y2 = data

            # Centro, largura e altura
            x_center = (x1 + x2) / 2.0 / img_w
            y_center = (y1 + y2) / 2.0 / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h

            desc = f"{i}: {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            self.listbox.insert(tk.END, desc)


    def on_listbox_click(self, event):
        widget = event.widget
        index = widget.nearest(event.y)
        if index < 0 or index >= len(self.detections):
            return
        self.selected_index = index
        self.draw_detections()
        self.show_image_on_canvas()

    def on_listbox_double_click(self, event):
        widget = event.widget
        index = widget.nearest(event.y)
        if index < 0 or index >= len(self.detections):
            return

        det = self.detections[index]
        if det not in self.accepted_detections:
            self.accepted_detections.append(det)
            print(f"[✔] Detecção {index + 1} aceita permanentemente.")
        self.draw_detections()
        self.show_image_on_canvas()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageToolApp(root)
    root.mainloop()
