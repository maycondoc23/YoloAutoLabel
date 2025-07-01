import os, sys, cv2, numpy as np


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
    threshold = 0.7
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

def on_listbox_click(self, event):
    widget = event.widget
    index = widget.nearest(event.y)
    if index < 0 or index >= len(self.detections):
        return
    self.selected_index = index
    self.draw_detections()
    self.show_image_on_canvas()