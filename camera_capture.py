from ultralytics import YOLO
from collections import defaultdict
import math
from componentes import carregar_componentes, calibrar
import sys, os, cv2
import json
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QRect
# import torch.multiprocessing
import threading
from ler_serial import ler_serial

# torch.multiprocessing.set_start_method('spawn', force=True)
import sys
import os
os.environ["GALAXY_GENICAM_ROOT"] = r"C:\\Program Files\\Daheng Imaging\\GalaxySDK\\GenICam"
import cv2
import gxipy as gx
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from skimage.metrics import structural_similarity as ssim
import os


serial = "Awaiting..."
Passed = "Awaiting..."
lista = ['BOSA_BOT1', 'BOSA_4P_BOT1']

def aprender(model):
    model.train(
        data=r'dataset.yaml',
        epochs=100,
        imgsz=2600,
        batch=4,
        device='0',
        workers=0,
        freeze=10,
        lr0=0.0005
    )

model = YOLO(r'C:\Users\mayconcosta\yolo-V8\runs\detect\train33\weights\best.pt')
# aprender(model)
componentes_esperados = carregar_componentes()


def centro_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)
def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


imagem_entrada = r'images\padrao2.jpg'
imagem_saida = r'images\padrao2-saida.jpg'


aprendizado = False
imagem_atual = None
class Principal(QMainWindow):
    def __init__(self):
        global imagem_saida
        super().__init__()
        self.exibir_macara = False
        self.exibir_resultado = True
        self.janelas_crops_abertas = set()

        self.setWindowTitle("Visualizador de Componentes")
        self.resize(1400, 800)
    
        self.caminho_json = "setup_componentes.json"

        layout_principal = QHBoxLayout()
        layout_imagem_botoes = QVBoxLayout()

        self.labelserial = QLabel("")
        self.labelserial.setMinimumHeight(30)
        self.labelserial.setMaximumHeight(30)
        self.labelserial.setStyleSheet("font-size: 30px;")

        self.label_imagem = QLabel()
        self.label_imagem.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_imagem.setAlignment(Qt.AlignCenter)

        layout_imagem_botoes.addWidget(self.labelserial)
        layout_imagem_botoes.addWidget(self.label_imagem)

        layout_botoes = QHBoxLayout()

        btn1 = QPushButton("INSPECIONAR")
        btn1.setMinimumHeight(80)
        self.btn1 = btn1

        btn2 = QPushButton("VOLTAR")
        btn2.setMinimumHeight(80)
        self.btn2 = btn2
        self.btn2.clicked.connect(self.recarregarimagem)

        btnmask = QPushButton("DEFINIR MASCARA PADRAO")
        btnmask.setMinimumHeight(80)
        self.btnmask = btnmask
        self.btnmask.clicked.connect(self.definirmask)

        layout_botoes.addWidget(btn1)
        layout_botoes.addWidget(btn2)
        layout_botoes.addWidget(btnmask)
        layout_imagem_botoes.addLayout(layout_botoes)

        layout_lista = QVBoxLayout()

        self.lista = QListWidget()
        self.lista.setMaximumWidth(420)
        self.lista.setSpacing(2)

        layout_lista.addWidget(self.lista)

        mover_layout = QHBoxLayout()
        self.combo_direcao = QComboBox()
        self.combo_direcao.addItems(["esquerda", "direita", "cima", "baixo"])
        mover_layout.addWidget(self.combo_direcao)

        btn_mover = QPushButton("Mover")
        btn_mover.clicked.connect(self.mover_componentes_direcao)
        mover_layout.addWidget(btn_mover)

        layout_lista.addLayout(mover_layout)

        btn3 = QPushButton("EXIBIR TODOS COMPONENTES")
        btn3.setMinimumHeight(80)
        self.btn3 = btn3
        # self.btn3.clicked.connect(self.exibir_todos_componentes)
        layout_lista.addWidget(btn3)

        layout_principal.addLayout(layout_imagem_botoes, 70)
        layout_principal.addLayout(layout_lista, 30)

        self.lista.itemClicked.connect(self.item_clique)
        self.lista.itemDoubleClicked.connect(self.item_duplo_clique)

        central = QWidget()
        central.setLayout(layout_principal)
        self.setCentralWidget(central)

        self.componentes = {}
        self.carregar_dados_json()

        # Inicializa câmera Daheng
        self.device_manager = gx.DeviceManager()
        self.device_manager.update_device_list()
        if self.device_manager.get_device_number() == 0:
            raise RuntimeError("Nenhuma câmera Daheng conectada.")

        self.cam = self.device_manager.open_device_by_index(1)
        self.cam.stream_on()

        self.timer = QTimer()
        self.timer.timeout.connect(self.atualizar_frame)
        self.timer.start(30)


    def mover_componentes_direcao(self):
        direcao = self.combo_direcao.currentText()
        deslocamento =5

        for nome, info in self.componentes.items():
            pos = info.get("posicoes", [[0, 0]])[0]

            if direcao == "direita":
                pos[0] += deslocamento
            elif direcao == "esquerda":
                print(pos)
                pos[0] -= deslocamento
                print(f"Movendo {nome} para a esquerda")
            elif direcao == "cima":
                pos[1] -= deslocamento
            elif direcao == "baixo":
                pos[1] += deslocamento

            info["posicoes"][0] = pos

        self.salvar_componentes()
        # self.exibir_todos_componentes()

    def carregar_dados_json(self):
        self.lista.clear()
        try:
            with open(self.caminho_json, 'r') as f:
                self.componentes = json.load(f)

            for nome, info in self.componentes.items():
                classe = info.get("classe", "")
                pos = info.get("posicoes", [[]])[0]
                item = QListWidgetItem(f"{nome} - Classe: {classe} \n x: {pos[0]}, y: {pos[1]} \n")
                item.setData(Qt.UserRole, nome)
                self.lista.addItem(item)
        except Exception as e:
            self.lista.addItem(f"Erro ao carregar JSON: {str(e)}")


    def item_duplo_clique(self, item):
        chave_antiga = item.data(Qt.UserRole)
        info = self.componentes.get(chave_antiga, {})
        if not info:
            return

        novo_nome, ok0 = QInputDialog.getText(self, "Editar Nome do Componente", "Nome do Componente:", text=chave_antiga)
        if ok0 and novo_nome and novo_nome != chave_antiga:
            if novo_nome in self.componentes:
                QMessageBox.warning(self, "Erro", f"O nome '{novo_nome}' já existe.")
                return
            self.componentes[novo_nome] = self.componentes.pop(chave_antiga)
            chave_antiga = novo_nome  # Atualiza para editar os demais campos
            info = self.componentes[novo_nome]

        self.salvar_componentes()
        self.carregar_dados_json()


    def item_clique(self, item):
        chave_antiga = item.data(Qt.UserRole)
        info = self.componentes.get(chave_antiga, {})
        if not info:
            return
        pos = info.get("posicoes", [[0, 0]])[0]
        tam = info.get("tamanho", [100, 100])
        # self.desenhar_retangulo(pos, tam)
            
    def definirmask(self):
        global componentes_esperados
        calibrar(model, imagem_entrada)
        self.carregar_dados_json()
        componentes_esperados = carregar_componentes()
        QMessageBox.information(self, "Done", f"Nova Mascara de teste definida")
        

    def atualizar_frame(self):
        global imagem_entrada
        global imagem_atual
        raw_image = self.cam.data_stream[0].get_image(timeout=100)
        
        if raw_image is None:
            return
        rgb_image = raw_image.convert("RGB").get_numpy_array()

        imagem_atual = rgb_image.copy()
        imagem_entrada = rgb_image.copy()
        # Máscara padrão
        if self.exibir_macara:
            for nome, info in self.componentes.items():
                pos = info.get("posicoes", [[0, 0]])[0]
                tam = info.get("tamanho", [100, 100])
                x, y = pos
                w, h = tam
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = x1 + w
                y2 = y1 + h

                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_image, nome, (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if self.exibir_resultado:
            self.desenhar_resultado_ia_em_tempo_real(rgb_image)

        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_imagem.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.label_imagem.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def recarregarimagem(self):
        self.exibir_macara = False
            
    def desenhar_retangulo(self, centro, tamanho):
        global imagem_atual
        x, y = centro
        w, h = tamanho
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        
        imagem_copia = imagem_atual.copy()

        painter = QPainter(imagem_copia)
        pen = QPen(Qt.green, 5)
        painter.setPen(pen)
        painter.drawRect(QRect(x1, y1, w, h))
        painter.end()

        self.pixmap_exibida = imagem_copia.scaledToWidth(980, Qt.SmoothTransformation)
        self.label_imagem.setPixmap(self.pixmap_exibida)

    def salvar_componentes(self):
        try:
            with open(self.caminho_json, 'w') as f:
                json.dump(self.componentes, f, indent=4)
            print("Componentes salvos com sucesso.")
        except Exception as e:
            QMessageBox.warning(self, "Erro", f"Falha ao salvar JSON: {str(e)}")


    def closeEvent(self, event):
        self.timer.stop()
        self.cam.stream_off()
        self.cam.close_device()
        event.accept()

    def desenhar_resultado_ia_em_tempo_real(self, img):
        global serial, componentes_esperados
        imagem_sem_anotacoes = img.copy()
        detectou = False
        contagem_classes = defaultdict(int)

        results = model.predict(img, conf=0.6)[0]
        boxes = results.boxes.xyxy
        centros_detectados = [centro_box(box) for box in boxes]

        detectados_nome = []
        # Desenhar objetos detectados pela IA
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            detectou = True
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = model.names[class_id]
            conf_pct = int(conf * 100)

            contagem_classes[label] += 1

            # Identificação do nome do componente detectado (ex: BOSA_BOT1, BOSA_4P_BOT1)
            for nome, info in componentes_esperados.items():
                if info["classe"] == label:
                    pos = info["posicoes"][0]
                    if distancia(pos, centro_box(box)) < 30:
                        detectados_nome.append(nome)

            if label == "DATAMATRIX":
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                texto = f"{label} {conf_pct}%"
                cv2.putText(img, texto, (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imwrite("serial.png", img[y1:y2, x1:x2])
                recorte = imagem_sem_anotacoes[y1:y2, x1:x2]
                if self.labelserial.text() == "Awaiting...":
                    serial = ler_serial(recorte)
                    print(f"Serial lido: {serial}")
                    self.labelserial.setText(f"Serial: {serial}")
                elif self.labelserial.text() == f"":
                    serial = ler_serial(recorte)
                    print(f"Serial lido: {serial}")
                    self.labelserial.setText(f"Serial: {serial}")
                elif self.labelserial.text() == f"Serial: ":
                    serial = ler_serial(recorte)
                    print(f"Serial lido: {serial}")
                    self.labelserial.setText(f"Serial: {serial}")
                else:
                    print(f"Serial ja lido")
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                texto = f"{label} {conf_pct}%"
                cv2.putText(img, texto, (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Desenhar componentes esperados faltando
        y_offset = 60
        faltando = []
        for nome, info in componentes_esperados.items():
            classe = info["classe"]
            posicoes = info["posicoes"]
            largura, altura = info.get("tamanho", [150, 120])
            achou = any(distancia(posicoes[0], d) < 30 for d in centros_detectados)

            if not achou:
                x, y = posicoes[0]
                falt_x1 = int(x - largura // 2)
                falt_y1 = int(y - altura // 2)
                falt_x2 = int(x + largura // 2)
                falt_y2 = int(y + altura // 2)
                cv2.rectangle(img, (falt_x1, falt_y1), (falt_x2, falt_y2), (255, 0, 0), 2)
                texto_faltando = f"{nome}"
                cv2.putText(img, texto_faltando, (falt_x1, falt_y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                faltando.append(nome)
            else:
                texto_contagem = f"{nome} ({classe}): {contagem_classes.get(classe, 0)} un."
                cv2.putText(img, texto_contagem, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                y_offset += 30

        if not detectou:
            cv2.putText(img, "Nenhum objeto detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            self.labelserial.setText("Awaiting...")

        # Se todos os componentes esperados foram detectados, faz o segundo teste
        if len(faltando) == 0:
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                class_id = int(cls)
                label = model.names[class_id]
                # Procura pelo nome do componente correspondente
                for nome, info in componentes_esperados.items():
                    if info["classe"] == label and nome in lista:
                        pos = info["posicoes"][0]
                        if distancia(pos, centro_box(box)) < 50:
                            # Crop em tempo real
                            x1, y1, x2, y2 = map(int, box)
                            crop_realtime = imagem_sem_anotacoes[y1:y2, x1:x2]
                            # Caminho do crop salvo
                            crop_path = os.path.join("componentes", nome, f"{nome}.jpg")
                            if os.path.exists(crop_path):
                                crop_salvo = cv2.imread(crop_path)
                                # Redimensiona para o mesmo tamanho
                                if crop_salvo is not None and crop_realtime.shape[:2] == crop_salvo.shape[:2]:
                                    crop_salvo_resized = crop_salvo
                                    crop_realtime_resized = crop_realtime
                                    # salvar crop salvo redimensionado
                                else:
                                    crop_salvo_resized = cv2.resize(crop_salvo, (crop_realtime.shape[1], crop_realtime.shape[0]))
                                    crop_realtime_resized = crop_realtime
                                # Converte para cinza
                                crop_salvo_gray = cv2.cvtColor(crop_salvo_resized, cv2.COLOR_BGR2GRAY)
                                crop_realtime_gray = cv2.cvtColor(crop_realtime_resized, cv2.COLOR_BGR2GRAY)
                                # salvar iamgem cropada em tempo real
                                cv2.imwrite(fr"debug\{nome}_crop_default.jpg", crop_salvo_gray)
                                cv2.imwrite(fr"debug\{nome}_crop_realtime.jpg", crop_realtime_gray)

                                # score, _ = ssim(crop_salvo_gray, crop_realtime_gray, full=True)
                                score, _ = ssim(crop_salvo_resized, crop_realtime_resized, channel_axis=-1, full=True)
                                comparacao = self.comparar(crop_salvo_resized,crop_realtime_resized, borda=0.1, area_min=100)
                                texto_ssim = f"SSIM {nome}: {score:.2f}"

                                logs_dir = "Logs"
                                os.makedirs(logs_dir, exist_ok=True)
                                log_file = os.path.join(logs_dir, f"{nome}.txt")
                                    
                                with open(log_file, "w", encoding="utf-8") as f:
                                    f.write(f"{score:.2f}")

                            if comparacao != None:
                                cv2.putText(img, texto_ssim, (x1, y1 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                                janela_nome = f"Crop em tempo real - {nome}"
                                try:
                                    if janela_nome not in self.janelas_crops_abertas:
                                        cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
                                        self.janelas_crops_abertas.add(janela_nome)

                                    cv2.imshow(janela_nome, comparacao)
                                    cv2.resizeWindow(janela_nome, 600, 400)

                                    if cv2.getWindowProperty(janela_nome, cv2.WND_PROP_VISIBLE) < 1:
                                        self.janelas_crops_abertas.discard(janela_nome)
                                except cv2.error as e:
                                    print(f"Erro ao abrir janela para {nome}: {e}")
                                    self.janelas_crops_abertas.discard(janela_nome)
                                else:
                                    cv2.putText(img, texto_ssim, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                                

                            else:
                                cv2.putText(img, f"Crop salvo não encontrado para {nome}", (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    
    def comparar(self, img_boa,img_comparar,borda,area_min):

        if img_boa is None or img_comparar is None:
            print("Erro ao carregar imagens.")
            return

        # Redimensiona para o mesmo tamanho
        if img_boa.shape != img_comparar.shape:
            img_comparar = cv2.resize(img_comparar, (img_boa.shape[1], img_boa.shape[0]))

        h, w = img_boa.shape[:2]

        # Define região central (desconsidera bordas)
        x_start = int(w * borda)
        y_start = int(h * borda)
        x_end = int(w * (1 - borda))
        y_end = int(h * (1 - borda))

        # Recorte da área central no CANAL VERDE
        canal_boa = img_boa[y_start:y_end, x_start:x_end, 1]
        canal_ruim = img_comparar[y_start:y_end, x_start:x_end, 1]

        # Suavização com filtro gaussiano
        canal_boa = cv2.GaussianBlur(canal_boa, (3, 3), 0)
        canal_ruim = cv2.GaussianBlur(canal_ruim, (3, 3), 0)

        # Diferença absoluta
        diff = cv2.absdiff(canal_boa, canal_ruim)

        media = np.mean(diff)
        _, diff_thresh = cv2.threshold(diff, media + 10, 255, cv2.THRESH_BINARY)

        # Remoção de ruídos pequenos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=1)

        # Contornos
        contornos, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_resultado = img_comparar.copy()
        diferencas = 0

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area >= area_min:
                # Critérios extras: aspecto e solidez
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = w_box / float(h_box) if h_box != 0 else 0
                hull = cv2.convexHull(cnt)
                solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

                if 0.2 < aspect_ratio < 5.0 and solidity > 0.3:
                    cnt += np.array([[[x_start, y_start]]])  # volta para coordenada original
                    cv2.drawContours(img_resultado, [cnt], -1, (0, 0, 255), 2)
                    diferencas += 1

        print(f"diferenca(s) detectada(s): {diferencas}")
        if diferencas > 0:
            empilhada = np.hstack([img_boa, img_resultado])
            return empilhada

        else:
            return None

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = Principal()
    janela.showMaximized()
    sys.exit(app.exec_())
