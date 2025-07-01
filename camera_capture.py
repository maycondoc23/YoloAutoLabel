from ultralytics import YOLO
import cv2
from collections import defaultdict
import math
from componentes import carregar_componentes, calibrar
import sys, os
import json
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QRect
import torch.multiprocessing
from ler_serial import ler_serial
torch.multiprocessing.set_start_method('spawn', force=True)
import sys
import os
os.environ["GALAXY_GENICAM_ROOT"] = r"C:\\Program Files\\Daheng Imaging\\GalaxySDK\\GenICam"
import cv2
import gxipy as gx
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QComboBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import json


serial = "Awaiting..."
Passed = "Awaiting..."

def aprender(model):
    model.train(
        data=r'dataset.yaml',
        epochs=100,
        imgsz=2600,
        batch=4,
        device='0',
        workers=0,
        # freeze=10,
        # lr0=0.0005
    )

model = YOLO(r'C:\Users\mayconcosta\yolo-V8\runs\detect\train27\weights\best.pt')
aprender(model)
componentes_esperados = carregar_componentes()


def centro_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)
def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


imagem_entrada = r'images\padrao2.jpg'
imagem_saida = r'images\padrao2-saida.jpg'


aprendizado = False

class Principal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.exibir_macara = False
        self.exibir_resultado = True
        self.setWindowTitle("Visualizador de Componentes")
        self.resize(1400, 800)

        self.caminho_json = "setup_componentes.json"

        layout_principal = QHBoxLayout()
        layout_imagem_botoes = QVBoxLayout()

        self.labelserial = QLabel("Serial: ")
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
        self.btn1.clicked.connect(self.salvar_imagem)

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
        self.btn3.clicked.connect(self.exibir_todos_componentes)
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

        self.imagem_atual = None

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
        self.exibir_todos_componentes()

    def carregar_dados_json(self):
        self.lista.clear()
        try:
            with open(self.caminho_json, 'r') as f:
                self.componentes = json.load(f)

            for nome, info in self.componentes.items():
                classe = info.get("classe", "")
                pos = info.get("posicoes", [[]])[0]
                tam = info.get("tamanho", [])
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

        pos = info.get("posicoes", [[0, 0]])[0]
        tam = info.get("tamanho", [100, 100])

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
        self.desenhar_retangulo(pos, tam)


    def exibir_todos_componentes(self):
        self.exibir_macara = True
        
        # if self.imagem_atual is None:
        #     return

        # # Copia da imagem atual da câmera
        # imagem_com_anotacoes = self.imagem_atual.copy()

        # for nome, info in self.componentes.items():
        #     pos = info.get("posicoes", [[0, 0]])[0]
        #     tam = info.get("tamanho", [100, 100])

        #     x, y = pos
        #     w, h = tam
        #     x1 = int(x - w / 2)
        #     y1 = int(y - h / 2)
        #     x2 = x1 + w
        #     y2 = y1 + h

        #     # Desenha o retângulo e o nome
        #     cv2.rectangle(imagem_com_anotacoes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(imagem_com_anotacoes, nome, (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # # Atualiza a imagem exibida com anotações
        # h, w, ch = imagem_com_anotacoes.shape
        # bytes_per_line = ch * w
        # qimg = QImage(imagem_com_anotacoes.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # self.label_imagem.setPixmap(QPixmap.fromImage(qimg).scaled(
        #     self.label_imagem.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def inspecaor_imagem(self):
        if self.inspecionar_salvar(imagem_entrada, imagem_saida):
            self.labelserial.setText(f"Serial: {serial}")
            print(f"Objeto(s) detectado(s). Resultado salvo em {imagem_saida}")
            self.pixmap_original = QPixmap("images/padrao2-saida.jpg")
            self.pixmap_exibida = self.pixmap_original.scaledToWidth(980, Qt.SmoothTransformation)
            self.label_imagem.setPixmap(self.pixmap_exibida)

        else:
            print("Nenhum objeto detectado.")
            
    def definirmask(self):
        global componentes_esperados
        calibrar(model, imagem_entrada)
        self.carregar_dados_json()
        componentes_esperados = carregar_componentes()
        QMessageBox.information(self, "Done", f"Nova Mascara de teste definida")
        

    def atualizar_frame(self):
        global imagem_entrada
        raw_image = self.cam.data_stream[0].get_image(timeout=100)
        

        if raw_image is None:
            return
        rgb_image = raw_image.convert("RGB").get_numpy_array()

        self.imagem_atual = rgb_image.copy()
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
                cv2.putText(rgb_image, nome, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # IA ativa
        if self.exibir_resultado:
            self.desenhar_resultado_ia_em_tempo_real(rgb_image)

        # Exibe imagem na interface
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label_imagem.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.label_imagem.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def recarregarimagem(self):
        self.exibir_macara = False
        # self.pixmap_original = QPixmap(self.caminho_imagem)
        # self.pixmap_exibida = self.pixmap_original.scaledToWidth(980, Qt.SmoothTransformation)
        # self.label_imagem.setPixmap(self.pixmap_exibida)
            
    def desenhar_retangulo(self, centro, tamanho):
        x, y = centro
        w, h = tamanho
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        
        imagem_copia = self.imagem_atual.copy()

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


    def inspecionar_salvar(self, imagem_path, salvar_path):
        global componentes_esperados, serial
        if not componentes_esperados:
            print("Aviso: componentes_esperados está vazio! Rode calibrar() primeiro.")

        results = model.predict(imagem_path, conf=0.4)[0]
        boxes = results.boxes.xyxy
        centros_detectados = [centro_box(box) for box in boxes]


        # img = cv2.imread(imagem_path)
        img = imagem_path.copy()
        detectou = False
        contagem_classes = defaultdict(int)

        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            detectou = True
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = model.names[class_id]
            conf_pct = int(conf * 100)

            contagem_classes[label] += 1
            cv2.rectangle   (img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            texto = f"{label} {conf_pct}%"
            cv2.putText(img, texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            if label == "DATAMATRIX":
                print(f"Processando {label}")
                serial = ler_serial(img[y1+10:y2+10, x1+10:x2+10])
                print(f"Serial lido: {serial}")
                print(f"Serial lido: {serial}")
                print(f"Serial lido: {serial}")
                print(f"Serial lido: {serial}")
                self.labelserial.setText(f"Serial: {serial}")

                
        y_offset = 80
        for nome, info in componentes_esperados.items():
            classe = info['classe']
            posicoes = info['posicoes']
            largura, altura = info.get("tamanho", [150, 120])
            achou = any(distancia(posicoes[0], d) < 30 for d in centros_detectados)

            if not achou:
                x, y = posicoes[0]
                falt_x1 = int(x - largura // 2)
                falt_y1 = int(y - altura // 2)
                falt_x2 = int(x + largura // 2)
                falt_y2 = int(y + altura // 2)
                cv2.rectangle(img, (falt_x1, falt_y1), (falt_x2, falt_y2), (0, 0, 255), 3)
                texto_faltando = f"{nome}"
                cv2.putText(img, texto_faltando, (falt_x1, falt_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
            else:

                texto_contagem = f"{nome} ({classe}): {contagem_classes.get(classe, 0)} un."
                cv2.putText(img, texto_contagem, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                y_offset += 40

        if not detectou:
            cv2.putText(img, "Nenhum objeto detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imwrite(salvar_path, img)

        if detectou:
            print("\nResumo da inspeção:")
            for classe, qtd in contagem_classes.items():
                print(f"  {classe}: {qtd} unidade(s)")
        else:
            print("Nenhum objeto detectado.")

        return detectou


    def salvar_imagem(self):
        global imagem_entrada
        if self.imagem_atual is not None:
            imagem_bgr = cv2.cvtColor(self.imagem_atual, cv2.COLOR_RGB2BGR)
            imagem_entrada = imagem_bgr
            cv2.imwrite("capture.png", imagem_bgr)
            self.inspecaor_imagem()

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

        results = model.predict(img, conf=0.15)[0]
        boxes = results.boxes.xyxy
        centros_detectados = [centro_box(box) for box in boxes]

        # Desenhar objetos detectados pela IA
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            detectou = True
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            label = model.names[class_id]
            conf_pct = int(conf * 100)

            contagem_classes[label] += 1

            cv2.rectangle(img, (x1+20, y1+20), (x2, y2), (0, 255, 0), 3)
            texto = f"{label} {conf_pct}%"
            cv2.putText(img, texto, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if label == "DATAMATRIX":

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                texto = f"{label} {conf_pct}%"
                cv2.putText(img, texto, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.imwrite("serial.png", img[y1:y2, x1:x2])
                recorte = imagem_sem_anotacoes[y1:y2, x1:x2]
                serial = ler_serial(recorte)
                self.labelserial.setText(f"Serial: {serial}")
            else:

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                texto = f"{label} {conf_pct}%"
                cv2.putText(img, texto, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Desenhar componentes esperados faltando
        y_offset = 60
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
                cv2.putText(img, texto_faltando, (falt_x1, falt_y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                texto_contagem = f"{nome} ({classe}): {contagem_classes.get(classe, 0)} un."
                cv2.putText(img, texto_contagem, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                y_offset += 30

        if not detectou:
            cv2.putText(img, "Nenhum objeto detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            self.labelserial.setText("Awaiting...")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = Principal()
    janela.showMaximized()
    sys.exit(app.exec_())
