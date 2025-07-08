from datetime import datetime

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
from ler_serial import ler_serial

import sys
import os
os.environ["GALAXY_GENICAM_ROOT"] = r"C:\\Program Files\\Daheng Imaging\\GalaxySDK\\GenICam"
import cv2
import gxipy as gx
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from skimage.metrics import structural_similarity as ssim
import os


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

class ItemFrame(QFrame):
    def __init__(self, nome, callback_selecao, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nome = nome
        self.callback_selecao = callback_selecao

    def mousePressEvent(self, event):
        self.callback_selecao(self.nome, self)  # chama a função que lida com a seleção
        super().mousePressEvent(event)

        
class Principal(QMainWindow):
    def __init__(self):
        global imagem_saida
        super().__init__()
        self.lista = []
        self.exibir_macara = False
        self.exibir_resultado = True
        self.janelas_crops_abertas = set()
        self.item_selecionado = None
        self.larg_img = None
        self.lista_passed = []
        self.lista_test = []
        
        for componente, info in componentes_esperados.items():
            if info["Compare"] == True:
                self.lista_test.append(componente)


        self.setWindowTitle("Visualizador de Componentes")
        self.resize(1400, 800)
    
        self.caminho_json = "setup_componentes.json"

        layout_principal = QHBoxLayout()
        self.layout_imagem_botoes = QVBoxLayout()

        self.labelserial = QLabel("Serial: ")
        self.labelserial.setMinimumHeight(50)
        self.labelserial.setMaximumHeight(50)
        self.labelserial.setStyleSheet("font-size: 30px;")

        self.label_imagem = QLabel()
        self.label_imagem.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_imagem.setAlignment(Qt.AlignCenter)

        self.layout_imagem_botoes.addWidget(self.labelserial)
        self.layout_imagem_botoes.addWidget(self.label_imagem)

        layout_botoes = QHBoxLayout()

        # btn1 = QPushButton("INSPECIONAR")
        # btn1.setMinimumHeight(80)
        # self.btn1 = btn1
        # self.btn1.setStyleSheet("background-color: blue; font-weight: bold; color;Black")

        btnmask = QPushButton("DEFINIR MASCARA PADRAO")
        btnmask.setMinimumHeight(30)
        self.btnmask = btnmask
        self.btnmask.clicked.connect(self.definirmask)  

        # layout_botoes.addWidget(btn1)
        self.layout_imagem_botoes.addLayout(layout_botoes)
            
        self.layout_lista_widget = QWidget()
        layout_lista = QVBoxLayout(self.layout_lista_widget)

        # ➕ TÍTULO DO FRAME
        titulo = QLabel("Setup Area")
        titulo.setStyleSheet("font-size: 22px; font-weight: bold; color: white; padding: 5px;")
        layout_lista.addWidget(titulo)

        self.scroll_area = QScrollArea()

        self.scroll_area.setWidgetResizable(True)
        self.lista_widget = QWidget()
        self.lista_layout = QVBoxLayout()
        self.lista_widget.setLayout(self.lista_layout)
        self.scroll_area.setWidget(self.lista_widget)
        layout_lista.addWidget(self.scroll_area)

        mover_layout = QHBoxLayout()
        self.combo_direcao = QComboBox()
        self.combo_direcao.addItems(["esquerda", "direita", "cima", "baixo"])
        mover_layout.addWidget(self.combo_direcao)

        btn_mover = QPushButton("Mover")
        btn_mover.clicked.connect(self.mover_componentes_direcao)
        mover_layout.addWidget(btn_mover)

        layout_lista.addLayout(mover_layout)

        btn3 = QPushButton("EXIBIR TODOS COMPONENTES")
        btn3.setMinimumHeight(30)
        self.btn3 = btn3

        # self.btn3.clicked.connect(self.exibir_todos_componentes)
        layout_lista.addWidget(btnmask)
        layout_lista.addWidget(btn3)

        # Botão para esconder/exibir Setup Area
        self.toggle_btn = QPushButton(">")
        self.toggle_btn.setFixedSize(30, 100)  # Largura = 30, Altura = 100
        self.toggle_btn.setStyleSheet("background-color: blue; font-weight: bold; color;Black")
        self.toggle_btn.clicked.connect(self.toggle_setup_area)

        layout_toggle = QVBoxLayout()
        layout_toggle.addStretch()               # empurra para baixo
        layout_toggle.addWidget(self.toggle_btn) # botão no meio
        layout_toggle.addStretch()   


        layout_principal.addLayout(self.layout_imagem_botoes, 70)
        layout_principal.addLayout(layout_toggle)  # botão entre a imagem e a área de setup
        layout_principal.addWidget(self.layout_lista_widget, 30)
        


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

        # larg_inspect = self.btn1.width()
        # self.btn1.setFixedWidth(larg_inspect)
        
        # larg_img = self.label_imagem.width()
        # self.label_imagem.setFixedWidth(larg_img)
        
    def toggle_Compare(self, nome, botao):
        if nome in self.componentes:
            novo_estado = not self.componentes[nome].get("Compare", False)
            if novo_estado == True:
                self.lista.append(nome)
            else:
                self.lista.remove(nome)
            self.componentes[nome]["Compare"] = novo_estado
            self.atualizar_cor_botao(botao, novo_estado)
            self.salvar_componentes()
            print(self.lista)
            # self.carregar_dados_json()

    def editar_nome_componente(self, nome_antigo):
        if nome_antigo not in self.componentes:
            return

        novo_nome, ok = QInputDialog.getText(self, "Editar Nome do Componente", "Nome do Componente:", text=nome_antigo)

        if ok and novo_nome and novo_nome != nome_antigo:
            if novo_nome in self.componentes:
                QMessageBox.warning(self, "Erro", f"O nome '{novo_nome}' já existe.")
                return
            self.componentes[novo_nome] = self.componentes.pop(nome_antigo)

        # Atualiza o nome também em componentes_esperados, se necessário
        for nome, info in list(componentes_esperados.items()):
            if nome == nome_antigo:
                componentes_esperados[novo_nome] = componentes_esperados.pop(nome_antigo)
                info = componentes_esperados[novo_nome]

        # Pergunta e salva nova área
        if novo_nome in componentes_esperados:
            area = componentes_esperados[novo_nome].get("area", 100)
            nova_area, ok = QInputDialog.getText(self, "Editar Area de Comparação", "Area (Numero Inteiro):", text=str(area))
            if ok and nova_area.isdigit():
                nova_area_int = int(nova_area)
                componentes_esperados[novo_nome]["area"] = nova_area_int
                if novo_nome in self.componentes:
                    self.componentes[novo_nome]["area"] = nova_area_int

        self.salvar_componentes()
        self.carregar_dados_json()
            
    def definirmask(self):
        global componentes_esperados
        calibrar(model, imagem_entrada)
        self.carregar_dados_json()
        componentes_esperados = carregar_componentes()
        for componente, info in componentes_esperados:
            if info["Compare"] == True:
                self.lista_test.append(componente)
        QMessageBox.information(self, "Done", f"Nova Mascara de teste definida")
        
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


    def toggle_setup_area(self):
        if self.layout_lista_widget.isVisible():
            self.label_imagem.setMaximumWidth(10000)
            
            self.layout_lista_widget.hide()
            self.toggle_btn.setText("<")  # mostrar botão para expandir
        else:
            self.label_imagem.setMaximumWidth(int(self.larg_img))
            self.layout_lista_widget.show()
            self.toggle_btn.setText(">")  # mostrar botão para esconder

    def carregar_dados_json(self):

        self.lista.clear()
        if not os.path.exists(self.caminho_json):
            return

        with open(self.caminho_json, 'r') as f:
            self.componentes = json.load(f)

        while self.lista_layout.count():
            item = self.lista_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for nome, dados in self.componentes.items():
            if "Compare" not in dados:
                dados["Compare"] = False  # Adiciona Compare se não existir

            if "area" not in dados:
                dados["area"] = 100  # Adiciona Compare se não existir

            if dados["Compare"] == True:
                self.lista.append(nome)

            

            item_frame = ItemFrame(nome, self.selecionar_item)
            item_layout = QHBoxLayout(item_frame)

            nome_label = QLabel(nome)
            nome_label.setFixedWidth(160)

            btn_short = QPushButton("Comparar")
            btn_short.setCheckable(True)
            btn_short.setChecked(dados["Compare"])
            btn_short.setObjectName(nome)  # Atribui o nome do componente ao botão
            self.atualizar_cor_botao(btn_short, dados["Compare"])
            btn_short.clicked.connect(lambda _, n=nome, b=btn_short: self.toggle_Compare(n, b))

            btn_edit = QPushButton("Editar")
            btn_edit.clicked.connect(lambda _, n=nome: self.editar_nome_componente(n))

            btn_excluir = QPushButton("Excluir")
            btn_excluir.clicked.connect(lambda _, n=nome: self.excluir_componente(n))
            
            item_layout.addWidget(nome_label)
            item_layout.addWidget(btn_short)
            item_layout.addWidget(btn_edit)
            item_layout.addWidget(btn_excluir)

            self.lista_layout.addWidget(item_frame)
        print(self.lista)
        self.salvar_componentes()

        
    def selecionar_item(self, nome, frame):
        # Remove destaque do item anterior, se houver
        if self.item_selecionado:
            self.item_selecionado.setStyleSheet("")
            for btn in self.item_selecionado.findChildren(QPushButton):
                nome_btn = btn.objectName()
                if nome_btn and nome_btn in self.componentes:
                    if self.componentes[nome_btn].get("Compare", False):
                        # Botão 'Comparar' ativo fica verde, com texto preto e bold
                        btn.setStyleSheet("background-color: lightgreen; color: black; font-weight: bold;")
                    else:
                        # Outros botões ficam branco e font normal
                        btn.setStyleSheet("color: white; font-weight: normal; background-color: none;")
                else:
                    btn.setStyleSheet("color: white; font-weight: normal; background-color: none;")
            for label in self.item_selecionado.findChildren(QLabel):
                # Label volta para fonte normal e cor branca
                label.setStyleSheet("color: white; font-weight: normal;")

        # Atualiza o item selecionado
        self.item_selecionado = frame
        frame.setStyleSheet("background-color: lightblue;")

        # Ajusta os botões do item selecionado
        for btn in frame.findChildren(QPushButton):
            if btn.objectName() == nome and btn.isChecked():
                btn.setStyleSheet("background-color: lightgreen; color: black; font-weight: bold;")
            else:
                btn.setStyleSheet("font-weight: bold; background-color: none;")
        
        # Ajusta os labels do item selecionado para bold e preto
        for label in frame.findChildren(QLabel):
            label.setStyleSheet("color: black; font-weight: bold;")

        print(f"Item selecionado: {nome}")


    def limpar_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
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
  
    def excluir_componente(self, nome):
        global componentes_esperados
        if nome in self.componentes:
            resposta = QMessageBox.question(
                self,
                "Confirmar Exclusão",
                f"Tem certeza que deseja excluir o componente '{nome}'?",
                QMessageBox.Yes | QMessageBox.No
            )

            if resposta == QMessageBox.Yes:
                del self.componentes[nome]
                self.salvar_componentes()
                self.carregar_dados_json()
                componentes_esperados = carregar_componentes()


    def atualizar_cor_botao(self, botao, ativado):
        if ativado:
            botao.setStyleSheet("background-color: lightgreen; font-weight: bold; color:black")
        else:
            botao.setStyleSheet("")  # Volta ao estilo padrão

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
        janelas_detectadas_atuais = set()

        if self.larg_img == None:
            self.larg_img = self.label_imagem.width()

        global serial, componentes_esperados
        imagem_sem_anotacoes = img.copy()
        detectou = False
        contagem_classes = defaultdict(int)

        results = model.predict(img, conf=0.65)[0]
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
                    self.lista_passed.clear()
                    serial = ler_serial(recorte)
                    print(f"Serial lido: {serial}")
                    self.labelserial.setText(f"Serial: {serial}")
                elif self.labelserial.text() == f"":
                    self.lista_passed.clear()
                    serial = ler_serial(recorte)
                    print(f"Serial lido: {serial}")
                    self.labelserial.setText(f"Serial: {serial}")
                elif self.labelserial.text() == f"Serial: ":
                    self.lista_passed.clear()
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
                # print(f"faltando  {nome}")
            else:
                texto_contagem = f"{nome} ({classe}): {contagem_classes.get(classe, 0)} un."
                cv2.putText(img, texto_contagem, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                y_offset += 30

        if not detectou:
            cv2.putText(img, "Nenhum objeto detectado", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            self.labelserial.setText("Awaiting...")
            self.lista_passed.clear()

        # Se todos os componentes esperados foram detectados, faz o segundo teste
        print(len(faltando))

        if len(faltando) == 0:
            janelas_detectadas_atuais = set()  # <- CRUCIAL: fora do loop

            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                class_id = int(cls)
                label = model.names[class_id]

                for nome, info in componentes_esperados.items():
                    if nome in self.lista_passed:
                        continue
                    if info["classe"] == label and nome in self.lista:
                        pos = info["posicoes"][0]
                        if distancia(pos, centro_box(box)) < 50:
                            x1, y1, x2, y2 = map(int, box)
                            crop_realtime = imagem_sem_anotacoes[y1:y2, x1:x2]
                            crop_path = os.path.join("componentes", nome, f"{nome}.bmp")

                            if os.path.exists(crop_path):
                                crop_salvo = cv2.imread(crop_path)
                                if crop_salvo is not None and crop_realtime.shape[:2] == crop_salvo.shape[:2]:
                                    crop_salvo_resized = crop_salvo
                                    crop_realtime_resized = crop_realtime
                                else:
                                    crop_salvo_resized = cv2.resize(crop_salvo, (crop_realtime.shape[1], crop_realtime.shape[0]))
                                    crop_realtime_resized = crop_realtime

                                crop_salvo_gray = cv2.cvtColor(crop_salvo_resized, cv2.COLOR_BGR2GRAY)
                                crop_realtime_gray = cv2.cvtColor(crop_realtime_resized, cv2.COLOR_BGR2GRAY)

                                cv2.imwrite(fr"debug\{nome}_crop_default.bmp", crop_salvo_gray)
                                cv2.imwrite(fr"debug\{nome}_crop_realtime.bmp", crop_realtime_gray)
                                path_boas = crop_path
                                score, _ = ssim(crop_salvo_resized, crop_realtime_resized, channel_axis=-1, full=True)
                                comparacao = self.comparar(crop_salvo_resized, crop_realtime_resized, 0.1, info["area"], path_boas=os.path.join("componentes", nome), nome=nome)
                                texto_ssim = f"SSIM {nome}: {score:.2f}"

                                log_file = os.path.join("Logs", f"{nome}.txt")
                                os.makedirs("Logs", exist_ok=True)
                                with open(log_file, "w", encoding="utf-8") as f:
                                    f.write(f"{score:.2f}")
                            else:
                                print("pasta crop nao encontrada")
                                comparacao = None

                            janela_nome = f"Crop em tempo real - {nome}"

                            # Se houver falha na comparação, MANTÉM a janela aberta
                            if comparacao is not None:
                                janelas_detectadas_atuais.add(janela_nome)
                                print("encontrado")
                                cv2.putText(img, texto_ssim, (x1, y1 + 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                                try:
                                    if janela_nome not in self.janelas_crops_abertas:
                                        cv2.namedWindow(janela_nome, cv2.WINDOW_NORMAL)
                                        self.janelas_crops_abertas.add(janela_nome)

                                    # Adiciona área abaixo da imagem
                                    h, w = comparacao.shape[:2]
                                    area_extra = 50
                                    comparacao_com_botao = np.zeros((h + area_extra, w, 3), dtype=np.uint8)
                                    comparacao_com_botao[:h, :, :] = comparacao

                                    # Centraliza o texto "PASS"
                                    text = "CLICK TO ACCEPT"
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_scale = 0.7
                                    thickness = 1
                                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                                    text_x = (w - text_width) // 2
                                    text_y = h + (area_extra + text_height) // 2

                                    cv2.putText(comparacao_com_botao, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                                    
                                    # Define callback de clique
                                    cv2.setMouseCallback(janela_nome, self.on_mouse_click, param=(nome, crop_realtime_resized, h))


                                    cv2.imshow(janela_nome, comparacao_com_botao)
                                    cv2.resizeWindow(janela_nome, 600, 450)

                                    if cv2.getWindowProperty(janela_nome, cv2.WND_PROP_VISIBLE) < 1:
                                        self.janelas_crops_abertas.discard(janela_nome)
                                except cv2.error as e:
                                    print(f"Erro ao abrir janela para {nome}: {e}")
                                    self.janelas_crops_abertas.discard(janela_nome)
                            else:
                                cv2.putText(img, texto_ssim, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            print(self.lista_passed)
            # FORA do for principal: fecha janelas cujas falhas desapareceram
            janelas_a_fechar = self.janelas_crops_abertas - janelas_detectadas_atuais
            for janela in janelas_a_fechar:
                try:
                    cv2.destroyWindow(janela)
                except cv2.error as e:
                    print(f"Erro ao fechar janela {janela}: {e}")
                self.janelas_crops_abertas.discard(janela)



            print(len(self.lista_test))
            print(len(self.lista_passed))
            if len(self.lista_test) == len(self.lista_passed):
                self.labelserial = f"{self.labelserial.text} = PASS"

                                    
    def comparar(self, img_boa, img_comparar, borda, area_min, path_boas, nome):
        if img_comparar is None:
            print("Imagem de comparação inválida.")
            return

        try:
            lista_boas = [os.path.join(path_boas, f) for f in os.listdir(path_boas)
                        if f.lower().endswith((".bmp", ".jpg", ".png"))]
        except Exception as e:
            print(f"Erro ao ler imagens do diretório '{path_boas}': {e}")
            return

        for path_img_boa in lista_boas:
            img_boa = cv2.imread(path_img_boa)
            if img_boa is None:
                print(f"Falha ao carregar imagem: {path_img_boa}")
                continue

            # Redimensiona
            if img_boa.shape != img_comparar.shape:
                img_comparar_resized = cv2.resize(img_comparar, (img_boa.shape[1], img_boa.shape[0]))
            else:
                img_comparar_resized = img_comparar.copy()

            h, w = img_boa.shape[:2]
            x_start = int(w * borda)
            y_start = int(h * borda)
            x_end = int(w * (1 - borda))
            y_end = int(h * (1 - borda))

            canal_boa = img_boa[y_start:y_end, x_start:x_end, 1]
            canal_ruim = img_comparar_resized[y_start:y_end, x_start:x_end, 1]

            canal_boa = cv2.GaussianBlur(canal_boa, (3, 3), 0)
            canal_ruim = cv2.GaussianBlur(canal_ruim, (3, 3), 0)

            diff = cv2.absdiff(canal_boa, canal_ruim)
            media = np.mean(diff)
            _, diff_thresh = cv2.threshold(diff, media + 10, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            diff_thresh = cv2.dilate(diff_thresh, kernel, iterations=1)

            contornos, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            diferencas = 0
            for cnt in contornos:
                area = cv2.contourArea(cnt)
                if area >= area_min:
                    x, y, w_box, h_box = cv2.boundingRect(cnt)
                    aspect_ratio = w_box / float(h_box) if h_box != 0 else 0
                    hull = cv2.convexHull(cnt)
                    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

                    if 0.2 < aspect_ratio < 5.0 and solidity > 0.3:
                        diferencas += 1
                        break  # já podemos interromper, essa imagem tem falha

            # Se essa imagem "boa" não encontrou diferença, a comparação é compatível
            if diferencas == 0:
                print(f"Imagem compatível encontrada: {os.path.basename(path_img_boa)}")
                self.lista_passed.append(nome)

                return None  # Não é falha, uma imagem boa bateu

        # Se chegou aqui, todas tinham diferenças
        print("Todas as imagens boas apresentaram diferenças.")
        img_resultado = img_comparar_resized.copy()
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area >= area_min:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                aspect_ratio = w_box / float(h_box) if h_box != 0 else 0
                hull = cv2.convexHull(cnt)
                solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0

                if 0.2 < aspect_ratio < 5.0 and solidity > 0.3:
                    cnt += np.array([[[x_start, y_start]]])  # volta para coord original
                    cv2.drawContours(img_resultado, [cnt], -1, (0, 0, 255), 2)

        empilhada = np.hstack([img_boa, img_resultado])
        return empilhada

            
    def salvar_crop_como_valido(self, nome, imagem):
        pasta = os.path.join("componentes", nome)
        os.makedirs(pasta, exist_ok=True)
        i = 1
        while True:
            caminho = os.path.join(pasta, f"{nome}_{i}.bmp")
            if not os.path.exists(caminho):
                cv2.imwrite(caminho, imagem)
                print(f"Imagem marcada como válida e salva em: {caminho}")
                break
            i += 1

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nome, crop_realtime_resized, altura = param

            if y >= altura:  # Qualquer clique abaixo da imagem
                print(f"[PASS] Clique detectado abaixo da imagem para {nome}")
                self.salvar_crop_como_valido(nome, crop_realtime_resized)

                
def aplicar_tema_escuro(app):
    dark_palette = QPalette()

    dark_palette.setColor(QPalette.Window, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(30, 30, 30))
    dark_palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(45, 45, 45))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Highlight, QColor(100, 100, 255))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(dark_palette)
    app.setStyle("Fusion")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = Principal()
    janela.showMaximized()
    aplicar_tema_escuro(app)
    sys.exit(app.exec_())
