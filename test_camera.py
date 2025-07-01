import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
import gxipy as gx


class DahengCameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Câmera Daheng ao Vivo")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setStyleSheet("background-color: black")
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Inicializar câmera Daheng
        self.device_manager = gx.DeviceManager()
        self.device_manager.update_device_list()

        if self.device_manager.get_device_number() == 0:
            raise RuntimeError("Nenhuma câmera Daheng conectada.")

        self.cam = self.device_manager.open_device_by_index(1)
        self.cam.stream_on()

        # Timer para capturar e atualizar frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        raw_image = self.cam.data_stream[0].get_image(timeout=1000)
        if raw_image is None:
            print("Falha ao capturar imagem.")
            return

        rgb_image = raw_image.convert("RGB").get_numpy_array()
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.timer.stop()
        self.cam.stream_off()
        self.cam.close_device()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DahengCameraViewer()
    viewer.show()
    sys.exit(app.exec_())
