import cv2
import numpy as np

def detectar_curto_aprimorado(img_path_boa,img_path_ruim,borda,area_min):
    img_boa = cv2.imread(img_path_boa)
    img_ruim = cv2.imread(img_path_ruim)

    if img_boa is None or img_ruim is None:
        print("Erro ao carregar imagens.")
        return

    # Redimensiona para o mesmo tamanho
    if img_boa.shape != img_ruim.shape:
        img_ruim = cv2.resize(img_ruim, (img_boa.shape[1], img_boa.shape[0]))

    h, w = img_boa.shape[:2]

    # Define região central (desconsidera bordas)
    x_start = int(w * borda)
    y_start = int(h * borda)
    x_end = int(w * (1 - borda))
    y_end = int(h * (1 - borda))

    # Recorte da área central no CANAL VERDE
    canal_boa = img_boa[y_start:y_end, x_start:x_end, 1]
    canal_ruim = img_ruim[y_start:y_end, x_start:x_end, 1]

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

    img_resultado = img_ruim.copy()
    curtos_encontrados = 0

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
                curtos_encontrados += 1

    print(f"Curto(s) detectado(s): {curtos_encontrados}")

    # Visualização
    empilhada = np.hstack([img_boa, img_resultado])
    cv2.imshow("Boa | Ruim | Curto Detectado", empilhada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemplo de uso
detectar_curto_aprimorado(
    img_path_boa=r"debug\padrao.jpg",
    img_path_ruim=r"debug\ruim.jpg",
    borda=0.1,
    area_min=100,
)
