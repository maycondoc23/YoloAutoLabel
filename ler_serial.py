import cv2
import json
import numpy as np
import datetime
from datetime import datetime

from pylibdmtx.pylibdmtx import decode

productsn="NOT_FOUND"

def run_model(imagem_ref_path, imagem_inspecao_path, rois_path):
    global productsn
    productsn="NOT_FOUND"
    imagem_ref = cv2.imread(imagem_ref_path, cv2.IMREAD_GRAYSCALE)
    imagem_inspecao = cv2.imread(imagem_inspecao_path, cv2.IMREAD_GRAYSCALE)

    if imagem_ref is None or imagem_inspecao is None:
        raise FileNotFoundError("Imagens de referência ou inspeção não foram encontradas corretamente.")

    with open(rois_path, 'r') as f:
        rois = json.load(f)

    imagem_saida = cv2.cvtColor(imagem_inspecao, cv2.COLOR_GRAY2BGR)
    falha = False
    relatorio = []

    for roi in rois:
        x1, y1, x2, y2 = roi['roi']
        nome = roi.get("nome", "ROI")
        tipo = roi.get("tipo", "default")
        rec_ref = imagem_ref[y1:y2, x1:x2]
        rec_inspecao = imagem_inspecao[y1:y2, x1:x2]

        valor = ""

        if tipo == "datamatrix":
            print(f"[{nome}] datamatrix")
            rec_zoom = cv2.resize(rec_inspecao, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
            _, rec_bin = cv2.threshold(rec_zoom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(f"{nome}_zoom.png", rec_zoom)
            cv2.imwrite(f"{nome}_zoom_bin.png", rec_bin)

            decoded_objects = decode(rec_bin, timeout=500, max_count=5)
            if decoded_objects:
                for obj in decoded_objects:
                    valor = obj.data.decode("utf-8")
                    print(f"[{nome}] Datamatrix NORMAL: {valor}")
                    productsn=valor
                    break

            if valor == "":
                falha = True
                print(f"[{nome}] Falha de leitura datamatrix")

            score = 0.0

        elif tipo in ["qrcode", "barcode"]:
            from pyzbar.pyzbar import decode as decode_pyzbar
            print(f"[{nome}] {tipo}")
            rec_zoom = cv2.resize(rec_inspecao, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
            _, rec_bin = cv2.threshold(rec_zoom, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(f"{nome}_zoom_bin.png", rec_bin)
            cv2.imwrite(f"{nome}_zoom.png", rec_zoom)

            decoded_objects = decode_pyzbar(rec_zoom)
            if decoded_objects:
                for obj in decoded_objects:
                    valor = obj.data.decode("utf-8")
                    print(f"[{nome}] {tipo} NORMAL: {valor}")
                    productsn=valor
                    break

            if valor == "":
                falha = True
                print(f"[{nome}] Falha de leitura {tipo}")

            score = 0.0

        else:
            diff = cv2.absdiff(rec_ref, rec_inspecao)
            score = np.mean(diff)

            if score >= 20:
                falha = True
                print(f"[{nome}] Falha visual (score={score:.1f})")

        cv2.rectangle(imagem_saida, (x1, y1), (x2, y2), (0, 255, 0) if score < 10 else (0, 255, 255) if score < 20 else (0, 0, 255), 2)
        cv2.putText(imagem_saida, f"{nome}: {score:.1f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if score < 10 else (0, 255, 255) if score < 20 else (0, 0, 255), 1)

        relatorio.append({"nome": nome, "score": score, "valor": valor})

    status = "Reprovado" if falha else "Aprovado"
    saida_path = imagem_inspecao_path.replace(".bmp", f"{productsn}_{status}_{datetime.now().strftime('%d%m%y_%H%M%S')}.jpg")
    cv2.imwrite(saida_path, imagem_saida)

    relatorio_path = imagem_inspecao_path.replace(".bmp", f"{productsn}_{status}_{datetime.now().strftime('%d%m%y_%H%M%S')}.csv")
    with open(relatorio_path, "w") as f:
        f.write("nome,score,valor\n")
        for item in relatorio:
            f.write(f"{item['nome']},{item['score']:.1f},{item['valor']}\n")

    return status, saida_path

    
def ler_serial(image_crop):
    rec_zoom = cv2.resize(image_crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    rec_gray = cv2.cvtColor(rec_zoom, cv2.COLOR_BGR2GRAY)
    _, rec_bin = cv2.threshold(rec_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # salvar foto
    cv2.imwrite("rec_bin.png", rec_bin)
    # cv2.imshow("image_crop.png", image_crop)
    cv2.imwrite("zoom.png", rec_zoom)
    decoded_objects = decode(rec_zoom, timeout=500, max_count=4)
    
    for obj in decoded_objects:
        valor = obj.data.decode("utf-8")
        print(f"Datamatrix NORMAL: {valor}")
        if valor.startswith("[)>"):
            valor = valor[8:]
            valor = valor[:12]
            
        return valor  # Retorna o primeiro valor lido com sucesso

    print("Falha de leitura datamatrix")
    return ""  # ou None, dependendo do que preferir tratar fora