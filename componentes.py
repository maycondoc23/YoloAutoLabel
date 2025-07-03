import json
import os
from collections import defaultdict
import math
import json
import os
from PIL import Image
import numpy as np

comp_json = 'setup_componentes.json'

def carregar_componentes():
    if os.path.exists(comp_json):
        with open(comp_json, 'r') as f:
            dados = json.load(f)
            # print(f"Componentes carregados de {comp_json}: {dados}")
            return dados
    else:
        print("Arquivo JSON de componentes esperados não encontrado. Usando vazio.")
        return {}


def salvar_componentes(componentes_dict):
    with open(comp_json, 'w') as f:
        json.dump(componentes_dict, f, indent=4)
    print(f"Componentes salvos em {comp_json}")


# def calibrar(model, imagem_path, conf=0.4):
#     results = model.predict(imagem_path, conf=conf)[0]

#     agrupados = defaultdict(list)
#     for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
#         class_id = int(cls)
#         label = model.names[class_id]
#         x_c, y_c = centro_box(box)
#         largura = int(box[2] - box[0])
#         altura = int(box[3] - box[1])

#         # Reduzir 10% da largura e altura
#         largura = int(largura * 0.75)
#         altura = int(altura * 0.75)

#         agrupados[label].append({
#             "posicao": [int(x_c), int(y_c)],
#             "tamanho": [largura, altura]
#         })
#         print(f"Calibração: Detectado {label} em ({int(x_c)}, {int(y_c)})")

#     componentes = {}
#     for classe, posicoes in agrupados.items():
#         for i, item in enumerate(posicoes, 1):
#             nome = f"{classe}{i}"
#             componentes[nome] = {
#                 "classe": classe,
#                 "posicoes": [item["posicao"]],
#                 "tamanho": item["tamanho"]
#             }

#     salvar_componentes(componentes)
#     return componentes


def calibrar(model, imagem_path, conf=0.6):
    results = model.predict(imagem_path, conf=conf)[0]

    agrupados = defaultdict(list)
    # Carregar imagem original
    if isinstance(imagem_path, np.ndarray):
        img = Image.fromarray(imagem_path)
    else:
        img = Image.open(imagem_path)

    for idx, (box, cls, conf) in enumerate(zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf), 1):
        class_id = int(cls)
        label = model.names[class_id]
        x_c, y_c = centro_box(box)
        largura = int(box[2] - box[0])
        altura = int(box[3] - box[1])

        # Reduzir 10% da largura e altura
        largura = int(largura * 0.75)
        altura = int(altura * 0.75)

        agrupados[label].append({
            "posicao": [int(x_c), int(y_c)],
            "tamanho": [largura, altura],
            "box": box
        })
        print(f"Calibração: Detectado {label} em ({int(x_c)}, {int(y_c)})")

    componentes = {}
    for classe, posicoes in agrupados.items():
        if classe == "LED":
            continue
        for i, item in enumerate(posicoes, 1):
            nome = f"{classe}{i}"
            componentes[nome] = {
                "classe": classe,
                "posicoes": [item["posicao"]],
                "tamanho": item["tamanho"]
            }

            # Criar pasta para o componente
            pasta = os.path.join("componentes", nome)
            os.makedirs(pasta, exist_ok=True)

            # Crop da imagem
            box = item["box"]
            x1, y1, x2, y2 = map(int, box)
            crop = img.crop((x1, y1, x2, y2))
            crop_path = os.path.join(pasta, f"{nome}.jpg")
            crop.save(crop_path)
            print(f"Imagem cropada salva em {crop_path}")

    salvar_componentes(componentes)
    return componentes

def distancia(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def centro_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


