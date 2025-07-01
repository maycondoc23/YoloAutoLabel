from roboflow import Roboflow
# Inicializa com sua API Key

rf = Roboflow(api_key="OywpycokmJlWMVUf3g6q")

# Conecta ao seu projeto e versão do modelo
project = rf.workspace("anotacoes").project("anotacoes-qvaav")
model = project.version("1").model

# Faz a predição em uma imagem local
resultado = model.predict(r"images/padrao4.jpg", confidence=40, overlap=30).json()

# Exibe os dados anotados
for pred in resultado['predictions']:
    print(f"Classe: {pred['class']}")
    print(f"Confiança: {pred['confidence']}")
    print(f"Bounding Box: x={pred['x']}, y={pred['y']}, w={pred['width']}, h={pred['height']}")
    print("------")
model.predict("saida_roboflow.jpg", confidence=40, overlap=30).save("saida_anotada.jpg")

