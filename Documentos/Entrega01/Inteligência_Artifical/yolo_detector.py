import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define e interpreta os argumentos de entrada do usuário

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

# Interpreta as entradas do usuário

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Verifica se o arquivo do modelo existe e é válido

if (not os.path.exists(model_path)):
    print('ERRO: O caminho do modelo é inválido ou o modelo não foi encontrado. Verifique se o nome do arquivo foi digitado corretamente.')
    sys.exit(0)

# Carrega o modelo na memória e obtém o mapa de labels

model = YOLO(model_path, task='detect')
labels = model.names

# Interpreta a entrada para determinar se a fonte é arquivo, pasta, vídeo ou câmera USB

img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'A extensão de arquivo {ext} não é suportada.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'A entrada {img_source} é inválida. Por favor, tente novamente.')
    sys.exit(0)

# Interpreta a resolução de exibição definida pelo usuário

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Verifica se a gravação é válida e configura o gravador

if record:
    if source_type not in ['video','usb']:
        print('A gravação funciona apenas para fontes de vídeo e câmera. Por favor, tente novamente.')
        sys.exit(0)
    if not user_res:
        print('Por favor, especifique a resolução para gravar o vídeo.')
        sys.exit(0)

    # Configura a gravação

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Carrega ou inicializa a fonte de imagem

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video':
        cap_arg = img_source
    elif source_type == 'usb':
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Define a resolução da câmera ou vídeo se especificada pelo usuário

    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Define as cores das caixas delimitadoras

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Inicializa variáveis de controle e status

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Inicia o loop de inferência

while True:
    t_start = time.perf_counter()

    # Carrega o frame da fonte de imagem

    if source_type == 'image' or source_type == 'folder':

        # Se a fonte for imagem ou pasta de imagens, carrega a imagem pelo nome do arquivo

        if img_count >= len(imgs_list):
            print('Todas as imagens foram processadas. Encerrando o programa.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1

    elif source_type == 'video':

        # Se a fonte for vídeo, carrega o próximo frame do arquivo

        ret, frame = cap.read()
        if not ret:
            print('Fim do arquivo de vídeo atingido. Encerrando o programa.')
            break

    elif source_type == 'usb':

        # Se a fonte for câmera USB, captura o frame da câmera

        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Não foi possível ler frames da câmera. Isso indica que a câmera está desconectada ou não está funcionando. Encerrando o programa.')
            break

    elif source_type == 'picamera':

        # Se a fonte for Picamera, captura frames pela interface da picamera

        frame = cap.capture_array()
        if (frame is None):
            print('Não foi possível ler frames da Picamera. Isso indica que a câmera está desconectada ou não está funcionando. Encerrando o programa.')
            break

    # Redimensiona o frame para a resolução de exibição desejada

    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Executa a inferência no frame

    results = model(frame, verbose=False)

    # Extrai os resultados

    detections = results[0].boxes

    # Inicializa variável para contagem básica de objetos

    object_count = 0

    # Percorre cada detecção e obtém coordenadas, confiança e classe

    for i in range(len(detections)):

        # Obtém as coordenadas da caixa delimitadora

        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Obtém o ID e o nome da classe da caixa delimitadora

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Obtém a confiança da caixa delimitadora

        conf = detections[i].conf.item()

        # Desenha a caixa se a confiança for suficiente

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Exemplo básico: conta o número de objetos na imagem

            object_count = object_count + 1

    # Calcula e desenha a taxa de frames (se usar fonte de vídeo, USB ou Picamera)

    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (232,120,55), 2) # Desenha a taxa de frames

    # Exibe os resultados da detecção

    cv2.putText(frame, f'Numero de objetos: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (232,120,55), 2) # Desenha o total de objetos detectados
    cv2.imshow('Resultados da deteccao YOLO', frame) # Exibe a imagem

    if record:
        recorder.write(frame)

    # Para vídeo/câmera, aguarda 5ms antes de passar para o próximo frame

    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):     # Pressione 'q' para sair
        break
    elif key == ord('s') or key == ord('S'):   # Pressione 's' para pausar a inferência
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):   # Pressione 'p' para salvar uma foto do frame atual
        cv2.imwrite('capture.png', frame)

    # Calcula o FPS deste frame

    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Adiciona o FPS ao buffer para calcular a média

    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)

    # Calcula o FPS médio dos últimos frames

    avg_frame_rate = np.mean(frame_rate_buffer)

# Encerra e libera os recursos

print(f'FPS médio do pipeline: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()