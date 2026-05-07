#!/usr/bin/env python3
"""
live_detect_v3.py — Detecção em tempo real com câmera usando o modelo v3_final.pt.

Uso:
    # Simples (webcam padrão)
    python scripts/live_detect_v3.py

    # Com modelo e câmera específicos
    python scripts/live_detect_v3.py --model modelo/v3_final.pt --source 0

    # Com threshold de confiança ajustado
    python scripts/live_detect_v3.py --conf 0.4

Controles durante a execução:
    q — sair
    s — pausar/retomar
    p — salvar screenshot (capture.png)
"""

import argparse
import time
import sys
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Detecção em tempo real com câmera")
    parser.add_argument("--model", default="modelo/v3_final.pt", help="Caminho do modelo .pt")
    parser.add_argument("--source", default="0", help="Fonte: índice da câmera (0, 1, ...) ou caminho de imagem/vídeo")
    parser.add_argument("--conf", type=float, default=0.35, help="Threshold de confiança")
    parser.add_argument("--resolution", default=None, help="Resolução (ex: 1280x720)")
    args = parser.parse_args()

    # Resolve caminho do modelo relativo ao treinamento-ml/
    model_path = args.model
    if not os.path.isabs(model_path):
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        model_path = os.path.join(project_root, args.model)

    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        sys.exit(1)

    print(f"Carregando modelo: {model_path}")
    model = YOLO(model_path, task="detect")
    classes = model.names
    print(f"Classes ({len(classes)}): {list(classes.values())}")

    # Cores para bounding boxes (uma por classe)
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(len(classes), 3)).tolist()

    # Interpreta fonte
    source = args.source
    try:
        cam_idx = int(source)
        source_type = "camera"
    except ValueError:
        source_type = "file"
        cam_idx = None

    if source_type == "camera":
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir a câmera {cam_idx}")
            sys.exit(1)

        if args.resolution:
            w, h = map(int, args.resolution.split("x"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        print(f"Câmera #{cam_idx} aberta. Pressione 'q' para sair.")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir {source}")
            sys.exit(1)
        print(f"Reproduzindo: {source}. Pressione 'q' para sair.")

    fps_buffer = []
    fps_avg_len = 100
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fim do stream.")
                break

        t_start = time.perf_counter()

        if not paused:
            results = model(frame, verbose=False)
            detections = results[0].boxes

            object_count = 0
            for box in detections:
                cls_id = int(box.cls.item())
                conf = box.conf.item()

                if conf < args.conf:
                    continue

                x1, y1, x2, y2 = box.xyxy.cpu().numpy().squeeze().astype(int)
                class_name = classes[cls_id]
                color = colors[cls_id % len(colors)]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name} {int(conf * 100)}%"
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1, lh + 10)
                cv2.rectangle(frame, (x1, label_y - lh - 10), (x1 + lw, label_y + baseline - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                object_count += 1

            # FPS
            avg_fps = np.mean(fps_buffer) if fps_buffer else 0
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
            cv2.putText(frame, f"Objetos: {object_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        # Exibe frame (sempre, mesmo pausado)
        paused_text = " [PAUSADO]" if paused else ""
        cv2.imshow(f"Deteccao — v3 (15 classes){paused_text}", frame)

        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            paused = not paused
            print("PAUSADO" if paused else "RETOMADO")
        elif key == ord("p"):
            cv2.imwrite("capture.png", frame)
            print("Screenshot salvo: capture.png")

        # Atualiza FPS
        t_stop = time.perf_counter()
        if not paused:
            frame_rate = 1.0 / (t_stop - t_start) if (t_stop - t_start) > 0 else 0
            fps_buffer.append(frame_rate)
            if len(fps_buffer) > fps_avg_len:
                fps_buffer.pop(0)

    cap.release()
    cv2.destroyAllWindows()
    print(f"FPS médio: {np.mean(fps_buffer):.1f}" if fps_buffer else "Encerrado.")


if __name__ == "__main__":
    main()
