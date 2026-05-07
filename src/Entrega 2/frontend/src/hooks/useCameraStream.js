import { useEffect, useRef } from 'react';

// Abre a câmera (facingMode environment) quando `ativo` vira true e libera
// os tracks no cleanup. O envio de frames ao backend é feito separadamente
// pelo useAuditoriaWS — este hook cuida apenas do <video> local.
export default function useCameraStream({ ativo, videoRef, onAberta, onErro }) {
  const streamRef = useRef(null);

  useEffect(() => {
    if (!ativo) return undefined;

    let cancelado = false;

    const abrir = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
          if (cancelado) {
            stream.getTracks().forEach(t => t.stop());
            return;
          }
          streamRef.current = stream;
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
          onAberta?.();
        }
      } catch (err) {
        console.error(err);
        onErro?.(err);
        alert("Não foi possível acessar a câmera.");
      }
    };

    abrir();

    const videoEl = videoRef.current;

    return () => {
      cancelado = true;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (videoEl) videoEl.srcObject = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps -- camera lifecycle tied to screen only
  }, [ativo]);
}
