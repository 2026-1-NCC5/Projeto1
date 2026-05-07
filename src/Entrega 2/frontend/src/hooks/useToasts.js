import { useCallback, useRef, useState } from 'react';

// Sistema de toast com auto-dismiss em 4s e animação de saída de 300ms.
// Retorna { toasts, addToast } — o overlay é renderizado por <ToastContainer/>.
export default function useToasts() {
  const [toasts, setToasts] = useState([]);
  const toastIdRef = useRef(0);

  const addToast = useCallback((message, type = 'info') => {
    const id = ++toastIdRef.current;
    setToasts(prev => [...prev, { id, message, type, exiting: false }]);
    setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, exiting: true } : t));
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 300);
    }, 4000);
  }, []);

  return { toasts, addToast };
}
