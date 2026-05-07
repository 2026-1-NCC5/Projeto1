import { useEffect, useRef } from 'react';

// Habilita drag (mouse + touch) num elemento referenciado por `popupRef`.
// Usa transform CSS direto (sem state) para evitar re-renders.
export default function useDraggablePopup({ ativo, popupRef }) {
  const draggingRef = useRef(false);
  const startPosRef = useRef({ x: 0, y: 0 });
  const translateRef = useRef({ x: 0, y: 0 });

  useEffect(() => {
    const handleUp = () => { draggingRef.current = false; };
    const handleMove = (e) => {
      if (!draggingRef.current || !popupRef.current) return;
      e.preventDefault();
      const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
      const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;
      const dx = clientX - startPosRef.current.x;
      const dy = clientY - startPosRef.current.y;

      popupRef.current.style.transform = `translate(calc(-50% + ${translateRef.current.x + dx}px), ${translateRef.current.y + dy}px)`;
    };

    if (ativo) {
      document.addEventListener('mousemove', handleMove);
      document.addEventListener('touchmove', handleMove, { passive: false });
      document.addEventListener('mouseup', handleUp);
      document.addEventListener('touchend', handleUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.removeEventListener('touchend', handleUp);
    };
  }, [ativo, popupRef]);

  const handlePopupDown = (e) => {
    if (e.target.closest('button') || e.target.closest('#session-items-list')) return;
    draggingRef.current = true;
    const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
    const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;
    startPosRef.current = { x: clientX, y: clientY };

    // Parse transform: CSS includes translateX(-50%); offset kept simple for demo.
  };

  return { handlePopupDown };
}
