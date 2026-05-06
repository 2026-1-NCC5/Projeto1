import React, { useState, useEffect, useRef } from 'react';
import LogoAbrace from './assets/logo_final_nova.png';

// Main Application
export default function App() {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [appState, setAppState] = useState([
    { id: 'A', title: 'Grupo A', members: [{ name: 'Maria Silva', ra: '12345' }, { name: 'João P.', ra: '54321' }], totalKg: 0, items: [] },
    { id: 'B', title: 'Grupo B', members: [], totalKg: 0, items: [] }
  ]);

  // Modal & Popup State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isUserPopupOpen, setIsUserPopupOpen] = useState(false);
  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [editingGroupId, setEditingGroupId] = useState(null);
  const [tempGroupName, setTempGroupName] = useState('');
  const [tempMembers, setTempMembers] = useState([]);
  const [memberInputName, setMemberInputName] = useState('');
  const [memberInputRa, setMemberInputRa] = useState('');

  // User Profile State
  const [userData, setUserData] = useState({
    nome: 'Admin',
    sobrenome: 'Central',
    email: 'admin@abraceai.com.br',
    ra: '00000000'
  });
  const [tempUserData, setTempUserData] = useState({});

  // Camera & Detection State
  const [activeGroupForCamera, setActiveGroupForCamera] = useState(null);
  const [currentSessionItems, setCurrentSessionItems] = useState([]);
  const [serverConectado, setServerConectado] = useState(false);

  const [leituraNome, setLeituraNome] = useState('Consultando Alimento...');
  const [leituraDesc, setLeituraDesc] = useState('Identificando pela balança...');
  const [btnLeituraText, setBtnLeituraText] = useState('AGUARDE...');
  const [currFood, setCurrFood] = useState('');
  const [currWeight, setCurrWeight] = useState(0);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const socketRef = useRef(null);
  const sendIntervalRef = useRef(null);
  const popupRef = useRef(null);

  // Popup Draggable State
  const draggingRef = useRef(false);
  const startPosRef = useRef({ x: 0, y: 0 });
  const translateRef = useRef({ x: 0, y: 0 });

  // Home Screen Nav
  const handleStart = () => setCurrentScreen('dashboard');
  const handleGoCadastro = () => setCurrentScreen('cadastro');

  // Config Screen Handlers
  const handleGoConfig = () => {
    setIsUserPopupOpen(false);
    setTempUserData({ ...userData });
    setCurrentScreen('config');
  };

  const handleConfigSave = () => {
    setUserData({ ...tempUserData });
    setCurrentScreen('dashboard');
  };

  // --- CAMERA LOGIC ---
  const foods = ['Arroz Agulhinha', 'Feijão Preto', 'Macarrão', 'Leite em Pó', 'Açúcar Refinado', 'Farinha de Trigo'];
  const weights = [1, 2, 5, 10];

  const generateDetection = () => {
    const f = foods[Math.floor(Math.random() * foods.length)];
    const w = weights[Math.floor(Math.random() * weights.length)];
    setCurrFood(f);
    setCurrWeight(w);
    setLeituraNome(f);
    setLeituraDesc(`Estimativa de Peso: ${w}kg`);
    setBtnLeituraText(`ADICIONAR +${w}KG`);
  };

  const connectToServer = () => {
    if (socketRef.current) return;
    try {
      const socket = window.io('http://localhost:5000', { transports: ['websocket'], reconnectionAttempts: 3, timeout: 5000 });
      socketRef.current = socket;
      socket.on('connect', () => { setServerConectado(true); console.log('Conectado YOLO'); });
      socket.on('disconnect', () => setServerConectado(false));
      socket.on('connect_error', () => { setServerConectado(false); });

      socket.on('deteccao', (data) => {
        if (data.detectado) {
          setCurrFood(data.nome);
          setCurrWeight(data.peso);
          setLeituraNome(data.nome);
          setLeituraDesc(`Estimativa de Peso: ${data.peso}kg (${data.confianca}%)`);
          setBtnLeituraText(`ADICIONAR +${data.peso}KG`);
        } else {
          setLeituraNome('Buscando alimento...');
          setLeituraDesc('Nenhum item detectado');
          setBtnLeituraText('AGUARDE...');
        }
      });
    } catch (e) {
      setServerConectado(false);
    }
  };

  const startSendingFrames = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const ctx = canvas.getContext('2d');

    sendIntervalRef.current = setInterval(() => {
      if (!video.videoWidth || !serverConectado) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      socketRef.current?.emit('frame', { image: canvas.toDataURL('image/jpeg', 0.6) });
    }, 500);
  };

  const stopSendingFrames = () => {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current);
      sendIntervalRef.current = null;
    }
  };

  const initCamera = async () => {
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        connectToServer();
      }
    } catch (err) {
      console.error(err);
      alert("Não foi possível acessar a câmera.");
    }
  };

  const stopCamera = () => {
    stopSendingFrames();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
  };

  // --- INTERACTIONS ---
  const handleStartCount = (groupId) => {
    setActiveGroupForCamera(groupId);
    setCurrentSessionItems([]);
    setLeituraNome('Consultando Alimento...');
    setLeituraDesc('Identificando pela balança...');
    setBtnLeituraText('AGUARDE...');
    setCurrFood('');
    setCurrWeight(0);
    setCurrentScreen('camera');
  };

  const handleFinishCount = () => {
    if (activeGroupForCamera && currentSessionItems.length > 0) {
      setAppState(prev => prev.map(group => {
        if (group.id === activeGroupForCamera) {
          const newItems = [...group.items, ...currentSessionItems];
          const addedKg = currentSessionItems.reduce((acc, obj) => acc + obj.weight, 0);
          return {
            ...group,
            items: newItems,
            totalKg: parseFloat((group.totalKg + addedKg).toFixed(2))
          };
        }
        return group;
      }));
    }
    stopCamera();
    setCurrentScreen('dashboard');
  };

  const handleSimularClick = () => {
    if (currFood && currWeight > 0) {
      setCurrentSessionItems(prev => [...prev, { name: currFood, weight: currWeight }]);
      setCurrFood('');
      setCurrWeight(0);
      setLeituraNome('Buscando próximo...');
      setLeituraDesc('--');
      setBtnLeituraText('AGUARDE...');
      if (!serverConectado) setTimeout(generateDetection, 1000);
    }
  };

  const handleRejectClick = () => {
    setLeituraNome('Descartado. Buscando...');
    setLeituraDesc('--');
    setBtnLeituraText('AGUARDE...');
    if (!serverConectado) setTimeout(generateDetection, 1000);
  };

  // Popup Drag Logic
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

    if (currentScreen === 'camera') {
      document.addEventListener('mousemove', handleMove);
      document.addEventListener('touchmove', handleMove, { passive: false });
      document.addEventListener('mouseup', handleUp);
      document.addEventListener('touchend', handleUp);
      initCamera();
    }

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.removeEventListener('touchend', handleUp);
      if (currentScreen !== 'camera') stopCamera();
    };
  }, [currentScreen]);

  const handlePopupDown = (e) => {
    if (e.target.closest('button') || e.target.closest('#session-items-list')) return;
    draggingRef.current = true;
    const clientX = e.type.includes('mouse') ? e.clientX : e.touches[0].clientX;
    const clientY = e.type.includes('mouse') ? e.clientY : e.touches[0].clientY;
    startPosRef.current = { x: clientX, y: clientY };

    // Parse transform
    if (popupRef.current) {
      const style = window.getComputedStyle(popupRef.current);
      const matrix = new DOMMatrixReadOnly(style.transform);
      // It includes translateX(-50%). We keep relative offset simple.
      // This is highly simplified and avoids complex matrix breakdown for demo.
    }
  };


  // --- RENDERING SCREENS ---

  if (currentScreen === 'home') {
    return (
      <div className="flex flex-col h-full bg-cover bg-center relative overflow-hidden" style={{ backgroundImage: "linear-gradient(rgba(0, 0, 0, 0.4), var(--dark)), url('https://wixmp-ed30a86b8c4ca887773594c2.wixmp.com/api/v1/fill/w_1300,h_812,al_c,q_81/dpvsc4c-b17f549c-f6ce-46e3-ae16-cdd231c51842.jpg')" }}>
        <div className="absolute inset-0 bg-gradient-to-t pointer-events-none"></div>

        <header className="flex justify-start align-center p-6 w-full absolute top-0 z-20 backdrop-blur-md bg-black/20 border-b border-white/5">
          <div className="flex align-center gap-4 mt-1">
            <div className="floating" style={{ width: '60px', height: '60px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <img src={LogoAbrace} alt="Logo AbraceAI" style={{ width: '130%', height: '130%', objectFit: 'contain', clipPath: 'inset(0 0 18% 0)', marginTop: '8px' }} />
            </div>
            <span className="font-black text-3xl tracking-tight text-white drop-shadow-md">ABRACE<span className="text-primary">AI</span></span>
          </div>
        </header>

        <div className="flex flex-col align-center justify-center p-6 pt-32 h-full pb-10 w-full max-w-2xl mx-auto z-10 relative">
          <div className="text-center slide-up-anim w-full flex flex-col align-center">
            <img src={LogoAbrace} alt="Logo AbraceAI" className="drop-shadow-xl pulse-glow" style={{ width: '220px', height: '220px', objectFit: 'contain', clipPath: 'inset(0 0 18% 0)', marginBottom: '-30px' }} />
            <h1 className="font-black text-white mb-6 leading-tight tracking-tighter drop-shadow-xl" style={{ fontSize: '3.5rem' }}>
              ABRACE<span className="text-primary">AI</span>
            </h1>
          </div>

          <div className="flex flex-wrap gap-5 slide-up-anim justify-center mt-6" style={{ animationDelay: '0.2s', animationFillMode: 'both' }}>
            <button className="btn btn-primary shadow-xl transition hover:scale-105 pulse-glow" onClick={handleStart} style={{ padding: '1.25rem 3.5rem', height: 'fit-content', borderRadius: '1.5rem' }}>
              <span className="font-bold tracking-wide text-md">COMEÇAR TRIAGEM</span> <i className="ph ph-arrow-right text-2xl"></i>
            </button>
            <button className="btn btn-outline shadow-xl transition hover:scale-105 bg-black/50 border-gray-medium text-white backdrop-blur-md hover:bg-white/10" onClick={handleGoCadastro} style={{ padding: '1.25rem 3.5rem', height: 'fit-content', borderRadius: '1.5rem' }}>
              <i className="ph ph-user-plus text-2xl text-primary"></i> <span className="font-bold tracking-wide text-md">CADASTRAR ADMIN</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // --- CONFIG SCREEN (User Profile) ---
  if (currentScreen === 'config') {
    return (
      <div className="flex flex-col h-full bg-dark">
        <header className="flex align-center p-6 border-b border-gray-medium bg-header shadow-md relative z-10">
          <button className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4" onClick={() => setCurrentScreen('dashboard')}>
            <i className="ph ph-arrow-left text-xl text-white"></i>
          </button>
          <div className="flex align-center gap-3">
            <i className="ph ph-gear text-2xl text-primary"></i>
            <span className="font-bold text-xl text-white tracking-tight">Configuração de Dados</span>
          </div>
        </header>

        <div className="flex-grow flex align-center justify-center p-6">
          <div className="w-full max-w-md bg-card border border-gray-medium rounded-2xl shadow-2xl p-8 slide-up-anim">
            <div className="flex align-center gap-3 mb-6">
              <div className="circle-bg-gray flex-center" style={{ width: '48px', height: '48px', borderRadius: '50%' }}>
                <i className="ph-fill ph-user-circle text-2xl text-primary"></i>
              </div>
              <div>
                <h2 className="font-bold text-xl text-white tracking-tight">Editar Perfil</h2>
                <p className="text-gray text-xs font-medium mt-1">Atualize seus dados de mantenedor.</p>
              </div>
            </div>

            <div className="flex flex-col gap-5 mb-8">
              <div className="flex gap-4">
                <div className="flex flex-col gap-1.5 w-full">
                  <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Nome</label>
                  <input type="text" className="input-dark w-full" value={tempUserData.nome || ''} onChange={e => setTempUserData({ ...tempUserData, nome: e.target.value })} placeholder="Ex: João" />
                </div>
                <div className="flex flex-col gap-1.5 w-full">
                  <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Sobrenome</label>
                  <input type="text" className="input-dark w-full" value={tempUserData.sobrenome || ''} onChange={e => setTempUserData({ ...tempUserData, sobrenome: e.target.value })} placeholder="Ex: Silva" />
                </div>
              </div>
              <div className="flex flex-col gap-1.5 w-full">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">E-mail Institucional</label>
                <input type="email" className="input-dark w-full" value={tempUserData.email || ''} onChange={e => setTempUserData({ ...tempUserData, email: e.target.value })} placeholder="joao.silva@usp.br" />
              </div>
              <div className="flex flex-col gap-1.5 w-full">
                <label className="text-[10px] text-gray font-bold uppercase tracking-widest ml-1">Matrícula (RA)</label>
                <input type="number" className="input-dark w-full" value={tempUserData.ra || ''} onChange={e => setTempUserData({ ...tempUserData, ra: e.target.value })} placeholder="Ex: 12345678" />
              </div>
            </div>

            <button className="btn btn-primary w-full py-4 rounded-xl shadow-red transition hover:scale-[1.02]" onClick={handleConfigSave}>
              <span className="font-bold">Salvar Alterações</span> <i className="ph ph-check-circle text-xl"></i>
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (currentScreen === 'cadastro') {
    return (
      <div className="flex flex-col h-full bg-dark">
        <header className="flex align-center p-6 border-b border-gray-medium bg-header shadow-md relative z-10">
          <button className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4" onClick={() => setCurrentScreen('home')}>
            <i className="ph ph-arrow-left text-xl text-white"></i>
          </button>
          <span className="font-bold text-xl text-white tracking-tight">Cadastro de Novo Usuário</span>
        </header>

        <div className="flex-grow flex flex-col p-6" style={{ overflow: 'auto' }}>
          <div className="w-full max-w-2xl bg-card border border-gray-medium shadow-2xl slide-up-anim flex flex-col" style={{ padding: '2.5rem 3rem', margin: '0 auto', flex: '1 1 auto', borderRadius: '2rem' }}>
            <div className="flex align-center gap-4 mb-8">
              <div className="circle-bg-gray flex-center" style={{ width: '64px', height: '64px', borderRadius: '50%' }}>
                <i className="ph ph-user-plus text-3xl text-primary"></i>
              </div>
              <div>
                <h2 className="font-bold text-3xl text-white tracking-tight">Criar Conta</h2>
                <p className="text-gray text-md font-medium mt-2">Insira as credenciais do novo mantenedor.</p>
              </div>
            </div>

            <div className="flex flex-col gap-8 flex-grow" style={{ justifyContent: 'space-evenly' }}>
              <div className="flex gap-6">
                <div className="flex flex-col gap-3 w-full">
                  <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Nome</label>
                  <input type="text" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="Ex: João" />
                </div>
                <div className="flex flex-col gap-3 w-full">
                  <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Sobrenome</label>
                  <input type="text" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="Ex: Silva" />
                </div>
              </div>
              <div className="flex flex-col gap-3 w-full">
                <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">E-mail Institucional</label>
                <input type="email" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="joao.silva@usp.br" />
              </div>
              <div className="flex flex-col gap-3 w-full">
                <label className="text-sm text-gray font-bold uppercase tracking-widest ml-1">Matrícula (RA)</label>
                <input type="number" className="input-dark w-full text-lg" style={{ padding: '1.1rem 1.25rem' }} placeholder="Ex: 12345678" />
              </div>
            </div>

            <button className="btn btn-primary w-full py-5 rounded-2xl shadow-red transition hover:scale-[1.02] mt-8" onClick={() => { alert('Cadastro Feito!'); setCurrentScreen('home'); }}>
              <span className="font-bold text-lg">Finalizar Cadastro</span> <i className="ph ph-check-circle text-2xl"></i>
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (currentScreen === 'camera') {
    return (
      <div className="flex flex-col h-full relative overflow-hidden bg-black">
        <div className="camera-feed-container">
          <video ref={videoRef} id="camera-feed" autoPlay playsInline muted
            onLoadedData={() => {
              startSendingFrames();
              setTimeout(() => { if (!serverConectado) generateDetection(); }, 2000);
            }}></video>
        </div>
        <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>

        <header className="flex justify-between align-center p-5 absolute top-0 w-full z-20 bg-gradient-to-b from-black/80 to-transparent">
          <button className="btn-icon circle-bg-dark border border-gray-medium shadow-sm transition hover:bg-white/10" onClick={() => { stopCamera(); setCurrentScreen('dashboard'); }}>
            <i className="ph ph-arrow-left text-white text-xl"></i>
          </button>
          <div className="text-xs font-bold text-white bg-black/60 backdrop-blur px-3 py-1.5 rounded-full border border-gray-medium shadow flex align-center gap-2">
            <div className="pulse-dot"></div> CAPTURANDO GRUPO {activeGroupForCamera}
          </div>
          <div style={{ width: '36px' }}></div>
        </header>

        <div className="flex-grow flex align-center justify-center relative w-full h-full pointer-events-none z-10">
          <div className="scan-frame relative pointer-events-none">
            <div className="corner top-left"></div>
            <div className="corner top-right"></div>
            <div className="corner bottom-left"></div>
            <div className="corner bottom-right"></div>
            <div className="scan-line"></div>
          </div>
        </div>

        <div ref={popupRef}
          onMouseDown={handlePopupDown} onTouchStart={handlePopupDown}
          className="absolute bg-black-80 backdrop-blur rounded-2xl p-5 shadow-2xl border border-gray-medium w-full max-w-sm flex flex-col max-h-[85vh] z-50 cursor-pointer"
          style={{ bottom: '40px', left: '50%', transform: 'translateX(-50%)', touchAction: 'none' }}>

          <div className="flex justify-between align-start mb-1 pointer-events-none">
            <div>
              <span className="text-primary text-[10px] font-bold tracking-widest uppercase flex align-center gap-1">
                <i className="ph-fill ph-scan"></i> DETECTANDO ALIMENTO
              </span>
              <h2 className="text-white font-bold text-xl mt-1 tracking-tight">{leituraNome}</h2>
              <p className="text-gray text-xs mt-1">{leituraDesc}</p>
            </div>
            <button className="btn-icon circle-bg-gray border border-gray-medium transition shadow-sm pointer-events-auto hover:bg-red-500"
              onClick={handleRejectClick} style={{ width: '34px', height: '34px' }}>
              <i className="ph ph-x text-white"></i>
            </button>
          </div>

          <button className="btn btn-outline w-full flex align-center justify-center gap-2 transition font-bold py-2 rounded-xl mb-4 mt-4 text-white border-gray-medium hover-bg-gray-light"
            onClick={handleSimularClick}>
            <i className="ph ph-plus-circle text-lg"></i> <span>{btnLeituraText}</span>
          </button>

          <div id="session-items-list" className="flex flex-col gap-2 mb-4 overflow-y-auto pointer-events-auto" style={{ maxHeight: '120px' }} onMouseDown={e => e.stopPropagation()} onTouchStart={e => e.stopPropagation()}>
            {currentSessionItems.length === 0 ? (
              <div className="text-center text-gray text-xs py-2 opacity-50 font-medium">Nenhum item adicionado nesta contagem.</div>
            ) : (
              currentSessionItems.map((item, index) => (
                <div key={index} className="flex justify-between align-center p-2 rounded text-sm text-white mb-2" style={{ background: 'rgba(0,0,0,0.3)' }}>
                  <span className="font-medium truncate flex-grow">{item.name}</span>
                  <div className="flex align-center gap-2 flex-shrink-0 ml-2">
                    <span className="text-primary font-bold">+{item.weight}kg</span>
                    <button className="btn-remove-session-item text-gray transition btn-icon hover:text-red-500" style={{ width: '24px', height: '24px' }}
                      onClick={() => setCurrentSessionItems(prev => prev.filter((_, i) => i !== index))}>
                      <i className="ph ph-trash"></i>
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          <button className="btn btn-primary w-full flex align-center justify-center gap-2 transition font-bold py-3 rounded-xl shadow-red mt-auto" onClick={handleFinishCount}>
            <i className="ph ph-check-circle text-lg"></i> <span>FINALIZAR CONTAGEM ({currentSessionItems.length})</span>
          </button>
        </div>
      </div>
    );
  }

  // Dashboard View
  const handleEditGroup = (group) => {
    setIsCreatingNew(false);
    setEditingGroupId(group.id);
    setTempGroupName(group.title);
    setTempMembers([...group.members]);
    setIsModalOpen(true);
  };

  const handleAddGroup = () => {
    setIsCreatingNew(true);
    setEditingGroupId(null);
    setTempGroupName('');
    setTempMembers([]);
    setIsModalOpen(true);
  };

  const handleSaveModal = () => {
    if (isCreatingNew) {
      const maxCode = appState.length > 0 ? Math.max(...appState.map(g => g.id.charCodeAt(0))) : 64;
      const newId = String.fromCharCode(maxCode + 1) || 'X';
      setAppState([...appState, {
        id: newId,
        title: tempGroupName || `Grupo ${newId}`,
        members: [...tempMembers],
        totalKg: 0,
        items: []
      }]);
    } else {
      setAppState(appState.map(g => g.id === editingGroupId ? {
        ...g,
        title: tempGroupName || `Grupo ${editingGroupId}`,
        members: [...tempMembers]
      } : g));
    }
    setIsModalOpen(false);
  };

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex justify-between align-center p-5 border-b border-gray-medium bg-header shadow-sm relative z-10">
        <div className="flex align-center gap-3 mt-1">
          <div style={{ width: '50px', height: '50px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <img src={LogoAbrace} alt="Logo" style={{ width: '130%', height: '130%', objectFit: 'contain', clipPath: 'inset(0 0 18% 0)', marginTop: '8px' }} />
          </div>
          <span className="font-black text-2xl tracking-tight text-white drop-shadow-md">ABRACE<span className="text-primary">AI</span></span>
        </div>
        <div className="flex align-center gap-5">
          <button className="btn btn-primary shadow-lg transition hover:scale-105" style={{ padding: '0.6rem 1.2rem', height: 'fit-content' }} onClick={() => console.log('Gerar Relatório ainda não implementado')}>
            <i className="ph ph-file-text text-lg"></i>
            <span className="font-bold text-xs tracking-wide">GERAR RELATÓRIO</span>
          </button>

          <div className="relative">
            <div className="flex align-center gap-3 cursor-pointer border py-2 px-4 rounded-full border-gray-medium bg-gray-light hover-opacity-100 opacity-80 transition hover-bg-gray-medium" onClick={() => setIsUserPopupOpen(!isUserPopupOpen)}>
              <i className="ph-fill ph-user-circle text-3xl text-primary"></i>
              <span className="text-sm font-bold text-white mr-1">{userData.nome}</span>
              <i className="ph ph-caret-down text-gray text-sm"></i>
            </div>

            {/* User Popup Modal */}
            <div className={`absolute mt-3 bg-card border border-gray-medium rounded-2xl shadow-2xl overflow-hidden transition ${isUserPopupOpen ? 'pop-up-show' : 'pop-up-hide'}`} style={{ zIndex: 50, cursor: 'default', right: '0', width: '260px' }}>
              <div className="p-4 border-b border-gray-medium bg-black/40">
                <div className="flex align-center gap-3">
                  <div className="circle-bg-gray flex-center" style={{ width: '45px', height: '45px', borderRadius: '50%' }}>
                    <i className="ph-fill ph-user-circle text-2xl text-primary"></i>
                  </div>
                  <div>
                    <h4 className="font-bold text-sm text-white">{userData.nome} {userData.sobrenome}</h4>
                    <p className="text-xs text-gray" style={{ userSelect: 'all' }}>{userData.email}</p>
                  </div>
                </div>
              </div>
              <div className="flex flex-col p-2">
                <button className="flex align-center gap-3 w-full p-3 text-sm text-gray hover:text-white rounded-xl transition text-left popup-btn-gray" onClick={handleGoConfig}>
                  <i className="ph ph-gear text-xl text-primary"></i>
                  <span className="font-medium">Configuração de Dados</span>
                </button>
                <div className="my-1 border-t border-gray-medium"></div>
                <button className="flex align-center gap-3 w-full p-3 text-sm text-red-500 rounded-xl transition text-left popup-btn-gray" onClick={() => { setIsUserPopupOpen(false); console.log('Conta marcada para deleção!'); }}>
                  <i className="ph ph-trash text-xl"></i>
                  <span className="font-bold tracking-wide">Deletar Conta</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-grow p-5 overflow-y-auto">
        <div className="flex justify-between align-center mb-6">
          <h2 className="font-bold text-2xl text-white tracking-tight">Esteiras em Aberto</h2>
          <span className="text-sm text-gray font-medium"><span className="text-white font-bold">{appState.length}</span> Grupos Ativos</span>
        </div>

        <div className="kanban-board">
          {appState.map(group => (
            <div key={group.id} className="kanban-card bg-card shadow-lg rounded-xl p-5 flex flex-col justify-between transition border-gray-medium" style={{ minHeight: '500px' }}>
              <div>
                <div className="flex justify-between align-center mb-2">
                  <h3 className="font-bold text-xl text-white tracking-tight">{group.title}</h3>
                  <button className="btn-icon text-gray transition hover-text-white hover-bg-gray-medium" onClick={() => handleEditGroup(group)}>
                    <i className="ph ph-pencil-simple text-lg"></i>
                  </button>
                </div>
                <div className="text-xs text-gray mb-4 flex gap-2 font-medium flex-wrap">
                  {group.members.length > 0 ? group.members.map((m, i) => (
                    <span key={i} className="text-white">{m.name} <span className="text-primary font-bold">({m.ra})</span></span>
                  )) : <span className="opacity-50">Sem integrantes</span>}
                </div>

                <div className="flex flex-col mt-4">
                  {group.items.length === 0 ? (
                    <div className="text-center text-gray py-8 text-sm opacity-50 border border-dashed border-gray-medium rounded-lg font-medium tracking-wide">
                      Aguardando contagem...
                    </div>
                  ) : (
                    group.items.map((item, idx) => (
                      <div key={idx} className="flex justify-between align-center p-3 bg-red-light rounded-lg text-sm transition hover:shadow-md border border-red-500 border-opacity-20 mb-3" style={{ border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                        <span className="flex align-center gap-2 font-bold text-white"><i className="ph-fill ph-check-circle text-primary text-xl"></i> {item.name}</span>
                        <span className="text-xs font-black text-primary badge-red px-2 py-1 rounded-full text-white">+{item.weight}kg</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
              <div className="mt-auto pt-4 relative">
                <div className="mb-4 mt-8 flex flex-col px-1">
                  <div className="font-bold text-white flex align-end gap-1">
                    <span className="text-primary text-3xl" style={{ lineHeight: 1, letterSpacing: '-1px' }}>{group.totalKg}kg</span>
                    <span className="text-sm pb-1 font-bold text-gray uppercase tracking-widest">Total</span>
                  </div>
                </div>
                <button className="btn btn-primary w-full shadow-red mt-4 py-3 rounded-xl font-bold text-md" onClick={() => handleStartCount(group.id)}>
                  <i className="ph ph-camera text-lg"></i> Iniciar Contagem
                </button>
              </div>
            </div>
          ))}

          <div className="kanban-card flex align-center justify-center p-5 text-center cursor-pointer transition border border-dashed border-gray-medium rounded-xl scale-hover hover-opacity-100 opacity-60"
            style={{ minHeight: '400px', background: 'transparent' }} onClick={handleAddGroup}>
            <div className="flex flex-col align-center justify-center items-center gap-3 h-full">
              <div className="circle-btn-red shadow-red mx-auto"><i className="ph ph-plus text-white text-2xl"></i></div>
              <span className="font-bold text-white text-md mt-2 tracking-tight block">Novo Grupo</span>
              <span className="text-xs text-gray font-medium max-w-[150px] inline-block">Adicionar novo Grupo de coleta</span>
            </div>
          </div>
        </div>
      </div>

      {isModalOpen && (
        <div className="modal-overlay active">
          <div className="modal-content">
            <div className="flex justify-between align-center mb-5">
              <h3 className="font-bold text-xl tracking-tight">{isCreatingNew ? 'Adicionar Novo Grupo' : 'Editar Grupo'}</h3>
              <button className="btn-icon circle-bg-gray border border-gray-medium transition hover-text-white" onClick={() => setIsModalOpen(false)}>
                <i className="ph ph-x"></i>
              </button>
            </div>

            <div className="flex flex-col gap-4 mb-6">
              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray font-bold uppercase tracking-widest">Nome do Grupo</label>
                <input type="text" className="input-dark" value={tempGroupName} onChange={e => setTempGroupName(e.target.value)} placeholder="Ex: Grupo C" />
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs text-gray font-bold uppercase tracking-widest">Integrantes</label>
                <div className="flex flex-col gap-2 mb-2" id="modal-members-list">
                  {tempMembers.length === 0 ? (
                    <div className="text-center text-xs text-gray opacity-50">Nenhum integrante adicionado.</div>
                  ) : (
                    tempMembers.map((member, i) => (
                      <div key={i} className="flex justify-between align-center text-sm p-2 bg-gray-light rounded border border-gray-medium">
                        <span className="text-white font-medium">{member.name} <span className="text-primary text-xs font-bold ml-1">RA: {member.ra}</span></span>
                        <button className="btn-icon" style={{ width: '20px', height: '20px' }} onClick={() => setTempMembers(prev => prev.filter((_, idx) => idx !== i))}>
                          <i className="ph ph-trash text-gray hover:text-red-500 transition"></i>
                        </button>
                      </div>
                    ))
                  )}
                </div>
                <div className="flex gap-2">
                  <input type="text" className="input-dark flex-grow" placeholder="Nome" value={memberInputName} onChange={e => setMemberInputName(e.target.value)} />
                  <input type="number" className="input-dark" style={{ width: '100px' }} placeholder="RA" value={memberInputRa} onChange={e => setMemberInputRa(e.target.value)} />
                  <button className="btn btn-outline" style={{ padding: '0 0.75rem', borderRadius: '0.5rem' }} onClick={() => {
                    if (memberInputName.trim() && memberInputRa.trim() && /^[0-9]+$/.test(memberInputRa.trim())) {
                      setTempMembers([...tempMembers, { name: memberInputName.trim(), ra: memberInputRa.trim() }]);
                      setMemberInputName(''); setMemberInputRa('');
                    } else alert("Nome e RA (apenas números) inválidos.");
                  }}><i className="ph ph-plus"></i></button>
                </div>
              </div>
            </div>

            <div className="flex gap-3 pt-2 border-t border-gray-medium mt-4 pt-4">
              {!isCreatingNew && (
                <button className="btn btn-dark w-full px-2" onClick={() => {
                  if (window.confirm("Certeza que deseja excluir o " + editingGroupId + "?")) {
                    setAppState(appState.filter(g => g.id !== editingGroupId));
                    setIsModalOpen(false);
                  }
                }}><i className="ph ph-trash text-red-500"></i> <span className="text-red-500">Excluir</span></button>
              )}
              <button className="btn btn-primary w-full" onClick={handleSaveModal}>
                {isCreatingNew ? 'Criar Grupo' : 'Salvar Alterações'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
