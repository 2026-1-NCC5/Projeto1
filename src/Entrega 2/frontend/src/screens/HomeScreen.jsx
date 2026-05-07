import React from 'react';
import LogoAbrace from '../assets/logo_final_nova.png';
import { useAppState } from '../context/appStateContextValue';

// Tela de boas-vindas. Dois CTAs: "Começar Triagem" (vai para Dashboard)
// e "Cadastrar Admin" (vai para Cadastro).
export default function HomeScreen() {
  const { setCurrentScreen } = useAppState();
  const handleStart = () => setCurrentScreen('dashboard');
  const handleGoCadastro = () => setCurrentScreen('cadastro');

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
