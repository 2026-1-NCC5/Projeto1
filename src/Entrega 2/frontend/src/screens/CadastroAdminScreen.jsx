import React, { useState } from 'react';
import LogoAbrace from '../assets/logo_final_nova.png';
import { useAppState } from '../context/appStateContextValue';
import { authCadastroAdmin } from '../services/api';

export default function CadastroAdminScreen() {
  const { setCurrentScreen, refreshMe, addToast } = useAppState();
  const [nome, setNome] = useState('');
  const [email, setEmail] = useState('');
  const [enviando, setEnviando] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setEnviando(true);
    try {
      const r = await authCadastroAdmin({ nome: nome.trim(), email: email.trim() });
      if (!r.ok) {
        const det = r.data?.detail;
        const msg = typeof det === 'string' ? det : 'Não foi possível cadastrar.';
        addToast(msg, 'error');
        return;
      }
      await refreshMe();
      setCurrentScreen('dashboard');
      addToast('Conta de administrador criada e sessão iniciada.', 'success');
    } catch {
      addToast('Servidor indisponível.', 'warning');
    } finally {
      setEnviando(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-dark">
      <header className="flex align-center p-6 border-b border-gray-medium bg-header shadow-md relative z-10">
        <button
          type="button"
          className="btn-icon circle-bg-gray border border-gray-dark shadow transition hover-bg-gray-light mr-4"
          onClick={() => setCurrentScreen('home')}
        >
          <i className="ph ph-arrow-left text-xl text-white"></i>
        </button>
        <span className="font-bold text-xl text-white tracking-tight">Cadastro de Administrador</span>
      </header>

      <div
        className="flex-grow flex flex-col align-center justify-center py-6 px-8 w-full"
        style={{ overflow: 'auto' }}
      >
        <div
          className="w-full max-w-md mx-auto bg-card border border-gray-medium shadow-2xl slide-up-anim flex flex-col"
          style={{ padding: '2.5rem 2rem', borderRadius: '2rem' }}
        >
          <div className="flex align-center gap-4 mb-6">
            <img
              src={LogoAbrace}
              alt="AbraceAI"
              style={{ width: '64px', height: '64px', objectFit: 'contain', clipPath: 'inset(0 0 18% 0)' }}
            />
            <div>
              <h2 className="font-bold text-2xl text-white tracking-tight">Nova conta</h2>
              <p className="text-gray text-sm mt-1">Nome completo e e-mail institucional permitido pelo sistema.</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-5">
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray font-bold uppercase tracking-widest ml-1">Nome completo</label>
              <input
                type="text"
                className="input-dark w-full text-lg"
                style={{ padding: '1rem 1.15rem' }}
                value={nome}
                onChange={(e) => setNome(e.target.value)}
                placeholder="Ex: Maria Silva"
                autoComplete="name"
                required
              />
            </div>
            <div className="flex flex-col gap-2">
              <label className="text-xs text-gray font-bold uppercase tracking-widest ml-1">E-mail institucional</label>
              <input
                type="email"
                className="input-dark w-full text-lg"
                style={{ padding: '1rem 1.15rem' }}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="nome@fecap.edu.br"
                autoComplete="email"
                required
              />
            </div>
            <button type="submit" className="btn btn-primary w-full py-4 rounded-2xl shadow-lg" disabled={enviando}>
              <span className="font-bold">{enviando ? 'Cadastrando…' : 'Criar conta e entrar'}</span>
              <i className="ph ph-check-circle text-xl"></i>
            </button>
          </form>

          <div className="auth-screen-footer">
            <p className="text-sm text-gray">
              Já tem conta?{' '}
              <button type="button" className="auth-screen-footer-link" onClick={() => setCurrentScreen('login')}>
                Fazer login
              </button>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
