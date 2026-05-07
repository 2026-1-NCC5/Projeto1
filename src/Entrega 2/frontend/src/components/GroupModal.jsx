import React, { useState } from 'react';

// Modal de criação/edição de grupo. Encapsula os inputs de nome do grupo +
// lista de integrantes (nome/RA). Não persiste nada por conta própria —
// chama onSave / onDelete quando o usuário confirma.
export default function GroupModal({
  aberto,
  isCreatingNew,
  initialTitle,
  initialMembers,
  editingGroupId,
  onSave,
  onDelete,
  onClose,
}) {
  const [tempGroupName, setTempGroupName] = useState(initialTitle);
  const [tempMembers, setTempMembers] = useState(initialMembers);
  const [memberInputName, setMemberInputName] = useState('');
  const [memberInputRa, setMemberInputRa] = useState('');

  // Re-sincroniza estado interno se a prop mudar enquanto o modal está aberto
  // (ex.: usuário fecha edit de A e abre edit de B sem desmontar).
  React.useEffect(() => {
    setTempGroupName(initialTitle);
    setTempMembers(initialMembers);
    setMemberInputName('');
    setMemberInputRa('');
  }, [initialTitle, initialMembers, editingGroupId, isCreatingNew]);

  if (!aberto) return null;

  const handleAddMember = () => {
    if (memberInputName.trim() && memberInputRa.trim() && /^[0-9]+$/.test(memberInputRa.trim())) {
      setTempMembers([...tempMembers, { name: memberInputName.trim(), ra: memberInputRa.trim() }]);
      setMemberInputName(''); setMemberInputRa('');
    } else alert("Nome e RA (apenas números) inválidos.");
  };

  const handleSave = () => {
    onSave({ title: tempGroupName, members: tempMembers });
  };

  return (
    <div className="modal-overlay active">
      <div className="modal-content">
        <div className="flex justify-between align-center mb-5">
          <h3 className="font-bold text-xl tracking-tight">{isCreatingNew ? 'Adicionar Novo Grupo' : 'Editar Grupo'}</h3>
          <button className="btn-icon circle-bg-gray border border-gray-medium transition hover-text-white" onClick={onClose}>
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
              <button className="btn btn-outline" style={{ padding: '0 0.75rem', borderRadius: '0.5rem' }} onClick={handleAddMember}>
                <i className="ph ph-plus"></i>
              </button>
            </div>
          </div>
        </div>

        <div className="flex gap-3 pt-2 border-t border-gray-medium mt-4 pt-4">
          {!isCreatingNew && (
            <button className="btn btn-dark w-full px-2" onClick={() => {
              if (window.confirm("Certeza que deseja excluir o " + editingGroupId + "?")) {
                onDelete(editingGroupId);
              }
            }}><i className="ph ph-trash text-red-500"></i> <span className="text-red-500">Excluir</span></button>
          )}
          <button className="btn btn-primary w-full" onClick={handleSave}>
            {isCreatingNew ? 'Criar Grupo' : 'Salvar Alterações'}
          </button>
        </div>
      </div>
    </div>
  );
}
