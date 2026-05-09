import sys
import os

bd_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_path = os.path.dirname(bd_path)
sys.path.insert(0, bd_path)
sys.path.insert(0, backend_path)

from models.database import SessionLocal, engine
from models.usuario import Usuario
from models.grupo import Grupo
from models.aluno import Aluno
from models.grupo_alimento import GrupoAlimento
from models.alimento import Alimento
from models.sessao import Sessao
from models.item_declarado import ItemDeclarado
from models.deteccao import Deteccao


def seed_grupos_alimentos(db):
    grupos = [
        GrupoAlimento(
            nome="Grãos",
            descricao="Arroz, feijão, lentilha e outros grãos",
            cor="#8B4513",
            icone="🌾",
        ),
        GrupoAlimento(
            nome="Enlatados",
            descricao="Alimentos em lata e conserva",
            cor="#4682B4",
            icone="🥫",
        ),
        GrupoAlimento(
            nome="Massas",
            descricao="Macarrão, lasanha e massas em geral",
            cor="#DAA520",
            icone="🍝",
        ),
        GrupoAlimento(
            nome="Laticínios",
            descricao="Leite em pó e derivados",
            cor="#F5F5DC",
            icone="🧀",
        ),
        GrupoAlimento(
            nome="Óleos",
            descricao="Óleos vegetais e azeites",
            cor="#9ACD32",
            icone="🫒",
        ),
        GrupoAlimento(
            nome="Farináceos",
            descricao="Farinhas, fubá e similares",
            cor="#DEB887",
            icone="🌽",
        ),
        GrupoAlimento(
            nome="Açúcar",
            descricao="Açúcar, adoçante e similares",
            cor="#FFB6C1",
            icone="🧂",
        ),
        GrupoAlimento(
            nome="Café",
            descricao="Café em pó e solúvel",
            cor="#3C1414",
            icone="☕",
        ),
    ]

    for g in grupos:
        existente = db.query(GrupoAlimento).filter_by(nome=g.nome).first()
        if not existente:
            db.add(g)

    db.commit()
    print("✅ Grupos de alimentos inseridos com sucesso.")


def seed_usuarios_e_equipes(db):
    existente_user = db.query(Usuario).filter_by(email="nutricionista@fecap.edu.br").first()
    if not existente_user:
        usuario = Usuario(
            nome="Nutricionista Teste",
            email="nutricionista@fecap.edu.br",
            perfil="operador"
        )
        db.add(usuario)
        
    existente_grupo = db.query(Grupo).filter_by(id=1).first()
    if not existente_grupo:
        grupo = Grupo(
            nome="Equipe Alfa",
            descricao="Equipe de teste"
        )
        db.add(grupo)
        
    db.commit()
    print("✅ Usuário nutricionista e Equipe inicial inseridos.")


def seed_alimentos(db):
    """Popula a tabela de alimentos com as 15 classes do modelo v3_final.pt.

    Cada classe YOLO vira exatamente um Alimento. Categorias agregadas
    (Arroz / Feijão / Outros) saem via grupos_alimentos.
    """
    # Mapa: classe_yolo -> (nome PT-BR, nome do grupo_alimento, peso médio kg)
    catalogo = [
        ("arroz", "Arroz", "Grãos", 5.0),
        ("feijao", "Feijão", "Grãos", 1.0),
        ("acucar", "Açúcar", "Açúcar", 1.0),
        ("sal", "Sal", "Açúcar", 1.0),
        ("cafe", "Café", "Café", 0.5),
        ("oleo", "Óleo", "Óleos", 0.9),
        ("macarrao", "Macarrão", "Massas", 0.5),
        ("farinha_trigo", "Farinha de Trigo", "Farináceos", 1.0),
        ("fuba", "Fubá", "Farináceos", 1.0),
        ("biscoito", "Biscoito", "Massas", 0.4),
        ("achocolatado", "Achocolatado", "Laticínios", 0.4),
        ("leite_em_po", "Leite em Pó", "Laticínios", 0.4),
        ("leite_condensado", "Leite Condensado", "Laticínios", 0.4),
        ("atum_sardinha", "Atum/Sardinha", "Enlatados", 0.17),
        ("molho_tomate", "Molho de Tomate", "Enlatados", 0.34),
    ]

    # Cache de grupo_alimento por nome para evitar consultas repetidas
    grupos_por_nome = {g.nome: g for g in db.query(GrupoAlimento).all()}

    for classe_yolo, nome_pt, grupo_nome, peso_kg in catalogo:
        grupo = grupos_por_nome.get(grupo_nome)
        # Procura por nome PT-BR ou pela classe YOLO (idempotência forte)
        existente = (
            db.query(Alimento)
            .filter((Alimento.nome == nome_pt) | (Alimento.classe_yolo == classe_yolo))
            .first()
        )
        if existente:
            # Garante que classe_yolo esteja preenchida em DBs antigos
            if not existente.classe_yolo:
                existente.classe_yolo = classe_yolo
            continue
        db.add(
            Alimento(
                nome=nome_pt,
                grupo_alimento_id=grupo.id if grupo else None,
                peso_padrao_kg=peso_kg,
                unidade="kg",
                classe_yolo=classe_yolo,
            )
        )

    db.commit()
    print(f"✅ {len(catalogo)} alimentos (mapeando todas as classes YOLO) inseridos.")


def main():
    print("🌱 Iniciando seed data...")
    db = SessionLocal()
    try:
        seed_grupos_alimentos(db)
        seed_alimentos(db)
        seed_usuarios_e_equipes(db)
        print("🎉 Seed data concluído!")
    except Exception as e:
        db.rollback()
        print(f"❌ Erro durante seed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
