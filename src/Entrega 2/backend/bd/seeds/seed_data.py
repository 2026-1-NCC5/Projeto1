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
import api.core.security as security


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
    existente_user = db.query(Usuario).filter_by(email="nutricionista").first()
    if not existente_user:
        usuario = Usuario(
            nome="Nutricionista Teste",
            email="nutricionista",
            senha_hash=security.get_password_hash("123456"),
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
    print("✅ Usuário (nutricionista / 123456) e Equipe inicial inseridos.")


def seed_alimentos(db):
    # Busca IDs dos grupos
    graos = db.query(GrupoAlimento).filter_by(nome="Grãos").first()
    cafe = db.query(GrupoAlimento).filter_by(nome="Café").first()

    alimentos = []
    if graos:
        alimentos.extend([
            Alimento(
                nome="Arroz",
                grupo_alimento_id=graos.id,
                peso_padrao_kg=5.0,
                unidade="kg",
                classe_yolo="arroz",
            ),
            Alimento(
                nome="Feijão",
                grupo_alimento_id=graos.id,
                peso_padrao_kg=1.0,
                unidade="kg",
                classe_yolo="feijao",
            ),
        ])
    if cafe:
        alimentos.append(
            Alimento(
                nome="Café",
                grupo_alimento_id=cafe.id,
                peso_padrao_kg=0.5,
                unidade="kg",
                classe_yolo="cafe",
            )
        )

    for a in alimentos:
        existente = db.query(Alimento).filter_by(nome=a.nome).first()
        if not existente:
            db.add(a)

    db.commit()
    print("✅ Alimentos iniciais inseridos com sucesso.")


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
