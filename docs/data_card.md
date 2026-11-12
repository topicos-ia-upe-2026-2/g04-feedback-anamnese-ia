# Data card — dados-base do projeto

Formato inspirado em *Datasheets for Datasets* (Gebru et al., 2021). Consulta às fontes em 12/11/2026.

## 1. MediTOD (dados do experimento textual)

| Campo | Valor |
|---|---|
| Origem | [github.com/dair-iitd/MediTOD](https://github.com/dair-iitd/MediTOD), revisão `0b7a87c3553c9056bc8371b0809811003f94b261` |
| Artigo | Saley et al. (2024), EMNLP. https://doi.org/10.18653/v1/2024.emnlp-main.936 |
| Arquivos usados | `raw_data/dialogs.json` (~4,4 MB) e `raw_data/annotations.json` (~4,3 MB): diálogos e anotações **não canonicalizadas** |
| Não usado | Conjunto canonicalizado com UMLS (exige licença UMLS e formulário); `data/pptod/` |
| Volume | 231 diálogos (216 com prefixo `r1`, 15 com `r3`), 22.317 falas, 10.082 atos `inquire` — medido em 12/11/2026 |
| SHA-256 | `dialogs.json` `9d79ecaa10e8da630f9accdb52a1be9a7b9bd812b9e3d19b77579e320a4123b6` · `annotations.json` `828a802da0c5d1c42e92ac15b0fb79b2d34c20e78c9d1c70e1b4423424982c59` |
| Idioma | Inglês |
| Domínio | Entrevistas simuladas de coleta de história clínica, predominantemente respiratórias |
| Licença | ⚠️ Sem arquivo LICENSE no repositório e sem licença identificada para as anotações |
| Condição de uso adotada | Uso local para pesquisa da disciplina; **sem redistribuição** de dados ou adaptadores derivados |

**Estrutura.** Diálogo → falas com identificador, texto e falante. Anotação → atos com `uid`, intenção (`inquire`,
`inform`…), categoria (ex.: `symptom`, `medication`, `habit`, `family history`), valor e atributos (ex.: `when check`).

Os atributos investigados aparecem como chaves booleanas `"<atributo> check": true` dentro do ato, por exemplo
`{"intent": "inquire", "slot": "symptom", "value": "cough", "characteristics check": true}`.

**Rótulos usados.** Somente atos `inquire` das falas do entrevistador. Uma amostra por fala do entrevistador, inclusive
quando a resposta esperada é lista vazia.

## Estatísticas descritivas (medidas em 12/11/2026)

Leitura direta dos dois JSON na revisão acima, sem treino nem alteração dos dados.

| Medida | Valor |
|---|---|
| Diálogos em `dialogs.json` e em `annotations.json` | 231 e 231 (mesmas chaves) |
| Diálogos com número de falas = número de listas de atos | 231 (100%) |
| Falas por falante | médico 11.097 · paciente 10.989 · controle 231 |
| Falas do médico com ≥ 1 `inquire` | 9.707 (87,5%) |
| Falas do médico sem `inquire` (amostras de lista vazia) | 1.390 (12,5%) |
| Atos `inquire` em falas que não são do médico | 12 (excluídos da tarefa; registrar na E2) |
| Total de atos `inquire` | 10.082 |
| Pares de diálogos com texto idêntico | 18 → 213 grupos únicos |

**Atos `inquire` por categoria (`slot`):**

| Categoria | Atos | Categoria | Atos |
|---|---|---|---|
| `symptom` | 5.527 | `residence` | 244 |
| `medical history` | 1.406 | `travel` | 199 |
| `habit` | 795 | `symptom (dermatology)` | 160 |
| `medication` | 520 | `medical test` | 106 |
| `exposure` | 488 | `occupation` | 91 |
| `family history` | 457 | `other` | 80 |
| | | `basic information` | 9 |

**Atributos mais investigados:** `when check` 526 · `location check` 384 · `ana factors check` 383 ·
`characteristics check` 252 · `progression check` 154 · `frequency check` 134 (+ 69 grafados `frquency check`).

**Implicações para o plano.**
- A classe `symptom` concentra 55% dos atos: reportar F1 também por categoria, não só o micro.
- Categorias raras (`basic information`, `other`, `occupation`) terão poucas amostras no teste; resultados nelas serão
  apresentados como indicativos.
- A variante `frquency check` confirma a necessidade do passo de normalização de grafia.
- Os 18 pares idênticos (ex.: `106`/`206`, `124`/`624`) tornam obrigatória a divisão por grupo.

<details>
<summary>Como reproduzir estas contagens</summary>

```bash
REV=0b7a87c3553c9056bc8371b0809811003f94b261
for f in dialogs annotations; do
  curl -sL -o /tmp/$f.json "https://raw.githubusercontent.com/dair-iitd/MediTOD/$REV/raw_data/$f.json"
done
shasum -a 256 /tmp/dialogs.json /tmp/annotations.json
python3 - <<'EOF'
import json, collections, hashlib
d = json.load(open('/tmp/dialogs.json')); a = json.load(open('/tmp/annotations.json'))
spk, slots = collections.Counter(), collections.Counter(); inq = 0; groups = set()
for k in d:
    U, A = d[k]['utterances'], a[k]
    groups.add(hashlib.sha256('\n'.join(u['text'] for u in U).encode()).hexdigest())
    for u, acts in zip(U, A):
        spk[u['speaker']] += 1
        for x in acts:
            if x.get('intent') == 'inquire':
                inq += 1; slots[x.get('slot')] += 1
print(len(d), dict(spk), inq, len(groups)); print(slots.most_common())
EOF
```
O script versionado que gera o manifesto entra em `src/` na E2.
</details>

**Divisão.** Os arquivos `data/*_ids.json` do repositório têm um id cada (`115`, `317`, `407`) e não reproduzem a
divisão do artigo. Divisão própria por grupo de diálogo (diálogos textualmente idênticos no mesmo grupo), separada
**antes** de gerar amostras, com semente registrada.

**Vieses e lacunas.** Recorte majoritariamente respiratório; atores simulando pacientes; inglês; alergias, vacinação
e revisão de sistemas sem categoria própria.

## 2. Fareez et al. (2022) (áudio, E3/E4)

| Campo | Valor |
|---|---|
| Origem | [Figshare, item 16550013](https://springernature.figshare.com/articles/dataset/Collection_of_simulated_medical_exams/16550013) |
| Artigo | Fareez et al. (2022), *Scientific Data*, 9, 313. https://doi.org/10.1038/s41597-022-01423-1 |
| Conteúdo | Entrevistas médicas simuladas com foco respiratório; `Data.zip` (~1,03 GB) |
| Licença | CC0 (informada pela API do item) |
| Papel | Fonte de áudio planejada; não é necessária para a primeira adaptação textual |

O CC0 dos diálogos de origem **não** determina a licença das anotações que o MediTOD acrescentou.

## 3. Dados futuros

- **Exemplos em português:** didáticos ou sintéticos; cada arquivo registra origem e finalidade. Servem para testes
  técnicos e não substituem validação humana.
- **Gravações da dupla:** ainda não coletadas. Antes: termo de consentimento, anonimização e regra de compartilhamento.

## Reprodutibilidade (E2)

Registrar no manifesto: revisão do repositório-fonte, SHA-256 de cada arquivo (valores acima), contagens por partição,
exclusões e motivo.
