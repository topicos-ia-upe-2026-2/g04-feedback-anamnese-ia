# Data card — dados-base do projeto

Formato inspirado em *Datasheets for Datasets* (Gebru et al., 2021). Consulta às fontes em 12/11/2026.

## 1. MediTOD (dados do experimento textual)

| Campo | Valor |
|---|---|
| Origem | [github.com/dair-iitd/MediTOD](https://github.com/dair-iitd/MediTOD), revisão `0b7a87c3553c9056bc8371b0809811003f94b261` |
| Artigo | Saley et al. (2024), EMNLP. https://doi.org/10.18653/v1/2024.emnlp-main.936 |
| Arquivos usados | `raw_data/dialogs.json` (~4,4 MB) e `raw_data/annotations.json` (~4,3 MB): diálogos e anotações **não canonicalizadas** |
| Não usado | Conjunto canonicalizado com UMLS (exige licença UMLS e formulário); `data/pptod/` |
| Volume | 231 diálogos alinhados (a confirmar no manifesto da E2) |
| Idioma | Inglês |
| Domínio | Entrevistas simuladas de coleta de história clínica, predominantemente respiratórias |
| Licença | ⚠️ Sem arquivo LICENSE no repositório e sem licença identificada para as anotações |
| Condição de uso adotada | Uso local para pesquisa da disciplina; **sem redistribuição** de dados ou adaptadores derivados |

**Estrutura.** Diálogo → falas com identificador, texto e falante. Anotação → atos com `uid`, intenção (`inquire`,
`inform`…), categoria (ex.: `symptom`, `medication`, `habit`, `family history`), valor e atributos (ex.: `when check`).

**Rótulos usados.** Somente atos `inquire` das falas do entrevistador. Uma amostra por fala do entrevistador, inclusive
quando a resposta esperada é lista vazia.

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

Registrar no manifesto: revisão do repositório-fonte, SHA-256 de cada arquivo, contagens por partição, exclusões e motivo.
