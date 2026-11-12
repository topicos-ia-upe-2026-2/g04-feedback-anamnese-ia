# Model card — extrator de perguntas de anamnese (planejado)

Formato inspirado em *Model Cards for Model Reporting* (Mitchell et al., 2019). Na E1 descreve o modelo **planejado**;
será atualizado com números medidos a partir da E2.

## Identificação

| Campo | Valor |
|---|---|
| Modelo-base | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revisão `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Distribuição usada | [mlx-community/Qwen3-4B-Instruct-2507-4bit](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit), revisão `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b` |
| Licença do modelo-base | Apache-2.0 (consultado em 12/11/2026) |
| Tarefa original | Geração de texto e seguimento de instruções, multilíngue |
| Adaptação | LoRA em parte das camadas, pesos-base congelados, pesos em 4 bits, `mlx-lm` |
| Artefato gerado | Adaptador LoRA (mantido local enquanto a licença das anotações do MediTOD não for esclarecida) |
| Responsáveis | Leonardo Marcelo e André Lucas |

## Uso pretendido

- Identificar, nas falas do entrevistador de uma anamnese **simulada**, quais investigações foram feitas
  (categoria, valor e atributos do CMAS), para alimentar feedback educacional revisável pelo preceptor.
- Usuários: estudantes de medicina em prática; preceptores como revisores.

## Fora de escopo

- Diagnóstico, prescrição, recomendação de conduta ou qualquer apoio à decisão clínica.
- Pacientes reais, dados de prontuário ou avaliação somativa (nota) sem revisão humana.
- Afirmar que a anamnese está completa: o CMAS não cobre bem alergias, vacinação e revisão de sistemas.

## Entrada e saída

- **Entrada:** histórico recente do diálogo + fala atual do entrevistador (texto, inglês na E2).
- **Saída:** lista JSON de atos `inquire`: `[{"slot": "...", "value": "...", "attributes": ["..."]}]`; lista vazia quando não há investigação.

## Dados de treino e avaliação

MediTOD `raw_data/` (revisão `0b7a87c`), partições próprias por grupo de diálogo. Ver [`data_card.md`](data_card.md).

## Avaliação planejada

Precisão, recall e F1 micro dos rótulos; taxa de JSON válido; perda; latência e memória — comparados com o mesmo
modelo sem adaptador. Protocolo em [`experimentos.md`](experimentos.md). **Sem resultados medidos ainda.**

## Limitações e riscos conhecidos

- Ajuste em inglês: desempenho em português desconhecido até ser medido.
- Corpus pequeno e predominantemente respiratório: pode não generalizar para outros casos.
- Erros de extração podem virar feedback errado. Mitigação: evidência por trecho e estado "inconclusivo" na aplicação.
- Números do artigo-base (ex.: F1 0,7139 do Llama3-8B) vêm de outra tarefa e não são comparáveis.
