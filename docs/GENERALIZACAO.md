# Generalização da interface RL — primeira etapa

## Aplicar esta atualização

Base: ZIP da branch webots-rl-interface enviado nesta conversa (commit d5fc2fcc),
com as correções posteriores de registro de execuções e recuperação da GUI.

1. Feche o aplicativo e as instâncias do Webots.
2. Extraia o conteúdo deste pacote na raiz `G:\UFSC\WeBots\Branch_App`, mesclando
   `app`, `training`, `rl_interface`, `docs` e `tools` e substituindo os arquivos correspondentes.
3. Abra o aplicativo pelo mesmo comando que já utiliza. Mantenha o ambiente virtual
   com as dependências do `requirements.txt` do projeto.

O pacote contém apenas arquivos alterados/novos, não o repositório completo.
A configuração existente é migrada ao carregar; os parâmetros que já estavam
salvos para algoritmos conhecidos são preservados. O arquivo de configuração
continua se chamando `spot_manager_config.json` para manter essa compatibilidade.

## Primeiro teste no projeto atual

- Pasta do ambiente: `G:\UFSC\WeBots\Branch_App\controllers\main`.
- Classe do ambiente: `spot_env:SpotEnv`.
- Algoritmo: `ppo`, com o modelo PPO que já funcionava.
- Mantenha um caminho de saída diferente do modelo de origem.
- Inicie o Webots, execute **Verificar compatibilidade**, faça um treino curto,
  solicite a parada e teste o arquivo de saída em tempo real.

Depois experimente A2C ou SAC com um caminho de origem que ainda não exista,
para criar um modelo novo. Trocar o algoritmo não converte um modelo PPO.
Não reutilize o mesmo arquivo de saída entre experimentos que deseja preservar.
DQN deve ser rejeitado para o Spot atual, cujo ambiente usa ações contínuas.

## O que foi separado

`rl_interface/algorithms.py` centraliza classes, parâmetros, validação de espaços,
escolha da política e carregamento dos modelos. GUI, treino, teste e inspeção
usam esse registro. Nesta etapa ele contém PPO, A2C, SAC e DQN.

`rl_interface/environments.py` carrega `modulo:Classe` da pasta selecionada.
O processo genérico `training/webots_worker.py` substitui o worker específico
como padrão do transporte. O antigo `controllers/main/parallel_worker.py` pode
permanecer no projeto, mas não é usado pelo fluxo padrão atualizado.

O campo **Robô** é um nome para identificar o experimento. O ambiente efetivamente
executado é determinado pela pasta e pela classe selecionadas. Alterar somente
o nome do robô não cria um novo ambiente.

## Conectar outro robô

Crie um módulo Python na pasta escolhida, por exemplo `meu_robo_env.py`, com uma
classe `MeuRoboEnv` que herde de `gymnasium.Env` e possa ser construída sem argumentos.
Na GUI informe `meu_robo_env:MeuRoboEnv`.

O ambiente é responsável por:

- Declarar `observation_space` e `action_space` coerentes com seus sensores e motores.
- Implementar `reset(self, *, seed=None, options=None)` e retornar `(observacao, info)`.
  Inicializar a aleatoriedade com `super().reset(seed=seed)` e usar `self.np_random`.
- Implementar `step(self, action)` e retornar
  `(observacao, recompensa, terminated, truncated, info)`.
- Avançar a simulação, aplicar comandos, calcular a recompensa, restaurar o estado
  do robô no reset e definir os critérios de término/truncamento.
- Implementar `close()` para liberar recursos próprios.

Para selecionar tempo real ou modo rápido no teste, exponha o Supervisor Webots
em `self.webots_supervisor`. O atributo `self.robot` do Spot continua aceito.
O worker reaplica o modo após reset e restaura o modo anterior ao fechar normalmente.

O mundo `.wbt` deve conter o robô correspondente aguardando um controller
`<extern>`. Nesta etapa use um único robô externo por instância do Webots, pois
o launcher ainda não seleciona um nome de robô entre vários controllers externos.
O ambiente novo e suas dependências precisam ser importáveis pelo Python do controller.

Não é necessário editar a GUI para registrar outro ambiente. É necessário escrever
seu adaptador: a interface não descobre automaticamente sensores, motores ou recompensas.

## Limites de suporte

| Algoritmo | Ações aceitas nesta etapa |
| --- | --- |
| PPO / A2C | Box vetorial, Discrete, MultiDiscrete unidimensional, MultiBinary unidimensional |
| SAC | Box vetorial com limites finitos |
| DQN | Discrete |

Observações: vetores Box, Discrete, MultiDiscrete/MultiBinary unidimensionais ou
Dict plano desses espaços. Dict seleciona `MultiInputPolicy`; os demais usam
`MlpPolicy`. Espaços discretos devem começar em zero. Imagens/CNN, Dict aninhado,
Tuple e múltiplos agentes não fazem parte desta etapa.

A verificação compara espaços e algoritmo. Ela não comprova que dois robôs com
espaços de mesmo tamanho usam a mesma ordem de sensores/motores ou a mesma tarefa.
Reutilizar pesos entre ambientes exige verificar também esse significado.

Modelos novos registram algoritmo e classe do ambiente. Modelos antigos dos quatro
algoritmos são identificados pelos campos do arquivo SB3, incluindo o PPO já usado.
Arquivos de algoritmos personalizados podem não ser identificados corretamente.

SAC e DQN retomam pesos e estado do otimizador, mas o replay buffer começa vazio.
Persistência do replay buffer e checkpoints periódicos ficam para uma etapa seguinte.
O salvamento ocorre ao fim do treino ou na parada graciosa; uma falha abrupta pode
perder o progresso desde o último arquivo salvo.

## Validação realizada

- Construção e troca de parâmetros na GUI com Qt simulado.
- Registro dos quatro algoritmos, parâmetros inválidos e incompatibilidade de modelos.
- Comunicação real por sockets/subprocessos usando ambientes e dependências simulados.
- Dois workers, observações Dict, ações discretas e contínuas, auto-reset e seleção
  de modo de simulação com Supervisor simulado.
- Timeout, erro de ambiente e encerramento dos processos.
- Fluxo de criação/salvamento do treino com backend de aprendizagem simulado.
- Regressão das correções de logs e metadados de execuções.
- Compilação dos arquivos Python.

Para repetir a verificação de transporte e registro, execute da raiz:

```powershell
python tools/verify_generalization.py
```

O script requer NumPy e usa substitutos temporários de Gymnasium/SB3. Uma exceção
`fixture error` aparece deliberadamente para validar a propagação de erros.
Não importa controllers reais nem inicia Webots. Estes testes não comprovam
aprendizagem, convergência, renderização real de Qt ou funcionamento físico de outro robô.
A validação final deve ser feita no Windows/Webots com o procedimento acima.
