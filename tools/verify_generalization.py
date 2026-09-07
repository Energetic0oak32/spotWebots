"""Integration checks with simulated Gymnasium/SB3 and real worker processes.
Run directly with Python and NumPy; does not validate real learning or Webots.
"""
import os,sys,tempfile,textwrap,json,time
from pathlib import Path
from zipfile import ZipFile
root=Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory() as temp:
 d=Path(temp)
 def put(name,text):
  p=d/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(textwrap.dedent(text));return p
 put('gymnasium/__init__.py','''
 from . import spaces
 class Env:
  @property
  def unwrapped(self): return self
  def close(self): pass
 ''')
 put('gymnasium/spaces.py','''
 import numpy as np
 class Box:
  def __init__(self,low=-1,high=1,shape=(3,)):
   self.shape=shape;self.low=np.full(shape,low);self.high=np.full(shape,high);self.dtype=np.dtype('float32')
  def __eq__(self,o): return type(self)==type(o) and self.shape==o.shape
 class Discrete:
  def __init__(self,n=3,start=0):self.n=n;self.start=start;self.shape=();self.dtype=np.dtype('int64')
  def __eq__(self,o):return type(self)==type(o) and self.n==o.n
 class MultiDiscrete:
  def __init__(self,nvec):self.nvec=np.array(nvec);self.start=np.zeros_like(self.nvec);self.dtype=np.dtype('int64')
 class MultiBinary:
  def __init__(self,n=3):self.shape=(n,);self.dtype=np.dtype('int8')
 class Dict:
  def __init__(self,spaces):self.spaces=spaces
  def __eq__(self,o):return type(self)==type(o) and self.spaces==o.spaces
 ''')
 put('stable_baselines3/__init__.py','''
 class Model:
  @classmethod
  def load(cls,path,**kwargs):return cls.__name__,kwargs
 class PPO(Model):pass
 class A2C(Model):pass
 class SAC(Model):pass
 class DQN(Model):pass
 ''')
 put('stable_baselines3/common/__init__.py','')
 put('stable_baselines3/common/monitor.py','''
 class Monitor:
  def __init__(self,env):self.env=env
  def __getattr__(self,name):return getattr(self.env,name)
 ''')
 put('stable_baselines3/common/vec_env.py','''
 class VecEnv:
  def __init__(self,n,obs,act):
   self.num_envs=n;self.observation_space=obs;self.action_space=act;self.reset_infos=[{} for _ in range(n)];self._reset_seeds();self._reset_options()
  def _reset_seeds(self):self._seeds=[None]*self.num_envs
  def _reset_options(self):self._options=[{} for _ in range(self.num_envs)]
  def _get_indices(self,indices):return range(self.num_envs) if indices is None else indices
 ''')
 put('sample.py','''
 import numpy as np,time,os
 from gymnasium import Env,spaces
 class Supervisor:
  SIMULATION_MODE_REAL_TIME=1;SIMULATION_MODE_FAST=2
  def __init__(self):self.mode=2
  def simulationGetMode(self):return self.mode
  def simulationSetMode(self,m):self.mode=m
  def step(self,n):pass
 class DiscreteEnv(Env):
  def __init__(self):
   self.observation_space=spaces.Dict({'state':spaces.Box()});self.action_space=spaces.Discrete();self.webots_supervisor=Supervisor();self.steps=0
  def reset(self,seed=None,options=None):self.steps=0;return {'state':np.zeros(3)}, {'seed':seed}
  def step(self,action):
   assert type(action) is int
   self.steps+=1
   return {'state':np.ones(3)*action},1.,False,self.steps==2,{'mode':self.webots_supervisor.mode}
 class ContinuousEnv(DiscreteEnv):
  def __init__(self):super().__init__();self.action_space=spaces.Box(shape=(2,))
  def step(self,action):
   assert action.dtype==np.float32 and action.shape==(2,)
   return {'state':np.ones(3)},1.,False,False,{}
 class SlowEnv(DiscreteEnv):
  def step(self,action):time.sleep(3);return super().step(action)
 class BrokenEnv(DiscreteEnv):
  def step(self,action):raise RuntimeError('fixture error')
 ''')
 sys.path[:0]=[str(d),str(root),str(root/'training')]
 os.environ['PYTHONPATH']=str(d)
 from gymnasium import spaces
 from rl_interface.algorithms import ALGORITHMS,validate_spaces,validate_parameters,defaults,load_model,policy_for
 from rl_interface.environments import load_environment
 for name in ALGORITHMS:
  action=spaces.Discrete() if name=='dqn' else spaces.Box()
  validate_spaces(name,spaces.Box(),action);validate_parameters(name,defaults(name),2)
  archive=d/f'{name}.zip'
  with ZipFile(archive,'w') as z:z.writestr('data',json.dumps({'rl_interface_algorithm':name}))
  assert load_model(name,archive)[0]==name.upper()
 try:validate_spaces('dqn',spaces.Box(),spaces.Box())
 except ValueError:pass
 else:raise AssertionError('DQN aceitou ações contínuas')
 try:load_model('sac',d/'ppo.zip')
 except ValueError:pass
 else:raise AssertionError('Modelo incompatível aceito')
 for values in [{'n_steps':0},{'learning_rate':float('nan')},{'n_epochs':1.5},{'invalid':2}]:
  try:validate_parameters('ppo',values,1)
  except ValueError:pass
  else:raise AssertionError(values)
 with ZipFile(d/'legacy.zip','w') as z:z.writestr('data',json.dumps({'clip_range':.2,'n_epochs':10}))
 assert load_model('ppo',d/'legacy.zip')[0]=='PPO'
 assert type(load_environment(d,'sample:DiscreteEnv')).__name__=='DiscreteEnv'
 print('PASS: registro dos quatro algoritmos, parâmetros inválidos, detecção de modelo, PPO legado e carregamento configurável')
 import webots_vec_env as transport
 original_popen = transport.subprocess.Popen
 def simulated_controller(command, **kwargs):
  # Replace only the Webots executable; keep the real Python worker/process.
  return original_popen([command[0], *command[2:]], **kwargs)
 transport.subprocess.Popen = simulated_controller
 from webots_vec_env import WebotsVecEnv,WorkerFailure
 def make(cls,n=1,**kw):return WebotsVecEnv(list(range(1234,1234+n)),controller_exe=sys.executable,env_path=d,env_class='sample:'+cls,startup_timeout=5,close_timeout=.2,**kw)
 with_env=make('DiscreteEnv',2,simulation_mode='realtime')
 try:
  obs=with_env.reset();assert obs['state'].shape==(2,3);assert policy_for(with_env.observation_space)=='MultiInputPolicy'
  with_env.step_async([1,2]);obs,_,done,info=with_env.step_wait();assert not done.any();assert all(i['mode']==1 for i in info)
  with_env.step_async([2,1]);obs,_,done,info=with_env.step_wait();assert done.all() and (obs['state']==0).all();assert all(i['TimeLimit.truncated'] for i in info)
  assert info[0]['terminal_observation']['state'][0]==2
 finally:with_env.close()
 assert all(p.poll() is not None for p in with_env.processes)
 env=make('ContinuousEnv')
 try:env.reset();env.step_async([[.1,.2]]);env.step_wait()
 finally:env.close()
 print('PASS: worker genérico em processos reais; dois ambientes; Dict; ações discretas/contínuas; auto-reset; modo realtime; encerramento')
 for cls,tag in [('SlowEnv','WORKER_TIMEOUT'),('BrokenEnv','WORKER_ERROR')]:
  env=make(cls,response_timeout=.2)
  try:
   env.reset();env.step_async([1]);env.step_wait();raise AssertionError('Esperava falha')
  except WorkerFailure as e:assert tag in str(e)
  finally:env.close()
  assert all(p.poll() is not None for p in env.processes)
 print('PASS: timeout e erro do ambiente não deixam processos ativos')
 put('stable_baselines3/common/callbacks.py','class BaseCallback: pass\n')
 import train_parallel as train
 class TestModel:
  def __init__(self,policy,env,**kw):self.env=env;self.policy=policy
  def set_random_seed(self,seed):self.seed=seed
  def learn(self,*args,**kw):self.env.reset()
  def save(self,path):
   with ZipFile(path,'w') as z:z.writestr('data',json.dumps({'rl_interface_algorithm':self.rl_interface_algorithm}))
 train.algorithm_class=lambda name:TestModel
 for name in ALGORITHMS:
  cls='DiscreteEnv' if name=='dqn' else 'ContinuousEnv'
  captured=[]
  def fixture_env(ports,**kwargs):
   env=make(cls);captured.append(env);return env
  train.WebotsVecEnv=fixture_env
  sys.argv=['train','--ports','1234','--algorithm',name,'--timesteps','5','--model-path',str(d/'new_model'),'--output-model-path',str(d/f'train_{name}.zip'),'--env-path',str(d),'--env-class','sample:'+cls]
  train.main();assert captured[0].closed
  from rl_interface.algorithms import saved_algorithm
  assert saved_algorithm(d/f'train_{name}.zip')==name
 print('PASS: comando de treino cria e salva os quatro algoritmos com backend de aprendizagem simulado')
