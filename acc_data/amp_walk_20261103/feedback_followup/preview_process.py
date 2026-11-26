"""Private local solver process; Isaac retains final physical WBC validation."""
import atexit,os,pickle,struct,subprocess,sys
from pathlib import Path

def receive(stream):
 header=stream.read(8)
 if not header:raise EOFError('Prediction worker closed its output')
 if len(header)!=8:raise EOFError('Incomplete prediction header')
 size=struct.unpack('!Q',header)[0];chunks=[]
 while size:
  chunk=stream.read(size)
  if not chunk:raise EOFError('Incomplete prediction payload')
  chunks.append(chunk);size-=len(chunk)
 return pickle.loads(b''.join(chunks))
def send(stream,message):
 payload=pickle.dumps(message,protocol=5);stream.write(struct.pack('!Q',len(payload)));stream.write(payload);stream.flush()

class ProcessPreview:
 def __init__(self,configuration,log_path):
  self.final_frame=len(configuration['modes'])-1
  self.configuration=configuration;self.log_path=Path(log_path);self.process=None;self.log=None
  self.log_path.with_suffix('.configuration.pkl').write_bytes(pickle.dumps(configuration,protocol=5))
  atexit.register(self.close);self._start()
 def _start(self):
  self.log=self.log_path.open('a');env=os.environ.copy()
  env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
  python=env.get('G1_PYTHON',sys.executable)
  self.process=subprocess.Popen([python,'-u',str(Path(__file__).with_name('preview_worker.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=env)
  send(self.process.stdin,self.configuration);reply=receive(self.process.stdout)
  if not reply.get('ready'):raise RuntimeError(reply.get('error','Prediction initialization failed'))
 def refine(self,q,v,frame,rate,first,iterations=20):
  try:
   if self.process is None or self.process.poll() is not None:self.close();self._start()
   send(self.process.stdin,dict(q=q,v=v,frame=frame,rate=rate,first=first,iterations=iterations))
   reply=receive(self.process.stdout)
   if 'error' in reply:raise RuntimeError(reply['error'])
   return reply['result']
  except (EOFError,BrokenPipeError,RuntimeError) as exc:
   if self.log:self.log.write(str(exc)+'\n');self.log.flush()
   self.close()
   return dict(success=False,status='PredictionWorkerFailed',linear_violation=1e30,tube_violation=1e30,objective=1e30,worker_error=str(exc),refinement_history=[])
 def close(self):
  process=self.process;self.process=None
  if process is not None:
   if process.stdin:
    try:process.stdin.close()
    except BrokenPipeError:pass
   try:process.wait(timeout=2.)
   except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5.)
   if process.stdout:process.stdout.close()
  if self.log:self.log.close();self.log=None
