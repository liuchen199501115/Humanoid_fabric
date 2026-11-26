"""Keep distant prediction nodes fixed as the 500 Hz source clock advances."""
import numpy as np

def anchored_grid(frame,rate,modes,origin,dt=.024,horizon=24,simulation_dt=.002):
 end=float(len(modes)-1)
 if not 0<=frame<end or rate<=0 or dt<=0 or horizon<2:raise ValueError('Invalid prediction domain')
 fine=min(.01,dt);coarse_start=origin+2*fine*rate;spacing=dt*rate
 desired=frame+(2*fine+(horizon-2)*dt)*rate
 last_index=max(0,int(np.ceil((desired-coarse_start)/spacing-1e-10)))
 finish=min(end,coarse_start+last_index*spacing)
 front=np.minimum(frame+np.array([0.,fine,2*fine])*rate,finish)
 first_index=max(0,int(np.floor((front[-1]-coarse_start)/spacing+1e-10))+1)
 coarse=coarse_start+np.arange(first_index,last_index+1)*spacing
 coarse=coarse[(coarse>front[-1]+1e-8)&(coarse<finish-1e-8)]
 changes=np.flatnonzero(np.asarray(modes)[1:]!=np.asarray(modes)[:-1])+1
 tick=simulation_dt*rate
 events=origin+np.ceil((changes-origin)/tick-1e-10)*tick
 events=events[(events>front[-1]+1e-8)&(events<finish-1e-8)]
 values=np.sort(np.r_[front,coarse,events,finish]);frames=values[np.r_[True,np.diff(values)>1e-8]]
 return frames,np.diff(frames)/rate

def anchored_control_grid(frame,rate,modes,origin,dt=.024,horizon=41,simulation_dt=.002,fine_steps=20):
 """Resolve every executed control in the 40 ms replanning window."""
 end=float(len(modes)-1)
 if not 0<=frame<end or rate<=0 or dt<=0 or horizon<=fine_steps:raise ValueError('Invalid dense prediction domain')
 fine_duration=fine_steps*simulation_dt;coarse_start=origin+fine_duration*rate;spacing=dt*rate
 desired=frame+(fine_duration+(horizon-fine_steps)*dt)*rate
 last_index=max(0,int(np.ceil((desired-coarse_start)/spacing-1e-10)))
 finish=min(end,coarse_start+last_index*spacing)
 front=np.minimum(frame+np.arange(fine_steps+1)*simulation_dt*rate,finish)
 first_index=max(0,int(np.floor((front[-1]-coarse_start)/spacing+1e-10))+1)
 coarse=coarse_start+np.arange(first_index,last_index+1)*spacing
 coarse=coarse[(coarse>front[-1]+1e-8)&(coarse<finish-1e-8)]
 changes=np.flatnonzero(np.asarray(modes)[1:]!=np.asarray(modes)[:-1])+1;tick=simulation_dt*rate
 events=origin+np.ceil((changes-origin)/tick-1e-10)*tick
 events=events[(events>front[-1]+1e-8)&(events<finish-1e-8)]
 values=np.sort(np.r_[front,coarse,events,finish]);frames=values[np.r_[True,np.diff(values)>1e-8]]
 return frames,np.diff(frames)/rate
