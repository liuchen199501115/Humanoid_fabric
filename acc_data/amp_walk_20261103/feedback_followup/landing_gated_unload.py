"""Keep the previous support until the incoming foot reaches its source goal.

This changes only the force objective; measured contact constraints and the
independent flight/landing/source-pose audits remain unchanged.
"""
import numpy as np

def intended_unload_mode(tracker,frame,feet,source_feet,planned_mode):
 if tracker is None or tracker.complete:return int(planned_mode),'source_plan'
 event=tracker.events[len(tracker.landings)]
 if frame<event['takeoff_frame']-1:return int(planned_mode),'before_takeoff'
 swing=1-event['support'];error=float(np.linalg.norm(np.asarray(feet)[swing]-np.asarray(source_feet)[swing]))
 if tracker.airborne and frame>=event['end_frame']-1 and error<=tracker.tolerance:
  return 3,'retain_both_for_landing_hold'
 return event['support']+1,'retain_previous_support_until_source_goal'

def anticipatory_unload_mode(tracker,frame,feet,source_feet,forces,fps,planned_mode,*,horizon=.08,error_limit=.03,load_limit=100.,hold=.01,hysteresis=False,allow_overdue=False):
 if tracker is None or tracker.complete:return int(planned_mode),False
 # Once a measured transfer has started, small load fluctuations must not
 # repeatedly restart the unloading ramp. Physical contact rows remain driven
 # by the measured state machine, and loss of the new support cancels the latch.
 if hysteresis:
  pending=getattr(tracker,'_anticipatory_transfer',None)
  if pending is not None:
   event_index,incoming=pending;event=tracker.events[event_index]
   error=float(np.linalg.norm(np.asarray(feet)[incoming]-np.asarray(source_feet)[incoming]))
   if (len(tracker.landings)<=event_index and frame<event['end_frame']
       and error<=tracker.tolerance and np.asarray(forces)[incoming]>=20.):
    return incoming+1,True
   tracker._anticipatory_transfer=None
 def commit(event_index,incoming):
  if hysteresis:tracker._anticipatory_transfer=(event_index,incoming)
  return incoming+1,True
 current=tracker.events[len(tracker.landings)]
 # After a landing is fully verified, it becomes the next event's support.
 # Keep the same early unloading intent through that tracker transition.
 if tracker.landings:
  until=(current['takeoff_frame']-frame)/fps;incoming=current['support']
  error=float(np.linalg.norm(np.asarray(feet)[incoming]-np.asarray(source_feet)[incoming]))
  if 0<=until<=horizon and error<=error_limit and np.asarray(forces)[incoming]>=load_limit:
   return commit(len(tracker.landings),incoming)
 if len(tracker.landings)+1>=len(tracker.events):return int(planned_mode),False
 following=tracker.events[len(tracker.landings)+1];incoming=1-current['support']
 until=(following['takeoff_frame']-frame)/fps
 # A measured landing can occur after the next nominal takeoff, especially
 # when the source has no double-support interval. Keep the same measured
 # evidence gates for such overdue transfers; never skip beyond the next swing.
 if (following['support']!=incoming or until>horizon
     or (until<0 and not allow_overdue) or frame>=following['end_frame']):
  return int(planned_mode),False
 error=float(np.linalg.norm(np.asarray(feet)[incoming]-np.asarray(source_feet)[incoming]))
 ready=(tracker.airborne and tracker.loaded_seconds>=hold-1e-9 and error<=error_limit and np.asarray(forces)[incoming]>=load_limit)
 return commit(len(tracker.landings)+1,incoming) if ready else (int(planned_mode),False)
