from pathlib import Path
import difflib
p=Path(__file__).parent
a=(p/'trial_flight_feedback.py').read_text().splitlines(True);b=(p/'trial_smooth_clock.py').read_text().splitlines(True);s=(p/'trial_tracking_tube.py').read_text()
for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b).get_opcodes():
 if tag=='equal':continue
 old=''.join(a[i:j]);new=''.join(b[k:l])
 if not old:
  anchor=''.join(a[max(0,i-1):i]);assert s.count(anchor)==1;s=s.replace(anchor,anchor+new,1)
 else:assert s.count(old)==1,(tag,old);s=s.replace(old,new,1)
for arg in ['foot_feedback_starts[foot]','late_leg_feedback_start','waist_feedback_start','com_feedback_metric_start','horizontal_feedback_start','global_posture_feedback_start','upper_posture_metric_start','late_foot_metric_start','late_foot_angular_metric_start','late_body_angular_metric_start','body_feedback_start','posture_total_accel_start','late_joint_tube_start','late_momentum_priority_start','joint_limit_priority_start','minimum_support_acceleration_start','synchronous_effort_budget_start','centroidal_preview_start']:
 s=s.replace('control_time-args.'+arg,'elapsed_motion_time-args.'+arg)
for arg in ['tracking_tube_start','late_joint_tube_start','position_tube_start','full_swing_pose_start','landing_gated_unload_start','anticipate_unload_start','late_unload_start','support_orientation_priority_start','late_balance_hierarchy_start','late_momentum_priority_start','late_pelvis_priority_start','transition_swing_priority_start','stance_posture_secondary_start','joint_limit_priority_start','minimum_support_acceleration_start','synchronous_effort_budget_start','centroidal_preview_start']:
 s=s.replace('control_time>=args.'+arg,'elapsed_motion_time>=args.'+arg)
s=s.replace('frame_position+frame_rate*offset+.5*frame_acceleration*offset**2,\n                        frame_rate+frame_acceleration*offset,frame_acceleration)',
    'motion_start_frame+motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed)[0],\n                        motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed)[1],\n                        motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed)[2])')
s=s.replace("    parser.add_argument('--clock-final-speed', type=float, default=.7)",
    "    parser.add_argument('--clock-final-speed', type=float, default=.7)\n    parser.add_argument('--clock-second-start',type=float,default=100.)\n    parser.add_argument('--clock-second-ramp',type=float,default=.3)\n    parser.add_argument('--clock-second-speed',type=float,default=None)\n    parser.add_argument('--clock-third-start',type=float,default=100.)\n    parser.add_argument('--clock-third-ramp',type=float,default=.2)\n    parser.add_argument('--clock-third-speed',type=float,default=None)")
s=s.replace('args.clock_final_speed)', 'args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)')
s=s.replace('control_time<args.tracking_tube_end','elapsed_motion_time<args.tracking_tube_end')
s=s.replace('effort_fraction(control_time,args.synchronous_effort_budget_start','effort_fraction(elapsed_motion_time,args.synchronous_effort_budget_start')
s=s.replace('                from smooth_clock import smooth_clock\n',"""                from smooth_clock import smooth_clock
                if args.contact_pause_end is not None:
                    from landing_pause_clock import LandingPauseClock
                    if args.contact_pause_early_resume:
                        from early_resume_clock import EarlyResumeLandingClock as LandingPauseClock
                    if 'source_contact_pause' not in locals():
                        source_contact_pause=LandingPauseClock(args.contact_pause_start,args.contact_pause_end,args.contact_pause_resume,args.contact_pause_landings)
                    speed_clock=smooth_clock
                    input_phase=speed_clock(wall_motion_time,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                    source_contact_pause.update(input_phase,len(measured_steps.landings) if measured_steps is not None else 0)
                    def smooth_clock(time,*parameters):
                        phase,rate,acceleration=speed_clock(time,*parameters)
                        out,phase_rate,phase_acceleration=source_contact_pause.sample(phase)
                        return out,phase_rate*rate,phase_acceleration*rate**2+phase_rate*acceleration
""")
compile(s,'trial_tracking_tube_clock.py','exec');(p/'trial_tracking_tube_clock.py').write_text(s)
