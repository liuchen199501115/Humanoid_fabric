"""Sequential closed-loop trials; original fidelity and landing checks stay on."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
# Pin the one-step baseline so promoting the main profile does not silently
# change these historical comparisons. Candidate options override these values.
BASE_OPTIONS = ['--posture-metric-scale', '1', '--joint-feedback-limit', '30',
                '--projection-joint-correction-weight', '0', '--qp-base-acceleration-weight', '1',
                '--source-pelvis-weight', '1', '--fabric-angular-feedback-limit', 'default',
                '--joint-tube-end', 'inf', '--com-acceleration-preservation-weight', '0']
TRIALS = {
    'posture10_feedback150': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150'],
    'posture10_capped150': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                            '--no-preserve-reference-acceleration'],
    'posture10_half_speed': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                            '--motion-speed', '.5', '--clock-final-speed', '1', '--duration', '24'],
    'half_speed_base100': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                          '--motion-speed', '.5', '--clock-final-speed', '1', '--duration', '24',
                          '--qp-base-acceleration-weight', '100'],
    'original_reference': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                           '--swing-midpoint-clearance', '0', '--smooth-stride', '0'],
    'original_half_speed': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                            '--swing-midpoint-clearance', '0', '--smooth-stride', '0',
                            '--motion-speed', '.5', '--clock-final-speed', '1', '--duration', '24'],
    'half_speed_attitude': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                            '--motion-speed', '.5', '--clock-final-speed', '1', '--duration', '24',
                            '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
                            '--fabric-angular-feedback-limit', '100'],
    'full_speed_attitude': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                            '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
                            '--fabric-angular-feedback-limit', '100'],
    'projection_joint1': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                          '--projection-joint-correction-weight', '1'],
    'projection_joint10': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                           '--projection-joint-correction-weight', '10'],
    'projection1_attitude': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                             '--projection-joint-correction-weight', '1',
                             '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
                             '--fabric-angular-feedback-limit', '100'],
    'projection1_base100': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                            '--projection-joint-correction-weight', '1',
                            '--qp-base-acceleration-weight', '100'],
    'projection1_attitude3': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                              '--projection-joint-correction-weight', '1',
                              '--qp-base-acceleration-weight', '10', '--source-pelvis-weight', '3',
                              '--fabric-angular-feedback-limit', '100'],
    'projection1_attitude_no_joint_tube': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                             '--projection-joint-correction-weight', '1',
                             '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
                             '--fabric-angular-feedback-limit', '100', '--joint-tube-end', '0'],
    'projection1_steady_clock': ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
                             '--projection-joint-correction-weight', '1',
                             '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
                             '--fabric-angular-feedback-limit', '100', '--joint-tube-end', '0',
                             '--clock-final-speed', '1'],
    'projection1_posture3': ['--posture-metric-scale', '3', '--joint-feedback-limit', '150',
                             '--projection-joint-correction-weight', '1',
                             '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
                             '--fabric-angular-feedback-limit', '100', '--joint-tube-end', '0'],
}
for com_weight in (100, 1000):
    TRIALS[f'projection1_com{com_weight}'] = [
        *TRIALS['projection1_attitude_no_joint_tube'],
        '--com-acceleration-preservation-weight', str(com_weight)]

def run(name, options):
    output = OUT / (name + '.npz')
    command = ['bash', 'examples/run_amp_full_motion.sh', '--output-path', str(output), *BASE_OPTIONS, *options]
    (OUT / (name + '.command.json')).write_text(json.dumps(command, indent=2)+'\n')
    sources = ['examples/test_g1.py', 'src/fabrics_sim/fabrics/dynamics_aware.py',
               'src/fabrics_sim/solver/dynamic_consistent_gpu.py',
               'src/fabrics_sim/solver/daqp_backend.py']
    (OUT / (name + '.sources.json')).write_text(json.dumps({
        source: hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
        for source in sources}, indent=2)+'\n')
    environment = dict(os.environ, G1_PYTHON='/home/sky/anaconda3/envs/humanoid_fabric/bin/python')
    with (OUT / (name + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=240)
    row = dict(name=name, returncode=result.returncode, options=options)
    for suffix in ('steps', 'tracking'):
        report = OUT / (name + '.' + suffix + '.json')
        if report.exists():
            data = json.loads(report.read_text())
            for key in ('measured_steps', 'last_time_s', 'stop_reason', 'fidelity_stop',
                        'full_motion_tracking_passed', 'peak_joint_rmse'):
                if key in data:
                    row[key] = data[key]
    (OUT / (name + '.result.json')).write_text(json.dumps(row, indent=2)+'\n')
    print(json.dumps(row), flush=True)

if __name__ == '__main__':
    for name in sys.argv[1:] or TRIALS:
        run(name, TRIALS[name])
