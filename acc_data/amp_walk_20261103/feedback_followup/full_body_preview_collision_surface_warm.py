"""Warm-start SQP from a previously verified plan, with a cold-start fallback."""
from full_body_preview_collision_surface import FullBodyPreview as BasePreview
from full_body_preview_collision_fine_release import integrate,difference
from preview_warm_start import shift_prediction

class FullBodyPreview(BasePreview):
    def solve(self, q, v, frame, rate, first, nominal=None):
        if nominal is None:
            nominal = getattr(self, '_initial_guess', None)
        return super().solve(q, v, frame, rate, first, nominal=nominal)

    def refine(self, q, v, frame, rate, first, iterations=20):
        previous = getattr(self, '_previous_solution', None)
        intervals=(self.prediction_intervals(frame,rate) if hasattr(self,'prediction_intervals') else
                   self.step_durations() if hasattr(self,'step_durations') else self.dt)
        count=len(intervals) if hasattr(intervals,'__len__') else self.N
        seed = None if previous is None else shift_prediction(
            previous, self.reference, frame, rate, intervals, count, integrate, difference,integration=previous.get('integration','continuous'))
        self._initial_guess = seed
        self.abort_contact_restoration_stall = seed is not None
        try:
            result = super().refine(q, v, frame, rate, first, iterations=iterations)
        finally:
            self._initial_guess = None
            self.abort_contact_restoration_stall = False
        warm_iterations = len(result.get('refinement_history', [])) if seed is not None else 0
        fallback = seed is not None and not result['success']
        warm_failure = {key:result.get(key) for key in ('status','refinement_history','contact_patch_release_history')} if fallback else None
        if fallback:
            result = super().refine(q, v, frame, rate, first, iterations=iterations)
        result.update(warm_failure=warm_failure,warm_started=seed is not None, warm_iterations=warm_iterations,
                      warm_start_fallback=fallback)
        if result['success']:
            self._previous_solution = result
        return result
