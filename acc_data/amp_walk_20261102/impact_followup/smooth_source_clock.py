"""Analytic monotone clock with a smooth rate reduction after the third landing."""
def source_clock(time, start=.92, duration=.4, final_rate=.65):
    if time<=start:return time,1.,0.
    if time>=start+duration:
        return start+duration*(1.+final_rate)/2+(time-start-duration)*final_rate,final_rate,0.
    u=(time-start)/duration;delta=1.-final_rate
    smooth=10*u**3-15*u**4+6*u**5
    integral=2.5*u**4-3*u**5+u**6
    return start+duration*(u-delta*integral),1.-delta*smooth,-delta/duration*30*u**2*(1-u)**2
