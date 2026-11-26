import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from g1_prepared_clock import PreparedSourceClock


class PreparedClockTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.reference = Path(self.directory.name)/'reference'
        self.reference.write_bytes(b'fixed reference')
        self.path = Path(self.directory.name)/'clock.npz'
        self.acc = np.array([.2,-.1,-.1,0.,0.])
        self.phase = np.arange(5)*.1+1.
        self.speed = np.sqrt(np.r_[1.,1.+np.cumsum(.2*self.acc[:-1])])
        self.time = np.r_[1.,1.+np.cumsum(.2/(self.speed[:-1]+self.speed[1:]))]
        np.savez(self.path,time=self.time,frame=self.phase*200,speed=self.speed,
                 phase_acceleration=self.acc,source_sha256='source',
                 parent_sha256=hashlib.sha256(self.reference.read_bytes()).hexdigest())
        self.base = lambda t: (1.+(t-1.)+.1*(t-1.)**2,1.+.2*(t-1.),.2)
        self.clock = PreparedSourceClock(self.path,self.reference,'source',200.,self.base)

    def sample(self,t):
        return self.clock.sample(t,self.base(t))

    def test_knots_prefix_and_chain_rule(self):
        for t in [0.,.5,1.]:
            self.assertEqual(self.sample(t),self.base(t))
        for t,p,v in zip(self.time,self.phase,self.speed):
            np.testing.assert_allclose(self.sample(t)[:2],[p,v],rtol=0,atol=1e-12)
        h=1e-6
        for t in np.linspace(self.time[0]+.001,self.time[-1]-.001,97):
            lo,hi=self.sample(t-h),self.sample(t+h)
            np.testing.assert_allclose([(hi[0]-lo[0])/(2*h),(hi[1]-lo[1])/(2*h)],
                                       self.sample(t)[1:],atol=2e-7,rtol=0)
        for t in self.time:
            np.testing.assert_allclose(self.sample(t-1e-9),self.sample(t+1e-9),atol=1e-6,rtol=0)
        self.assertGreater(self.clock.report['minimum_speed'],0)
        self.assertAlmostEqual(self.sample(10)[0],self.phase[-1]+(10-self.time[-1])*self.speed[-1])

    def test_reject_wrong_source_reference_and_prefix(self):
        for source,base in [('wrong',self.base),('source',lambda t:(0.,1.,0.))]:
            with self.assertRaises(ValueError):
                PreparedSourceClock(self.path,self.reference,source,200.,base)
        self.reference.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            PreparedSourceClock(self.path,self.reference,'source',200.,self.base)


if __name__=='__main__':
    unittest.main()
