import json,tempfile,unittest
from pathlib import Path
import numpy as np
from charm.publication import training_frame,figure_series,tables,validate_paper_data

class PublicationTests(unittest.TestCase):
    def test_archived_inputs_have_expected_hashes(self):
        validate_paper_data()

    def test_complete_training_campaign_and_joint_compliance(self):
        f=training_frame()
        self.assertEqual(len(f),15*801)
        for metric in ['safety','validity','nrf']:
            self.assertTrue((f.joint<=f[metric]+1e-12).all())

    def test_smoothing_precedes_across_seed_statistics(self):
        f=training_frame();mean,sd=figure_series(f,'sum_rate','SLAPP')
        # At episode zero, a centered 20-point window uses episodes 0..9.
        values=[f[(f.method=='SLAPP')&(f.seed==s)&(f.episode<10)].sum_rate.mean() for s in range(1,6)]
        self.assertAlmostEqual(mean[0],np.mean(values),places=12)
        self.assertAlmostEqual(sd[0],np.std(values,ddof=1),places=12)

    def test_all_accepted_table_values(self):
        with tempfile.TemporaryDirectory() as d:
            result=tables(d)
            self.assertEqual(result,{'printed_scalar_checks':240,'passed':240,'failed':0})
            self.assertEqual(len(list(Path(d).glob('table*.tex'))),3)

if __name__=='__main__':unittest.main()
