import unittest
from analyze_grpo_learnability import early_mixed, period, describe, selection_permutation


def group(step, gold, kept=True, invalid=False):
    return {'step':step,'gold_rollouts':gold,'group_size':4,'kept':kept,
            'drop_reason':'sampling_invalid' if invalid else '', 'trained':False,'mean_steps':8}


class LearnabilityTests(unittest.TestCase):
    def test_membership_never_uses_late_outcomes(self):
        self.assertFalse(early_mixed([group(20,4),group(450,2)]))
        self.assertTrue(early_mixed([group(20,2),group(450,0)]))

    def test_invalid_mixed_group_does_not_qualify(self):
        self.assertFalse(early_mixed([group(20,2,kept=False,invalid=True)]))

    def test_primary_keeps_invalid_observations(self):
        gs=[group(401,0,kept=False,invalid=True),group(402,4)]
        self.assertEqual(period(gs,401,500)['p'],.5)
        self.assertEqual(period(gs,401,500,True)['p'],1)

    def test_task_weighting_is_not_rollout_weighting(self):
        a={'p':0,'rollouts':4,'gold':0,'invalid_groups':0}
        b={'p':1,'rollouts':4,'gold':4,'invalid_groups':0}
        c={'p':1,'rollouts':40,'gold':40,'invalid_groups':0}
        d={'p':0,'rollouts':40,'gold':0,'invalid_groups':0}
        result=describe([(a,b),(c,d)],200)
        self.assertEqual(result['delta_task_mean'],0)
        self.assertNotEqual(result['early_rollout_rate'],result['late_rollout_rate'])

    def test_reselection_null_detects_positive_selection_baseline(self):
        # Under shuffled time labels, selection still requires the .5 group
        # to land early; the remaining all-gold group then gives +.5 by design.
        result=selection_permutation({i:[group(20,2),group(450,4)] for i in range(20)},100)
        self.assertEqual(result['early_mixed_reselected']['null_mean'],.5)
        self.assertEqual(result['early_mixed_reselected']['one_sided_ge_observed'],1)


if __name__=='__main__':unittest.main()
