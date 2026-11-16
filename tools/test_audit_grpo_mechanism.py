"""Regression checks for balanced diagnostics with duplicate rollout contents."""
import copy
import unittest
from audit_grpo_mechanism import fingerprint,reindexed_content_count


def records(actions):
    return [{'uid':'group','rollout_index':i,'actions':[a],'reward':{'total':1 if a=='good'else -.85}}
            for i,a in enumerate(actions)]


class IdentityTests(unittest.TestCase):
    def test_duplicates_without_content_change_are_not_counted_as_reordering(self):
        generated=records(['good','good','bad','bad'])
        self.assertEqual(reindexed_content_count(generated,copy.deepcopy(generated)),0)
    def test_rebalanced_ordinal_must_not_assign_other_trajectories_reward(self):
        generated=records(['good','good','bad','bad']);consumed=records(['bad','good','good','bad'])
        self.assertEqual(reindexed_content_count(generated,consumed),2)
        self.assertNotEqual(fingerprint(generated[0]),fingerprint(consumed[0]))
        self.assertEqual(fingerprint(generated[0]),fingerprint(consumed[2]))


if __name__=='__main__':unittest.main()
