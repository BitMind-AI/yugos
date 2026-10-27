# Pack revisions and competition resets

A competition may add questions through an explicit pack revision. The current
king answers the new questions; its verified answers, spend and working time on
retained questions carry forward. The model submission stays the same.

`CompetitionHistory.pack_revision` links the previous frozen history, old/new pack
digests, contest identity and complete expanded-pack incumbent measurement. Replay
checks the original decisions under their original policies. The revision preserves
the scoring rule and requires addition-only question coverage and unchanged
hard/private membership for retained questions. The hosting operator additionally
verifies that retained IDs still bind identical prompts and grading contracts.

`lineage()` contains actual promotions across revisions and determines reward seats.
`comparison_lineage()` contains the current pack's baseline and later promotions.
`champion()` returns its latest comparison measurement. Updating a baseline creates
no promotion and does not shift the 85% / 10% / 5% seats. The incumbent baseline has
zero relative Net Value on the new pack; historic winning Net Values remain recorded.

Quality floors for the new pool start from its frozen calibration and incumbent
baseline, then rise with current-pack champions. Previous floors stay in their
original histories. A previous king without answers to the added questions is not
a complete measurement on the new pool.

A full reset remains available: initialize a new contest and a history without a
pack revision or challenges. It starts with no king, promotions or reward seats.
Existing histories serialize identically when `pack_revision` is absent.

Pack generation, private questions, source seeds and operator execution stay in
the hosting service. This public contract contains competition measurements and
history, without task generation logic.
