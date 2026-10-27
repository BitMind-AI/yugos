# Rebenchmarking an incumbent

`CompetitionHistory.baseline` seeds comparisons on an expanded pack without
recording a challenge or promotion. `prior_kings` retains at most three actual
promotions for reward shares. New promotions append normally; quality floors use
the baseline and subsequent measurements on the current pack.

The host retains historical decisions under their original policies, checks the
addition-only pack transition and supplies the settled incumbent baseline. Older
histories are not embedded or rescored. A normal `CompetitionHistory(policy=...)`
continues to start with no king or reward seats, and its serialization is unchanged.
