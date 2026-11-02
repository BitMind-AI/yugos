# Scoring

There is one competition and one submitted router. Chat/QA, math/code, and
agentic are parts of a single score. They are not separate championships.

A challenger is compared with the current champion on the same questions.
Promotion requires every category gate to pass and the lower bound on net
value to clear the minimum. Validators then pay the last three champions
**85%**, **10%**, and **5%** (current, previous, and the one before). Shares
for seats that do not exist yet stay with the current champion. The same
hotkey keeps every seat it holds.

The live policy is part of the published result. A change to those numbers is
a new policy, which validators of this major version (`yugos-overall-v2`)
follow without an upgrade.

## Net value

The next competition pack, `v0.2.0`, uses **m = 30**, **k = 100**, and
the same overall dollar penalty in every category. Its calibrated pack contains
2,642 questions: 1,438 public and 1,204 private. Evaluations use up to eight
concurrent questions, with Anthropic prompt caching enabled for submission calls.
Every submission, including the previous champion, must complete the new pack.
The contest starts without an incumbent; earlier results, decisions and champion history remain archived. The
first eligible champion receives 100% of rewards, followed by 90/10 and then
85/10/5 as the new champion history grows.

For this pack, let `B` be the sum of the calibrated category leaders' model spend on
the pack, `w_c` the category weight, and `n_c` its question count. Preparation
freezes `V_c = m * B * w_c / n_c`. The resulting overall spend contribution is
`-100 * (challenger spend - champion spend) / (m * B)`: a dollar has the same
penalty whichever category spends it. Working time retains a separate anchor
for each category. Older policies retain their original scoring terms.

```text
c ∈ {chat_qa, math_code, agentic}
g_{A,i}, g_{K,i} ∈ [0, 1]
s_{A,i}, s_{K,i}              USD spent on question i
t_{A,i}, t_{K,i}              working time on question i
V_c = m * B * w_c / n_c                  (v0.2.0 pack)
V_c = m * anchor_cost_per_question[c]    (earlier policies)
T_c = k * anchor_time_per_question[c]
u_i = 2 if question i is hard, else 1

NV_c = 100 * Σ_i [ u_i * (g_{A,i} - g_{K,i}) - (s_{A,i} - s_{K,i}) / V_c - (t_{A,i} - t_{K,i}) / T_c ] / n_c

NV = 0.40 * NV_chat_qa + 0.35 * NV_math_code + 0.25 * NV_agentic

   = 0.40 * 100 * Σ_{i ∈ chat_qa}   [ u_i * (g_{A,i} - g_{K,i}) - (s_{A,i} - s_{K,i}) / V_chat_qa - (t_{A,i} - t_{K,i}) / T_chat_qa ]   / n_chat_qa
   + 0.35 * 100 * Σ_{i ∈ math_code} [ u_i * (g_{A,i} - g_{K,i}) - (s_{A,i} - s_{K,i}) / V_math_code - (t_{A,i} - t_{K,i}) / T_math_code ] / n_math_code
   + 0.25 * 100 * Σ_{i ∈ agentic}   [ u_i * (g_{A,i} - g_{K,i}) - (s_{A,i} - s_{K,i}) / V_agentic   - (t_{A,i} - t_{K,i}) / T_agentic ]   / n_agentic
```

A wrong or missing answer has grade 0 and still counts its spend and time. A
hard question is one the calibration panel did not solve. `NV_c = +1` is one
extra percentage point of that category, after spend and time. One extra
correct answer's spend value is the frozen `V_c`; its time value is `T_c`.
For this pack, extra spend equal to `B` costs about 3.33 overall points, and saving
30% of `B` adds one point. Taking the category anchor's time again on every
question costs one point within that category before its overall weight.

The library default, used when a policy leaves these fields unset, is a fixed
`V` of $1 / $1 / $10 and no priced time (`k` omitted, so time is only a gate).
`V` on the anchor basis is `m` times the anchor's mean spend per question, and
`T` is `k` times the anchor's mean working time per question. The published
result carries the policy that was actually used. A submission that misses a
category is not scored.

The math scorer compares supported finite factorial products structurally,
including shifted indices and explicit indexed definitions, without expanding
enormous factorials. It records an ungraded result when it cannot prove equality
or a mismatch within its resource limits. Such a grading failure is retained for
regrading from the saved answer; it does not become an incorrect answer or require
another miner inference.

For this pack, the reported bound uses 70% one-sided confidence: the 30th percentile
of `NV` over 2,000 paired resamples
(questions drawn with replacement inside each category, both routers' answers
kept together). The challenger is promoted when every gate passes and that
lower bound is **greater than 0.2**.

Net value is antisymmetric. An identical router has a lower bound of 0 and
does not win. Two routers cannot both beat each other on the same questions.

### Examples

With `V = $1`, `H = 2`, and 1,000 questions, a challenger with 6 more regular
correct answers, 8 more hard correct answers, and $21 more spend scores
`100 * (6 + 16 - 21) / 1000 = +0.1` in that category.

A challenger that spends $14.50 less can drop 10 regular correct answers and
still gain `100 * (-10 + 14.5) / 1000 = +0.45`.

## Gates

Every gate must pass in every category.

- **Evidence.** The assigned questions were executed, spend is resolved, and
  measured cost and time are positive. Question spend covers the graded
  questions and sums to the category cost. When time is priced, question time
  covers the graded questions and sums to the category time.
- **Time.** Category execution time stays within an absolute budget, or within
  **2×** the calibration panel leader's time when no absolute budget is set.
- **Quality.** Mean quality stays within a tolerance of the reference. The
  tolerance is 25% of the reference's remaining error, clamped to between 2
  and 5 percentage points.
- **Hard quality.** The same rule on the hard subset, once that subset has at
  least 100 questions.
- **Private quality.** The same rule on the private subset.

The v0.2.0 contest has absolute working-time ceilings of 52,000 seconds for
chat/QA, 117,000 for math/code, and 145,000 for agentic work. They are twice the
category leader's measured time on the selected v0.2.0 calibration questions,
rounded up. They sum question working time across concurrent slots; the separate
elapsed job deadline is 86,000 seconds. These settings do not promise a fixed
completion time.

The reference is the category's high-water mark, starting from the panel
reference frozen by its policy. The v0.2.0 pack starts quality floors from
the weaker panel model; the category leader still supplies cost and time anchors.
After each promotion the mark rises to the new champion's lower
confidence bound, so a lucky run does not lift the floor. Marks only increase.

A question is hard when the calibration panel's solve share is at most the
policy threshold. The default threshold is 0: no panel model solved it.
Questions with the same solve share stay on the same side of the line.

### Wording check

A private wording check runs before the scored pack. It compares the router's
model choices on questions, hidden rewordings and identical repeats. The check
is **enforced for newly dispatched evaluations**: a failed check stops the run
before the scored pack. Jobs already dispatched in report-only mode keep that
setting; existing results and champion history are retained.

The check discourages overfitting to a question's exact wording or presentation.
Equivalent questions should lead to broadly consistent model choices. A run is
rejected when the check finds, with **95% confidence**, that rewording increases
the model-choice change rate by **more than 20 percentage points**, after
allowing for any randomness in the router.

These probes are outside the 2,375 scored questions. Their provider
calls consume the run's budget, while their grades, spend and time are excluded
from the competition measurement. Existing evidence, quality and time gates
continue to apply.

## What the validator publishes

The evaluation service publishes a confirmed result. Validators check the
content hash, the subnet, the competition major version, and the confirmation
time. They turn the published champion list into weights and map hotkeys to current
UIDs.

A paused result, an empty history, or an unregistered current champion
produces no weight transaction. Existing on-chain weights stay until a later
valid result is published.

Cost is actual provider spend, including failed attempts that belong to the
submission. Missing bills are not treated as free. Time is working time before
grading, in nanoseconds, covering solving, routing, inference, and task tools.
