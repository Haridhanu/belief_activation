# TGN dynamics — how the BeliefStore activity becomes TGN training signal

This document explains *where* the TGN operates, *what* "TGN update" means
mechanically, and *how* coherence/dissonance activity in the broader
BeliefStore flows into the TGN's parameters.

## Where the TGN operates

The TGN does **not** read from or write to the BeliefStore directly. There
are three distinct layers, and the TGN sits in the middle one:

```
┌────────────────────────────────────────────────────────────┐
│  BeliefStore                                               │
│  • persistent (Postgres + Redis)                           │
│  • holds beliefs across all sessions                       │
│  • API surface for the rest of the product (pi_infer, etc) │
│  • TGN never reads or writes here directly                 │
└────────────┬─────────────────────────────────▲─────────────┘
             │                                 │
─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─
             │ feed beliefs                    │ push committed
             │ for this session                │ edges back
             ▼                                 │
┌────────────────────────────────────────────────────────────┐
│  In-process Graph (multi_agent.graph.Graph)                │
│  • per-session, in-memory                                  │
│  • _raw, _z, _adj, _edges, _edge_count, _edge_timestamps   │
│  • the trainer's working scratchpad                        │
│  • TGN reads node IDs/embeddings here, signals back        │
│    "here are the edges I want committed"                   │
└─────────────────────────▲──────────────────────────────────┘
                          │
                          │ tgn.update() fires per
                          │ committed edge event
                          ▼
┌────────────────────────────────────────────────────────────┐
│  TGN (multi_agent.tgn.TGNModule)                           │
│  • per-node memory dict                                    │
│  • learnable parameters (msg encoder, GRU, link head)      │
│  • does NOT store edges; does NOT own a graph              │
│  • only relates to the in-process Graph through edge       │
│    events flowing in and predictions flowing out           │
└────────────────────────────────────────────────────────────┘
```

The BeliefStore is persistent and shared; the in-process Graph is
ephemeral working state; the TGN is the reasoner sitting on top of the
working state. Only the **winnowed result** — committed edges — gets
pushed back to the BeliefStore at session end.

## Two distinct "TGN updates"

There are two completely different operations both called "update,"
and confusing them is the source of most fuzzy thinking. They are:

### Update type 1 — memory propagation (forward state)

Triggered by every committed edge:

```
tgn.update(src, dst, sign, timestamp, weight)
```

This is **forward-only**, mutates the per-node memory dict, no gradient,
no supervision. Step by step:

```
INPUT:                 src_id, dst_id, sign ∈ {-1, +1}, timestamp, |weight|

  1. Read:             m_src = NodeMemory[src_id]   (zero if unseen)
                       m_dst = NodeMemory[dst_id]

  2. Time encode:      t_enc = TimeEncoder(timestamp - last_ref_time)
                       (sinusoidal, no parameters)

  3. Build messages:   msg_for_dst = MessageEncoder(m_src, m_dst, sign, t_enc, |w|)
                       msg_for_src = MessageEncoder(m_dst, m_src, sign, t_enc, |w|)

  4. GRU step:         new_m_dst = GRUCell(msg_for_dst, m_dst)
                       new_m_src = GRUCell(msg_for_src, m_src)

  5. Detach + store:   NodeMemory[dst_id] = new_m_dst.detach()
                       NodeMemory[src_id] = new_m_src.detach()
                                         ^^^^^^^^ autograd graph cut here

OUTPUT:                nothing returned; memory dict mutated
```

This runs once per committed edge. It costs a forward pass through the
GRU and message encoder (~tens of microseconds). The weights of those
modules are *not* changing — they are being *applied*.

### Update type 2 — parameter training (gradient)

Triggered once per `Trainer.step` call, on judge-revealed pairs:

```python
loss = tgn.link_loss(judged_triples)
loss.backward()
optimizer.step()
```

Step by step:

```
INPUT:                 list of [(u, v, y_truth), ...] from judge

  1. For each (u, v, y):
        h_u = NodeMemory[u]              (current memory, detached)
        h_v = NodeMemory[v]              (current memory, detached)
        prediction = link_head(cat([h_u, h_v]))   ← gradient enabled here
                   ∈ [-1, 1] via Tanh

  2. loss = mean( (prediction - y_truth)² )

  3. loss.backward()
       gradient flows backward through:
         - link_head's Linear → ReLU → Linear → Tanh
         - the cat operation
         - STOPS at NodeMemory[u] and NodeMemory[v]
                    (because they were stored detached)

  4. optimizer.step()
       link_head's weights nudged toward producing y_truth
       msg encoder & GRU updater weights NOT touched in this gradient step
       (they didn't appear in the forward path of link_head(cat([h_u, h_v])))

OUTPUT:                a scalar loss; module parameters mutated
```

So **the only parameters that get gradient from a judge call are the link
head's**. The msg encoder and GRU updater are updated *only* when they
run forward (during `tgn.update()`), and even then they are being applied,
not trained.

This is a real architectural limit of the current design — and it is why
empirical training shows fast loss drop but unstable held-out accuracy:
the link head is fitting the seen pairs while the encoder + GRU stay at
random init.

### How they compose in one step

The order inside `TGNTrainer.step(batch)`:

```
PHASE                  OPERATIONS                                 AFFECTS
─────                  ──────────                                 ───────

1. Add nodes           graph._raw[id] = embedding                  in-process Graph
                       (no edges committed yet)

2. Score candidates    for each (u, v): predict_link(u, v)         NodeMemory (read)
                       [forward only, no_grad]                     link_head (read)

3. Pick uncertain      rank by 1 - |predict_link|, take top-K      —

4. Judge               oracle returns y for K pairs                —

5. TRAIN ◀──────────   loss = link_loss(K judged triples)          link_head.weights
                       loss.backward(); optimizer.step()           (only)

6. PROPAGATE ◀─────    for each (u, v, y) in judged:               NodeMemory.set()
                         tgn.update(u, v, sign(y), t, |y|)         (forward, detached)

7. Commit edges        graph.extend([], [], edges)                 in-process Graph
                       where edges = judged + confident_predicted   _edges, _adj
```

Phase 5 (gradient) runs *first*, then Phase 6 (memory propagation). Future
steps see both: the slightly-better link head AND the slightly-richer
memory.

## How memory accumulates — concrete trace

Three nodes A, B, C. Initial memory dict is empty.

```
EVENT 1:               edge (A, B, sign=+1, t=1, w=0.8)

  Read:                m_A = zeros(memory_dim)   (default for unseen)
                       m_B = zeros(memory_dim)

  Build messages:      msg_for_B = MsgEnc(0, 0, +1, t_enc(1), 0.8)
                       msg_for_A = MsgEnc(0, 0, +1, t_enc(1), 0.8)

  GRU update:          new_A = GRU(msg_for_A, zeros)
                       new_B = GRU(msg_for_B, zeros)

  Store:               NodeMemory["A"] = new_A.detach()
                       NodeMemory["B"] = new_B.detach()

────────────────────────────────────────────────────────────────────────

EVENT 2:               edge (B, C, sign=-1, t=2, w=0.6)

  Read:                m_B = NodeMemory["B"]   ← from event 1, has +1
                                                  history with A
                       m_C = zeros

  Build messages:      msg_for_C = MsgEnc(m_B, 0, -1, t_enc(1), 0.6)
                       msg_for_B = MsgEnc(0, m_B, -1, t_enc(1), 0.6)

  GRU update:          new_B = GRU(msg_for_B, m_B)   ← B's memory now
                                                       carries +1 with A
                                                       AND -1 with C
                       new_C = GRU(msg_for_C, zeros) ← C's memory carries
                                                       -1 with B,
                                                       AND traces of A
                                                       through m_B

────────────────────────────────────────────────────────────────────────

EVENT 3:               predict_link(A, C)

  Read:                h_A = NodeMemory["A"]   ← carries +1 with B
                       h_C = NodeMemory["C"]   ← carries -1 with B,
                                                 traces of A via event 2

  prediction = link_head(cat([h_A, h_C]))

  The model's prediction reflects: "A was +1 with B; C was -1 with B;
  by transitivity, A and C should be -1." This is multi-hop reasoning
  emerging implicitly from memory accumulation.
```

Information propagates through the chain of memory updates, not through
explicit graph traversal. Two nodes that never interact directly can
still influence each other's memory if a third node served as a relay.

## The route from BeliefStore activity to TGN training

This is the bridge that ties it all together. The BeliefStore is the
live system: new beliefs land continuously, edges get added and adjusted
as the LLM judge weighs in. The TGN doesn't watch the BeliefStore
directly, but it is *fed* by exactly that activity.

```
USER / SYSTEM ACTIVITY
──────────────────────

POST /beliefs                  ┌───────────────────────────────┐
     │                         │  BeliefStore                  │
     ├────────────────────────▶│  • new belief written         │
     │                         │  • indexed, embedded          │
     │                         │  • PI Lite proposes initial   │
     │                         │    coherence/dissonance       │
     │                         │  • LLM judge confirms/refines │
     │                         │  • edges committed/adjusted   │
     │                         └───────────┬───────────────────┘
     │                                     │
     │                                     │ session triggers a
     │                                     │ daydream training run
     │                                     ▼
     │                         ┌───────────────────────────────┐
     │                         │  Cloud Tasks invokes          │
     │                         │  run_activation()             │
     │                         └───────────┬───────────────────┘
     │                                     │ pulls beliefs +
     │                                     │ existing edges for
     │                                     │ this session
     │                                     ▼
     │                         ┌───────────────────────────────┐
     │                         │  TGNTrainer (per session)     │
     │                         │  • streams beliefs in batches │
     │                         │  • TGN memory propagates per  │
     │                         │    judged event               │
     │                         │  • link head trains on judge  │
     │                         │    truth                      │
     │                         └───────────┬───────────────────┘
     │                                     │
     │                                     │ committed edges
     │                                     │ pushed back
     │                                     ▼
     │                         ┌───────────────────────────────┐
     │                         │  BeliefStore (now richer)     │
     │                         │  + new coherence/dissonance   │
     │                         │    edges from this session    │
     │                         └───────────────────────────────┘
     │
─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     │
POST /infer                    ┌───────────────────────────────┐
     │                         │  Router.from_redis(snapshot)  │
     └────────────────────────▶│  loads the TGN that just      │
                               │  trained on session activity  │
                               └───────────────────────────────┘
```

Two channels carry the dynamics:

### Channel 1 — beliefs become TGN nodes

`run_activation()` (in `perseverate_api/belief_activation.py`) pulls
session-relevant beliefs from BeliefStore, chunks them into batches, and
feeds them to a `TGNTrainer`. Every belief that participates in this
session **becomes a node** from the TGN's perspective. The node ID is the
BeliefStore's `bid`; the node embedding is the same vector BeliefStore
uses for retrieval. No translation layer.

### Channel 2 — coherence/dissonance edges become training events

The judge is the bridge. When the trainer runs, the same judge that
weighs in on edges in the BeliefStore weighs in on edges proposed by the
TGN. Each verdict serves three simultaneous purposes:

```
┌──────────────────────────────────────────┐
│  judge(u, v) returns y ∈ [-1, 1]         │
│     +0.8 = strong coherence              │
│     -0.8 = strong dissonance             │
│      0.0 = unrelated                     │
└────────────┬─────────────────────────────┘
             │
             │ this y is used for:
             │
 ┌───────────┼───────────┐
 ▼           ▼           ▼
 1.          2.          3.
 training    memory      edge to commit
 signal      event       to in-process Graph
 for TGN     for TGN     (and later to BeliefStore)
 link head   memory
 (gradient)  (forward
              only)
```

The same signed value `y` is used three ways in the same step:

1. **Training the link head.** `loss = MSE(predict_link(u, v), y)`,
   then `backward()`, then `optimizer.step()`. This teaches the link
   head to map memory pairs to signed predictions.

2. **Propagating memory.** `tgn.update(u, v, sign=sign(y), ...)`. The
   sign argument tells the message encoder whether this is a coherence
   or dissonance event. Coherence pulls memories together; dissonance
   pushes them apart.

3. **Committing the edge.** Written to the in-process Graph and (at
   session end) to BeliefStore. The signed weight encodes the
   relationship type.

This is the precise mechanism by which BeliefStore's coherence/dissonance
dynamics become TGN learning. The TGN doesn't know about the BeliefStore
— it only knows about the judge — but the judge is the same oracle that
determines what gets written to the BeliefStore. So they are in lock-step.

### How the dynamics manifest inside the TGN

Three concrete mechanisms encode coherence vs dissonance:

- **Sign in messages.** `TemporalMessageEncoder` takes
  `(src_mem, dst_mem, sign, time_enc, |w|)` and produces a message vector.
  The sign is a `+1` or `-1` scalar concatenated into the input. Coherence
  events produce structurally different messages from dissonance events —
  they hit different regions of the message space. The GRU absorbs
  different dynamics.

- **Time encoder distinguishes recent from old.** Two events with the
  same sign and weight produce *different* messages if they arrive at
  different timestamps. Recently-confirmed coherence has different
  gravitational pull than old coherence.

- **Weight in messages.** `|y|` (the magnitude of the verdict) goes into
  the message encoder. A weak `+0.3` produces a different message than a
  strong `+0.9`. The GRU weights confident judgments more heavily.

Memory becomes a function of *what kinds of relationships this belief
has had, when, and with how much confidence* — not just a count.

## Connection adjustments — re-judgment

The BeliefStore can adjust an existing edge as more evidence comes in
(e.g., initial coherence later contradicted). When that happens, the
trainer fires another `tgn.update()` event with the new sign and weight.
The GRU updates memory based on the *new* event, layered onto the
existing memory. So the second judgment doesn't overwrite the first; it
composes with it. Memory carries traces of both. The link head learns
to read the combined memory and produce a calibrated prediction.

## What persists, what resets

| | Persists across sessions | Resets per session |
|---|:---:|:---:|
| BeliefStore beliefs and edges | ✓ | ✗ |
| In-process Graph (trainer's scratchpad) | ✗ | ✓ |
| TGN per-node memory (NodeMemory dict) | ✗ | ✓ (`tgn.reset()` between sessions) |
| TGN parameters (link head, msg encoder, GRU) | ✓ (via snapshot) | ✗ |

So while the BeliefStore accumulates knowledge over time, the TGN's
*memory* doesn't — it rebuilds from scratch each session by replaying
that session's edge events. What persists in the TGN is the *learned
mapping* from memory pairs to predictions. That mapping benefits from
each session's training, and arrives ready-to-use in the next.

## In one paragraph

When a belief is added to the BeliefStore and the judge weighs in on its
coherence/dissonance with other beliefs, that activity triggers a
daydream training run; `run_activation()` pulls the session's beliefs
and edges out of BeliefStore, hands them to a `TGNTrainer`, which streams
the beliefs as batches and asks the same judge for verdicts on uncertain
pairs — and each verdict simultaneously trains the TGN's link head
(gradient on MSE), propagates the TGN's memory through the GRU (sign and
weight encode coherence vs dissonance), and commits the edge to the
trainer's graph; at session end the trainer pushes the committed edges
back to BeliefStore and snapshots the TGN's *parameters* (not memory) to
Redis so the next session inherits what was learned. The dynamics carry
through because the judge is the same oracle in both worlds: whatever
verdict shapes the BeliefStore's edges is also the supervised signal
teaching the TGN how those verdicts relate to belief-pair representations.
