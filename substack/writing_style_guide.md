# Writing Style Guide — Matching Terence's Actual Voice

## 0. Primary calibration anchor

This is ground truth, not one data point among several. Terence, describing in his own words how he'd open this week's post referencing last week's:

> "So last week we talked about multi-arm bandit as a way to minimise regrets when dealing with multiple choices of uncertain rewards - this week we look at how this framework can be used in deciding which moves to choose while playing games."

Everything else in this guide is calibrated against this sentence. Where older material (the DPhil thesis, the MaThRAD research statement) agrees with it, that material is kept as supporting evidence. Where it disagrees, this sentence wins.

### What makes it work

**Recap framing.** It opens "So last week we talked about..." — a spoken discourse marker ("So") leading into a plain recap of what the previous post covered. This is how you'd start explaining something to someone who half-remembers the last conversation, not how an essay's abstract announces its topic.

**"We" as shared exploration, not lecture.** "We talked about", "we look at" — first person plural throughout, and not the royal/co-authored "we" of academic papers. It reads as Terence and the reader having gone through the bandit problem together last week, and now going through the next thing together. This is a real shift from treating "I" as the default register (which fit the more personal, autobiographical passages) — for the recap-and-explain passages that make up most of this series, "we" is the natural voice.

**Plain, direct statement of the technical idea.** "multi-arm bandit as a way to minimise regrets when dealing with multiple choices of uncertain rewards" states what the concept *is*, in one breath, with no scaffolding before it ("It's worth noting that...", "One way to think about this is...") and no hedge after it. Trust the reader to take the idea as given and move on.

**Natural connective tissue, including a plain dash.** The two halves of the sentence are joined by " - " (a plain hyphen with spaces, not a typeset em-dash), functioning like a breath or a "and now" in speech. This is different from the AI habit of using an em-dash to insert a tidy parenthetical aside mid-sentence — it's just how one clause runs into the next when talking. It's also different from a deliberately near-zero dash policy: casual dashes as connectors are fine and natural; engineered em-dash parentheticals are the thing to avoid (see 2.3 below).

**Medium, flowing sentence length.** The whole quote is one sentence, but it's built from two coordinate halves, each medium-length, each reading like something said aloud in one breath. It is neither the thesis's long chains of subordinate clauses nor a series of short punchy fragments. Sentence length should follow how the idea would actually be said, not a fixed target in either direction.

---

## 1. Supporting evidence from the academic writing (secondary)

The DPhil thesis and the MaThRAD research statement (`/Users/holungtsui/Documents/GitHub/DPhil-Thesis/OxThesis-master/`) were the original calibration source before the anchor sentence above existed. Where they agree with the anchor, they're useful confirming evidence:

- **No contractions.** The anchor sentence has none ("minimise" not "minimize"-with-a-shortcut, no "we're", no "it's"). Across ~10,500 words of thesis/research-statement prose, contractions appear essentially never. Keep expanding to full forms in body prose.
- **No "not X, but Y" contrastive pairs.** Not in the anchor sentence, not in the academic sample. Direct positive statements only.
- **No tidy rhetorical closers or list-intro announcements** ("Two things matter here", "What's remarkable is..."). The academic writing states things and moves on; so does the anchor sentence.
- **Ideas attributed inline, not flagged separately** — "borrowed from KataGo" folded into the sentence, not called out as its own aside.

Where the academic writing pulled toward something the anchor sentence doesn't support, the thesis material is now downgraded:

- **Long, cumulative, subordinate-clause-heavy sentences** were the thesis's natural rhythm, but the anchor sentence isn't built that way — it's two medium clauses joined by a plain dash, not three or four clauses stacked with "which" and "since". Don't chase long sentences as a goal in themselves.
- **"I" as the default first person** was based on the research statement being solo-authored. The anchor sentence uses "we" for exactly this kind of recap-and-explain passage, which is most of what this series does. Reserve "I" for genuinely personal, autobiographical material (e.g. "I used to play this game with classmates at school"); default to "we" when walking through an idea together with the reader.
- **Near-zero em-dashes as a blanket rule** is too strict. The anchor sentence uses a plain dash as a natural connector. The actual thing to avoid is the specific AI habit in 2.3 below — using an em-dash to bolt a tidy parenthetical onto the middle of a sentence — not dashes in general.

---

## 2. Patterns to remove from the current drafts

Constructions that show up repeatedly across the drafts and don't appear in either the anchor sentence or the academic sample. Treat this as a find-and-fix checklist.

### 2.1 The "not X, but Y" / "not just X" contrastive pair

The single most common tell.

> **Before** (from "The Full System — Network, MCTS, and Self-Play," loss function section): "By adding score margin and ownership as auxiliary targets, every position now contributes richer signal — not just 'did we win' but 'how much did we win by' and 'which parts of the board did we control.'"
>
> **After**: "Score margin and ownership give every position richer signal: how much the game was won or lost by, and which parts of the board each player controlled."

> **Before** (from "Teaching an Agent to Play — From Q-Tables to Deep Q-Networks," closing): "The neural Q-network we built is not wasted work."
>
> **After**: "The neural Q-network we built carries over directly."

### 2.2 Sentence-fragment punchlines

Short (2-4 word) standalone sentences used for dramatic emphasis. Not present in the anchor sentence or the academic sample.

> **Before** (from "Randomness by Design — Temperature, Noise, and the Self-Play Loop"): "The network learns nothing, so the next MCTS round provides no better signal. A perfect deadlock."
>
> **After**: fold the idea into the preceding sentence rather than trailing it with a fragment.

### 2.3 Em-dash used as a tidy parenthetical device

This is the specific pattern to avoid — not dashes in general (see 0 above on the anchor sentence's plain dash). The tell is an em-dash bracketing an aside in the *middle* of a sentence, engineered for tidiness:

> **Before**: "Two auxiliary outputs — `score_margin` (how much we win/lose by) and `ownership` (which sub-boards each player controls) — are training targets only and are not used during play."
>
> **After**: "Two auxiliary outputs, `score_margin` (how much we win or lose by) and `ownership` (which sub-boards each player controls), are training targets only; they are not used during play."

A plain dash used once, casually, to join two flowing clauses (as in the anchor sentence) is fine and should not be stripped out reflexively.

### 2.4 Formulaic "X things matter here" list-intros

> **Before** (from "Randomness by Design — Temperature, Noise, and the Self-Play Loop"): "Two design choices matter here: Noise only at the root... α = 0.3 produces spiky samples..."
>
> **After**: "Two choices are worth explaining. Noise is added only at the root: deeper nodes are left unperturbed, so the diversity is concentrated at the move actually being played rather than corrupting the evaluation of what follows. And α = 0.3 produces spiky samples: a small α concentrates the noise's mass on a few moves unpredictably, rather than uniformly boosting every alternative."

### 2.5 Tidy, aphoristic paragraph- or essay-closers

> **Before** (from "The Full System — Network, MCTS, and Self-Play," final line of the whole essay): "Three posts ago we had a Q-table that couldn't generalise. This is what it took to actually get there."
>
> **After**: end on the last substantive claim, or a plain forward pointer, not a rhetorical bow.

### 2.6 Rhetorical "no X needed" / "what's not happening" framing

> **Before** (from "Inside One MCTS Simulation — How AlphaZero Thinks Move by Move"): "Note what is not happening: no random rollout. Classic MCTS would simulate a random game from the leaf to the end and use the outcome as the value estimate."
>
> **After**: "Classic MCTS simulates a random game from the leaf to the end and uses the outcome as the value estimate. AlphaZero replaces this random rollout with a direct network evaluation."

### 2.7 Contractions

Convert to full forms in body prose ("can't" → "cannot", "it's" → "it is"). Some judgment needed in code comments or dialogue-style asides, where full forms would look stilted.

### 2.8 What to leave alone

- Defining notation/terms plainly before using them.
- Bullet/numbered lists for genuinely itemizable technical content ("The Full System" essay's three failure modes, "Randomness by Design" essay's four-step self-play process) — legitimate, not padding.
- Attributing techniques to their source inline ("borrowed from KataGo").
- A sentence running a bit long when the idea genuinely needs the room — length should follow the idea, not a fixed rule in either direction.

---

## 3. Practical checklist for the rewrite

1. Open recap-style passages (linking to the previous post, or previewing this one) with a spoken framing like the anchor sentence — "So last week we...", "this week we..." — rather than a formal topic announcement.
2. Default to "we" for passages that walk through an idea with the reader; reserve "I" for genuinely personal, autobiographical material.
3. State technical ideas directly, in one breath, with no "it's worth noting" scaffolding before and no "in other words" reframing after.
4. Read sentence length against how the idea would actually be said aloud — not stretched into long subordinate-clause chains, not chopped into short fragments.
5. Search for em-dashes used as tidy mid-sentence parentheticals (2.3) and rewrite as commas, colons, semicolons, or a plain connecting dash if genuinely casual.
6. Search for "not just", "not only", "isn't just", "not X but Y"; rewrite as direct positive statements.
7. Search for contractions in body prose; expand to full forms.
8. Flag any sentence under ~6 words standing alone as its own paragraph or ending a section; fold it in or cut it.
9. Check every section/essay ending for a tidy aphoristic callback; replace with a plain technical closer or forward pointer.
10. Check list-intro phrases ("X things matter here"); cut the announcement and go straight into the list, or fold the count into a sentence naturally.

This guide applies to all current drafts — local markdown and live Substack — with technical content and accuracy fully preserved. Only prose style changes; no code, numbers, or claims should change as a result of a rewrite pass.
