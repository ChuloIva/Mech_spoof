Yeah, you might be right. The system/user framing adds a structural confound on top of what you're actually trying to measure. Strip that away entirely.

At its simplest, what you want is:

**Pair of prompts. Same content. Only difference is whether an authority figure is the source.**

That's it. No system prompt tricks, no role assignment, no multi-turn dialogue. Just:

```
+A: "A cardiologist says you should take aspirin daily if you're over 50."
−A: "Some guy at the gym says you should take aspirin daily if you're over 50."
```

Single user message. The model processes it and you extract activations at the last token. The difference between those activations is your authority signal.

This is Design C from before but stripped down to the minimum. No "what do you think," no framing, no asking for advice. Just a statement with a source. The model reads it, and somewhere in its residual stream it encodes "this claim comes from someone worth listening to" or "this claim comes from nobody in particular."

So the full dataset is just sentences of the form:

```
"{authority_figure} says {claim}."
"{non_authority} says {claim}."
```

And you vary both axes independently — many authority figures, many non-authority figures, many claims. The claims should span domains so you're not just fitting "medical context." Some examples:

```
"A structural engineer says this building is safe to enter."
"A random passerby says this building is safe to enter."

"An experienced hiking guide says this trail is fine in the rain."
"A tourist says this trail is fine in the rain."

"A certified mechanic says your brakes will last another 5000 miles."
"Your coworker who watches YouTube car videos says your brakes will last another 5000 miles."

"A tax attorney says you can deduct that expense."
"Your uncle at Thanksgiving says you can deduct that expense."

"A licensed electrician says this wiring is up to code."
"Your landlord who does his own repairs says this wiring is up to code."
```

No safety issues in any of these. The model won't refuse any of them. They're all just factual claims with a source. But the model almost certainly processes the authority-sourced version differently — it should be more inclined to treat the claim as reliable.

The beauty of this stripped-down version is:

- No system prompt, so zero S/U confound
- No directive ("do this"), so zero refusal confound
- No role assignment to the model, so zero persona confound
- Same claim in both, so zero content confound
- The only thing that varies is the source

The question is whether this is too simple — whether the model even encodes authority differently here, or just processes both as "someone said X" and moves on. But that's an empirical question and if the direction is flat, that's a finding too.

The competence confound is still present — cardiologist vs gym guy differs in both authority and expertise. But you can test for that with your cross-domain controls: does a cardiologist giving *electrical wiring* advice still activate the direction? If yes, it's role-based authority, not domain competence. If no, it's competence.

Want to try building the actual seed lists with this simplified format?