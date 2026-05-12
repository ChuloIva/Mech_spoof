Ah, I see what you're asking. Not just "does the model detect authority" but "what does the model actually treat as authoritative when it generates responses." Like — when the model chooses to say one thing over another, what sources in its context is it treating as having weight?

That's a different question from what the dataset as designed would give you.

The dataset as designed gives you: the model *recognizes* authority. It detects that a cardiologist has more standing than a gym bro. The probe tells you the strength of that recognition. But recognition and *following* are different things. The model might perfectly recognize that someone is authoritative and then completely ignore them. Or it might follow instructions from someone it doesn't represent as authoritative at all — like it follows system prompt instructions not because the system prompt has "authority" in any semantic sense, but because of structural training.

What you actually want to know is: when the model generates token N+1, what in the context influenced that choice, and does the thing we call "authority" predict what gets influence?

That's really a causal question, not a representational one. The probe tells you about representation. Influence on generation is about the computational graph — which parts of the context actually push the output distribution.

But here's where it gets interesting. If you have the authority probe, you can do something like this:

Take a prompt where multiple sources give conflicting information:

```
"Your doctor says you should avoid coffee. Your personal trainer 
says coffee before workouts is beneficial. Your neighbor says 
coffee doesn't matter either way. Should you drink coffee?"
```

The model generates a response. It probably sides with the doctor. Now you can do two things:

**Thing 1:** Run the authority probe at each source-mention position. Does "your doctor" light up more than "your personal trainer" which lights up more than "your neighbor"? If yes, the probe tracks the authority gradient in a multi-source setting.

**Thing 2:** Look at what the model's response actually *does* — whose advice does it endorse? If the model says "your doctor's advice should take priority," and the probe correctly predicted that the doctor tokens carry the highest authority signal, then you've shown that the probe predicts which source the model follows.

But here's the harder version: what if there's no explicit authority figure at all? What if the prompt is just:

```
"Some studies show coffee is harmful. The WHO recommends moderate 
consumption. A recent meta-analysis found no significant health risks. 
My grandmother always said coffee was bad for you."
```

There's no "doctor" or "police officer" here. But the model still weights these sources differently — it probably trusts the meta-analysis over grandma. Does the authority probe fire on "meta-analysis" more than on "my grandmother"? If yes, then you've found something deeper than role-based authority — you've found the model's general *source credibility* representation, and named authority figures are just one instance of it.

And that might be the real finding. Maybe what the model encodes isn't "this person is an authority figure" but something more general like "this source should be weighted heavily in my reasoning." Doctors activate it, but so do peer-reviewed studies, official organizations, direct empirical evidence, and established consensus. The probe would then be a *credibility detector*, and authority figures are just high-credibility sources.

To test this you'd extend the dataset beyond people:

```
+credibility: "According to a peer-reviewed study in The Lancet, {claim}."
−credibility: "According to a blog post I found, {claim}."

+credibility: "The manufacturer's manual states {claim}."
−credibility: "An Amazon review says {claim}."

+credibility: "Federal law requires {claim}."
−credibility: "I heard somewhere that {claim}."
```

If these activate the same direction as doctor-vs-gym-bro, then your direction isn't "authority" in the narrow sense of a person with a role — it's "source credibility" or "epistemic weight," and human authority figures are one subcategory. That's arguably a more interesting and more general finding.

But — and this is the key tension — "source credibility" is very close to "epistemic trust," which is very close to "this source is probably right." And at that point your probe might just be detecting the model's Bayesian prior about source reliability, which is a competence/accuracy judgment, not an authority judgment.

The way to pull these apart: include authoritative sources that say wrong things.

```
+A: "A board-certified dermatologist says you should apply 
     sunscreen indoors even with no windows."
−A: "A random Reddit user says you should apply sunscreen 
     indoors even with no windows."
```

The claim is dubious regardless of source. If the probe fires equally on both — the model doesn't give extra credibility to the dermatologist for a weird claim — then your probe tracks epistemic plausibility, not authority. If the probe still fires higher for the dermatologist even though the claim is weird, then you've genuinely captured something about role-based authority independent of content accuracy.

So the final dataset probably needs three conditions, not two:

```
+A correct:   "A {authority} says {true claim in their domain}."
+A incorrect: "A {authority} says {false/dubious claim in their domain}."
−A correct:   "A {non-authority} says {true claim}."
−A incorrect: "A {non-authority} says {false/dubious claim}."
```

The direction that separates +A from −A *regardless of whether the claim is correct* is pure authority. The direction that separates correct from incorrect *regardless of who says it* is epistemic accuracy. And if they're different directions, you've cleanly decomposed authority from competence.

Does this get at what you were asking? The probe doesn't automatically tell you "what authority does the model follow" — but with the right dataset design and the right downstream analysis (multi-source conflicts, wrong-authority scenarios), you can use it to answer that question.