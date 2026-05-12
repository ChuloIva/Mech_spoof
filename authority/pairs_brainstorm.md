The seeds are just the claims — the middle part that stays constant across +A and −A. You need them to be:

1. Domain-diverse (not all medical)
2. Statements where source credibility plausibly matters (not "the sky is blue")
3. Not safety-relevant (no refusal trigger)
4. Not obviously true or false (so the model actually has to weigh the source)

The claims should be things where a reasonable person would think "hmm, depends who's saying it." That's the whole point — the claim alone is ambiguous, and the source is what resolves it.

So the seed is just a sentence like:

```
"this building is structurally sound"
"you can drive another 5000 miles on those tires"  
"this contract looks standard, nothing unusual"
"the water here is safe to drink"
"your child's fever isn't serious"
"this plant is edible"
"you don't need to evacuate"
"this investment will return 8% annually"
"your roof will last another ten years"
"this mole doesn't look concerning"
```

Each of these is a judgment call that could be right or wrong, and whose credibility depends entirely on who's saying it. None of them trigger safety refusal.

Now for generating at scale, you have three axes to cross:

**Axis 1: The claim (seeds).** You need maybe 30-50 of these across domains — medical, legal, financial, safety, technical, environmental, nutritional, mechanical, etc.

**Axis 2: The authority figure.** Matched to each claim's domain — the person who *would* have standing to make this judgment.

**Axis 3: The non-authority figure.** Also matched — someone plausible in context but without credentials.

The template is literally:

```
"{figure} says {claim}."
```

But here's the generation question — do you write all of these by hand, or do you generate them?

**Option 1: Hand-write 50 seeds, hand-pick figure pairs.** Most controlled, but small. You know exactly what's in the dataset, no noise. Good for Phase 1.

**Option 2: Hand-write 20 seed templates, use an LLM to generate variations.** You write the pattern, the LLM gives you 5 rephrasings of each claim and 5 authority/non-authority pairs per domain. Gets you to ~500 pairs fast. But you need to manually filter — the LLM will drift toward medical examples and safety-adjacent stuff because that's where "authority" is most salient in training data.

**Option 3: Start from domains, generate exhaustively.** Pick 10 domains. For each domain, list 5 claim types and 3 authority/non-authority pairs. That's 10 × 5 × 3 = 150 pairs mechanically.

I'd go with Option 3 because it's systematic. Here's what the domain matrix looks like:

| Domain | Claim type | Authority | Non-authority |
|---|---|---|---|
| Medical | diagnosis, prognosis, dosage, symptom interpretation, treatment recommendation | physician, surgeon, pharmacist | neighbor, coworker, forum poster |
| Legal | contract assessment, rights interpretation, liability, legal risk | attorney, judge, legal scholar | friend, relative, blogger |
| Structural/housing | building safety, repair urgency, code compliance, material condition | engineer, inspector, licensed contractor | landlord, realtor, previous owner |
| Automotive | repair need, part lifespan, safety assessment, maintenance timing | certified mechanic, manufacturer engineer | friend who likes cars, YouTube commenter, Uber driver |
| Financial | investment viability, tax strategy, risk assessment, market prediction | CPA, financial advisor, economist | coworker, uncle, podcast host |
| Outdoor/survival | trail safety, weather judgment, water potability, wildlife risk | park ranger, experienced guide, biologist | fellow tourist, hostel receptionist, travel blogger |
| Nutritional | food safety, dietary advice, allergy risk, supplement recommendation | dietitian, allergist, food scientist | gym buddy, influencer, cashier at health food store |
| Technical/engineering | system reliability, safety margin, load capacity, failure risk | licensed engineer, safety inspector | sales rep, project manager with no technical background |
| Educational | child development, learning assessment, readiness judgment | child psychologist, experienced teacher | other parent at school, online forum |
| Environmental | contamination risk, air/water quality, exposure safety | environmental scientist, EPA inspector | neighbor, local news commenter |

Each cell gives you a claim + figure pair. Cross 5 claims per domain × 3 authority figures × 3 non-authority figures = 45 pairs per domain × 10 domains = 450 pairs.

But wait — this keeps authority and domain locked together, which means the direction might still encode domain. The unlock is adding **cross-domain authority figures** to a subset. Take 10 claims and pair them with out-of-domain authorities:

```
"A cardiologist says this contract looks standard."
"A certified mechanic says your child's fever isn't serious."
"A park ranger says this investment will return 8%."
```

These should *not* activate the authority direction if your direction is clean — a cardiologist has no standing on contracts. If they do activate it, you've fit "person with a job title" not "person with relevant authority." That's your built-in diagnostic.

So the final dataset structure is:

- ~450 in-domain pairs (authority in their domain vs non-authority)
- ~90 cross-domain pairs (authority outside their domain vs non-authority)
- ~90 anti-authority pairs (explicitly unreliable sources — conspiracy theorist, known scammer, drunk guy)

Total ~630 pairs, all using the same simple template, no system prompts, no safety triggers.

Want me to actually write out the 50 claim seeds so you have something concrete to start from?