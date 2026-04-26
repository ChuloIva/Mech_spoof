# Difference-in-Means Probes: Theory, Implementation, and Why They Matter

## Reference paper

Marks & Tegmark, "The Geometry of Truth" (COLM 2024, arXiv:2310.06824)

Key finding: difference-in-means (which they call "mass-mean probing" or MM) produces
directions that are **more causally faithful** than logistic regression (LR) or CCS probes,
despite all three methods achieving similar classification accuracy. MM outperformed LR and
CCS in 7/8 causal intervention experiments.

---

## 1. What is a difference-in-means probe?

A difference-in-means probe is the simplest possible way to find a direction in activation
space that separates two classes. No optimization, no training loop, no hyperparameters.

Given two sets of activation vectors — one for class A (e.g., "system-following") and one
for class B (e.g., "user-following") — the probe direction is just:

```
θ = mean(activations_A) - mean(activations_B)
```

That's it. The direction θ points from the centroid of class B toward the centroid of class A.
To classify a new activation vector x, you project it onto θ:

```
score = (x - midpoint) · θ_hat
```

where `θ_hat = θ / ||θ||` is the unit direction and `midpoint = (mean_A + mean_B) / 2`.

If score > 0, classify as A. If score < 0, classify as B. The magnitude of the score
gives a continuous confidence measure.


## 2. Why MM over logistic regression?

Three reasons, all empirically demonstrated in Marks & Tegmark:

### 2.1 More causally faithful

When you intervene along the MM direction (add θ to a class-B activation to make the
model treat it as class A), it changes model behavior more reliably than intervening along
the LR direction. This is the key finding: MM directions are better aligned with the
model's actual causal processing, not just better at classification.

Why? Logistic regression maximizes classification margin, which means it finds a direction
that separates the *training data* as cleanly as possible. This can include spurious
correlations — features that happen to co-occur with the class label in your training
set but aren't causally involved in the model's computation. LR exploits these features
because they help classification, even though intervening along them doesn't change behavior.

MM doesn't optimize anything. It just finds the centroid shift. This means it naturally
weights features by how consistently they differ between classes, which turns out to be
a better proxy for causal relevance.

### 2.2 No hyperparameters

LR has a regularization parameter C, a solver choice, convergence criteria, and random
seed for initialization. These all affect the resulting direction. MM has none. Given the
same data, MM always produces the same direction. This eliminates an entire class of
researcher degrees of freedom.

### 2.3 Training-free

MM requires one pass through the data to compute two means. No iterative optimization.
This makes it trivially fast and eliminates concerns about overfitting — there's nothing
to overfit.


## 3. Full implementation

```python
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiffMeanProbe:
    """
    A difference-in-means (mass-mean) probe for a single layer.
    
    Stores the direction vector, the midpoint for classification,
    and metadata about how it was computed.
    """
    direction: np.ndarray       # unit direction vector, shape (d_model,)
    midpoint: np.ndarray        # classification threshold point, shape (d_model,)
    raw_direction: np.ndarray   # un-normalized θ = mean_A - mean_B
    mean_A: np.ndarray          # centroid of class A
    mean_B: np.ndarray          # centroid of class B
    n_A: int                    # number of class A examples
    n_B: int                    # number of class B examples
    layer: int                  # which layer this probe is for
    
    def score(self, x: np.ndarray) -> float:
        """
        Project activation vector x onto the probe direction.
        
        Returns a scalar:
          > 0 means x is closer to class A (e.g., system-following)
          < 0 means x is closer to class B (e.g., user-following)
          magnitude indicates confidence
        
        x: shape (d_model,) — a single activation vector
        """
        return float(np.dot(x - self.midpoint, self.direction))
    
    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Score a batch of activation vectors.
        
        X: shape (n_samples, d_model)
        Returns: shape (n_samples,) array of scores
        """
        return (X - self.midpoint[np.newaxis, :]) @ self.direction
    
    def classify(self, x: np.ndarray) -> int:
        """Binary classification. 1 = class A, 0 = class B."""
        return int(self.score(x) > 0)
    
    def accuracy(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Classification accuracy on a labeled dataset."""
        predictions = (self.score_batch(X) > 0).astype(int)
        return float(np.mean(predictions == labels))


def compute_diff_mean_probes(
    activations_A: List[np.ndarray],
    activations_B: List[np.ndarray],
    n_layers: int,
    normalize_activations: bool = False
) -> Dict[int, DiffMeanProbe]:
    """
    Compute difference-in-means probe directions at every layer.
    
    Parameters
    ----------
    activations_A : list of np.ndarray
        Each element has shape (n_layers, d_model). These are activations
        from class A examples (e.g., system-following traces).
    activations_B : list of np.ndarray
        Same format, for class B examples (e.g., user-following traces).
    n_layers : int
        Number of layers in the model.
    normalize_activations : bool
        If True, L2-normalize each activation vector before computing means.
        Marks & Tegmark do NOT normalize — they use raw activations.
        Normalizing changes the geometry: it projects everything onto the
        unit hypersphere, which removes magnitude information. This can
        help if activation norms vary wildly across examples, but it
        discards information the model might use.
        
        Recommendation: try both, report both. Default to unnormalized
        to match the reference paper.
    
    Returns
    -------
    probes : dict mapping layer_idx -> DiffMeanProbe
    """
    probes = {}
    
    for layer in range(n_layers):
        # Extract activations at this layer for both classes
        X_A = np.array([act[layer] for act in activations_A])  # (n_A, d_model)
        X_B = np.array([act[layer] for act in activations_B])  # (n_B, d_model)
        
        if normalize_activations:
            norms_A = np.linalg.norm(X_A, axis=1, keepdims=True) + 1e-8
            norms_B = np.linalg.norm(X_B, axis=1, keepdims=True) + 1e-8
            X_A = X_A / norms_A
            X_B = X_B / norms_B
        
        # Compute centroids
        mean_A = X_A.mean(axis=0)
        mean_B = X_B.mean(axis=0)
        
        # The direction: from B centroid toward A centroid
        raw_direction = mean_A - mean_B
        
        # Normalize to unit vector
        norm = np.linalg.norm(raw_direction)
        if norm < 1e-10:
            # Degenerate case: centroids are identical
            direction = np.zeros_like(raw_direction)
        else:
            direction = raw_direction / norm
        
        # Midpoint for classification threshold
        midpoint = (mean_A + mean_B) / 2
        
        probes[layer] = DiffMeanProbe(
            direction=direction,
            midpoint=midpoint,
            raw_direction=raw_direction,
            mean_A=mean_A,
            mean_B=mean_B,
            n_A=len(activations_A),
            n_B=len(activations_B),
            layer=layer
        )
    
    return probes
```


## 4. Selecting which layer to use

Marks & Tegmark found that truth representations are localized to specific layers —
they don't exist uniformly across the network. The same will be true for your
authority/commitment direction.

```python
def find_best_layer(
    probes: Dict[int, DiffMeanProbe],
    X_val_A: List[np.ndarray],
    X_val_B: List[np.ndarray]
) -> Tuple[int, Dict[int, float]]:
    """
    Find the layer where the probe achieves the best classification accuracy
    on held-out validation data.
    
    Returns (best_layer, {layer: accuracy} dict)
    """
    accuracies = {}
    
    for layer, probe in probes.items():
        # Validation activations at this layer
        X_val = np.concatenate([
            np.array([act[layer] for act in X_val_A]),
            np.array([act[layer] for act in X_val_B])
        ], axis=0)
        
        labels = np.array(
            [1] * len(X_val_A) + [0] * len(X_val_B)
        )
        
        accuracies[layer] = probe.accuracy(X_val, labels)
    
    best_layer = max(accuracies, key=accuracies.get)
    return best_layer, accuracies
```

But classification accuracy alone is insufficient — the COLM paper's key lesson. You also
want to check **causal relevance** at each layer. A probe with 95% accuracy at layer 20
but no causal effect when you intervene is worse than a probe with 85% accuracy at layer 14
that actually changes behavior when you steer along it.


## 5. Causal intervention (the critical validation step)

This is what separates a diagnostic observation from a mechanistic finding.

The intervention is simple: take an activation vector from class B, add the probe
direction scaled by some strength α, and see if the model's behavior changes to
match class A.

```python
import torch

def intervene_along_direction(
    model,
    tokens: torch.Tensor,
    probe: DiffMeanProbe,
    layer: int,
    alpha: float = 1.0,
    position: Optional[int] = None
) -> torch.Tensor:
    """
    Run a forward pass with an intervention: at the specified layer,
    add alpha * probe.direction to the residual stream.
    
    Parameters
    ----------
    model : HookedTransformer or nn.Module with hooks
        The target model.
    tokens : torch.Tensor
        Input token IDs, shape (1, seq_len).
    probe : DiffMeanProbe
        The probe whose direction we'll intervene with.
    layer : int
        Which layer to intervene at.
    alpha : float
        Intervention strength. 
        alpha > 0 pushes toward class A (e.g., system-following).
        alpha < 0 pushes toward class B (e.g., user-following).
        
        The natural scale for alpha is the norm of the raw
        (un-normalized) direction vector:
            ||mean_A - mean_B||
        This represents "one full standard shift." Start with
        alpha = ||raw_direction|| and adjust from there.
    position : int, optional
        If set, only intervene at this token position. 
        If None, intervene at all positions.
    
    Returns
    -------
    logits : torch.Tensor
        The model's output logits after intervention.
    """
    direction_tensor = torch.tensor(
        probe.direction, dtype=torch.float32
    ).to(model.device)
    
    def hook_fn(module, input, output, layer_idx=layer):
        if layer_idx != layer:
            return output
        
        # output is the residual stream at this layer
        # shape: (batch, seq_len, d_model)
        modified = output.clone() if isinstance(output, torch.Tensor) else output[0].clone()
        
        if position is not None:
            modified[0, position, :] += alpha * direction_tensor
        else:
            modified[0, :, :] += alpha * direction_tensor
        
        if isinstance(output, torch.Tensor):
            return modified
        else:
            return (modified,) + output[1:]
    
    # Register hook
    # (Adapt this to your model's architecture — shown for a generic transformer)
    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        logits = model(tokens).logits
    
    handle.remove()
    return logits


def sweep_intervention_strength(
    model, tokenizer, probe, layer,
    test_prompts,  # list of (tokens, expected_behavior_before, expected_behavior_after)
    alphas=None
):
    """
    Sweep intervention strength to find the point where behavior flips.
    
    For your project: take prompts where the model follows the user
    instruction (ignores system), intervene with positive alpha to push
    toward system-following, and measure at what strength the model
    starts obeying the system instruction.
    
    If behavior flips at reasonable alpha (within 1-3x the natural scale),
    the direction is causally load-bearing.
    If behavior never flips even at very high alpha, the direction is
    diagnostic but not causal.
    """
    if alphas is None:
        natural_scale = np.linalg.norm(probe.raw_direction)
        alphas = np.linspace(-2 * natural_scale, 2 * natural_scale, 20)
    
    results = []
    for alpha in alphas:
        for tokens, _, _ in test_prompts:
            logits = intervene_along_direction(
                model, tokens, probe, layer, alpha
            )
            # Decode and evaluate behavior
            # (model-specific — generate a few tokens and check)
            ...
    
    return results
```


## 6. Comparison with logistic regression: when to use which

| Property | Difference-in-means (MM) | Logistic regression (LR) |
|----------|--------------------------|--------------------------|
| Training | No optimization, compute two means | Iterative optimization (L-BFGS or similar) |
| Hyperparameters | None | Regularization C, solver, max_iter |
| Classification accuracy | Good (comparable to LR) | Slightly better (maximizes margin) |
| Causal faithfulness | **Better** (empirically, 7/8 conditions) | Worse (exploits spurious correlations) |
| Interpretability | Direction = centroid shift, easy to reason about | Direction = max-margin hyperplane, harder to interpret |
| Sensitivity to outliers | Moderate (means are affected by outliers) | Lower (regularization helps) |
| Risk of overfitting | None (no parameters to overfit) | Low but nonzero (especially with small datasets) |
| Works with imbalanced classes | Yes, but midpoint shifts toward larger class | Better (LR naturally handles class weights) |

**Use MM** as your primary probe direction. It's simpler, more causally faithful, and
gives you a direction that's directly interpretable as "the average shift between
system-following and user-following representations."

**Use LR** as a comparison to report. If MM and LR give similar directions
(high cosine similarity), that's evidence the direction is robust. If they diverge,
the LR direction may be exploiting a spurious feature that MM ignores.


## 7. Additional checks from the reference paper

### 7.1 Opposite-pair training (from Marks & Tegmark Section 4.2)

Training on "statements and their opposites" improves generalization. In your context:
for each instruction pair (s_instruction, u_instruction), include both orderings
(s-in-system/u-in-user AND u-in-system/s-in-user) when computing the means. This is
exactly your 4-trace design — it naturally builds in the opposite-pair structure.

The COLM paper found this helps because it forces the direction to be invariant to
content and instead track the target concept (truth vs. falsehood for them,
authority vs. non-authority for you).

### 7.2 Token position matters

Marks & Tegmark found that truth representations are localized to specific token
positions — specifically the final token of the statement and the end-of-clause token.
They extract activations at these specific positions, not by mean-pooling.

For your project, you've already found that `response_first` and `response_last`
behave differently. The COLM paper's approach suggests being principled about
token position: choose the position based on where the model must have committed
to a behavioral decision, and be consistent between training and evaluation
(your exp2c position-matching insight).

### 7.3 Visualize with PCA first

Before computing the probe, project your activations into 2D with PCA and visually
inspect whether the classes separate. If they do, the linear direction will work.
If they form clusters that aren't linearly separable, you need nonlinear methods.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_class_separation(activations_A, activations_B, layer, title=""):
    """
    PCA visualization of class A vs class B at a given layer.
    """
    X_A = np.array([act[layer] for act in activations_A])
    X_B = np.array([act[layer] for act in activations_B])
    
    X_all = np.concatenate([X_A, X_B], axis=0)
    labels = ['A'] * len(X_A) + ['B'] * len(X_B)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_all)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:len(X_A), 0], X_2d[:len(X_A), 1], 
                c='blue', alpha=0.5, label='Class A (system-following)')
    plt.scatter(X_2d[len(X_A):, 0], X_2d[len(X_A):, 1],
                c='red', alpha=0.5, label='Class B (user-following)')
    
    # Draw the MM direction projected into PCA space
    mm_direction = X_A.mean(axis=0) - X_B.mean(axis=0)
    mm_2d = pca.transform(mm_direction.reshape(1, -1))[0]
    midpoint_2d = X_2d.mean(axis=0)
    plt.arrow(midpoint_2d[0], midpoint_2d[1], 
              mm_2d[0] * 0.3, mm_2d[1] * 0.3,
              head_width=0.1, head_length=0.05, fc='black', ec='black',
              label='MM direction')
    
    plt.legend()
    plt.title(f'{title} — Layer {layer} (PCA explained var: '
              f'{pca.explained_variance_ratio_[:2].sum():.1%})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    return plt.gcf()
```


## 8. Adapting MM to your project's 4-trace design

Your 4-trace contrastive design maps directly onto the MM framework.

```python
def compute_authority_direction_from_traces(
    traces: List[dict],
    n_layers: int,
    position: str = "response_first"
) -> Dict[int, DiffMeanProbe]:
    """
    Compute the authority direction using the 4-trace contrastive design.
    
    Each trace has:
      - 'activations': shape (n_layers, d_model) at the chosen position
      - 'label': 1 = system-following, 0 = user-following
    
    The 4-trace design means for each instruction pair (A, B):
      Trace 1: sys=A, user=B, response=A_aligned → label=1
      Trace 2: sys=A, user=B, response=B_aligned → label=0
      Trace 3: sys=B, user=A, response=B_aligned → label=1
      Trace 4: sys=B, user=A, response=A_aligned → label=0
    
    System-following traces: {1, 3} — response matches system instruction
    User-following traces:   {2, 4} — response matches user instruction
    
    Content is controlled: instruction A appears in system-following (trace 1)
    AND user-following (trace 4), so the direction can't latch onto content.
    
    Position is controlled: instruction A appears in system slot (traces 1,2)
    AND user slot (traces 3,4). BUT if the merge template always puts s_instruction
    first, position is NOT controlled in NONE/FAKE conditions.
    See exp2d for why this matters.
    """
    system_following = [t['activations'] for t in traces if t['label'] == 1]
    user_following   = [t['activations'] for t in traces if t['label'] == 0]
    
    return compute_diff_mean_probes(
        activations_A=system_following,
        activations_B=user_following,
        n_layers=n_layers
    )


def compute_authority_direction_structural(
    system_activations: List[np.ndarray],
    user_activations: List[np.ndarray],
    n_layers: int
) -> Dict[int, DiffMeanProbe]:
    """
    Compute the authority direction from purely structural contrast
    (REAL-only training, recovery path 2).
    
    Same instruction, two conditions:
      - system_activations: instruction in real system field
      - user_activations: same instruction in user field
    
    No behavioral labels needed. The direction captures "the model
    internally represents this text as coming from the system role."
    """
    return compute_diff_mean_probes(
        activations_A=system_activations,
        activations_B=user_activations,
        n_layers=n_layers
    )
```


## 9. The multi-layer ensemble extension

Since your exp findings showed signal at layers 16-18 with an unexplained negative
correlation at layer 8, a multi-layer approach might capture more structure.

```python
def multi_layer_mm_probe(
    activations_A: List[np.ndarray],
    activations_B: List[np.ndarray],
    layers: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate activations across multiple layers and compute a single
    MM direction in the concatenated space.
    
    This captures features that rotate across layers (as found by
    the multi-layer ensembling paper, arXiv:2604.13386).
    
    Parameters
    ----------
    layers : list of int
        Which layers to include. Recommendation: start with the
        top-5 layers by single-layer probe accuracy.
    
    Returns
    -------
    direction : np.ndarray, shape (len(layers) * d_model,)
    midpoint : np.ndarray, same shape
    """
    # Concatenate across selected layers
    X_A = np.array([
        np.concatenate([act[l] for l in layers])
        for act in activations_A
    ])  # shape: (n_A, len(layers) * d_model)
    
    X_B = np.array([
        np.concatenate([act[l] for l in layers])
        for act in activations_B
    ])
    
    mean_A = X_A.mean(axis=0)
    mean_B = X_B.mean(axis=0)
    
    raw_direction = mean_A - mean_B
    norm = np.linalg.norm(raw_direction)
    direction = raw_direction / (norm + 1e-10)
    midpoint = (mean_A + mean_B) / 2
    
    return direction, midpoint


def score_with_multi_layer(
    activation_stack: np.ndarray,
    direction: np.ndarray,
    midpoint: np.ndarray,
    layers: List[int]
) -> float:
    """
    Score using the multi-layer probe.
    
    activation_stack: shape (n_layers, d_model) — full activation cache
    """
    x_concat = np.concatenate([activation_stack[l] for l in layers])
    return float(np.dot(x_concat - midpoint, direction))
```


## 10. Summary: practical recipe for your project

1. **Extract activations** at `response_first` position for all training traces.
   Store as `(n_layers, d_model)` per trace.

2. **Split** into class A (system-following) and class B (user-following).

3. **Compute MM probes** at every layer using `compute_diff_mean_probes()`.
   No training loop, no hyperparameters.

4. **Visualize** with PCA at the top-3 layers by centroid separation distance.
   Eyeball whether the classes separate linearly.

5. **Find best layer** by held-out classification accuracy on validation set.

6. **Compare with LR** direction at the same layer. Compute cosine similarity.
   If cos_sim > 0.9, the directions agree and MM is preferred (simpler, more causal).
   If cos_sim < 0.7, investigate what feature LR found that MM didn't (likely spurious).

7. **Causal intervention** at the best layer. Add α * direction to user-following
   activations. Does behavior flip to system-following? Sweep α from 0 to 3× natural
   scale. Report the flip rate as a function of α.

8. **Multi-layer ensemble** as a robustness check. Concatenate layers 14-20,
   compute one MM direction, and check if it improves classification or causal
   intervention over the single-layer version.

9. **Transfer evaluation** to FAKE and attack conditions. Score fake-delimiter
   injection prompts with the MM probe. Does the score correlate with attack success?