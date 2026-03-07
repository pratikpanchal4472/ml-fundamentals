# Linear Algebra in Machine Learning & Linear Regression

## Vectors as Data Points

In ML, every data sample is a **vector**. If you have a dataset with 3 features (age, income, spend), each row is a vector in ℝ³:

```
x = [age, income, spend] = [25, 50000, 3200]
```

---

## Linear Combination → The Prediction Equation

A **linear combination** is: `c₁v₁ + c₂v₂ + ... + cₙvₙ`

In linear regression, your prediction is _exactly_ a linear combination of feature vectors:

```
ŷ = θ₀·1 + θ₁·x₁ + θ₂·x₂ + ... + θₙ·xₙ
```

Where θ (weights) are the **scalar coefficients** and x (features) are the **vectors** being combined. Training the model = finding the right coefficients.

---

## Span → What Your Model Can Learn

The **span** of a set of vectors is all possible linear combinations you can form from them.

In regression terms:

- Your feature columns (vectors) define a **column space** (their span)
- The model can only predict values that lie **within this span**
- If your target `y` lies outside the span → you can't fit it perfectly → **residual error exists**

```
Feature matrix X has columns [x₁, x₂, ..., xₙ]
Span(x₁, x₂, ..., xₙ) = Column Space of X

Best prediction ŷ = projection of y onto Column Space(X)
```

This is geometrically why **least squares regression finds the closest point** in the column space to the actual target.

---

## Linear Independence → Feature Quality

If your feature vectors are **linearly dependent** (one feature = combination of others), bad things happen:

| Situation                                  | ML Consequence                    |
| ------------------------------------------ | --------------------------------- |
| Features are independent                   | Unique, stable solution θ         |
| Features are dependent (multicollinearity) | Infinite solutions, unstable θ    |
| Redundant feature                          | Adds no new span, wastes capacity |

Example: if `x₃ = 2x₁ + x₂`, then adding x₃ doesn't expand the span — your model gains nothing.

---

## The Normal Equation — Pure Linear Algebra

The closed-form solution to linear regression is:

```
θ = (XᵀX)⁻¹ Xᵀy
```

This only works when **XᵀX is invertible**, which requires the columns of X to be **linearly independent** (full rank). If not → use pseudoinverse (SVD).

---

## Span in Neural Networks

This extends to deep learning too:

- Each **neuron layer** computes linear combinations of inputs
- Without non-linear activations, stacking layers just produces another linear combination — **the span doesn't grow**
- This is why activation functions (ReLU, sigmoid) are essential — they break out of the linear span

---

## Quick Mental Model

```
Span        →   "What can my model express?"
Lin. Combo  →   "How does my model make predictions?"
Independence →  "Are my features carrying unique information?"
Column Space →  "Where does the best prediction live?"
```

Linear algebra isn't just background math — it _is_ the skeleton of how regression and most of ML works under the hood.

# Vector Spaces in Machine Learning & Linear Regression

## What is a Vector Space?

A vector space is a set V with two operations (addition, scalar multiplication) satisfying 8 axioms. The key intuition: **a vector space is any "universe" where linear combinations stay inside the same universe.**

---

## 1. Feature Space — Your Data Lives in a Vector Space

Every dataset defines a vector space ℝⁿ where n = number of features.

```
Sample 1: x¹ = [age=25, income=50k, spend=3200]  ∈ ℝ³
Sample 2: x² = [age=35, income=80k, spend=5100]  ∈ ℝ³
Sample 3: x³ = [age=45, income=60k, spend=4000]  ∈ ℝ³
```

The entire dataset is a **collection of vectors inside ℝ³**. This space has:

- **Closure**: any linear combination of samples stays in ℝ³
- **Origin**: zero vector [0,0,0] exists
- **Dimension**: 3 (one per feature)

---

## 2. Column Space — The Heart of Linear Regression

The **column space** (range) of matrix X is the vector space spanned by its feature columns:

```
X = | 1  25  50000 |        col₁ = bias column
    | 1  35  80000 |        col₂ = age column
    | 1  45  60000 |        col₃ = income column
```

```
Column Space(X) = all vectors of the form Xθ = θ₀col₁ + θ₁col₂ + θ₂col₃
```

**This is a subspace of ℝᵐ** (m = number of samples).

### Why this matters for regression:

```
y (actual target) ──── usually NOT in Column Space(X)
                                    │
                                    ▼
ŷ = Xθ ──────────────── LIVES in Column Space(X)

Residual (y - ŷ) ────── PERPENDICULAR to Column Space(X)
```

Least squares literally finds the **orthogonal projection of y onto the column space**. The normal equation `θ = (XᵀX)⁻¹Xᵀy` is just the algebraic solution to this geometric projection.

---

## 3. Null Space — When Things Break

The **null space** of X = all vectors θ where `Xθ = 0`

```
Null Space(X) = { θ : Xθ = 0 }
```

| Null Space Condition              | ML Meaning                            |
| --------------------------------- | ------------------------------------- |
| Null space = {0} only             | Unique solution θ, healthy model      |
| Null space has other vectors      | Infinite solutions, multicollinearity |
| Two features perfectly correlated | They share null space directions      |

If `Null(X) ≠ {0}`, you can add any null space vector to θ and get the **same predictions** — meaning your weights are not uniquely determined. This is the linear algebra root of **multicollinearity**.

---

## 4. Subspaces in ML — Everywhere You Look

### Hypothesis Space as a Subspace

All possible linear models form a **subspace** of all possible functions:

```
All functions f: ℝⁿ → ℝ          (huge, infinite space)
        ⊃
Linear functions { f(x) = θᵀx }   (a subspace — closed under addition & scaling)
        ⊃
Models with ≤ k nonzero θ          (sparse subspace — used in Lasso)
```

### PCA and Subspace Projection

PCA finds the **subspace of maximum variance**:

```
High-dim data in ℝ¹⁰⁰⁰
        │
        ▼  Project onto subspace
Low-dim data in ℝ²  (principal subspace)
```

The principal components are **basis vectors of this subspace**.

---

## 5. Row Space vs Column Space — Two Perspectives

```
Matrix X (m samples × n features)

Column Space (in ℝᵐ):         Row Space (in ℝⁿ):
"combinations of features"     "combinations of samples"
Used to find predictions ŷ     Used in SVD, data compression
```

**Rank-Nullity Theorem** ties them together:

```
dim(Column Space) + dim(Null Space) = n (total features)

In ML: useful features + redundant features = total features
```

---

## 6. Function Spaces — Extending to Kernel Methods & Neural Nets

Vector spaces don't have to contain just coordinate vectors. **Functions themselves form vector spaces.**

```python
# Two functions in function space:
f(x) = 3x² + 2x + 1
g(x) = x² - 5x + 4

# Linear combination — still a function (stays in the space):
2f(x) + g(x) = 7x² - x + 6  ✓
```

| ML Concept        | Vector Space Used                       |
| ----------------- | --------------------------------------- |
| Linear Regression | ℝⁿ feature space                        |
| Kernel SVM        | Reproducing Kernel Hilbert Space (RKHS) |
| Neural Networks   | Layered composition of vector spaces    |
| Fourier Features  | Frequency function space                |

Kernel methods implicitly map data into a **very high (or infinite) dimensional vector space** where linear separation becomes possible — this is the kernel trick.

---

## 7. The Full Picture

```
Your Dataset
     │
     ▼
Feature Space ℝⁿ  ←── vector space where each sample lives
     │
     │  X matrix formed
     ▼
Column Space(X)   ←── subspace where all predictions live
     │
     │  Project y onto it
     ▼
ŷ = Xθ            ←── best achievable prediction (least squares)
     │
     │  Residual y - ŷ
     ▼
Orthogonal Complement of Col(X)  ←── irreducible error lives here
```

**The entire pipeline of linear regression is a story about vector spaces** — finding which subspace your features define, projecting your target into it, and measuring how far off you are in the orthogonal complement.
