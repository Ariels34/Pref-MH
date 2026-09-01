# When Metropolis and Hastings Meet Bradley and Terry: Exact MCMC From Preference Voting

### Abstract

Sampling from distributions conditioned on desired semantic properties is an emerging challenge in modern generative modeling. Metropolis--Hastings (MH) provides a principled route to conditional sampling, but requires access to exact pointwise target-density evaluations, which are not available in generative settings. Meanwhile, pairwise comparisons by humans or model “judges” are highly accessible and have proved valuable across diverse applications. 
We introduce **Pref-MH**, a general exact MH sampler for judge-induced conditional distributions using only stochastic binary pairwise comparisons.
Our key observation is that the MH unnormalized density ratio matches the preference odds of the Bradley--Terry (BT) choice model. The central challenge is that while MH requires precise ratio computation, BT judges provide only sampled binary feedback.
To this end, we develop a valid accept/reject rule whose resulting Markov chain provably converges to the target distribution. 
We further show that, for a fixed proposal kernel and budget, **Pref-MH** is optimal in the Peskun--Tierney sense among this class of exact reversible acceptance rules. Experiments on text generation and molecular design with LLM judges, as well as image generation with VLM judges, demonstrate that Pref-MH provides a practical and flexible approach to conditional sampling when comparative feedback is relatively easy to obtain.

## Repository Overview

This repository contains the code used to reproduce the theoretical illustrations and experiments in the paper. Each experiment is self-contained within its corresponding directory. 


```text
.
├── synthetic_experiment/
├── text_generation/
├── image_generation/
├── de_novo_molecular_design/
├── machine_translation/
```

### `synthetic_experiment/`

Synthetic experiments used to verify the theoretical behavior of Pref-MH in a setting where the target distribution is known exactly.

The experiments compare Pref-MH with oracle Metropolis--Hastings and plug-in alternatives, and illustrate the effect of the number of pairwise preference queries.

### `text_generation/`

Text-generation experiments demonstrating how Pref-MH can steer a generative model toward desired semantic properties using only pairwise judgments.

The experiments consider both individual and simultaneous semantic constraints, and compare preference-based guidance with pointwise scoring and direct generation.

### `image_generation/`

Image-generation experiments demonstrating the use of Pref-MH in a continuous generative space and with multiple specialized preference criteria.

The experiments evaluate whether pairwise visual judgments can guide generation toward images satisfying several semantic conditions simultaneously while maintaining visual quality.

### `de_novo_molecular_design/`

De novo molecular-design experiments demonstrating the use of Pref-MH on a real-world scientific generation task.

The experiments compare Pref-MH with pointwise LLM guidance and a strong MCMC baseline for molecular design, evaluating whether pairwise judgments can provide an effective signal for generating promising molecules.

### `machine_translation/`

Machine-translation experiments demonstrating the use of Pref-MH for improving structured model outputs through preference feedback.

The experiments compare Pref-MH with pointwise LLM guidance and a strong MCMC baseline for machine translation, demonstrating the benefit of pairwise judgments under the same judge-query budget.


