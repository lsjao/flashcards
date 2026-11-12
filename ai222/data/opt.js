// Optimization cards
window.CARDS = (window.CARDS || []).concat([
{t:"opt",k:"concept",q:String.raw`Why is driving training loss $L(\theta)$ to its minimum <b>not</b> the actual goal in ML optimization?`,
a:String.raw`Because $L$ is only a proxy for the performance measure $P$ we care about (on unseen data). Pure optimization ends at min $L$; ML optimizes $L$ indirectly hoping to improve $P$.`,
r:String.raw`A training loss of ~0 can mean memorization. Goodfellow §8.1 frames this as "indirect" optimization, which is also why surrogate losses and early stopping are used.`,s:"Slide 2 · Goodfellow §8.1"},

{t:"opt",k:"formula",q:String.raw`Write the empirical risk minimization (ERM) objective, starting from the expectation over the empirical distribution.`,
a:String.raw`$$L(\theta)=\mathbb{E}_{(x,y)\sim \hat p_{data}}\,L(f(x;\theta),y)=\frac{1}{m}\sum_{i=1}^{m}L\big(f(x^{(i)};\theta),y^{(i)}\big)$$`,
r:String.raw`The expectation over the training set's empirical distribution is just an average. If $m$ is the full dataset size it is exact ERM; if $m$ is a minibatch it is an unbiased <i>estimate</i> of it.`,s:"Slides 3–4"},

{t:"opt",k:"concept",q:String.raw`Give three reasons minibatches beat full-batch gradient descent, and one cost of very small batches.`,
a:String.raw`<b>Pros:</b> (1) full batch is expensive with less-than-linear return; (2) minibatches parallelize well on GPUs (powers of 2, 32–256); (3) small batches add noise that regularizes and can improve generalization.<br><b>Cost:</b> higher gradient variance, so the learning rate must be reduced.`,
r:String.raw`Standard error of a mean falls as $\sigma/\sqrt m$, so 4× the batch only halves the noise. Goodfellow: generalization is often best near batch size 1, but that needs a small LR. Ruder quotes 50–256 instead of the slide's 32–256.`,s:"Slide 5 · Goodfellow · Ruder"},

{t:"opt",k:"proof",q:String.raw`Show that the minibatch gradient is an unbiased estimate of the full gradient, and explain why returns from larger batches are sub-linear.`,
a:String.raw`With $m$ i.i.d. samples, $\hat g=\frac1m\sum_i \nabla L_i$. By linearity, $\mathbb E[\hat g]=\frac1m\sum_i\mathbb E[\nabla L_i]=\nabla L$. Its variance is $\mathrm{Var}(\hat g)=\sigma^2/m$, so the standard error is $\sigma/\sqrt m$.`,
r:String.raw`Going from $m=100$ to $m=10{,}000$ costs 100× more compute but reduces noise only 10×. That is the "no linear return" point on slide 5 (Goodfellow §8.1.3).`,s:"Slide 5 · Goodfellow §8.1.3"},

{t:"opt",k:"concept",q:String.raw`What does convexity guarantee, and what do we settle for in deep learning?`,
a:String.raw`In a convex loss every local minimum is global. Deep losses are non-convex, so we accept any local minimum (or low region) that meets the performance metric.`,
r:String.raw`Goodfellow §8.2.2/8.2.7: many local minima are roughly equally good in large nets; the real trouble is often poor correspondence between local structure and the global solution.`,s:"Slide 6"},

{t:"opt",k:"concept",q:String.raw`Define a saddle point. Why are they common in high dimensions, and is the slide's claim "easily overcome by SGD" settled?`,
a:String.raw`A critical point whose Hessian has both positive and negative eigenvalues. In high dimensions, saddles outnumber local minima exponentially. The claim is <b>contested</b>: Goodfellow says gradient descent often escapes empirically (Newton is attracted to them); Ruder (citing Dauphin) says saddles plus plateaus are the main difficulty and SGD/Momentum/NAG struggle.`,
r:String.raw`For a random critical point, each eigenvalue being positive is like a coin flip, so "all positive" (a minimum) gets exponentially unlikely as dimension grows. Slide 31's own animation shows SGD stuck at a saddle.`,s:"Slides 7, 31 · Goodfellow · Ruder"},

{t:"opt",k:"concept",q:String.raw`What is a "cliff" in the loss landscape, what goes wrong, and what is the fix?`,
a:String.raw`A very steep region: the gradient is huge, so a step overshoots far past the minimum (exploding gradient). Fix: <b>gradient clipping</b> (cap the gradient norm). Most common in RNNs.`,
r:String.raw`The gradient gives a direction that is only valid locally; clipping keeps the step inside that region of validity. RNNs suffer because the same $W$ is multiplied many times (slide 9).`,s:"Slide 8 · Goodfellow §8.2.4"},

{t:"opt",k:"proof",q:String.raw`Using eigendecomposition, show why repeatedly applying the same $W$ (as in an RNN) makes gradients vanish or explode.`,
a:String.raw`If $W=V\,\mathrm{diag}(\lambda)\,V^{-1}$ then $$W^t=V\,\mathrm{diag}(\lambda)\,V^{-1}V\,\mathrm{diag}(\lambda)V^{-1}\cdots=V\,\mathrm{diag}(\lambda)^t\,V^{-1}$$ since the inner $V^{-1}V=I$ cancel. Components along eigenvectors scale by $\lambda_i^t$: $|\lambda_i|\lt 1\Rightarrow\to0$ (vanish), $|\lambda_i|\gt1\Rightarrow\to\infty$ (explode).`,
r:String.raw`A deep MLP uses a different $W$ per layer, so the product doesn't share one eigenbasis and the effect is milder. Only $|\lambda|\approx1$ is stable, which motivates gated units (LSTM).`,s:"Slide 9 · Goodfellow §8.2.5"},

{t:"opt",k:"concept",q:String.raw`What causes inexact gradients, and where does training usually stop as a result?`,
a:String.raw`Noisy/biased minibatch estimates, small batches, or intractable objectives (only approximate gradients exist). We usually stop at a point with near-zero gradient and acceptable performance, not a true critical point.`,
r:String.raw`Slide 10's plot shows spiky loss that settles over iterations. Goodfellow §8.2.6 gives contrastive divergence as an example of an approximate gradient.`,s:"Slide 10"},

{t:"opt",k:"concept",q:String.raw`What is the "wrong side of the mountain" problem, and what is the slide's solution?`,
a:String.raw`A bad initial point sends local descent into a basin that never reaches the good valley (a peak sits in between). Solution: a principled algorithm for choosing initial parameters.`,
r:String.raw`Local descent only follows local slope. Goodfellow notes descent can fail even without local minima or saddles when the gradient points away from the good solution. This leads into slides 12–15 on initialization.`,s:"Slide 11"},

{t:"opt",k:"concept",q:String.raw`Why must weights be initialized randomly, and what is the simplest scheme on the slides?`,
a:String.raw`To <b>break symmetry</b>: units with identical weights and inputs get identical gradients and stay identical forever. Simplest: Gaussian, zero mean, small std (e.g. 0.01).`,
r:String.raw`If two hidden units start equal, backprop updates them equally, so the layer behaves like one unit. Glorot, He and LeCun are scaled alternatives.`,s:"Slide 12 · Goodfellow"},

{t:"opt",k:"numeric",q:String.raw`Compute the Glorot uniform bound for a layer with 256 inputs and 128 outputs, and the He normal std for fan-in 512.`,
a:String.raw`Glorot: $\sqrt{6/(256+128)}=\sqrt{6/384}=\sqrt{0.015625}=0.125$, so $W\sim\mathcal U(-0.125,\,0.125)$.<br>He normal: $\sqrt{2/512}=\sqrt{1/256}=0.0625$.`,
r:String.raw`Glorot uses fan-in $m$ + fan-out $n$; He uses only fan-in $m$. Notation caveat: the slide's $\mathcal N(w;0,\sqrt{2/(m+n)})$ means std (variance $2/(m+n)$) in common use.`,s:"Slides 13–14"},

{t:"opt",k:"proof",q:String.raw`Show that Glorot uniform and Glorot normal have the same variance, and explain where $2/(m+n)$ and He's $2/m$ come from.`,
a:String.raw`For $\mathcal U(-a,a)$, $\mathrm{Var}=a^2/3$. With $a=\sqrt{6/(m+n)}$: $\mathrm{Var}=\frac{6}{3(m+n)}=\frac{2}{m+n}$, matching std $\sqrt{2/(m+n)}$.<br>For $y=\sum_{i=1}^{m}w_ix_i$, $\mathrm{Var}(y)=m\,\mathrm{Var}(w)\mathrm{Var}(x)$. Keeping forward variance needs $\mathrm{Var}(w)=1/m$; backward needs $1/n$; Glorot compromises with $2/(m+n)$. ReLU zeroes half the inputs, halving the variance, so He uses $2/m$.`,
r:String.raw`This is why He "drops the fan-out term": it targets ReLU forward signal preservation. <i>Derivation is added context; the slides give only the formulas.</i>`,s:"Slides 13–14 · Added context"},

{t:"opt",k:"concept",q:String.raw`Give the bias initializations on the slides and the reason for each.`,
a:String.raw`<ul><li><b>0</b> for linear activations.</li><li><b>~0.1</b> for ReLU, so units start active (avoid initial saturation at 0).</li><li><b>1</b> for the LSTM forget gate, so the gate starts open and remembers.</li></ul>`,
r:String.raw`Goodfellow also suggests setting output biases so softmax(b) matches class frequencies, and starting variance/precision parameters at 1.`,s:"Slide 15 · Goodfellow §8.4"},

{t:"opt",k:"formula",q:String.raw`Write the SGD algorithm from the slides, and the conditions on the learning-rate schedule for convergence.`,
a:String.raw`Sample a minibatch, $g=\frac1m\nabla_\theta\sum_i L(f(x^{(i)};\theta),y^{(i)})$, $\theta\leftarrow\theta-\varepsilon_k g$, $k\leftarrow k+1$.<br>Convergence (Goodfellow): $$\sum_k\varepsilon_k=\infty,\qquad\sum_k\varepsilon_k^2\lt\infty$$`,
r:String.raw`The LR must decay because minibatch noise does not vanish at the minimum: the true gradient goes to 0 but the estimate doesn't. Typical initial LR 0.1 to 0.001; decay by cosine or multistep.`,s:"Slides 17–18"},

{t:"opt",k:"proof",q:String.raw`Explain why each Robbins–Monro condition is needed, and check $\varepsilon_k=1/k$, constant $\varepsilon$, and $\varepsilon_k=1/k^2$.`,
a:String.raw`$\sum\varepsilon_k=\infty$: the total distance you can travel is unbounded, so you can reach the minimum from anywhere. $\sum\varepsilon_k^2\lt\infty$: the accumulated noise variance (∝ $\sum\varepsilon_k^2\sigma^2$) stays finite, so iterates settle.<br>$1/k$: harmonic series diverges ✓, $\sum 1/k^2=\pi^2/6$ ✓ → converges.<br>Constant: $\sum\varepsilon^2=\infty$ ✗ (keeps bouncing).<br>$1/k^2$: $\sum$ finite ✗ (may stop short).`,
r:String.raw`This is the formal reason behind "gradually decrease LR because the noise term dominates later" on slide 17. <i>The check of each schedule is added reasoning.</i>`,s:"Slide 17 · Goodfellow · Added reasoning"},

{t:"opt",k:"formula",q:String.raw`Write the momentum update (slide convention) and state what $\alpha$ controls.`,
a:String.raw`$$v\leftarrow\alpha v-\varepsilon g,\qquad\theta\leftarrow\theta+v$$ $\alpha\in[0,1)$, typically 0.5, 0.9, 0.99. The larger $\alpha$ is relative to $\varepsilon$, the more past gradients affect the current direction.`,
r:String.raw`$v$ is an exponentially decaying sum of past gradients ("snowballing"). Along consistent directions it accelerates; across oscillating directions the signs cancel, damping zigzags in ravines (slide 20).`,s:"Slides 19–21"},

{t:"opt",k:"proof",q:String.raw`Derive the terminal velocity of momentum when the gradient $g$ is constant. What speed-up does $\alpha=0.9$ give? And $\alpha=0.99$?`,
a:String.raw`At steady state $v=\alpha v-\varepsilon g\Rightarrow v(1-\alpha)=-\varepsilon g\Rightarrow\|v\|=\dfrac{\varepsilon\|g\|}{1-\alpha}$.<br>$\alpha=0.9$: $1/(1-0.9)=10\times$ plain GD. $\alpha=0.99$: $100\times$.`,
r:String.raw`This is why momentum can be thought of as multiplying the effective step size by $1/(1-\alpha)$ along consistent directions (Goodfellow §8.3.2).`,s:"Slide 19 · Goodfellow §8.3.2"},

{t:"opt",k:"concept",q:String.raw`How does Nesterov momentum differ from standard momentum?`,
a:String.raw`The gradient is evaluated <b>after</b> applying the momentum jump, at the look-ahead point $\theta+\alpha v$:<br>$g=\frac1m\nabla_\theta\sum L(f(x;\theta+\alpha v),y)$, then $v\leftarrow\alpha v-\varepsilon g$, $\theta\leftarrow\theta+v$.`,
r:String.raw`Looking ahead lets it correct before overshooting (Ruder: "anticipates where parameters are going"). In Ruder's visualizations, NAG goes off course less and corrects sooner than plain momentum.`,s:"Slides 22–23 · Ruder"},

{t:"opt",k:"proof",q:String.raw`Show that the slide's momentum ($v\leftarrow\alpha v-\varepsilon g$, $\theta\leftarrow\theta+v$) and Ruder's ($v_t=\gamma v_{t-1}+\eta\nabla J$, $\theta\leftarrow\theta-v_t$) are the same algorithm.`,
a:String.raw`Let $u=-v$ (Ruder's velocity), $\gamma=\alpha$, $\eta=\varepsilon$. Then $-u_t=-\alpha u_{t-1}-\varepsilon g\Rightarrow u_t=\alpha u_{t-1}+\varepsilon g$, and $\theta\leftarrow\theta+v=\theta-u$. Identical updates; only the sign of the stored velocity differs.`,
r:String.raw`Same for Nesterov: Ruder's look-ahead $\theta-\gamma v_{t-1}$ equals the slide's $\theta+\alpha v$. Don't get tricked by sign conventions on an exam.`,s:"Slides 19, 23 · Ruder"},

{t:"opt",k:"formula",q:String.raw`Write the AdaGrad update and explain its motivation.`,
a:String.raw`$$r\leftarrow r+g\odot g,\quad\Delta\theta=-\frac{\varepsilon}{\delta+\sqrt r}\odot g,\quad\theta\leftarrow\theta+\Delta\theta$$ with $\delta\approx10^{-7}$. Each parameter's LR shrinks in proportion to $1/\sqrt{\text{sum of its past squared gradients}}$, an "equalizer" between parameters with big and small gradients.`,
r:String.raw`Parameters that rarely get gradient (sparse features) keep a large step; frequently updated ones get smaller steps. Ruder: good for sparse data, typical $\eta=0.01$ (and he quotes $\delta\approx10^{-8}$).`,s:"Slides 24–25 · Ruder"},

{t:"opt",k:"numeric",q:String.raw`Two parameters under AdaGrad get constant gradients $g_1=1$ and $g_2=0.01$ every step. With $\varepsilon=0.1$ and ignoring $\delta$, what is each step size at $t=100$?`,
a:String.raw`$r_i=t\,g_i^2$, so step $=\varepsilon\,|g_i|/\sqrt{t g_i^2}=\varepsilon/\sqrt t$. At $t=100$: $0.1/10=0.01$ for <b>both</b> parameters.`,
r:String.raw`This shows the "equalizer" effect: a 100× difference in gradient scale disappears. It also shows the flaw: the step keeps falling as $1/\sqrt t$ no matter what.`,s:"Slide 24 · Added worked example"},

{t:"opt",k:"proof",q:String.raw`Prove that AdaGrad's effective learning rate goes to zero, and show how RMSProp avoids this.`,
a:String.raw`AdaGrad: if $|g_t|\ge g_{min}\gt0$, then $r_t\ge t\,g_{min}^2\to\infty$, so $\varepsilon/(\delta+\sqrt{r_t})\to0$ and learning stalls.<br>RMSProp: $r\leftarrow\rho r+(1-\rho)g^2$ is an exponential moving average. For stationary gradients $r\to\mathbb E[g^2]$ (bounded), so the step stays near $\varepsilon\,g/\mathrm{RMS}(g)$.`,
r:String.raw`The sum never forgets; the EMA discards extreme-past history (slide 26). That makes RMSProp better for non-convex problems where early gradients are irrelevant later.`,s:"Slides 24–27 · Ruder"},

{t:"opt",k:"formula",q:String.raw`Write the RMSProp update with the slide's defaults. What inconsistency between the slide and the textbook should you know?`,
a:String.raw`$$r\leftarrow\rho r+(1-\rho)g\odot g,\quad\Delta\theta=-\frac{\varepsilon}{\delta+\sqrt r}\odot g$$ $\rho\approx0.9$, $\delta\approx10^{-6}$. The slide puts $\delta$ <b>outside</b> the root ($\delta+\sqrt r$); Goodfellow Alg. 8.5 and Ruder put it <b>inside</b> ($\sqrt{\delta+r}$).`,
r:String.raw`Both just prevent division by zero; the effect differs only when $r$ is tiny ($\sqrt{10^{-6}}=10^{-3}$ vs $10^{-6}$). Ruder: from Hinton's Coursera lecture, suggested $\eta=0.001$.`,s:"Slides 26–27 · Goodfellow · Ruder"},

{t:"opt",k:"formula",q:String.raw`Write the full Adam update with bias correction, and its four default hyperparameters (slide notation).`,
a:String.raw`$t\leftarrow t+1$<br>$s\leftarrow\rho_1 s+(1-\rho_1)g,\quad \hat s=s/(1-\rho_1^t)$<br>$r\leftarrow\rho_2 r+(1-\rho_2)g\odot g,\quad \hat r=r/(1-\rho_2^t)$<br>$\Delta\theta=-\varepsilon\,\hat s/(\delta+\sqrt{\hat r})$<br>Defaults: $\varepsilon=0.001$, $\rho_1=0.9$, $\rho_2=0.999$, $\delta=10^{-8}$.`,
r:String.raw`Adam = RMSProp's second moment + a momentum-like first moment + bias correction for the zero initialization. Ruder uses $\eta,\beta_1,\beta_2,\epsilon$ for the same symbols.`,s:"Slides 28–29 · Kingma & Ba · Ruder"},

{t:"opt",k:"proof",q:String.raw`Prove that Adam's first moment is biased toward zero by the factor $(1-\rho_1^t)$, which justifies dividing by it.`,
a:String.raw`With $s_0=0$: $s_t=(1-\rho_1)\sum_{i=1}^{t}\rho_1^{t-i}g_i$. If $\mathbb E[g_i]=\mathbb E[g]$ (stationary), $$\mathbb E[s_t]=(1-\rho_1)\mathbb E[g]\sum_{i=1}^{t}\rho_1^{t-i}$$ $$=(1-\rho_1)\frac{1-\rho_1^t}{1-\rho_1}\mathbb E[g]=(1-\rho_1^t)\,\mathbb E[g]$$ So $\hat s_t=s_t/(1-\rho_1^t)$ is unbiased. The same argument with $g^2$ gives $\hat r$.`,
r:String.raw`At $t=1$: $s_1=(1-\rho_1)g$ and $\hat s_1=g$ exactly (the slide 28 QUESTION). The bias vanishes as $t$ grows since $\rho_1^t\to0$.`,s:"Slide 28 · Ruder · Added derivation"},

{t:"opt",k:"numeric",q:String.raw`Adam, first step, gradient $g=4$ for one parameter, defaults ($\varepsilon=0.001$). Compute $\hat s$, $\hat r$ and $\Delta\theta$. What would the step be <b>without</b> bias correction?`,
a:String.raw`$s_1=0.1\cdot4=0.4\Rightarrow\hat s=0.4/0.1=4$. $r_1=0.001\cdot16=0.016\Rightarrow\hat r=0.016/0.001=16$, $\sqrt{\hat r}=4$. $\Delta\theta\approx-0.001\cdot4/4=-0.001$.<br>Without correction: $-0.001\cdot0.4/\sqrt{0.016}=-0.001\cdot0.4/0.1265\approx-0.00316$, about 3.16× too large.`,
r:String.raw`The uncorrected ratio is $(1-\rho_1)/\sqrt{1-\rho_2}=0.1/0.0316=3.16$. Notice the corrected first step is $\approx\varepsilon\cdot\mathrm{sign}(g)$, independent of gradient scale.`,s:"Slides 28–29 · Added worked example"},

{t:"opt",k:"concept",q:String.raw`On the saddle-point animation (slide 31), which optimizers escape quickly and which struggle?`,
a:String.raw`Adagrad, RMSProp and Adadelta escape quickly. SGD, Momentum and NAG struggle to break symmetry and linger at the saddle.`,
r:String.raw`At a saddle the gradient is near zero in the escape direction. Adaptive methods divide by a small $\sqrt r$ there, amplifying the tiny gradient. This figure contradicts slide 7's "easily overcome by SGD."`,s:"Slides 30–31 · Ruder"},

{t:"opt",k:"concept",q:String.raw`Which optimizer is best? Give the answer from slide 44 and each link.`,
a:String.raw`No clear winner. SGD+Momentum and Adam are the usual defaults; it depends on the task (Schmidt et al. 2020 plot: lines cross across benchmarks). Goodfellow: no consensus, often chosen by familiarity; Adam fairly robust to hyperparameters. Ruder: Adam is a good default; adaptive methods for sparse data; SGD+momentum with annealing can generalize well but is slower and more sensitive to init and schedule.`,
r:String.raw`Ruder also notes adaptive methods can sometimes converge to worse solutions than SGD+momentum, and Adam's moving average can wash out rare large gradients (the motivation for AMSGrad).`,s:"Slide 44 · Goodfellow · Ruder"},

{t:"opt",k:"formula",q:String.raw`Write the cosine-decay LR formula from the slide code.`,
a:String.raw`$$lr=lr_{final}+\tfrac12\,(lr_{init}-lr_{final})\Big(1+\cos\frac{\pi\cdot epoch}{total\_epochs}\Big)$$`,
r:String.raw`At epoch 0: $\cos0=1$ → $lr_{init}$. At the end: $\cos\pi=-1$ → $lr_{final}$. The curve is flat at both ends and steepest in the middle (inflection at the midpoint).`,s:"Slides 33–34"},

{t:"opt",k:"numeric",q:String.raw`Cosine decay with $lr_{init}=0.1$, $lr_{final}=0.01$, 150 epochs. What is the LR at epoch 75 and at epoch 50?`,
a:String.raw`Epoch 75: $\cos(\pi/2)=0$ → $0.01+0.5(0.09)(1)=0.055$.<br>Epoch 50: $\cos(\pi/3)=0.5$ → $0.01+0.5(0.09)(1.5)=0.0775$.`,
r:String.raw`Matches the slide 33 plot (≈0.055 at the midpoint). Being able to evaluate scheduler code at a given epoch is a listed priority.`,s:"Slides 33–34"},

{t:"opt",k:"numeric",q:String.raw`Multistep decay: $lr_{init}=0.1$, milestones [30, 60, 90], decay factor 0.1. What is the LR at epochs 29, 30, 65 and 140?`,
a:String.raw`Epoch 29: 0.1. Epoch 30: 0.01 (≥30). Epoch 65: 0.001 (≥30 and ≥60). Epoch 140: 0.0001 (≥ all three).`,
r:String.raw`The code multiplies by the factor once for <i>every</i> milestone already passed (<code>if epoch &gt;= milestone: lr *= decay_factor</code>), producing the staircase on slide 35.`,s:"Slides 35–36"},

{t:"opt",k:"numeric",q:String.raw`Linear decay ($lr_{init}=0.1$, 150 epochs): LR at epoch 120? How does Goodfellow's linear schedule differ at the end?`,
a:String.raw`$0.1-(0.1/150)\cdot120=0.1-0.08=0.02$. The slide goes to exactly 0 at epoch 150; Goodfellow stops at $\varepsilon_\tau\approx1\%$ of $\varepsilon_0$ and holds it constant after $\tau$.`,
r:String.raw`Goodfellow: $\varepsilon_k=(1-\alpha)\varepsilon_0+\alpha\varepsilon_\tau$ with $\alpha=k/\tau$. Reaching 0 only at the very last epoch is harmless because no further updates happen.`,s:"Slides 37–38 · Goodfellow"},

{t:"opt",k:"numeric",q:String.raw`Cosine with restarts (period 50, $lr_{init}=0.1$, $lr_{min}=0.01$): LR at epoch 125? And at epoch 100?`,
a:String.raw`Epoch 125: $125\bmod50=25$ → decay $=0.5(1+\cos(\pi/2))=0.5$ → $0.01+0.09\cdot0.5=0.055$.<br>Epoch 100: $100\bmod50=0$ → decay $=1$ → back to 0.1 (a restart).`,
r:String.raw`The jumps at 50 and 100 give the sawtooth on slide 39. Restarts let the optimizer leave a sharp basin and re-explore.`,s:"Slides 39–40"},

{t:"opt",k:"numeric",q:String.raw`Cosine with 5 warmup epochs, $lr_{init}=0.1$, $lr_{min}=0.01$, 150 total. LR at epochs 0, 2 and 5?`,
a:String.raw`Epoch 0: $0.1\cdot0/5=0$. Epoch 2: $0.1\cdot2/5=0.04$. Epoch 5: warmup over, epoch→0, $\cos0=1$ → $0.01+0.5(0.09)(2)=0.1$.`,
r:String.raw`Warmup avoids huge early steps while Adam/BN statistics are unreliable and the weights are random. Note the quirk: at epoch 0 the LR is exactly 0, so that first epoch updates nothing (slide 42 QUESTION).`,s:"Slides 41–42"},

{t:"opt",k:"concept",q:String.raw`What problem does AMSGrad target, and is it settled that it helps?`,
a:String.raw`Adam's exponential average can forget rare but large, informative gradients, causing poor convergence. AMSGrad keeps the running <b>max</b> of past second moments, $\hat v_t=\max(\hat v_{t-1},v_t)$, so the step never increases. Not settled: other experiments found similar or worse results than Adam.`,
r:String.raw`Extension material from Ruder (Part C). Also know: Adadelta removes the global LR, AdaMax uses an $\ell_\infty$ second moment, Nadam adds Nesterov to Adam.`,s:"Ruder (Part C)"},

{t:"opt",k:"concept",q:String.raw`How does Adadelta avoid a global learning rate?`,
a:String.raw`It scales the step by the ratio of the RMS of past <i>updates</i> to the RMS of gradients: $\Delta\theta_t=-\dfrac{\mathrm{RMS}[\Delta\theta]_{t-1}}{\mathrm{RMS}[g]_t}\,g_t$, with $\gamma\approx0.9$.`,
r:String.raw`The numerator has the units of $\theta$, so the update has correct units with no hand-set $\eta$. Its first update coincides with RMSProp's, which is why they look alike in the visualizations.`,s:"Ruder (Part C)"},

{t:"opt",k:"concept",q:String.raw`Why is Newton's method impractical for large networks, and what failure mode does it have near saddles?`,
a:String.raw`Update $\theta^*=\theta_0-H^{-1}\nabla J(\theta_0)$ requires inverting a $k\times k$ Hessian: $O(k^3)$ per step, impossible for millions of parameters. Near saddles it is <b>attracted</b> to them, because it jumps to any critical point of the local quadratic.`,
r:String.raw`With negative eigenvalues, $-H^{-1}g$ points <i>toward</i> the saddle along those directions. Regularized versions add $\alpha I$ (Levenberg–Marquardt). Second-order methods also need much larger batches (~10,000).`,s:"Goodfellow §8.6 (Part C)"},

{t:"opt",k:"formula",q:String.raw`Write the gradient-noise schedule from Ruder and explain why the noise should shrink.`,
a:String.raw`$g_{t,i}\leftarrow g_{t,i}+\mathcal N(0,\sigma_t^2)$ with $\sigma_t^2=\dfrac{\eta}{(1+t)^\gamma}$.`,
r:String.raw`Early noise helps escape poor local minima and saddles in deep nets; later it must decay so the iterates can settle (the same logic as the Robbins–Monro conditions).`,s:"Ruder (Part C)"},

{t:"opt",k:"proof",q:String.raw`Theory: why do adaptive methods (AdaGrad/RMSProp/Adam) help on sparse features? Argue from the update rule.`,
a:String.raw`A rarely active feature gets few nonzero gradients, so its accumulator $r_i$ stays small, and its effective LR $\varepsilon/\sqrt{r_i}$ stays large. When it does appear, it gets a big update. A frequently active feature has a large $r_i$ and gets small, careful steps.`,
r:String.raw`Plain SGD uses one LR for all parameters, so rare features barely learn. Ruder: with sparse inputs, adaptive methods usually work well at their default LR.`,s:"Slide 24 · Ruder"},

{t:"opt",k:"proof",q:String.raw`Out of the box: Adam's corrected first step is $\approx-\varepsilon\,\mathrm{sign}(g)$. What does this imply about Adam's sensitivity to gradient scale? Prove it.`,
a:String.raw`If every gradient is multiplied by $c\gt0$: $\hat s\to c\hat s$, $\hat r\to c^2\hat r$, so $\hat s/\sqrt{\hat r}$ is unchanged (ignoring $\delta$). Adam is invariant to rescaling the loss; step size is controlled by $\varepsilon$, not by gradient magnitude.`,
r:String.raw`This is part of why Adam is "fairly robust to hyperparameters" (Goodfellow). It also means $\varepsilon$ roughly bounds the per-parameter step size, unlike SGD where the step is $\varepsilon\|g\|$.`,s:"Slides 28–29 · Added reasoning"},

{t:"opt",k:"concept",q:String.raw`In which situation should you pick cosine-with-restarts over plain cosine, and warmup over no warmup?`,
a:String.raw`Restarts: when you suspect convergence to sharp or poor basins and want periodic re-exploration (also gives several snapshot models). Warmup: when early training is unstable, e.g. large batches, high LR, Adam or transformers, where random weights plus big steps can diverge.`,
r:String.raw`The slides give the shapes and code only; the "when" is practical guidance. Know how each shape looks and its key parameter: restart_period = 50, warmup_epochs = 5.`,s:"Slides 39–42 · Added context"}
]);
