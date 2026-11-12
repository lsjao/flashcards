// Regularization cards
window.CARDS = (window.CARDS || []).concat([
{t:"reg",k:"concept",q:String.raw`What is the real goal behind regularization, and which number must you always report?`,
a:String.raw`<b>Generalization</b>: same model, same dataset, tune the training configuration so it performs well on unseen data. Report <b>test/validation</b> performance on an accepted, "highly defensible" split, never training performance.`,
r:String.raw`The professor called reporting training performance in a capstone "not correct." A self-invented test split is not acceptable either.`,s:"Slide 2 · Lecture"},

{t:"reg",k:"concept",q:String.raw`Define overfitting. If training loss is near 0, what single measurement tells you whether the model memorized?`,
a:String.raw`Overfitting = a model with enough complexity <b>memorizes</b> the input-output pairs. The test (unseen) split tells you: good test performance means it learned something.`,
r:String.raw`A very large model is fine if test performance is good; size alone isn't the problem, failing on unseen data is.`,s:"Slide 3"},

{t:"reg",k:"concept",q:String.raw`Organize the regularization toolbox by <i>where</i> each technique acts.`,
a:String.raw`<ul><li><b>Loss:</b> weight penalty.</li><li><b>Data:</b> augmentation (CutOut, MixUp, CutMix, RandAugment, STRAug), input noise, ADR.</li><li><b>Weights:</b> weight noise.</li><li><b>Architecture:</b> latent bottleneck, dropout, multitask, parameter sharing.</li><li><b>Schedule:</b> early stopping.</li><li><b>Evaluation:</b> cross-validation.</li></ul>`,
r:String.raw`Regularizers can be combined, but stacking too many makes training worse. Pick the ones that suit your model and dataset.`,s:"Slides 4, 46 · Lecture"},

{t:"reg",k:"formula",q:String.raw`Write the empirical risk with weight penalty. What exactly is penalized?`,
a:String.raw`$$R_{emp}(f,X,y)=\frac1N\sum_{n=1}^{N}\big(y_n-f(x_n,\theta)\big)^2+\lambda\,\theta^\top\theta$$ Only the <b>weights</b> (not biases), via their scaled dot product with themselves.`,
r:String.raw`Slide 5's sum index is inconsistent (written $i=1$ to $n$); read it as $n=1$ to $N$. $\lambda$ is a small constant.`,s:"Slide 5"},

{t:"reg",k:"numeric",q:String.raw`Three weight sets give the same minimal empirical risk: $\theta_A=(0.5,0.5)$, $\theta_B=(1,0)$, $\theta_C=(0.2,0.9)$. Which does the weight penalty choose? If each input gets noise $e=0.1$, how much does each output shift?`,
a:String.raw`$\theta^\top\theta$: A = 0.50, B = 1.00, C = 0.85 → choose <b>A</b>.<br>Output shift $=\theta^\top e$: A = 0.1, B = 0.1, C = 0.11. For random-sign noise, the shift's variance is $\sigma^2\theta^\top\theta$: smallest for A.`,
r:String.raw`Small weights multiply input perturbations by small numbers, so decisions move less. That is the robustness argument in the lecture.`,s:"Slide 5 · Added worked example"},

{t:"reg",k:"concept",q:String.raw`Define MLE as on the slides. Why is minimizing $-\log p(\mathcal D|\theta)$ the same as maximizing $p(\mathcal D|\theta)$?`,
a:String.raw`MLE: choose $\theta$ that best explains the data, $\mathcal L_{\mathcal D}(\theta)=-\log p(\mathcal D|\theta)$. Since $\log$ is strictly increasing, the argmax of $p$ equals the argmax of $\log p$, which equals the argmin of $-\log p$.`,
r:String.raw`The log doesn't move the optimum, but it turns products into sums (numerically stable, easy gradients). The loss is minimized as $p\to1$.`,s:"Slide 7"},

{t:"reg",k:"concept",q:String.raw`What do the two parts of the IID assumption each buy you in the MLE derivation?`,
a:String.raw`<b>Independent:</b> the joint likelihood factorizes into a product, $p(\mathcal Y|\mathcal X,\theta)=\prod_n p(\hat y_n|x_n,\theta)$.<br><b>Identically distributed:</b> every point uses the same distribution and the same parameters $\theta$.`,
r:String.raw`After the product, $-\log$ turns it into a sum, $-\sum_n\log p(\hat y_n|x_n,\theta)$, which is a per-example loss you can minimize.`,s:"Slides 9–10"},

{t:"reg",k:"proof",q:String.raw`Derive that MLE with Gaussian noise $p(\hat y_n|x_n,\theta)=\mathcal N(x_n^\top\theta,\sigma^2)$ gives the squared loss.`,
a:String.raw`$$\mathcal L=-\sum_n\log\Big[\tfrac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(\hat y_n-x_n^\top\theta)^2}{2\sigma^2}}\Big]=\sum_n\frac{(\hat y_n-x_n^\top\theta)^2}{2\sigma^2}-\sum_n\log\frac{1}{\sqrt{2\pi\sigma^2}}$$ The second sum is constant in $\theta$. Minimizing $\mathcal L$ is minimizing $\sum_n(\hat y_n-x_n^\top\theta)^2$.`,
r:String.raw`The log cancels the exponential. This gives the "risk" part of ERM; the weight penalty is still missing, which MAP supplies next.`,s:"Slides 11–12"},

{t:"reg",k:"proof",q:String.raw`Derive the weight penalty from MAP with a zero-mean Gaussian prior on $\theta$.`,
a:String.raw`Bayes: $p(\theta|\mathcal D)=\frac{p(\mathcal D|\theta)p(\theta)}{p(\mathcal D)}\propto p(\mathcal D|\theta)p(\theta)$ ($p(\mathcal D)$ has no $\theta$).<br>$\arg\max_\theta\log p(\theta|\mathcal D)=\arg\min_\theta\big[\underbrace{-\log p(\mathcal D|\theta)}_{\text{MLE}}\underbrace{-\log p(\theta)}_{\text{regularizer}}\big]$.<br>With $p(\theta)=\mathcal N(0,\tau^2 I)$: $-\log p(\theta)=\frac{1}{2\tau^2}\theta^\top\theta+\text{const}=\lambda\,\theta^\top\theta$.`,
r:String.raw`"Hindi naman pwedeng palabasin lang": the penalty has a derivation. It is "risk (MLE) plus a tunable knob" preferring small weights.`,s:"Slides 13–14"},

{t:"reg",k:"proof",q:String.raw`The slides show $p(\theta)=\mathcal N(0,\Sigma)$ but never relate $\lambda$ to it. With noise variance $\sigma^2$ and prior $\mathcal N(0,\tau^2I)$, find $\lambda$.`,
a:String.raw`MAP objective: $\frac{1}{2\sigma^2}\sum_n(y_n-x_n^\top\theta)^2+\frac{1}{2\tau^2}\theta^\top\theta$. Multiply by $2\sigma^2$ (doesn't move the argmin): $\sum_n(y_n-x_n^\top\theta)^2+\frac{\sigma^2}{\tau^2}\theta^\top\theta$. So $$\lambda=\sigma^2/\tau^2$$ e.g. $\sigma^2=1,\tau^2=10\Rightarrow\lambda=0.1$.`,
r:String.raw`A tighter prior (small $\tau$, strong belief that weights are small) or noisier data (large $\sigma$) means stronger regularization. <i>Added derivation; fills the gap flagged on slide 14.</i>`,s:"Slide 14 · Added derivation"},

{t:"reg",k:"proof",q:String.raw`Which prior gives an <b>L1</b> penalty instead? Why does L1 produce sparse weights?`,
a:String.raw`A Laplace prior $p(\theta_j)\propto e^{-|\theta_j|/b}$ gives $-\log p(\theta)=\frac1b\sum_j|\theta_j|+\text{const}$, the L1 penalty.<br>Sparsity: L1's gradient is a constant $\pm\lambda$ no matter how small $\theta_j$ is, so it pushes weights all the way to 0. L2's gradient $2\lambda\theta_j$ fades near 0, so weights shrink but rarely hit exactly 0.`,
r:String.raw`Slide 46 lists L2/L1, but the lecture derived only the L2 ($\theta^\top\theta$) case. <i>Added context, not covered in lecture.</i>`,s:"Slide 46 · Added context"},

{t:"reg",k:"proof",q:String.raw`Derive the closed-form L2-regularized (ridge) solution for linear regression and explain how it relates to slide 38's $w=(X^\top X)^{-1}X^\top y$.`,
a:String.raw`$J=(Xw-y)^\top(Xw-y)+\lambda w^\top w$. $\nabla_w J=2X^\top(Xw-y)+2\lambda w=0\Rightarrow (X^\top X+\lambda I)w=X^\top y$, so $$w^*=(X^\top X+\lambda I)^{-1}X^\top y$$ With $\lambda=0$ it reduces to slide 38's least-squares solution.`,
r:String.raw`Adding $\lambda I$ shrinks $w$ and makes the matrix invertible even when $X^\top X$ is singular (more features than samples). <i>Added derivation.</i>`,s:"Slides 5, 38 · Added derivation"},

{t:"reg",k:"proof",q:String.raw`Prove that adding Gaussian noise $\epsilon\sim\mathcal N(0,\sigma^2I)$ to the <b>input</b> of a linear model is equivalent, in expectation, to an L2 penalty.`,
a:String.raw`$\mathbb E_\epsilon[(y-(x+\epsilon)^\top\theta)^2]=\mathbb E[(y-x^\top\theta-\epsilon^\top\theta)^2]$<br>$=(y-x^\top\theta)^2-2(y-x^\top\theta)\mathbb E[\epsilon^\top\theta]+\mathbb E[(\epsilon^\top\theta)^2]$<br>$=(y-x^\top\theta)^2+0+\sigma^2\theta^\top\theta$.<br>So the expected noisy loss is the clean loss plus $\lambda\theta^\top\theta$ with $\lambda=\sigma^2$.`,
r:String.raw`This ties slide 33 (input noise) to slide 5 (weight penalty): both prefer small weights. <i>Added result (Bishop 1995), not shown in lecture.</i>`,s:"Slides 5, 33 · Added context"},

{t:"reg",k:"concept",q:String.raw`State the capacity rule for underfitting, overfitting and optimal fit in terms of train and test loss.`,
a:String.raw`<ul><li>Low capacity → <b>underfit</b>: high train loss, high test loss.</li><li>High capacity → <b>overfit</b>: low train loss, high test loss.</li><li>Right capacity → <b>optimal</b>: low train and low test loss.</li></ul>`,
r:String.raw`Analogy: a genius child taught only to memorize fails on new real-world problems. We want enough capacity to learn patterns but not to memorize.`,s:"Slide 15"},

{t:"reg",k:"concept",q:String.raw`Data come from $y=x^2+x+1+\lambda\sin(\beta x)$, $x\in[-3,3]$. Describe the 1st-, 6th- and 2nd-degree fits.`,
a:String.raw`<b>1st degree:</b> straight line, underfits (big train and test error). <b>6th degree:</b> wiggly curve that hugs training points but diverges beyond the data ends, so it overfits. <b>2nd degree:</b> parabola, optimal, keeps following the pattern when extrapolated.`,
r:String.raw`The sine term is noise. Watch the notation clash: $\lambda$ here is noise amplitude, but on slides 5 and 14 it's the penalty scale.`,s:"Slides 16–19"},

{t:"reg",k:"concept",q:String.raw`What is data augmentation, and what constraint must every augmentation satisfy? Give the slide's counterexample.`,
a:String.raw`Perturb existing data whose label you <b>know stays the same</b>, giving extra exposure without new data. The transform must not change the meaning of the input or the label. Counterexample: "b" rotated/flipped becomes "q", mirrored becomes "d", so geometric flips break text recognition.`,
r:String.raw`Augmentation attacks the data, not the loss. There are infinitely many augmentations, but time is limited, so choose an optimal set.`,s:"Slides 20–22"},

{t:"reg",k:"concept",q:String.raw`Why does training on deliberately degraded or hard data make clean-data prediction easier?`,
a:String.raw`The model must still decide correctly when informative parts are missing or corrupted, so it learns more robust, redundant features. Clean test data is then an "easy" case of what it already handles.`,
r:String.raw`Lecture analogy: students who do well on difficult exams generalize after graduation. The same logic underlies CutOut, noise injection and ADR.`,s:"Slides 4, 27, 33"},

{t:"reg",k:"formula",q:String.raw`Write MixUp's input and label formulas. Is $\tilde y$ a number or a vector?`,
a:String.raw`$$\tilde x=\lambda x_1+(1-\lambda)x_2,\qquad\tilde y=\lambda y_1+(1-\lambda)y_2$$ $\tilde y$ is a <b>vector</b> over classes (soft label). The lecture corrected the "+" to read as combining one-hot vectors.`,
r:String.raw`$\lambda$ sets the opacity of the blend (Zhang et al., ICLR 2018). The slide writes a plain sum, but with one-hot $y$'s it's a probability vector.`,s:"Slide 28"},

{t:"reg",k:"numeric",q:String.raw`MixUp with $\lambda=0.7$, $x_1$ = bird (one-hot [1,0]), $x_2$ = cat ([0,1]). A pixel is 200 in the bird image and 100 in the cat image. Give $\tilde y$ and the mixed pixel.`,
a:String.raw`$\tilde y=0.7[1,0]+0.3[0,1]=[0.7,\,0.3]$. Pixel: $0.7\cdot200+0.3\cdot100=140+30=170$.`,
r:String.raw`The label proportions equal the blend proportions, so the model is told the image is "70% bird."`,s:"Slide 28 · Worked example"},

{t:"reg",k:"numeric",q:String.raw`CutMix: a $112\times112$ cat patch is pasted onto a $224\times224$ bird image. What is $\tilde y$?`,
a:String.raw`Patch area fraction $=\frac{112^2}{224^2}=\frac14$. So $\tilde y=0.75\cdot\text{bird}+0.25\cdot\text{cat}=[0.75,\,0.25]$.`,
r:String.raw`$\tilde x=M\odot x_1+(1-M)\odot x_2$, with $\lambda$ proportional to the area kept from $x_1$ (Yun et al., CVPR 2019).`,s:"Slide 29"},

{t:"reg",k:"concept",q:String.raw`Compare CutOut, MixUp and CutMix on how the input and the label change.`,
a:String.raw`<ul><li><b>CutOut:</b> remove a random region (black square); label unchanged.</li><li><b>MixUp:</b> global opacity blend of two images; label mixed by $\lambda$.</li><li><b>CutMix:</b> paste a real patch from another image; label mixed by patch area.</li></ul>`,
r:String.raw`CutMix keeps CutOut's spatial occlusion but fills the hole with useful content instead of wasted black pixels, and avoids MixUp's unnatural ghosted images.`,s:"Slides 27–29"},

{t:"reg",k:"concept",q:String.raw`What constraint does AgMax add, and where (input, latent or output) is it applied?`,
a:String.raw`Two different augmentations of the same input pass through <b>one shared encoder</b> $\mathcal E(\cdot;\theta)$; their latent codes $z_1,z_2$ must <b>agree</b> (maximize mutual information) and also share the label. It is applied in the <b>latent</b> space, before prediction.`,
r:String.raw`ImageNet ResNet50, 90 epochs: AgMax lifts every baseline at no data cost (e.g. MixUp 76.5 → 77.6; Standard 76.4 → 76.9). It's the same shared-θ idea as parameter sharing.`,s:"Slides 30–31 (Atienza, WACV 2022)"},

{t:"reg",k:"concept",q:String.raw`In the AgMax results, AutoAugment alone is <i>below</i> the baseline in every group. Give a plausible reason.`,
a:String.raw`Open question (not answered in lecture). Plausible: the AA policy was searched for a different setup, and at 90 epochs strong augmentation underfits; combining AA with another strong augmentation (MixUp/CutMix) over-regularizes. Adding AgMax recovers it (e.g. CutMix +AA 75.5 → +AA+AgMax 77.0).`,
r:String.raw`This illustrates the lecture's warning: regularizers stacked too heavily can make training worse.`,s:"Slide 31 · Open"},

{t:"reg",k:"concept",q:String.raw`What are RandAugment and STRAug?`,
a:String.raw`<b>RandAugment:</b> apply a random sequence of $N$ augmentations (e.g. ShearX, AutoContrast) at a chosen magnitude; label unchanged (Cubuk 2020; slide uses $N=2$, magnitudes 9/17/28).<br><b>STRAug:</b> Atienza's 36 augmentations for scene text recognition in groups (warp, geometry, noise, blur, weather, camera, process); ViTSTR-Base 74.7 → 81.3.`,
r:String.raw`STRAug exists because geometric flips break text. Its transforms keep a word readable to a human ("DELIVERY" is still readable after warping).`,s:"Slides 23–25, 32"},

{t:"reg",k:"concept",q:String.raw`Noise injection: where can noise go, when is it applied, and how often?`,
a:String.raw`In the <b>input data</b> or in the <b>weights</b> (or both). Only during <b>training</b>; removed at test. Every time you sample (each iteration), fresh noise is injected.`,
r:String.raw`From the student Q&A. Slide 33's "not insensitive to noise" is a likely typo; networks <i>are</i> sensitive to noise, which is why training with noise helps.`,s:"Slides 33, 38 · Q&A"},

{t:"reg",k:"concept",q:String.raw`Why does noise in the weights act as a regularizer? Explain "flat region."`,
a:String.raw`Training with perturbed weights rewards solutions where small weight changes don't change the prediction. That drives $w$ to a <b>flat region</b> of the loss landscape (low curvature), where a small perturbation leaves the loss nearly unchanged.`,
r:String.raw`Flat minima are robust: if test data shift the effective loss slightly, a flat basin still gives low loss, while a sharp one may not. Bayesian view: weights are stochastic.`,s:"Slide 38"},

{t:"reg",k:"concept",q:String.raw`What is (Automatic) Domain Randomization, and what project of the professor used the idea?`,
a:String.raw`Randomize simulated environments over an ever-growing distribution, train there, then transfer to the real world with little tuning (OpenAI Rubik's-cube hand). The professor's <b>GOO</b> gaze-object dataset used it: lots of generated data plus a small real set.`,
r:String.raw`If the real world looks like just one more random variation, the model generalizes to it (Sim2Real).`,s:"Slides 34–37"},

{t:"reg",k:"concept",q:String.raw`How is multitask learning a regularizer? Give the slide's example.`,
a:String.raw`One shared body with several heads (classification, boxes, masks). The body must learn features useful for all tasks: a soft constraint that limits overfitting to any one task. Example: <b>Mask R-CNN</b>.`,
r:String.raw`Parameters are shared, and some factors explaining the data's variation are shared across tasks.`,s:"Slide 39"},

{t:"reg",k:"concept",q:String.raw`Explain parameter sharing with the slide's stereo-disparity example.`,
a:String.raw`Networks A (left image) and B (right image) share parameters, effectively one network with two inputs; network C predicts disparity (a proxy for depth). Sharing forces robustness because the same weights must work on different images from the same distribution.`,
r:String.raw`Common in Siamese networks (Atienza, ICRA 2018). Halving the free parameters also limits capacity.`,s:"Slide 40"},

{t:"reg",k:"concept",q:String.raw`State the early-stopping rule, and how "wait 10 epochs" fits in.`,
a:String.raw`Watch validation error; when it starts to rise after a minimum, stop and keep the weights from the minimum. "10 epochs" is a <b>patience</b> example (how long you tolerate rising loss), not a fixed rule.`,
r:String.raw`It limits the reachable parameter space, so it's a regularizer. "Don't waste time": a rising loss means you've already passed the best point.`,s:"Slide 41"},

{t:"reg",k:"numeric",q:String.raw`Validation loss is minimal at epoch 12 and rises every epoch after. With patience 10, when do you stop and which weights do you keep?`,
a:String.raw`Stop at epoch <b>22</b> (10 epochs without improvement) and restore the epoch-<b>12</b> weights.`,
r:String.raw`Saving a checkpoint at each new minimum is what makes "keep the best weights" possible.`,s:"Slide 41 · Worked example"},

{t:"reg",k:"proof",q:String.raw`Theory: why is early stopping similar to L2 regularization? (Goodfellow, ch. 7)`,
a:String.raw`Starting near $\theta=0$ and taking $\tau$ gradient steps with LR $\varepsilon$ limits how far $\theta$ can travel. For a quadratic loss, the solution after $\tau$ steps matches L2-regularized training with $\lambda\approx\frac{1}{\tau\varepsilon}$. Fewer steps means stronger regularization.`,
r:String.raw`This matches the slide's phrase "limits the parameter space." <i>Added context from the course textbook (Goodfellow §7.8), not derived in lecture.</i>`,s:"Slide 41 · Goodfellow §7.8"},

{t:"reg",k:"concept",q:String.raw`Dropout: what happens in training vs test, and what convention trap should you watch for?`,
a:String.raw`Training: each neuron is kept with probability $p$ (zeroed otherwise). Test: all neurons are used. Trap: the slide's $p$ = <b>keep</b> probability, but the lecture (and PyTorch's <code>nn.Dropout(p)</code>) uses $p$ = <b>drop</b> probability.`,
r:String.raw`Inspired by neurons dying in the brain while it keeps working. The authors couldn't prove why it works; the paper was reportedly rejected once for that.`,s:"Slides 42–43"},

{t:"reg",k:"numeric",q:String.raw`A neuron outputs 5 and has keep probability $p=0.8$. What is its expected output in training? How is train/test consistency kept?`,
a:String.raw`Expected training output $=0.8\cdot5=4$. At test all units are on (output 5), so either scale weights by $p$ at test (classic, gives 4), or use <b>inverted dropout</b>: divide by $p$ during training (kept output $5/0.8=6.25$, expectation 5) and do nothing at test.`,
r:String.raw`Without rescaling, the next layer would see systematically larger inputs at test time. <i>Scaling detail is added context (Srivastava et al. 2014, in references).</i>`,s:"Slide 43 · Added context"},

{t:"reg",k:"proof",q:String.raw`Theory: why can dropout be seen as training an ensemble? How many sub-networks for $n$ droppable units?`,
a:String.raw`Each dropout mask selects a thinned sub-network that shares weights with all others. With $n$ units there are $2^n$ possible sub-networks. Training samples one per step; using all units at test (with scaling) approximates averaging the ensemble's predictions.`,
r:String.raw`Ensembles reduce variance, which explains the regularization effect. It also prevents co-adaptation: no unit can rely on a specific partner being present. <i>Added context (Srivastava et al. 2014).</i>`,s:"Slides 42–43 · Added context"},

{t:"reg",k:"numeric",q:String.raw`4-fold cross-validation gives validation scores 0.82, 0.85, 0.80, 0.83. Report the estimate. How many models are trained, and how often is each sample validated?`,
a:String.raw`Mean $=(0.82+0.85+0.80+0.83)/4=3.30/4=0.825$. Four models; each sample is in the validation fold exactly <b>once</b>.`,
r:String.raw`Use when data is small (100s to a few 1000s). The cost is $k\times$ training; the benefit is a lower-variance estimate than one fixed split.`,s:"Slide 44"},

{t:"reg",k:"concept",q:String.raw`How is a latent bottleneck (encoder → small latent → decoder) a regularizer?`,
a:String.raw`The small latent space squeezes information, so only what the downstream task needs survives. This is implicit regularization through architecture ("inverted clock" shape).`,
r:String.raw`It limits capacity at one point in the network, much like the capacity argument on slide 15.`,s:"Slide 4 · Lecture"},

{t:"reg",k:"proof",q:String.raw`Out of the box: argue why MixUp mixes the <b>label</b> instead of keeping the label of the dominant image.`,
a:String.raw`If a 70/30 bird-cat blend were labeled 100% bird, the model would be trained to be overconfident on inputs that truly contain cat evidence. Mixing labels tells it to respond <i>linearly</i> between training examples: halfway inputs get halfway predictions. That smooths the decision function between classes.`,
r:String.raw`Smoother functions between data points generalize better and are better calibrated. This is the "vicinal risk" view of MixUp. <i>Added reasoning.</i>`,s:"Slide 28 · Added reasoning"},

{t:"reg",k:"proof",q:String.raw`Theory: CutMix's label weight equals the patch area. Argue why area (not, say, 50/50) is the right weight.`,
a:String.raw`The classifier sees evidence roughly in proportion to how many pixels show each object. A tiny pasted cat patch carries little cat evidence, so a 50/50 label would contradict the image. Matching $\lambda$ to the area keeps the target consistent with the visible evidence, the same principle MixUp uses with opacity.`,
r:String.raw`Edge case: if the pasted patch happens to cover background only, the area label overstates that class. That is a known limitation. <i>Added reasoning.</i>`,s:"Slide 29 · Added reasoning"},

{t:"reg",k:"concept",q:String.raw`Name two notation traps in the Regularization deck.`,
a:String.raw`(1) $\lambda$ is the noise amplitude on slide 16 but the weight-penalty scale on slides 5 and 14 (and the MixUp/CutMix ratio on 28–29). (2) Dropout $p$ is keep-probability on slide 43 but drop-probability in the lecture.`,
r:String.raw`Also, slide 2's set is indexed by $m$ but written with $n$, and slide 5's summation index is inconsistent.`,s:"Slides 2, 5, 16, 43"},

{t:"reg",k:"proof",q:String.raw`Out of the box: does a bigger model always overfit more? Use the lecture's statements to argue.`,
a:String.raw`No. The lecture says a very large model is fine if test performance is good. Overfitting depends on capacity <i>relative</i> to the data and the regularization used: with augmentation, weight penalty, dropout and early stopping, a large model can generalize. The 6th-degree polynomial overfits a small noisy 2nd-degree dataset with no regularization.`,
r:String.raw`With a penalty $\lambda\theta^\top\theta$, the 6th-degree model's higher coefficients would be shrunk toward 0, pulling it back toward the parabola.`,s:"Slides 3, 15–19"}
]);
