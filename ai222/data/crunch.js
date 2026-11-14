// 1-hour Crunch plan: must-know sheets + highest-yield drill cards per block.
// Block "end" = cumulative minute mark for pacing.
window.CRUNCH = [
{id:"opt", title:"Optimization", end:13, mins:13,
drill:["opt-6","opt-12","opt-17","opt-18","opt-19","opt-23","opt-25","opt-26","opt-31","opt-32"],
sheet:String.raw`
<h3>1 · The setup</h3>
<ul>
<li>$L(\theta)$ is a <b>proxy</b> for the real target $P$. ERM: $L(\theta)=\frac1m\sum_{i}L(f(x^{(i)};\theta),y^{(i)})$.</li>
<li>Minibatch gradient is <b>unbiased</b>, variance $\sigma^2/m$, so 4× batch only halves noise. Batches 32 to 256 (powers of 2). Small batches regularize but need a smaller LR.</li>
</ul>
<h3>2 · Why it's hard</h3>
<table><tr><th>Challenge</th><th>Key fact</th><th>Fix</th></tr>
<tr><td>Saddle point</td><td>Hessian has ± eigenvalues; dominate in high dims</td><td>Contested: slide says SGD escapes; Ruder says SGD/Momentum/NAG struggle, adaptive escape</td></tr>
<tr><td>Cliff</td><td>Huge step overshoots (exploding grad), RNNs</td><td>Gradient clipping</td></tr>
<tr><td>Long-term dep.</td><td>$W^t=V\mathrm{diag}(\lambda)^tV^{-1}$: $|\lambda|\lt1$ vanish, $\gt1$ explode</td><td>(LSTM)</td></tr>
<tr><td>Inexact grads</td><td>Noisy minibatches</td><td>Stop at near-zero grad with good $P$</td></tr>
<tr><td>Wrong side of mountain</td><td>Bad start</td><td>Initialization</td></tr></table>
<h3>3 · Initialization</h3>
<ul>
<li>Random to <b>break symmetry</b>. Simplest $\mathcal N(0,0.01^2)$.</li>
<li><b>Glorot</b> ($m$ in, $n$ out): $\mathcal U\big(\pm\sqrt{6/(m+n)}\big)$, normal std $\sqrt{2/(m+n)}$.</li>
<li><b>He</b> (fan-in only, ReLU): $\mathcal U\big(\pm\sqrt{6/m}\big)$, std $\sqrt{2/m}$.</li>
<li>Biases: <b>0</b> linear · <b>0.1</b> ReLU · <b>1</b> LSTM forget gate.</li>
</ul>
<h3>4 · Optimizers (memorize these)</h3>
<ul>
<li><b>SGD</b>: $\theta\leftarrow\theta-\varepsilon_k g$. Decay LR; convergence needs $\sum\varepsilon_k=\infty,\ \sum\varepsilon_k^2\lt\infty$.</li>
<li><b>Momentum</b>: $v\leftarrow\alpha v-\varepsilon g,\ \theta\leftarrow\theta+v$. $\alpha\in\{0.5,0.9,0.99\}$. Terminal speed $\varepsilon\|g\|/(1-\alpha)$ → 10× at 0.9.</li>
<li><b>Nesterov</b>: same, but $g$ computed at the look-ahead $\theta+\alpha v$.</li>
<li><b>AdaGrad</b>: $r\leftarrow r+g\odot g$, $\Delta\theta=-\frac{\varepsilon}{\delta+\sqrt r}g$, $\delta=10^{-7}$. Sum never forgets → LR → 0.</li>
<li><b>RMSProp</b>: $r\leftarrow\rho r+(1-\rho)g\odot g$, $\rho=0.9$, $\delta=10^{-6}$. EMA fixes AdaGrad.</li>
<li><b>Adam</b>: $s\leftarrow\rho_1s+(1-\rho_1)g$, $r\leftarrow\rho_2r+(1-\rho_2)g^2$, $\hat s=\frac{s}{1-\rho_1^t}$, $\hat r=\frac{r}{1-\rho_2^t}$, $\Delta\theta=-\varepsilon\frac{\hat s}{\delta+\sqrt{\hat r}}$. Defaults $\varepsilon=0.001,\rho_1=0.9,\rho_2=0.999,\delta=10^{-8}$. First step $\approx-\varepsilon\,\mathrm{sign}(g)$.</li>
<li>Why bias-correct: $\mathbb E[s_t]=(1-\rho_1^t)\mathbb E[g]$ (zero init).</li>
<li><b>Best?</b> No winner. SGD+momentum and Adam are the defaults (Schmidt 2020).</li>
</ul>
<h3>5 · LR schedules (initial 0.1, 150 epochs)</h3>
<ul>
<li>Cosine: $lr_f+\frac12(lr_0-lr_f)(1+\cos\frac{\pi e}{E})$. Midpoint = average of the two.</li>
<li>Multistep: ×factor for <i>each</i> milestone passed. Linear: $lr_0-\frac{lr_0}{E}e$ (hits 0).</li>
<li>Restarts: use $e \bmod P$. Warmup: $lr_0\cdot e/W$, then cosine over $E-W$.</li>
</ul>
<p class="trap"><b>Traps:</b> Ruder's momentum uses $\theta\leftarrow\theta-v$ (same algorithm, flipped sign). RMSProp $\delta$: slide $\delta+\sqrt r$, book $\sqrt{\delta+r}$.</p>`},

{id:"reg", title:"Regularization", end:25, mins:12,
drill:["reg-4","reg-8","reg-9","reg-13","reg-14","reg-18","reg-20","reg-22","reg-30","reg-33"],
sheet:String.raw`
<h3>1 · Goal</h3>
<ul>
<li><b>Generalization.</b> Report <b>test/validation</b> on a defensible split, never training.</li>
<li>Overfitting = <b>memorization</b>. A big model is fine if test performance is good.</li>
</ul>
<h3>2 · Toolbox by where it acts</h3>
<p>Loss: weight penalty · Data: augmentation, input noise, ADR · Weights: noise · Architecture: bottleneck, dropout, multitask, parameter sharing · Schedule: early stopping · Evaluation: cross-validation. Combine, but not too many.</p>
<h3>3 · Weight penalty and its derivation (top priority)</h3>
<ul>
<li>$R_{emp}=\frac1N\sum(y_n-f(x_n,\theta))^2+\lambda\theta^\top\theta$. Weights only. Equal risk → pick smallest $\theta^\top\theta$ (robust to input noise).</li>
<li><b>MLE</b>: $\mathcal L=-\log p(\mathcal D|\theta)$. Log is monotonic → same optimum.</li>
<li><b>IID</b>: independent → product; identical → shared $\theta$. $-\log$ turns the product into a sum.</li>
<li><b>Gaussian</b> $\mathcal N(x^\top\theta,\sigma^2)$ → $\sum\frac{(\hat y-x^\top\theta)^2}{2\sigma^2}$ + const = <b>squared loss</b>.</li>
<li><b>MAP</b>: $p(\theta|\mathcal D)\propto p(\mathcal D|\theta)p(\theta)$; $p(\mathcal D)$ drops (no $\theta$). $\arg\min[-\log p(\mathcal D|\theta)-\log p(\theta)]$.</li>
<li>Prior $\mathcal N(0,\tau^2I)$ → $\frac{1}{2\tau^2}\theta^\top\theta$ = <b>L2</b>, with $\lambda=\sigma^2/\tau^2$. Laplace prior → <b>L1</b> (sparse).</li>
<li>Ridge: $w^*=(X^\top X+\lambda I)^{-1}X^\top y$. Input noise $\mathcal N(0,\sigma^2)$ ≡ L2 with $\lambda=\sigma^2$.</li>
</ul>
<h3>4 · Capacity</h3>
<table><tr><th>Capacity</th><th>Train</th><th>Test</th><th>Polynomial ($y=x^2+x+1+$noise)</th></tr>
<tr><td>Low → underfit</td><td>high</td><td>high</td><td>1st degree</td></tr>
<tr><td>High → overfit</td><td>low</td><td>high</td><td>6th degree (diverges past data)</td></tr>
<tr><td>Right → optimal</td><td>low</td><td>low</td><td>2nd degree</td></tr></table>
<h3>5 · Augmentation (label must not change: b → q, d)</h3>
<ul>
<li><b>CutOut</b>: black out a region, same label.</li>
<li><b>MixUp</b>: $\tilde x=\lambda x_1+(1-\lambda)x_2$, $\tilde y=\lambda y_1+(1-\lambda)y_2$ (vector).</li>
<li><b>CutMix</b>: $\tilde x=M\odot x_1+(1-M)\odot x_2$, label weight = <b>area</b>.</li>
<li><b>AgMax</b>: two augmented views, shared encoder, latents $z_1,z_2$ must <b>agree</b> (max mutual info). MixUp 76.5 → 77.6.</li>
<li><b>RandAugment</b>: random $N$ ops at magnitude $M$. <b>STRAug</b>: 36 ops for scene text; ViTSTR-B 74.7 → 81.3.</li>
</ul>
<h3>6 · The rest</h3>
<ul>
<li><b>Noise</b>: input or weights, <b>training only</b>, every sample. Weight noise → <b>flat region</b>. ADR: randomize simulation (GOO dataset).</li>
<li><b>Multitask</b>: shared body, many heads (Mask R-CNN). <b>Parameter sharing</b>: Siamese A=B, C → disparity.</li>
<li><b>Early stopping</b>: stop when val error rises, keep best weights; 10 epochs = patience example.</li>
<li><b>Dropout</b>: train keeps each unit w.p. $p$; test uses all. Inverted dropout divides by $p$ in training.</li>
<li><b>Cross-validation</b>: small data; $k$ folds, average.</li>
</ul>
<p class="trap"><b>Traps:</b> $\lambda$ = penalty scale (slide 5), noise amplitude (slide 16), mix ratio (MixUp). Dropout $p$ = keep on slide, drop in lecture/PyTorch.</p>`},

{id:"det", title:"Object Detection", end:40, mins:15,
drill:["det-8","det-9","det-10","det-11","det-20","det-25","det-30","det-32","det-35","det-39","det-40","det-46"],
sheet:String.raw`
<h3>1 · Anchors (SSD)</h3>
<ul>
<li>Detection = <b>class + box</b>; origin top-left. Untrained objects → <b>background class 0</b>.</li>
<li>Per anchor: classify + regress <b>offsets</b> (low variance; conv-friendly).</li>
<li>Grids 2×1…40×30 → <b>1,608</b> cells; ×5 ratios = <b>8,040</b>; +1 square = <b>9,648</b>. 40×30 on 640×480 → 16 px stride.</li>
<li>Ratio rule $(ws\sqrt a,\ hs/\sqrt a)$ keeps area. Extra square uses geometric mean $\sqrt{s_js_{j+1}}$.</li>
<li>Assignment: rule 1 = best-IoU anchor per GT (every object gets one); rule 2 = any IoU &gt; threshold.</li>
</ul>
<h3>2 · Loss and targets</h3>
<ul>
<li>$\mathcal L=\frac1N(\mathcal L_{cls}+\alpha[u\ge1]\mathcal L_{off})$, $N$ = matched boxes.</li>
<li>Offsets: $\big(\frac{c_{bx}-c_{ax}}{w_a},\frac{c_{by}-c_{ay}}{h_a},\log\frac{w_b}{w_a},\log\frac{h_b}{h_a}\big)\div(0.1,0.1,0.2,0.2)$.</li>
<li>Smooth L1: $\frac{(\sigma u)^2}{2}$ if $|u|\lt1/\sigma^2$, else $|u|-\frac{1}{2\sigma^2}$.</li>
<li>IoU = intersection / union. IoU loss $-\ln\text{IoU}$ is scale-invariant.</li>
</ul>
<h3>3 · Post-processing and metrics</h3>
<ul>
<li><b>NMS</b>: keep top box, delete IoU ≥ $N_t$. <b>Soft NMS</b>: $p_i\leftarrow p_ie^{-IoU^2/\sigma}$.</li>
<li>mIoU ignores false positives. $P=\frac{TP}{TP+FP}$ ("bad guys out"), $R=\frac{TP}{TP+FN}$ ("good guys in").</li>
<li>VOC AP: 11-point, IoU 0.5. COCO AP: 80 classes × IoU 0.5:0.05:0.95.</li>
</ul>
<h3>4 · The lineage</h3>
<table>
<tr><th>Model</th><th>Key idea</th><th>Numbers</th></tr>
<tr><td>R-CNN</td><td>CNN per selective-search proposal + SVM</td><td>47 s/img</td></tr>
<tr><td>Fast R-CNN</td><td>1 CNN pass, RoI pooling (7×7), multi-task loss, K×4 offsets</td><td>FG IoU ≥ 0.5, BG [0.1, 0.5)</td></tr>
<tr><td>Faster R-CNN</td><td>RPN: 3×3 window, $k=9$ anchors, 2k scores + 4k offsets; 4-step training</td><td>+ &gt; 0.7, − &lt; 0.3; λ = 10; 256 samples</td></tr>
<tr><td>FPN</td><td>Bottom-up + top-down + lateral 1×1 (256 ch)</td><td>$k=4+\log_2\frac{\sqrt{wh}}{224}$; 33.9 AP</td></tr>
<tr><td>RetinaNet</td><td>Focal loss $-\alpha_t(1-p_t)^\gamma\log p_t$</td><td>γ = 2, α = 0.25 (foreground); bias $-\log 99=-4.595$; 39.1 AP</td></tr>
<tr><td>FSAF</td><td>Anchor-free branch; train on the min-loss level</td><td>effective 0.2, ignore 0.5; IoU loss</td></tr>
<tr><td>FCOS</td><td>Anchor-free; $(l,t,r,b)$ per point; smallest box wins</td><td>neg if max &gt; $m_i$ or max &lt; $m_{i-1}$; centerness $\sqrt{\frac{\min(l,r)}{\max(l,r)}\frac{\min(t,b)}{\max(t,b)}}$</td></tr>
</table>
<ul>
<li>Imbalance: 1-stage sees ~1:1000 easy background; focal factor $(1-p_t)^\gamma$ ≈ 0 for easy examples (100× smaller at $p_t$ = 0.9).</li>
<li>Selective Search: hierarchical grouping by <b>colour, texture, size, fill</b>.</li>
</ul>
<p class="trap"><b>Traps:</b> slide's focal α convention is flipped vs the paper. FCOS slide uses min on one side; the paper uses max on both. Multi-scale test AP ≠ single-scale AP.</p>`},

{id:"seg", title:"Segmentation", end:53, mins:13,
drill:["seg-1","seg-4","seg-8","seg-12","seg-13","seg-14","seg-18","seg-23","seg-29","seg-33","seg-40"],
sheet:String.raw`
<h3>1 · Tasks</h3>
<ul>
<li><b>Semantic</b>: class per pixel. <b>Instance</b>: mask + class per object. <b>Panoptic</b>: both (class per pixel + instance id).</li>
<li><b>Stuff</b> (sky, road) suits semantic; <b>things</b> (people, cars) suit instance.</li>
<li>Pixel accuracy is inflated by background; prefer <b>mIoU</b> (all-background at 85% → PA 0.85, mIoU ≈ 0.21).</li>
</ul>
<h3>2 · Semantic: FCN and PSPNet</h3>
<ul>
<li><b>FCN</b>: all-conv (any size), per-pixel CE. 32s → 16s (+pool4) → 8s (+pool3): skips restore <i>where</i>. 62.2 IoU.</li>
<li><b>PSPNet</b> fixes context errors: relationship (boat → car), confusion (building/skyscraper), inconspicuous (pillow). Pool bins <b>1, 2, 3, 6</b> → conv → upsample → <b>concat</b> with original. 85.4 mIoU.</li>
</ul>
<h3>3 · Instance: two-stage</h3>
<ul>
<li><b>Mask R-CNN</b>: RPN → <b>RoIAlign</b> (no quantization; bilinear) → box head + FCN mask head ($K\times m\times m$, per-class <b>sigmoid</b>). $\mathcal L=\mathcal L_{cls}+\mathcal L_{box}+\mathcal L_{mask}$; mask loss on the <b>GT class only</b>. Inference: top 100, mask of the predicted class, resize, threshold 0.5.</li>
<li>Sigmoid decouples mask from class (softmax would make classes compete).</li>
<li><b>MS R-CNN</b>: class score ≠ mask quality. MaskIoU head (L2) → $s_{mask}=s_{cls}\cdot s_{iou}$. Helps AP@0.75, not @0.5.</li>
<li><b>Cascade R-CNN</b>: $u$ = 0.5 is noisy, 0.7 starves positives. Stages 0.5 → 0.6 → 0.7, each trained on the previous stage's refined boxes (resampling).</li>
<li><b>HTC</b>: interleaved box/mask, mask information flow, semantic branch. Loss weights $\alpha=[1,0.5,0.25]$.</li>
<li><b>Rotated Mask R-CNN</b>: dense, tilted objects (PPC 13.2 → 19.3); slightly worse on COCO.</li>
</ul>
<h3>4 · Instance: one-stage (YOLACT)</h3>
<ul>
<li>ProtoNet: $k$ prototypes $P$ · head: $c$ + 4 + $k$ per anchor, coefficients via <b>tanh</b> (can subtract).</li>
<li>$M=\sigma(PC^\top)$: one matrix multiply for all masks → <b>crop by box</b> → threshold.</li>
<li>$\mathcal L_{mask}$ ÷ GT area; OHEM 1:3. 29.8 AP at 33 fps.</li>
<li><b>Fast NMS</b>: upper-triangle IoU, column max, threshold → removes a <b>superset</b> of NMS.</li>
</ul>
<h3>5 · SAM 2</h3>
<ul>
<li>Promptable visual segmentation (points, boxes, masks) → mask or <b>masklet</b> (video).</li>
<li>Hiera encoder → memory attention → mask decoder → memory encoder → memory bank (<b>two FIFOs</b>: $N$ recent + $M$ prompted frames).</li>
<li>One point is ambiguous → <b>multimask</b> output + scores. Labels 1 = foreground, 0 = background. <code>set_image</code> once, prompt many times.</li>
<li>$\mathcal J$ = region IoU, $\mathcal F$ = boundary F. 130 FPS vs SAM 21.7; data engine 8.4× faster.</li>
</ul>`},

{id:"mix", title:"Mixed final round", end:60, mins:7,
drill:["opt-20","opt-29","reg-11","reg-40","det-41","det-49","seg-15","seg-31"],
sheet:String.raw`
<p>Last 7 minutes: 8 tricky cross-topic cards (sign conventions, notation traps, "why" questions). Answer each out loud before revealing.</p>
<p><b>One-line recap to say to yourself:</b> Adam = momentum + RMSProp + bias correction · L2 = Gaussian prior (MAP) · focal loss down-weights easy negatives · Mask R-CNN = Faster R-CNN + RoIAlign + per-class sigmoid masks · SAM 2 = prompts + memory bank.</p>`}
];
