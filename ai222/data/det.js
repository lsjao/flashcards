// Object Detection cards
window.CARDS = (window.CARDS || []).concat([
{t:"det",k:"concept",q:String.raw`Object detection = ? + ?. How is a box represented and where is the origin?`,
a:String.raw`<b>Identification</b> (class id) + <b>localization</b> (bounding box $(x_{min},y_{min}),(x_{max},y_{max})$). Origin at the image's <b>upper-left</b> corner, with $x$ right and $y$ down.`,
r:String.raw`A detector must also find multiple objects, ignore untrained objects, and be robust to illumination, distortion and orientation.`,s:"Slides 5–8"},

{t:"det",k:"concept",q:String.raw`What is the key idea of single-shot detection with anchor boxes?`,
a:String.raw`Divide the image into anchor regions. For each anchor: (1) classify it (background or object class), (2) if not background, predict <b>offsets</b> of the object's box relative to the anchor.`,
r:String.raw`Offsets relative to a nearby anchor are low-variance (small, similar numbers), so they are much easier to regress than raw coordinates.`,s:"Slide 9"},

{t:"det",k:"concept",q:String.raw`How does a detector "ignore" objects it was never trained on?`,
a:String.raw`Through the <b>background class</b> (class id 0). Every anchor not assigned to a labeled object is trained as background, so untrained objects get learned as background.`,
r:String.raw`It's a learned behavior, not a hard-coded rule. A soda detector learns that juice cans and bottles look like "background."`,s:"Slides 7, 20–24"},

{t:"det",k:"numeric",q:String.raw`Verify the SSD anchor totals 1,608 → 8,040 → 9,648 for grids 2×1, 3×2, 5×4, 10×8, 20×15, 40×30.`,
a:String.raw`Cells: $2+6+20+80+300+1200=1{,}608$.<br>× 5 aspect ratios: $1{,}608\times5=8{,}040$.<br>+ 1 extra square anchor per cell (6 per cell): $1{,}608\times6=9{,}648$.`,
r:String.raw`The grid choice changes <b>scale</b>; aspect ratios change <b>shape</b>. Real SSD uses linearly spaced scales in [0.2, 0.9] instead of these grid-derived scales.`,s:"Slides 14–17"},

{t:"det",k:"numeric",q:String.raw`A 40×30 grid on a 640×480 image: what size is each anchor cell, and what does it correspond to in the backbone?`,
a:String.raw`$640/40\times480/30=16\times16$ pixels, a <b>16× downsampling</b>. One feature-map unit after the backbone's conv and pooling corresponds to one cell.`,
r:String.raw`The slide calls this the "receptive field." Strictly, 16 is the <i>stride</i> (cell spacing); a deep unit's true receptive field is usually larger.`,s:"Slide 15"},

{t:"det",k:"proof",q:String.raw`Prove that the aspect-ratio rule $(w_i,h_i)=(w\,s_{xj}\sqrt{a_i},\;h\,s_{yj}/\sqrt{a_i})$ keeps the anchor area fixed and gives aspect ratio $a_i$.`,
a:String.raw`Area: $w_ih_i=w s_{xj}\sqrt{a_i}\cdot h s_{yj}/\sqrt{a_i}=wh\,s_{xj}s_{yj}$, independent of $a_i$.<br>Ratio: $\frac{w_i}{h_i}=\frac{w s_{xj}}{h s_{yj}}\cdot a_i$, i.e. the base cell's ratio times $a_i$.`,
r:String.raw`If you scaled only width by $a_i$, a ratio of 3 would triple the area and mix up "scale" with "shape."`,s:"Slide 16"},

{t:"det",k:"proof",q:String.raw`Why does SSD's extra square anchor use the geometric mean $\sqrt{s_js_{j+1}}$ rather than the arithmetic mean?`,
a:String.raw`Scales are spaced multiplicatively, so the natural "midpoint" between two scales is in log space: $\log\sqrt{s_js_{j+1}}=\frac12(\log s_j+\log s_{j+1})$. The geometric mean is exactly halfway in log-scale.`,
r:String.raw`Example: between 0.2 and 0.8 the geometric mean is 0.4 (2× each way), while the arithmetic mean 0.5 is 2.5× above one and 1.6× below the other. <i>Added reasoning.</i>`,s:"Slide 17 · Added reasoning"},

{t:"det",k:"formula",q:String.raw`Define IoU and compute it for boxes $A=(0,0,4,4)$ and $B=(2,2,6,6)$ (xmin, ymin, xmax, ymax).`,
a:String.raw`$IoU=\dfrac{|A\cap B|}{|A\cup B|}$ (Jaccard index).<br>Intersection: $[2,4]\times[2,4]$ → area 4. Union: $16+16-4=28$. $IoU=4/28=1/7\approx0.143$.`,
r:String.raw`Dividing by the union stops a huge box from scoring well just by covering the object. Selective Search's "Overlap" uses the same formula.`,s:"Slide 22 · Uijlings 2013"},

{t:"det",k:"concept",q:String.raw`State the two-rule ground-truth assignment for anchors. Why is rule 1 still needed when rule 2 exists?`,
a:String.raw`<b>Rule 1:</b> for each ground-truth box, the anchor with maximum IoU is positive. <b>Rule 2 ("second chance"):</b> any anchor with IoU &gt; threshold is also positive. All others are negative (class 0, no offsets).<br>Rule 1 guarantees every object at least one positive, even a small or oddly shaped object whose best IoU is below the threshold.`,
r:String.raw`Rule 2 adds more positive signal; the resulting duplicates are cleaned up by NMS at inference. Note: the slide writes $\max_j$ but means $\arg\max_j$ (the anchor index).`,s:"Slides 23–24"},

{t:"det",k:"formula",q:String.raw`Write the SSD loss and explain the Iverson bracket and the normalizer.`,
a:String.raw`$$\mathcal L=\frac1N\big(\mathcal L_{cls}+\alpha[u\ge1]\mathcal L_{off}\big)$$ $\mathcal L_{cls}$: categorical CE; $\mathcal L_{off}$: L1/L2/smooth L1. $[u\ge1]$ = 1 only for positive (non-background) anchors. $N$ = number of matched boxes. $\alpha=1$.`,
r:String.raw`Normalizing by matched boxes keeps the loss per object constant regardless of how many anchors exist (9,648 anchors, 2 objects → divide by 2, not 9,648). Edge case: $N=0$ needs a guard such as $\max(N,1)$.`,s:"Slide 28"},

{t:"det",k:"formula",q:String.raw`Write the normalized offset parameterization (center + log size) and the standardized version.`,
a:String.raw`$$y_{gt}=\Big(\frac{c_{bx}-c_{ax}}{w_a},\ \frac{c_{by}-c_{ay}}{h_a},\ \log\frac{w_b}{w_a},\ \log\frac{h_b}{h_a}\Big)$$ Standardized: divide by $(\sigma_x,\sigma_y,\sigma_w,\sigma_h)=(0.1,0.1,0.2,0.2)$.`,
r:String.raw`Center offsets are scale-normalized by anchor size; log makes size changes additive and guarantees positive width after $e^{\Delta}$. Dividing by σ brings targets to order 1, balancing them with the class loss.`,s:"Slides 29–30"},

{t:"det",k:"numeric",q:String.raw`Anchor $(100,100,200,300)$, box $(110,120,230,300)$. Compute the normalized offsets and the σ-standardized ones.`,
a:String.raw`Anchor: $c=(150,200)$, $w_a=100$, $h_a=200$. Box: $c=(170,210)$, $w_b=120$, $h_b=180$.<br>$y_{gt}=(20/100,\ 10/200,\ \ln1.2,\ \ln0.9)=(0.200,\,0.050,\,0.182,\,-0.105)$.<br>÷(0.1, 0.1, 0.2, 0.2): $(2.00,\,0.50,\,0.91,\,-0.53)$.`,
r:String.raw`All four end up order 1, which is the purpose of standardization.`,s:"Slides 29–30 · Worked check"},

{t:"det",k:"proof",q:String.raw`Derive the decoding equations that turn predicted standardized offsets $(\Delta_x,\Delta_y,\Delta_w,\Delta_h)$ back into a box.`,
a:String.raw`Undo σ: $d_x=\sigma_x\Delta_x$, etc. Then invert each term:<br>$c_{bx}=c_{ax}+w_a d_x$, $c_{by}=c_{ay}+h_a d_y$, $w_b=w_a e^{d_w}$, $h_b=h_a e^{d_h}$.<br>Corners: $x_{min}=c_{bx}-w_b/2$, $x_{max}=c_{bx}+w_b/2$ (same for $y$).`,
r:String.raw`$e^{d_w}\gt0$ always, so predicted widths can never be negative, a built-in advantage of the log parameterization.`,s:"Slides 29–30 · Added derivation"},

{t:"det",k:"proof",q:String.raw`Why regress offsets relative to an anchor rather than absolute coordinates, given the predictor is a convolution?`,
a:String.raw`A convolution applies the same weights everywhere (translation equivariance): the same local pattern produces the same output wherever it is. An absolute target needs the same pattern to output different numbers at different positions, which the filter can't do. A relative offset target is the same wherever the object is.`,
r:String.raw`Slide 9 gives the statistical reason (low variance). This is the structural one. <i>Added reasoning.</i>`,s:"Slide 9 · Added reasoning"},

{t:"det",k:"formula",q:String.raw`Write smooth L1 as on the slide.`,
a:String.raw`$$L1_{smooth}(u)=\begin{cases}\dfrac{(\sigma u)^2}{2}&|u|\lt\dfrac1{\sigma^2}\\[4pt]|u|-\dfrac1{2\sigma^2}&\text{otherwise}\end{cases}$$`,
r:String.raw`Quadratic near zero (smooth, shrinking gradients for small errors); linear far away (bounded gradient, less sensitive to outliers than L2). Larger σ makes it closer to pure L1.`,s:"Slide 31"},

{t:"det",k:"proof",q:String.raw`Show that smooth L1 is continuous and has a continuous derivative at $|u|=1/\sigma^2$. Then evaluate it at $u=2$ for $\sigma=0.5,1,10$.`,
a:String.raw`At $u=1/\sigma^2$: quadratic $=\frac{(\sigma/\sigma^2)^2}{2}=\frac1{2\sigma^2}$; linear $=\frac1{\sigma^2}-\frac1{2\sigma^2}=\frac1{2\sigma^2}$ ✓. Slopes: quadratic $\sigma^2u=1$, linear 1 ✓.<br>$u=2$: σ = 0.5 (threshold 4, quadratic) → $(1)^2/2=0.5$; σ = 1 → $2-0.5=1.5$; σ = 10 → $2-0.005=1.995$.`,
r:String.raw`These match the three curves on slide 31. Continuity of the gradient is what makes it "smooth."`,s:"Slide 31 · Worked check"},

{t:"det",k:"concept",q:String.raw`Slide 31 says smooth L1 "yields better results," but slide 46 shows it lowering mIoU, precision and recall. How can both be true?`,
a:String.raw`The claim comes from the papers' large-scale results; slide 46 is the course's own 50-image test. Differences of 0.01–0.03 on 50 images may be noise. Also, with σ-standardized targets of order 1, smooth L1 (σ = 1) sits partly in its quadratic zone, giving weaker gradients for small errors than L1.`,
r:String.raw`Open question per the notes: answering it needs the training code, the σ used, and repeated runs with variance estimates.`,s:"Slides 31, 46"},

{t:"det",k:"concept",q:String.raw`In SSD, why does each successive head see a coarser grid, and which objects does each handle?`,
a:String.raw`Each extra Conv2D(stride 2) halves the feature map, so deeper heads have fewer, larger cells. Fine early maps (e.g. $w/16$) handle <b>small</b> objects; coarse deep maps handle <b>large</b> objects with more global context.`,
r:String.raw`All heads' predictions are combined, then NMS. FPN later fixes the weakness that SSD's fine maps are semantically weak.`,s:"Slides 32, 34"},

{t:"det",k:"concept",q:String.raw`How does YOLO's per-cell confidence differ from SSD's background class?`,
a:String.raw`YOLO predicts, per grid cell, box center/size, a separate <b>objectness confidence</b> and class probabilities. SSD has no separate confidence: "no object" is just class 0 in the softmax.`,
r:String.raw`YOLO's confidence is a quality/objectness score, similar in spirit to FCOS centerness. YOLOv3+ predicts at grids like 13, 26, 52; CSPDarkNet splits channels into a residual path plus an identity shortcut to cut FLOPs.`,s:"Slides 35–36"},

{t:"det",k:"concept",q:String.raw`What is NMS, and what does Soft NMS change?`,
a:String.raw`<b>NMS:</b> repeatedly keep the highest-scoring box and delete every remaining box with IoU ≥ $N_t$ to it. <b>Soft NMS:</b> instead of deleting, decay overlapping scores: $p_i\leftarrow p_i\,e^{-IoU(\mathcal M,b_i)^2/\sigma}$.`,
r:String.raw`Soft NMS keeps a genuinely different but overlapping object (the second horse) with a reduced score. A final score threshold is still needed to remove duplicates, since the soft branch never removes anything.`,s:"Slides 38–41"},

{t:"det",k:"numeric",q:String.raw`Boxes A (0.95), B (0.80, IoU with A = 0.6), C (0.70, IoU with A = 0.1). Run greedy NMS ($N_t=0.5$) and Soft NMS ($\sigma=0.5$).`,
a:String.raw`<b>NMS:</b> keep A; B has 0.6 ≥ 0.5 → removed; C stays → output {A, C}.<br><b>Soft NMS:</b> B → $0.80e^{-0.36/0.5}=0.80\times0.487=0.39$; C → $0.70e^{-0.01/0.5}=0.686$. Output all three: A 0.95, C 0.686, B 0.39.`,
r:String.raw`Whether B survives now depends on a later score cutoff.`,s:"Slide 41 · Worked check"},

{t:"det",k:"proof",q:String.raw`Explain why detection needs NMS at all, as a consequence of training design.`,
a:String.raw`The assignment rules deliberately make several anchors positive for one object ("second chance," slide 24). At inference all of those anchors fire, producing overlapping boxes for the same object. NMS is the inference-time counterpart of that training choice.`,
r:String.raw`Anchor-free FCOS has the same issue (every interior point is positive), which is why it also uses NMS, helped by centerness.`,s:"Slides 24, 38–39"},

{t:"det",k:"formula",q:String.raw`Write the mIoU metric. Why is it fine for evaluating proposals but insufficient for a detector?`,
a:String.raw`$$mIoU=\frac{1}{n_{box}}\sum_{i}\max_{j}IoU(b_i,d_j)$$ It takes the best prediction for each ground-truth box, so extra (false-positive) predictions can never lower it. For proposals, extra boxes are fine; for a detector, false positives matter.`,
r:String.raw`Same structure as Selective Search's Average Best Overlap (ABO/MABO). Precision is the metric that exposes false positives.`,s:"Slide 42 · Uijlings 2013"},

{t:"det",k:"numeric",q:String.raw`A detector produces 10 detections: 8 are true positives, 2 false positives; 4 ground-truth objects are missed. Compute precision and recall.`,
a:String.raw`Precision $=\frac{8}{8+2}=0.80$. Recall $=\frac{8}{8+4}=0.667$.`,
r:String.raw`Slide slogans: precision = "bad guys out" (penalizes false positives); recall = "good guys in" (penalizes misses).`,s:"Slide 43 · Worked example"},

{t:"det",k:"numeric",q:String.raw`11-point VOC AP: precision is 1.0 at recalls 0–0.5, 0.5 at recalls 0.6–0.8, and 0 at 0.9 and 1.0. Compute AP. How does COCO's AP differ?`,
a:String.raw`$AP=\frac{1}{11}(6\times1.0+3\times0.5+2\times0)=\frac{7.5}{11}=0.682$.<br>VOC2007: one IoU threshold (0.5), 20 classes. COCO: averaged over 80 categories <b>and</b> 10 IoU thresholds (0.5:0.05:0.95).`,
r:String.raw`COCO's averaging rewards tight localization; at IoU 0.5 alone the metric saturates (Uijlings makes the same argument about recall at 0.5).`,s:"Slide 48 · Worked check"},

{t:"det",k:"concept",q:String.raw`Normalized offsets lowered mIoU (0.64 → 0.53) but raised precision (0.87 → 0.90). How can these move in opposite directions?`,
a:String.raw`mIoU measures tightness of the best box per object and ignores false positives. Precision counts false positives but, above the IoU cutoff, doesn't care how tight a correct box is. A change that removes spurious boxes while making correct boxes slightly looser raises precision and lowers mIoU.`,
r:String.raw`The slides give only the numbers; this interpretation is derived. Lesson: always report more than one metric.`,s:"Slide 46"},

{t:"det",k:"concept",q:String.raw`Why was R-CNN slow, and what three things did Fast R-CNN change?`,
a:String.raw`R-CNN warped ~2k selective-search crops and ran the CNN <b>once per proposal</b>, classified with per-class SVMs (not parallel), and cached features to disk: 47 s/image.<br>Fast R-CNN: (1) one CNN pass per image with RoI pooling on the shared feature map; (2) single-stage training with a multi-task loss updating all layers; (3) no disk caching.`,
r:String.raw`Uijlings reports a similar SVM cost profile for its pre-CNN pipeline: ~40 s/image features and ~8 h per class to train the final SVM.`,s:"Slides 52–54 · Uijlings 2013"},

{t:"det",k:"concept",q:String.raw`How does RoI pooling make a fixed-size output from any RoI, and what is lost for small RoIs?`,
a:String.raw`Divide the projected $h\times w$ RoI into an $H\times W$ grid (e.g. 7×7) of cells of size $(h/H,w/W)$ and max-pool each cell, giving a fixed $7\times7$ map.<br>For a small RoI (e.g. 2×2 feature cells at stride 16 for a 32-px object), each output cell covers less than one feature cell, so values repeat and spatial detail is lost.`,
r:String.raw`FPN mitigates this by routing small RoIs to finer pyramid levels. Mask R-CNN's RoIAlign removes the coarse quantization.`,s:"Slides 55–56"},

{t:"det",k:"concept",q:String.raw`Fast R-CNN: give the label rules for RoIs and the output heads. Why predict $K$ separate 4-d offset vectors?`,
a:String.raw`IoU ≥ 0.5 → foreground ($u\ge1$); $0.1\le IoU\lt0.5$ → background ($u=0$). Heads: a $(K+1)$-way softmax and $K$ class-specific 4-d regressors (log-space, zero mean, unit variance). Loss $\mathcal L_{cls}+\alpha[u\ge1]\mathcal L_{off}$ with $\mathcal L_{cls}=-\log p_u$.<br>Per-class offsets let each class learn its own typical box correction (a person's box differs from a car's); the predicted class selects which one is applied.`,
r:String.raw`Excluding IoU &lt; 0.1 focuses negatives on informative near-misses; Uijlings uses a similar 20–50% overlap band for negatives.`,s:"Slides 57–60"},

{t:"det",k:"concept",q:String.raw`Why did Faster R-CNN introduce the RPN? How does the RPN work?`,
a:String.raw`Selective Search was the bottleneck: ~2 s/image on CPU vs 0.2 s for detection. The RPN slides an $n\times n$ ($n=3$) window over the last feature map → 256/512-d code → $2k$ objectness scores and $4k$ box offsets for $k=9$ anchors (3 scales × 3 ratios) per location.`,
r:String.raw`Faster R-CNN = RPN + Fast R-CNN sharing convolutional features. The RPN acts as the network's "attention."`,s:"Slides 65–69"},

{t:"det",k:"numeric",q:String.raw`RPN loss: $N_{cls}\approx256$, $N_{reg}\approx2{,}400$, $\lambda=10$. Show that $\lambda$ balances the two terms. How many anchors before and after removing cross-boundary ones?`,
a:String.raw`$1/N_{cls}=1/256\approx0.0039$; $\lambda/N_{reg}=10/2400\approx0.0042$ → roughly equal weights ✓.<br>Anchors: $2{,}400\times9=21{,}600$; about 6,000 after ignoring cross-boundary anchors (~70% removed).`,
r:String.raw`$\mathcal L=\frac{1}{N_{cls}}\mathcal L_{cls}+\frac{\lambda}{N_{reg}}[u\ge1]\mathcal L_{off}$ with smooth L1 and the same parameterization as SSD.`,s:"Slides 68, 72, 76 · Worked check"},

{t:"det",k:"concept",q:String.raw`Why does the RPN use two thresholds (positive IoU &gt; 0.7, negative &lt; 0.3) with an ignored band, instead of one cutoff at 0.5?`,
a:String.raw`An anchor at IoU 0.5 covers half an object: calling it "object" teaches firing on partial views, calling it "background" punishes a reasonable proposal. Ignoring 0.3–0.7 removes these contradictory signals: positives are clear objects, negatives clearly empty.`,
r:String.raw`Cost: ignored anchors give no gradient. Training samples 256 anchors at ~1:1 positive:negative because of imbalance.`,s:"Slides 71, 73"},

{t:"det",k:"concept",q:String.raw`Describe the 4-step alternating training of Faster R-CNN and why it's done.`,
a:String.raw`(1) Train RPN end-to-end. (2) Train a separate Fast R-CNN on step-1 proposals. (3) Copy the shared conv layers from step 2, freeze them, fine-tune only RPN-specific layers. (4) Keeping shared layers frozen, fine-tune only Fast R-CNN layers.<br>Why: if the RPN changes while Fast R-CNN trains, the detector's input proposals keep shifting. Alternating gives each a stable partner and ends with one shared backbone.`,
r:String.raw`The result is a "single, unified network" (slide 66).`,s:"Slides 74–75"},

{t:"det",k:"concept",q:String.raw`Why is a "pyramid of anchors" (multiple reference boxes on one feature map) cheaper than an image pyramid or a filter pyramid?`,
a:String.raw`The backbone runs once on one image at one scale, and the conv filters are a single size. Multi-scale handling is pushed into cheap anchor references at the output. An image pyramid reruns the backbone per scale; a filter pyramid needs several filter sizes.`,
r:String.raw`What it gives up: all scales share one feature resolution, which FPN later fixes.`,s:"Slide 70"},

{t:"det",k:"concept",q:String.raw`Describe FPN's bottom-up, top-down and lateral pathways. Why the 1×1 conv on the lateral path?`,
a:String.raw`<b>Bottom-up:</b> ResNet $C_1$–$C_5$ (decreasing resolution, stronger semantics). <b>Top-down:</b> $P_5$ from $C_5$, then repeatedly upsample ×2. <b>Lateral:</b> add a 1×1-conv projection of $C_i$ to the upsampled map, giving $P_4,P_3,P_2$. Predict at every level.<br>The 1×1 conv maps different backbone channel counts to a common depth (256) so element-wise addition works.`,
r:String.raw`A common depth also lets one prediction head be shared across levels (RetinaNet, FCOS). FPN combines strong low-res semantics with high-res localization.`,s:"Slides 80–85"},

{t:"det",k:"concept",q:String.raw`FPN ablation: FPN 33.9 AP, bottom-up only 24.9, top-down without lateral 31.3, only finest level 33.4. What does each tell you?`,
a:String.raw`Bottom-up only (24.9): high-res backbone maps are semantically weak. Without laterals (31.3): upsampled coarse maps have lost precise locations. Finest level only (33.4): most of the gain comes from the semantic + localization fusion, and multi-level prediction adds a bit more.`,
r:String.raw`FPN's biggest gain is on small objects (AP$_s$ 17.8 vs 9.6–15.7 for single-level baselines).`,s:"Slide 89"},

{t:"det",k:"numeric",q:String.raw`FPN routes an RoI to level $k=k_0+\log_2(\sqrt{wh}/224)$, $k_0=4$. Which level for RoIs of 224², 112², 448² and 300²?`,
a:String.raw`224 → $4+0=4$ ($P_4$). 112 → $4-1=3$. 448 → $4+1=5$. 300 → $4+\log_2(1.34)=4.42$ → round to 4 (and clip to available levels).`,
r:String.raw`224 is the ImageNet pre-training size. Near boundaries ($\sqrt{wh}\approx316$, $k\approx4.5$) a few pixels flip the level, which FSAF criticizes.`,s:"Slide 87 · Worked check"},

{t:"det",k:"concept",q:String.raw`What is the class-imbalance problem in one-stage detectors, and how do two-stage detectors reduce it?`,
a:String.raw`One-stage detectors score 10k–100k anchors, nearly all easy background (~1:1000), so the summed loss is dominated by easy negatives. Two-stage detectors filter most background with proposals (Selective Search/RPN) and then sample fixed ratios (1:1 or 1:3).`,
r:String.raw`Slide 91 says two-stage detectors have "no imbalance"; more accurately they <i>reduce</i> it (they still sample and mine hard negatives).`,s:"Slides 91–94"},

{t:"det",k:"formula",q:String.raw`Write CE in terms of $p_t$, the α-balanced CE, and the focal loss. Which values work best? State the paper's α convention.`,
a:String.raw`$p_t=p$ if $y=1$, else $1-p$. $CE=-\log p_t$.<br>Balanced: $-\alpha_t\log p_t$. Focal: $$FL(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t$$ Best: $\gamma=2$, $\alpha=0.25$. In the paper, $\alpha_t=\alpha$ for the foreground class and $1-\alpha$ for background.`,
r:String.raw`The slide's transcription (foreground weight 1, background $1-\alpha$) conflicts with this; the focal-loss paper defines α for class 1. α = 0.25 gives foreground <i>less</i> weight, which works because $(1-p_t)^\gamma$ has already suppressed most background loss.`,s:"Slides 95–98 · Lin et al. 2017 (paper)"},

{t:"det",k:"numeric",q:String.raw`Compare CE and FL ($\gamma=2$, ignore α) for an easy negative at $p_t=0.9$ and a hard example at $p_t=0.1$. With 1,000 easy negatives and 1 hard positive, who dominates?`,
a:String.raw`Easy: CE $=0.105$; FL $=0.01\times0.105=0.00105$ (100× smaller).<br>Hard: CE $=2.303$; FL $=0.81\times2.303=1.865$ (1.23× smaller).<br>Totals: CE 105 vs 2.3 (negatives dominate 45:1); FL 1.05 vs 1.87 (<b>hard example</b> now dominates).`,
r:String.raw`The modulating factor $(1-p_t)^\gamma$ is ≈0 for confident correct predictions and ≈1 for mistakes, shifting the gradient budget to hard examples.`,s:"Slides 96–98 · Worked check"},

{t:"det",k:"numeric",q:String.raw`RetinaNet initializes the final classification bias to $b=-\log\frac{1-\pi}{\pi}$, $\pi=0.01$. Compute $b$ and the initial loss of one background anchor, vs bias 0.`,
a:String.raw`$b=-\ln99=-4.595$; $\sigma(-4.595)=1/(1+99)=0.01$. Background CE $=-\ln0.99=0.010$.<br>With bias 0: $p=0.5$, CE $=\ln2=0.693$. Over 100k anchors: ~1,005 vs ~69,300, a 69× smaller initial loss.`,
r:String.raw`Prevents the huge background loss from destabilizing early training. It also makes background anchors immediately "easy" for focal loss ($(1-0.99)^2=10^{-4}$).`,s:"Slide 100 · Worked check"},

{t:"det",k:"concept",q:String.raw`Describe RetinaNet's architecture and its reported results.`,
a:String.raw`ResNet + FPN ($P_3$–$P_7$), with two small fully convolutional subnets per level: class (4 × 256-ch convs → $KA$ outputs) and box (4 convs → $4A$), trained with focal loss. COCO test-dev 39.1 AP (ResNet-101-FPN).`,
r:String.raw`Careful: 39.1 comes from scale jitter plus 1.5× longer training, while the speed plot's best is 37.8 AP at 198 ms (~5 fps). "39.1 AP at 5 fps" merges two results.`,s:"Slides 99–102"},

{t:"det",k:"concept",q:String.raw`What problem does FSAF address, and how does it pick a pyramid level for each instance?`,
a:String.raw`Anchor-based heuristics pick levels by box size, so a 40×40 and a 50×50 car may share a level that's poor for one. FSAF adds anchor-free branches per level, computes each instance's focal + IoU loss at every level, and trains only on the <b>lowest-loss</b> level: $l^*=\arg\min_l L^I_{FL}(l)+L^I_{IoU}(l)$.`,
r:String.raw`FSAF adds only two conv layers per level and is merged with RetinaNet's anchor-based outputs at inference (NMS 0.5).`,s:"Slides 104–115"},

{t:"det",k:"numeric",q:String.raw`FSAF: box $b=[x,y,w,h]=[320,240,128,96]$. Project to $P_3$ and $P_5$ and give the effective ($\epsilon_e=0.2$) and ignoring ($\epsilon_i=0.5$) box sizes.`,
a:String.raw`$P_3$ (÷8): $b_p=[40,30,16,12]$; effective $3.2\times2.4$ cells; ignoring $8\times6$ cells.<br>$P_5$ (÷32): $b_p=[10,7.5,4,3]$; effective $0.8\times0.6$ cells, less than one cell, so at most one positive location there.`,
r:String.raw`Positive = central 20%; the ring out to 50% is excluded from the loss; the rest is negative. Box targets are distances to the four sides ÷ $S=4$, trained with IoU loss.`,s:"Slides 108–110 · Worked check"},

{t:"det",k:"proof",q:String.raw`Why is the IoU loss $-\ln\frac{\text{Intersection}}{\text{Union}}$ scale-invariant while an $\ell_2$ loss on the four distances is not?`,
a:String.raw`Scale both boxes by $c$: intersection and union both scale by $c^2$, so their ratio (and the loss) is unchanged. The $\ell_2$ loss on distances scales by $c^2$: the same relative error costs far more on a big box. IoU loss also treats the four sides jointly as one box instead of four independent numbers.`,
r:String.raw`From UnitBox (Yu et al. 2016). It's why FSAF and FCOS don't need slide 30's σ normalization.`,s:"Slide 111 · Added proof"},

{t:"det",k:"concept",q:String.raw`FCOS: how are positives defined, how is ambiguity resolved, and how is the box regressed?`,
a:String.raw`Location $(x,y)$ maps to $(\lfloor s/2\rfloor+xs,\ \lfloor s/2\rfloor+ys)$. It's positive if that point is inside a ground-truth box; target $(l^*,t^*,r^*,b^*)$ = distances to the four sides. Ambiguity: objects are first split across FPN levels by size range; if still inside several boxes, the <b>smallest</b> box wins. Outputs pass through exp() to stay positive; IoU loss; focal loss for classes.`,
r:String.raw`No anchors, so no anchor hyperparameters or IoU matching, and many more positives per object (~180k anchors avoided for FPN).`,s:"Slides 120–127"},

{t:"det",k:"numeric",q:String.raw`FCOS at $P_3$ (stride 8): location (10, 5), box $(x_0,y_0,x_1,y_1)=(60,20,200,100)$. Compute the image point, targets and centerness.`,
a:String.raw`Point: $(4+80,\ 4+40)=(84,44)$. $l^*=24$, $t^*=24$, $r^*=116$, $b^*=56$.<br>Centerness $=\sqrt{\frac{24}{116}\cdot\frac{24}{56}}=\sqrt{0.207\times0.429}=\sqrt{0.0887}=0.298$.`,
r:String.raw`Low centerness: the point is near the left/top edge. At inference, class score × centerness down-ranks such boxes before NMS.`,s:"Slides 123–129 · Worked check"},

{t:"det",k:"proof",q:String.raw`Slide 128 says a point is negative if $\max(t^*)\gt m_i$ or $\min(t^*)\lt m_{i-1}$. Show with an example why the <i>min</i> version is suspicious, and give the paper's rule.`,
a:String.raw`Take the box above at $P_4$ (stride 16): location (5, 2) → (88, 40), $t^*=(28,20,112,60)$. $\max=112\le m_4=128$ ✓ but $\min=20\lt m_3=64$ → negative. At every higher level $\min\lt m_{i-1}$ still holds, so this interior point is negative <b>everywhere</b>.<br>FCOS paper: negative if $\max(l^*,t^*,r^*,b^*)\gt m_i$ <b>or</b> $\max(l^*,t^*,r^*,b^*)\lt m_{i-1}$, with $(m_2..m_7)=(0,64,128,256,512,\infty)$.`,
r:String.raw`Bounding the same statistic (max) on both sides gives each level a clean size range. <i>The paper's rule is cited from the FCOS paper; the paper page could not be re-fetched today.</i>`,s:"Slide 128 · Tian et al. 2019"},

{t:"det",k:"concept",q:String.raw`Why does FCOS need a centerness branch while FSAF does not?`,
a:String.raw`FCOS labels <i>every</i> interior point positive, so edge points learn to predict boxes from poor evidence and produce low-quality boxes; centerness down-weights them at inference. FSAF trains positives only in the central 20% and ignores the ring, so edge locations never learn to fire confidently.`,
r:String.raw`Same problem, solved at different stages: FSAF at labeling time, FCOS at scoring time. Centerness is trained with BCE as an extra sigmoid output.`,s:"Slides 108, 129"},

{t:"det",k:"concept",q:String.raw`Selective Search: how does it generate proposals, and what are its four similarity measures?`,
a:String.raw`Start from Felzenszwalb–Huttenlocher initial regions; repeatedly merge the most similar neighboring pair until one region remains; every region ever created gives a box. Similarities: <b>colour</b>, <b>texture</b>, <b>size</b>, <b>fill</b> (the slide says "shape").`,
r:String.raw`Size merges small regions first everywhere (balanced hierarchy); fill prefers regions that fit snugly in their joint bounding box. Fast mode: ~2,134 windows, recall 0.980, 3.79 s, matching Fast R-CNN's "2,000 ROIs."`,s:"Slide 61 · Uijlings 2013"},

{t:"det",k:"proof",q:String.raw`Using Uijlings' upper-bound experiment, argue whether better proposals or a better classifier was the key need around 2013.`,
a:String.raw`Scoring only ground-truth boxes gives an upper bound of 0.592 MAP. Adding the GT boxes to all ~10k selective-search windows gave 0.377, only +0.017 over not adding them. So near-perfect boxes were already among the proposals; accuracy was lost because the <b>classifier</b> was confused by many windows.`,
r:String.raw`That's why CNN classifiers (R-CNN: 35.1% → 62.9% on VOC 2010) gave the big jump, and the RPN later mainly bought <i>speed</i> (66.9 → 69.9 mAP but ~10× faster).`,s:"Slides 52, 65 · Uijlings 2013"},

{t:"det",k:"concept",q:String.raw`FSAF reports 42.8 AP and FCOS 44.7 AP. Why can't you just compare these numbers?`,
a:String.raw`42.8 (ResNet-101) and 44.6 (ResNeXt) for FSAF are <b>multi-scale</b> test results; single-scale are 40.9 and 42.9. 44.7 is "FCOS w/ improvements"; plain FCOS with the same ResNeXt-64x4d backbone is 43.2. Compare same backbone, same test settings: 42.9 vs 43.2.`,
r:String.raw`Headline numbers mix training schedules, multi-scale testing and backbones. Always check the table row.`,s:"Slides 116–117, 130"}
]);
