// Segmentation cards
window.CARDS = (window.CARDS || []).concat([
{t:"seg",k:"concept",q:String.raw`Define semantic, instance and panoptic segmentation, and "stuff" vs "things."`,
a:String.raw`<b>Semantic:</b> class label per pixel (no object identity). <b>Instance:</b> mask + class per countable object. <b>Panoptic:</b> per-pixel class + instance id for things (Kirillov 2019).<br><b>Things:</b> countable objects (people, cars). <b>Stuff:</b> amorphous regions (sky, road, grass).`,
r:String.raw`Counting "instances" of sky is ill-defined, so stuff suits semantic segmentation and things suit instance segmentation. Panoptic covers both.`,s:"Slides 2–3"},

{t:"seg",k:"concept",q:String.raw`How does treating segmentation as per-pixel classification change the network's output compared with image classification?`,
a:String.raw`Instead of one $K$-vector per image, the output is an $H\times W\times K$ map: a class distribution at every pixel, with a one-hot ground truth per pixel (e.g. 0 background, 1 water, 2 soda, 3 juice).`,
r:String.raw`The course example uses 4 classes (3 drink objects + background), matching <code>n_classes=4</code> in <code>build_fcn</code>.`,s:"Slides 7–8, 20"},

{t:"seg",k:"concept",q:String.raw`What made FCN "fully convolutional," and why does that allow any input size and a spatial output?`,
a:String.raw`It replaces fully connected layers with convolutions, so every layer works on a spatial grid. Convolutions slide over any input size and preserve spatial layout, so the deepest layer is a coarse grid of class scores rather than a single vector. Trained end-to-end, pixel-to-pixel, with per-pixel categorical CE.`,
r:String.raw`FC layers need a fixed-size flattened input and destroy spatial structure. Novel in 2015: skips, multi-level upsampling and fusion, fully conv layers, pixel-wise loss.`,s:"Slides 9–11"},

{t:"seg",k:"concept",q:String.raw`Explain FCN-32s, 16s and 8s. Why does 8s give sharper boundaries?`,
a:String.raw`<b>32s:</b> upsample conv7 scores 32× in one step. <b>16s:</b> 2× upsample conv7, add pool4 scores, then 16×. <b>8s:</b> also add pool3 scores, then 8×.<br>Deep layers know <i>what</i> but lost <i>where</i>; shallow layers keep <i>where</i>. Fusing them restores boundary detail, so 8s is sharpest (62.2 IoU, VOC 2012).`,
r:String.raw`No upsampler can recover detail that pooling has already thrown away; the shallow skips supply it.`,s:"Slides 10–11"},

{t:"seg",k:"numeric",q:String.raw`A 512×512 image through FCN: what is the score-map size at stride 32? At what resolutions are pool4/pool3 fused, and what is the final upsampling in FCN-8s?`,
a:String.raw`Stride 32: $16\times16$. pool4 (stride 16): $32\times32$. pool3 (stride 8): $64\times64$. FCN-8s upsamples the fused $64\times64$ map by 8× to $512\times512$.`,
r:String.raw`Element-wise addition needs equal grids, which forces the 2× upsampling before each fusion.`,s:"Slide 10 · Worked example"},

{t:"seg",k:"proof",q:String.raw`Argue why FCN must upsample before adding a skip, and why the skip acts as a "correction."`,
a:String.raw`Fusion is element-wise addition, so both maps must be on the same spatial grid; the coarse map is upsampled to the skip's stride first. The upsampled coarse scores already give a rough class layout; the shallow-layer scores only need to supply the residual boundary detail the coarse map lacks, much like a residual correction.`,
r:String.raw`This also means each skip contributes what it uniquely has (fine location), not a full prediction. <i>Derived reasoning, consistent with slide 10.</i>`,s:"Slide 10"},

{t:"seg",k:"concept",q:String.raw`What three failure types does PSPNet target, and what is the root cause in FCN?`,
a:String.raw`(1) <b>Mismatched relationship</b>: boat on water labeled "car." (2) <b>Confusion categories</b>: building vs skyscraper, mountain vs hill. (3) <b>Inconspicuous classes</b>: small objects like a pillow absorbed into the bed.<br>Root cause: FCN does not use the global scene category or context.`,
r:String.raw`A global scene prior ("this is a harbor") can change a single pixel's label. Small objects need fine detail as well as context.`,s:"Slides 13–15"},

{t:"seg",k:"concept",q:String.raw`Describe PSPNet's pyramid pooling module. Why several bin sizes, and why concatenate the original map?`,
a:String.raw`ResNet features → average-pool into 1×1, 2×2, 3×3, 6×6 bins → conv each → upsample to feature size → concatenate with the original feature map → conv → per-pixel prediction.<br>1×1 = global scene prior; finer bins = sub-region context. Concatenating the original keeps local detail so strong local evidence can still win.`,
r:String.raw`85.4 mIoU on VOC 2012; 1st place on VOC 2012 and Cityscapes. Downside: a co-occurrence prior could push an unusual scene (car on a flooded street) toward "boat."`,s:"Slides 16–17"},

{t:"seg",k:"numeric",q:String.raw`PSPNet: how many pooled cells does the pyramid produce? If the feature map has 2048 channels and each of 4 branches is reduced to 512 channels, how many channels after concatenation?`,
a:String.raw`Cells: $1+4+9+36=50$. Channels: $2048+4\times512=4096$.`,
r:String.raw`The 1/N channel reduction per branch (N = 4 levels) is from the PSPNet paper, so that global context doesn't outweigh local features. <i>Added context; the slide shows only the bin sizes.</i>`,s:"Slide 16 · Added context"},

{t:"seg",k:"numeric",q:String.raw`On the cans dataset, a model predicts "background" everywhere and background is 85% of pixels. What are the pixel accuracy and mIoU (4 classes)?`,
a:String.raw`Pixel accuracy $=0.85$. mIoU: background IoU $=0.85$, each of 3 object classes $=0$ → $0.85/4\approx0.21$.`,
r:String.raw`That's why the course's 0.91 mIoU is more informative than its 97.9% pixel accuracy: PA is inflated by the dominant background class.`,s:"Slide 22 · Worked example"},

{t:"seg",k:"numeric",q:String.raw`In <code>features_pyramid(x, n_layers=4)</code> with <code>x</code> at 1/4 scale, how many outputs are returned and at what scales? How does <code>build_fcn</code> bring them back to 1/4?`,
a:String.raw`<code>outputs=[x]</code> (1/4), plus AveragePooling (1/8), plus $n\_layers-1=3$ stride-2 convs (1/16, 1/32, 1/64): <b>5 outputs</b>.<br><code>build_fcn</code> applies a 256-filter conv to each extra level and bilinearly upsamples by 2, 4, 8, … so all land at 1/4, concatenates them, then two transposed convs (×2, ×2) reach full resolution, then 1×1 transposed conv (<code>n_classes</code>) + softmax.`,
r:String.raw`Inconsistency: the slide 18 diagram uses Conv2D(stride 2) for every level, but the code uses AveragePooling for the first extra level.`,s:"Slides 18–20"},

{t:"seg",k:"concept",q:String.raw`List Mask R-CNN's pipeline, step by step.`,
a:String.raw`(1) Backbone (ResNet/ResNeXt, optionally FPN). (2) RPN proposes RoIs. (3) <b>RoIAlign</b> extracts fixed-size features. (4) Box head: class + box offsets. (5) Mask head: a small FCN outputs $K$ masks of $m\times m$ with per-pixel sigmoid. (6) Loss $\mathcal L_{cls}+\mathcal L_{off}+\mathcal L_{mask}$.`,
r:String.raw`It extends Faster R-CNN by applying an FCN on each RoI, with mask and class predicted in parallel and decoupled. COCO mask AP: 33.1 (R101-C4), 35.7 (R101-FPN), 37.1 (ResNeXt-101-FPN).`,s:"Slides 26–32, 39"},

{t:"seg",k:"concept",q:String.raw`What does RoIAlign fix in RoI Pool, and why does it matter more for masks than for boxes?`,
a:String.raw`RoI Pool <b>quantizes</b> (rounds) the RoI boundaries and bin edges to the feature grid. RoIAlign keeps exact floating-point coordinates and samples features by bilinear interpolation at regular points in each bin, then pools.<br>A mask is pixel-aligned output: a sub-cell shift at stride 16–32 becomes several pixels of boundary error, while the class barely changes.`,
r:String.raw`The slides only say RoI Pool is "coarse spatial quantization." <i>The bilinear-sampling mechanism is from the Mask R-CNN paper (added context).</i>`,s:"Slide 26 · He et al. 2017"},

{t:"seg",k:"concept",q:String.raw`Mask R-CNN's mask loss: which of the $K$ masks contributes in training, and which mask is used at inference?`,
a:String.raw`Training: only the mask of the <b>ground-truth</b> class $k$ contributes to $\mathcal L_{mask}$ (per-pixel binary CE). Inference: take the top 100 detections; for each, pick the mask of its <b>predicted</b> class, resize to the RoI, binarize at 0.5.`,
r:String.raw`Each mask channel only ever learns from its own class, so a misclassified box at test time gets the wrong class's mask.`,s:"Slides 29–33"},

{t:"seg",k:"proof",q:String.raw`Prove/argue why per-class sigmoids "decouple" mask from class, whereas a per-pixel softmax over $K$ classes would not.`,
a:String.raw`Softmax: $p_k=e^{z_k}/\sum_j e^{z_j}$, so $\partial p_k/\partial z_j=-p_kp_j\ne0$: raising one class's score lowers the others. Every pixel would be forced to choose among classes, re-solving classification inside the mask head.<br>Sigmoid: $p_k=\sigma(z_k)$ depends only on $z_k$; $\partial p_k/\partial z_j=0$ for $j\ne k$. Each mask just answers "object pixel or not," and the box head decides the class.`,
r:String.raw`Semantic segmentation (FCN) correctly uses per-pixel softmax because each pixel must take exactly one class. Mask R-CNN departs from that.`,s:"Slides 11, 29"},

{t:"seg",k:"numeric",q:String.raw`Mask R-CNN with FPN on COCO: how many mask values per RoI, and per image with 100 detections?`,
a:String.raw`$K\,m\times m=80\times28\times28=62{,}720$ per RoI; $\times100=6{,}272{,}000$ per image, though only one channel per detection is used.`,
r:String.raw`Output memory and final-conv parameters grow linearly with $K$; a huge vocabulary would leave most channels with few training examples. ($m=14$ for the C4 head.)`,s:"Slide 32"},

{t:"seg",k:"concept",q:String.raw`Why is the mask branch a conv (FCN) head rather than FC layers?`,
a:String.raw`A conv head keeps the $m\times m$ spatial layout, so each output pixel depends on its local neighborhood, and it needs fewer parameters. FC layers that output $m^2$ values per mask flatten the input and lose spatial structure.`,
r:String.raw`It's the same lesson as FCN: spatial outputs need spatial (convolutional) computation.`,s:"Slide 31"},

{t:"seg",k:"concept",q:String.raw`What problem does Mask Scoring R-CNN fix, and how does its MaskIoU head work?`,
a:String.raw`Mask R-CNN ranks masks by classification score, but that score is <b>not correlated</b> with mask quality (MaskIoU). The MaskIoU head takes the RoI feature concatenated with the max-pooled predicted mask (14×14×257), runs convs + FCs, and regresses MaskIoU with an <b>L2 loss</b>. Final score: $s_{mask}=s_{cls}\cdot s_{iou}$.`,
r:String.raw`R101-FPN: 38.3 vs 37.0 AP (COCO 2017 test-dev). Inference: top-100 boxes after SoftNMS → mask head → MaskIoU head.`,s:"Slides 42–49"},

{t:"seg",k:"numeric",q:String.raw`Detection X: $s_{cls}=0.95$, predicted MaskIoU 0.60. Detection Y: $s_{cls}=0.85$, MaskIoU 0.90. Rank them under Mask R-CNN and under MS R-CNN.`,
a:String.raw`Mask R-CNN (class score): X (0.95) &gt; Y (0.85).<br>MS R-CNN: X $=0.95\times0.60=0.57$; Y $=0.85\times0.90=0.765$ → <b>Y ranks first</b>.`,
r:String.raw`The product is high only when the class is right <i>and</i> the mask is good, which is what AP across IoU thresholds rewards.`,s:"Slides 44, 47 · Worked example"},

{t:"seg",k:"proof",q:String.raw`Why does MS R-CNN raise AP@0.75 but not AP@0.5 (e.g. R101: 55.0 → 54.9 at 0.5, 36.6 → 38.1 at 0.75)?`,
a:String.raw`The masks themselves are unchanged; only the <b>ranking</b> changes. At IoU 0.5 most masks already count as correct, so reordering helps little (and may demote loose-but-correct masks). At 0.75 only high-quality masks count; moving them up raises precision at each recall level.`,
r:String.raw`Lesson: a scoring change can move AP without improving any segmentation. Rotated MS R-CNN's README shows a similar-sized MS gain (35.3 vs 34.1).`,s:"Slide 49 · rotated_maskrcnn README"},

{t:"seg",k:"numeric",q:String.raw`A predicted mask has 1,000 pixels, the ground truth 1,200, and they overlap on 900 pixels. Compute MaskIoU.`,
a:String.raw`Union $=1000+1200-900=1300$. MaskIoU $=900/1300\approx0.692$.`,
r:String.raw`This is the regression target of the MaskIoU head; mask score multiplies it by the class score.`,s:"Slide 44 · Worked example"},

{t:"seg",k:"concept",q:String.raw`YOLACT's key idea: what two parallel tasks replace the RPN → RoI stage?`,
a:String.raw`(1) <b>ProtoNet</b> produces $k$ image-size prototype masks ("where"). (2) The prediction head outputs, per anchor, $c$ class scores, 4 box offsets and $k$ mask coefficients ("what"). After NMS, masks are assembled as linear combinations of prototypes, then cropped and thresholded.`,
r:String.raw`One stage, 33 fps at 550², 29.8 AP. Based on RetinaNet with ResNet-101 + FPN; anchors 24–384 px, ratios 1, 2, 1/2.`,s:"Slides 51–56, 61"},

{t:"seg",k:"proof",q:String.raw`Show that $M=\sigma(PC^\top)$ yields one mask per detection with a single matrix multiply. Give the shapes.`,
a:String.raw`$P$: $h\times w\times k$, reshape to $(hw)\times k$. $C$: $n\times k$. $PC^\top$: $(hw)\times n$ → reshape to $h\times w\times n$. Column $j$ at pixel $p$ is $\sum_{i=1}^{k}P_{p,i}C_{j,i}$: a weighted sum of the $k$ prototype values at that pixel, using detection $j$'s coefficients. σ makes it a probability.`,
r:String.raw`All $n$ instances are assembled in one GEMM, far cheaper than running a mask head per RoI. Example: $P$ 138×138×32, $C$ 10×32 → $M$ 138×138×10.`,s:"Slide 56"},

{t:"seg",k:"numeric",q:String.raw`YOLACT with $c=80$ classes, $k=4$ prototypes and 3 anchors per location: how many output channels does the head produce per location?`,
a:String.raw`Per anchor $c+4+k=80+4+4=88$; × 3 anchors $=264$ channels per location (class $W\times H\times ca$, box $4a$, mask $ka$).`,
r:String.raw`Per-instance identity costs only $k$ extra numbers per anchor, which is why YOLACT is cheap.`,s:"Slides 55, 61"},

{t:"seg",k:"concept",q:String.raw`ProtoNet has no explicit loss. How is it trained, and what do prototypes end up representing?`,
a:String.raw`Supervision arrives only after assembly: $\mathcal L_{mask}=BCE(M,M_{gt})$ on assembled masks, and gradients flow into $P$ through the coefficient-weighted sums. Prototypes become whatever basis reconstructs GT masks well: boundary maps, directional (e.g. bottom-left) maps, background/contour maps, ground maps.`,
r:String.raw`Open puzzle: an FCN is translation-equivariant, yet prototype 2 is position-dependent; padding at image borders is a candidate source of position information.`,s:"Slides 54, 59–60"},

{t:"seg",k:"concept",q:String.raw`Why does YOLACT use tanh for mask coefficients?`,
a:String.raw`tanh gives coefficients in $(-1,1)$, so a detection can <b>subtract</b> a prototype as well as add it ("Mask 4 − Mask 5 removes the red umbrella"). With sigmoid (0 to 1), prototypes could only add activation, making it hard to separate an object from a neighbor lit up by the same prototype.`,
r:String.raw`Subtraction is what turns shared, image-wide prototypes into instance-specific masks.`,s:"Slides 55, 60"},

{t:"seg",k:"concept",q:String.raw`YOLACT: why crop the assembled mask with the predicted box, and why divide $\mathcal L_{mask}$ by the GT mask area?`,
a:String.raw`<b>Crop:</b> prototypes span the whole image, so one person's combination can also fire on another person elsewhere; the box crop removes distant activations.<br><b>Area normalization:</b> summed BCE grows with pixel count, so large objects would dominate; dividing by area makes each instance contribute a per-pixel average.`,
r:String.raw`Area normalization matters more here than in Mask R-CNN because YOLACT masks are image-size, not fixed $m\times m$.`,s:"Slides 52, 57"},

{t:"seg",k:"concept",q:String.raw`What is OHEM, and why does YOLACT use it with a 1:3 positive:negative ratio?`,
a:String.raw`Online Hard Example Mining: compute the loss for all RoIs/anchors with a read-only copy, then backprop only on the highest-loss ones. A one-stage detector has huge numbers of easy background anchors; picking the hardest negatives, capped at 3 per positive, keeps training focused without letting negatives swamp the loss.`,
r:String.raw`Shrivastava, Gupta, Girshick (CVPR 2016). Compare RetinaNet, which uses focal loss instead of sampling.`,s:"Slides 57–58"},

{t:"seg",k:"concept",q:String.raw`List the three steps of Fast NMS.`,
a:String.raw`Build a sorted IoU matrix $X\in\mathbb R^{c\times n\times n}$ (top $n$ per class, descending score).<br>1. Zero the lower triangle and diagonal: $X_{kij}=0$ for $i\ge j$ (<code>torch.triu</code>).<br>2. Column max: $K_{kj}=\max_i X_{kij}$, the highest IoU each box has with any higher-scoring box.<br>3. Threshold: remove detections with $K_{kj}\ge t$.`,
r:String.raw`All matrix ops, so fully parallel on GPU. The slide doesn't say keep vs remove; by construction the thresholded boxes are the ones to remove.`,s:"Slide 62"},

{t:"seg",k:"numeric",q:String.raw`Fast NMS vs NMS, one class: A &gt; B &gt; C by score, IoU(A,B) = 0.7, IoU(A,C) = 0.2, IoU(B,C) = 0.6, $t=0.5$. What survives in each?`,
a:String.raw`<b>Fast NMS:</b> column maxima A: 0, B: 0.7, C: max(0.2, 0.6) = 0.6 → remove B and C → {A}.<br><b>NMS:</b> keep A; remove B (0.7); check C only against kept boxes (A: 0.2) → keep C → {A, C}.`,
r:String.raw`Fast NMS counts suppression by an already-suppressed box (B), so it can remove slightly more.`,s:"Slide 62 · Worked example"},

{t:"seg",k:"proof",q:String.raw`Prove that every box removed by standard NMS is also removed by Fast NMS (Fast NMS removes a superset).`,
a:String.raw`Standard NMS removes box $j$ only if some <b>kept</b> box $i$ with higher score has $IoU(i,j)\ge t$. Since $i$ scores higher, $X_{ij}$ is in the upper triangle and survives the <code>triu</code> step, so $K_j=\max_{i'}X_{i'j}\ge X_{ij}\ge t$, and Fast NMS removes $j$ too. ∎<br>The converse fails: Fast NMS also counts suppressed boxes as suppressors (previous card).`,
r:String.raw`So Fast NMS trades a little recall for parallelism. <i>Added proof.</i>`,s:"Slide 62 · Added proof"},

{t:"seg",k:"concept",q:String.raw`YOLACT gets 29.8 AP at 33 fps vs Mask R-CNN's 35.7 AP. Why the AP drop, and what's wrong with the "3.8× faster" claim?`,
a:String.raw`YOLACT's masks come from a 1/4-resolution shared basis plus a box crop; Mask R-CNN gives every RoI its own aligned features and dedicated mask head, so fine detail and separating touching objects are easier.<br>Claim check: slide 40 says Mask R-CNN runs at 5 fps, and 33/5 ≈ 6.6×, not 3.8×; the basis isn't stated. Slide 63 also mixes tables (38.3 is MS R-CNN on 2017 test-dev).`,
r:String.raw`Always check backbone, hardware and test split before quoting speedups.`,s:"Slides 40, 63"},

{t:"seg",k:"concept",q:String.raw`Cascade R-CNN: what is the IoU-threshold dilemma for training detectors?`,
a:String.raw`At $u=0.5$, detectors accept "close" false positives (noisy detections). Raising $u$ (e.g. 0.7) starves training of positives (few high-IoU proposals), causing overfitting and low detection rates. AP: $u$ = 0.5 → 0.349, 0.6 → 0.354, 0.7 → 0.319. Each regressor works best near the IoU range it was trained on.`,
r:String.raw`So simply raising the threshold does not give higher-quality detections.`,s:"Slides 66–67"},

{t:"seg",k:"numeric",q:String.raw`From slide 67c: at input IoU ≈ 0.5–0.55, the $u=0.7$ regressor outputs ≈0.62 while the $u=0.5$ regressor outputs ≈0.74. What does this imply for stage ordering?`,
a:String.raw`A regressor trained only on near-perfect boxes under-corrects poor boxes (it never saw large offsets). So stages must go from <b>low to high</b> threshold: early stages fix poor proposals, later stages refine already-good ones.`,
r:String.raw`This is the quantitative case for the cascade instead of starting with a strict detector.`,s:"Slide 67"},

{t:"seg",k:"concept",q:String.raw`Compare Iterative BBox, Integral loss and Cascade R-CNN.`,
a:String.raw`<ul><li><b>Iterative BBox:</b> reuse the same head H1 (trained at $u=0.5$) repeatedly; mismatched to tighter later boxes, and outliers grow.</li><li><b>Integral loss:</b> several classifiers at $u\in\{0.5,\dots,0.75\}$ on the same stage-1 proposals; high-$u$ classifiers have few positives and overfit.</li><li><b>Cascade:</b> stage $t$ has its own head $f_t$ trained at $u^t\gt u^{t-1}$ on boxes refined by stage $t-1$.</li></ul>`,
r:String.raw`Cascade: $f=f_T\circ\dots\circ f_1$, loss per stage $\mathcal L^t_{cls}+\lambda[c\ge1]\mathcal L^t_{off}$. 42.8 AP on COCO 2017 test-dev (ResNet-101).`,s:"Slides 68–78"},

{t:"seg",k:"proof",q:String.raw`Argue why Cascade R-CNN's resampling increases the number of high-IoU positives at later stages.`,
a:String.raw`On slide 67c the regressor curves lie <b>above the diagonal</b>: output IoU &gt; input IoU for most inputs. Applying stage $t-1$'s regressor shifts the whole proposal IoU distribution to the right. So the fraction of boxes above a stricter $u^t$ can stay large even though $u^t\gt u^{t-1}$ (slides 74, 76). Each stage's training distribution then matches its test distribution.`,
r:String.raw`Integral loss changes only the label threshold, not the data distribution, which is why it overfits.`,s:"Slides 67, 74–76"},

{t:"seg",k:"concept",q:String.raw`HTC: name the three ideas and how the stage-$t$ mask features change across variants.`,
a:String.raw`<ul><li>Cascade Mask R-CNN: $\mathcal P(x,r_{t-1})$ (boxes before refinement).</li><li><b>Interleaved:</b> $\mathcal P(x,r_t)$, the box just refined by $B_t$.</li><li><b>Mask information flow:</b> add $\mathcal G(m^-_{t-1})$, the previous mask head's features via 1×1 conv.</li><li><b>Semantic branch:</b> add $\mathcal P(\mathcal S(x),\cdot)$ to box and mask inputs.</li></ul>`,
r:String.raw`Plain cascading on Mask R-CNN gives small gains because mask heads work in isolation. HTC mask AP: 38.4 (R50), 39.7 (R101), 47.1 (ResNeXt-101).`,s:"Slides 80–90"},

{t:"seg",k:"formula",q:String.raw`Write HTC's loss with its stage weights. What do the weights suggest?`,
a:String.raw`$$\mathcal L=\sum_{t=1}^{3}\alpha_t\big(\mathcal L^t_{bbox}+\mathcal L^t_{mask}\big)+\beta\,\mathcal L_{seg}$$ $\alpha=[1,0.5,0.25]$, $\beta=1$. $\mathcal L_{mask}$ = BCE, $\mathcal L_{seg}$ = CE.`,
r:String.raw`Halving each stage means stage 1 gets the strongest gradient, plausibly because every later stage depends on stage 1's boxes. Note slide 87's last line is circular; it should read $m^-_t=M^-_t(\mathcal F(x^{mask}_t,m^-_{t-1}))$.`,s:"Slides 87, 89"},

{t:"seg",k:"concept",q:String.raw`When should you use Rotated Mask R-CNN instead of standard Mask R-CNN? Give the evidence.`,
a:String.raw`For <b>dense, elongated, tilted</b> objects (surfboards, pencils). Axis-aligned boxes overlap heavily, so NMS removes true neighbors and each box contains several same-class objects. Evidence: on the dense PPC set, MS R-CNN 13.2 → Rotated MS R-CNN 19.3 mask mAP; on COCO, rotated is slightly <i>worse</i> (34.7 vs 35.3).`,
r:String.raw`Costs: ~10% slower inference, 30–50% slower training, similar memory. PPC is small with 2 classes, so don't over-read the gain.`,s:"Slides 92–94 · rotated_maskrcnn README"},

{t:"seg",k:"concept",q:String.raw`What is SAM 2's task, and what are its main components?`,
a:String.raw`<b>Promptable Visual Segmentation</b>: given an image/video and prompts (points, boxes, masks), output a mask, or a <b>masklet</b> (spatio-temporal mask) for video.<br>Components: Hiera image encoder → memory attention (to memory bank + object pointers) → mask decoder (with prompt encoder) → memory encoder → memory bank.`,
r:String.raw`New vs SAM: memory to track the object across frames. Hiera = hierarchical ViT pretrained with MAE.`,s:"Slides 96–105"},

{t:"seg",k:"concept",q:String.raw`Why does SAM 2's memory bank keep two separate FIFOs?`,
a:String.raw`One FIFO holds up to $N$ <b>recent frames</b> (the object's current appearance, pose, position); another holds up to $M$ <b>prompted frames</b> (human-verified evidence of the target). A single recent-frames queue would eventually evict the prompt; a prompts-only queue couldn't follow appearance changes.`,
r:String.raw`With memory, one click in a later frame can repair a lost track (slide 98) instead of re-annotating from scratch.`,s:"Slides 98, 105"},

{t:"seg",k:"concept",q:String.raw`Why should a single-point prompt return several masks? What does the course notebook recommend?`,
a:String.raw`A point is ambiguous (one click on a wheel fits the magwheel, the whole wheel or the whole car). The decoder predicts multiple masks with quality scores. Notebook: keep <code>multimask_output=True</code> for a single point and pick the highest score; use <code>False</code> when several prompts already pin down one object.`,
r:String.raw`Demo scores: magwheel 0.770, wheel 0.630, car 0.043. Forcing one output would make training average the interpretations into a blurry compromise.`,s:"Slide 103 · SAM2 demo notebook"},

{t:"seg",k:"concept",q:String.raw`Practical SAM 2 (course notebook): prompt formats, labels, and why <code>set_image</code> runs once.`,
a:String.raw`Points: (x, y) with labels <b>1 = foreground, 0 = background</b>. Boxes: xyxy. Previous low-res logits can be fed back as <code>mask_input</code> to refine. <code>set_image</code> runs the heavy image encoder once and caches the embedding ([1, 256, 64, 64]); each <code>predict</code> only runs the light prompt encoder + mask decoder, which makes interactive clicking cheap.`,
r:String.raw`A negative click on the car body plus a positive on the wheel leaves only the magwheel. Notebook quirk: cells 31/35 show a stale "Score: 0.770."`,s:"SAM2 demo notebook (Part C)"},

{t:"seg",k:"numeric",q:String.raw`SAM 2 data engine: Phase 1 took 37.8 s/frame and Phase 3 took 4.5 s/frame. Speed-up? And SAM 2 vs SAM image FPS (130.1 vs 21.7)?`,
a:String.raw`Annotation: $37.8/4.5=8.4\times$ faster. Inference: $130.1/21.7\approx6.0\times$ faster.`,
r:String.raw`Phases: SAM per frame (100% edited, 4.80 clicks) → SAM + SAM 2 propagation (23.25%, 7.4 s) → SAM 2 in the loop (19.04%, 2.68 clicks). SA-V: 50.9K videos, 642.6K masklets, 35.5M masks.`,s:"Slides 107–110"},

{t:"seg",k:"concept",q:String.raw`Define the $\mathcal J\&\mathcal F$ VOS metric. Why average the two?`,
a:String.raw`$\mathcal J$ = region IoU (Jaccard) between masks. $\mathcal F$ = boundary F-measure from bipartite matching of boundary pixels (precision/recall). A mask can cover the object's bulk (high $\mathcal J$) but have ragged edges (low $\mathcal F$); averaging penalizes both missing area and sloppy contours.`,
r:String.raw`SAM 2 Hiera-L: MOSE val 77.9, SA-V test 78.4 vs best prior Cutie-base+ 71.7 and 62.8.`,s:"Slides 111–112"},

{t:"seg",k:"proof",q:String.raw`Why is Phase-3 mask alignment lowest for small masks (72.8%) and highest for large ones (100%)? Connect to another failure in this lecture.`,
a:String.raw`Alignment = IoU &gt; 0.75 with Phase 1. For a small mask, a few pixels of boundary disagreement are a large fraction of its area, so IoU drops fast; large masks have many interior pixels that dominate IoU. This is the same reason semantic segmentation fails on <b>inconspicuous</b> (small) classes, the pillow on the bed.`,
r:String.raw`IoU-type metrics are inherently harsher on small objects (also why COCO reports AP$_s$ separately).`,s:"Slides 14, 108"},

{t:"seg",k:"proof",q:String.raw`Out of the box: Cascade R-CNN refines boxes in stages; HTC passes mask features stage to stage; SAM 2 refines via extra clicks. Is user feedback a kind of cascade stage? Argue both sides.`,
a:String.raw`<b>Yes:</b> each click conditions the next prediction on the previous mask (<code>mask_input</code>), progressively tightening it, like stage $t$ using stage $t-1$'s output. <b>No:</b> cascade stages are separate trained heads with stricter thresholds; SAM 2 reuses the same decoder, and the new information comes from a human, not from a learned distribution shift.`,
r:String.raw`Open/stretch question from the notes. A good answer names the mechanism (distribution matching vs external evidence).`,s:"Slides 75, 85, 98 · Stretch"}
]);
