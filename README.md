# MNIST LeNet-BN v2.0

Run `python run_experiments.py --execute` with a CUDA-enabled environment matching requirements.txt. The default configuration is configs/ts_compaction.yaml.

30 epochs; seed 1; batch size 64; Adadelta lr 1.0, rho 0.9, eps 1e-6, weight decay 0.0001. Learning-rate decay is explicitly disabled. SGD remains supported. Global BN-scale pruning removes approximately 50% of channels (strict cutoff ties can remove more).

MNIST: 54,000 training / 6,000 validation examples, fixed seed split; 10,000 official test examples. All splits use Pad(2), ToTensor and Normalize(mean=0.1307, std=0.3081), producing 1x32x32 inputs. No random augmentation. Two Conv-BN-ReLU-Pool blocks (6 and 16 channels), followed by Linear 400->120->84->10.

Dense baseline records candidate masks at completed epochs 0..29. Early-Bird uses normalized Hamming distance, epsilon 0.1, window 5. Comparisons: offsets [-10,-5,0,5,10] from detected t_m, online/posthoc, dense-late/train-time compaction: 20 runs. Actual t_s is clipped to [1,29]; offsets can therefore coincide near the boundary. C4 now denotes +5, not the removed fixed-time condition. Posthoc uses the baseline t_m mask, while retaining each comparison run's current weights.

Validation/logging every epoch, initial/final checkpoints, final test and model metrics follow the v1.1 runner. LeNet compaction slices the first classifier's channel-spatial columns and migrates optimizer state. Run `python check_compaction.py` to check gated/compact output equivalence, parameter/FLOPs counts and optimizer-state shapes.

Repair notes: reused the validated v1.1 runner and shared utilities to correct incompatible imports/configuration helpers; corrected LeNet constructors/forward/BN names and sequential classifier slicing. Summary rows use online then posthoc, C1..C5, dense-late then train-time. CUDA/PyTorch environment is recorded per run. MNIST download needs network access on first execution. Run python check_adadelta.py to verify Adadelta state slicing and constant learning rate.
