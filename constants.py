SEED = 42
BATCH_SIZE = 8 # large batch size like 128, given 1/100 target class, will cause
# amnesiacML to flag virtually every batch and undo all learning
LR = 1e-3
PRINT_ITERS = 6000 # ~1x per epoch
EPOCHS = 25

## For membership inference attacks
NUM_SHADOW_MODELS = 20
SHADOW_EPOCHS = 10
ATTACK_EPOCHS = 10

## for Incompetent-teacher unlearning
TEACHER_STEPS = 600
STUDENT_LR = 7.5e-4
# tuned for best unlearning on forget vs. performance on val data tradeoff

## for UNSIR
NOISE_EPOCHS = 5
NOISE_STEPS = 10
NOISE_LAMBDA = 0.1
NUM_NOISE_BATCHES = 100

IMPAIR_LR = 0.02
REPAIR_LR = 0.01

IMPAIR_EPOCHS = 1
REPAIR_EPOCHS = 1

## for gated knowledge transfer (GKT)
GEN_LR = 1e-3
Z_DIM = 128
ATTN_BETA = 250 # they used 250 on MNIST
BAND_PASS_THRESH = 0.01







