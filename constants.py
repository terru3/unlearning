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

## for incompetent-teacher unlearning
TEACHER_STEPS = 600
STUDENT_LR = 7.5e-4
# tuned for best unlearning on forget vs. performance on val data tradeoff