import matplotlib.pyplot as plt
import numpy as np
val_acc_l = [1,2,3,4,5]
steps = np.arange(1, len(val_acc_l) + 1)

plt.figure(figsize=(50, 20))
plt.plot(steps,val_acc_l, marker='o', linewidth=1.8, markersize=4, label='Loss')
plt.title('Training Loss per Step. gumbel softmax tau 1 5l size 128k}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('figure.png')