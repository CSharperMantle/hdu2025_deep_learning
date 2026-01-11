import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "../../../output/train.csv"

df = pd.read_csv(CSV_PATH)
df = df.sort_values("epoch")

epoch = df["epoch"]
fake = df["fake_loss"]
neg_real = -df["real_loss"]  # plot -real_loss
neg_critic = -df["critic_loss"]  # plot -critic_loss
neg_gen = -df["generator_loss"]  # plot -generator_loss

# --- Fig 1: -real_loss vs fake_loss with shaded gap ---
plt.figure(figsize=(5, 5))
plt.plot(epoch, neg_real, label="-real_loss", linewidth=2)
plt.plot(epoch, fake, label="fake_loss", linewidth=2)
plt.fill_between(epoch, neg_real, fake, alpha=0.25, label="gap")
plt.xlabel("epoch")
plt.ylabel("loss (scaled as plotted)")
plt.title("Real vs Fake (shaded gap)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# --- Fig 2: -critic_loss ---
plt.figure(figsize=(5, 5))
plt.plot(epoch, neg_critic, label="-critic_loss", linewidth=2)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Critic loss (negated)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# --- Fig 3: -generator_loss ---
plt.figure(figsize=(5, 5))
plt.plot(epoch, neg_gen, label="-generator_loss", linewidth=2)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Generator loss (negated)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()
