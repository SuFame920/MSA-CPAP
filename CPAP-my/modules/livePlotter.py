import matplotlib
# 在无显示环境也不报错（如远程/服务器）
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt

class LivePlotter:
    def __init__(self, title="Loss Curves"):
        # 尝试启用交互；若是纯终端也不影响训练
        try:
            plt.ion()
        except Exception:
            pass
        self.fig, self.ax = plt.subplots(figsize=(7,4))
        self.ax.set_title(title)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        (self.l_train,) = self.ax.plot([], [], label="Train", linewidth=2)
        (self.l_val,)   = self.ax.plot([], [], label="Valid", linewidth=2)
        (self.l_test,)  = self.ax.plot([], [], label="Test",  linewidth=2)
        self.ax.legend(loc="best")
        self.train_y, self.val_y, self.test_y = [], [], []
        self.x = []
        self.fig.tight_layout()

    def update_epoch(self, epoch, train_loss, val_loss, test_loss):
        self.x.append(epoch)
        self.train_y.append(float(train_loss))
        self.val_y.append(float(val_loss))
        self.test_y.append(float(test_loss))

        self.l_train.set_data(self.x, self.train_y)
        self.l_val.set_data(self.x, self.val_y)
        self.l_test.set_data(self.x, self.test_y)

        # 自适应坐标
        xmin, xmax = min(self.x), max(self.x) if self.x else 1
        ymin = min(self.train_y + self.val_y + self.test_y)
        ymax = max(self.train_y + self.val_y + self.test_y)
        pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        self.ax.set_xlim(0, max(self.x)+0.5)
        self.ax.set_ylim(ymin - pad, ymax + pad)

        # 刷新显示（交互环境会实时更新），并存一份到文件，便于纯终端查看
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass
        self.fig.savefig("live_losses.png", dpi=150)
