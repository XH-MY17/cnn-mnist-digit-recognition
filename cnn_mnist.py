# -*- coding: utf-8 -*-
"""
基于 CNN 的手写数字识别系统 / CNN Handwritten Digit Recognition

自动从 Jupyter Notebook 导出
"""



"""
CNN-based Handwritten Digit Recognition System
=============================================
Course: Artificial Intelligence
Author: Xiong Hu  Student ID: 20231300125

Usage:
  pip install tensorflow numpy matplotlib pillow scikit-learn seaborn
  python cnn_digit_recognition.py
  python cnn_digit_recognition.py --gui
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("CNN Handwritten Digit Recognition System")
print("=" * 60)

# ═══════════════════ 1. Data Loading & Preprocessing ═══════════════════
print("\n[1/5] Loading MNIST dataset...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(f"  Train: {x_train.shape[0]} samples  Test: {x_test.shape[0]} samples")

# Normalize [0,255] -> [0,1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# Reshape to 4D (batch, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)

# One-hot encoding
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test,  10)

# ═══════════════════ 2. Data Augmentation ═══════════════════
print("\n[2/5] Configuring data augmentation...")
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

# Split validation set (5000 from training)
x_val = x_train[:5000];  y_val_cat = y_train_cat[:5000]
x_train_aug = x_train[5000:];  y_train_aug_cat = y_train_cat[5000:]
print(f"  Train:{x_train_aug.shape[0]}  Val:{x_val.shape[0]}  Test:{x_test.shape[0]}")

# ═══════════════════ 3. CNN Model Construction ═══════════════════
print("\n[3/5] Building CNN model...")

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Model Architecture:
      Conv2D(32,3x3)+ReLU -> MaxPool(2x2)
      -> Conv2D(64,3x3)+ReLU -> MaxPool(2x2)
      -> Flatten -> Dense(128)+ReLU -> Dropout(0.5)
      -> Dense(10)+Softmax
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape,
                      padding='same', name='conv1'),
        layers.MaxPooling2D((2,2), name='pool1'),

        layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2'),
        layers.MaxPooling2D((2,2), name='pool2'),

        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    return model

model = build_cnn_model()
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ═══════════════════ 4. Model Training ═══════════════════
print("\n[4/5] Starting model training...")

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
model_ckpt = callbacks.ModelCheckpoint(
    'best_cnn_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

history = model.fit(
    datagen.flow(x_train_aug, y_train_aug_cat, batch_size=128),
    steps_per_epoch=len(x_train_aug) // 128,
    epochs=20,
    validation_data=(x_val, y_val_cat),
    callbacks=[early_stop, reduce_lr, model_ckpt],
    verbose=1
)

# ═══════════════════ 5. Model Evaluation ═══════════════════
print("\n[5/5] Evaluating model...")
model.load_weights('best_cnn_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\n{'='*60}")
print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Test loss:     {test_loss:.4f}")
print(f"{'='*60}")

# ═══════════════════ 6. Training History Plot ═══════════════════
print("\n  Generating training history plot...")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'], 'b-', label='训练集准确率', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r--', label='验证集准确率', linewidth=2)
axes[0].set_title('模型准确率变化曲线'); axes[0].set_xlabel('训练轮数')
axes[0].set_ylabel('准确率'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], 'b-', label='训练集损失', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r--', label='验证集损失', linewidth=2)
axes[1].set_title('模型损失变化曲线'); axes[1].set_xlabel('训练轮数')
axes[1].set_ylabel('损失值'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('training_history.png', dpi=150); plt.close()
print("  -> training_history.png")

# ═══════════════════ 7. Confusion Matrix ═══════════════════
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('混淆矩阵 (Confusion Matrix)')
plt.xlabel('预测标签'); plt.ylabel('真实标签')
plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=150); plt.close()
print("  -> confusion_matrix.png")

print("\n  Classification Report:")
print(classification_report(y_test, y_pred_classes, digits=4))

# ═══════════════════ 8. Misclassified Samples ═══════════════════
misclassified = np.where(y_test != y_pred_classes)[0]
print(f"\n  Misclassified: {len(misclassified)}/{len(y_test)} ({len(misclassified)/len(y_test)*100:.2f}%)")

plt.figure(figsize=(12, 8))
for i, idx in enumerate(misclassified[:16]):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f'真实:{y_test[idx]} 预测:{y_pred_classes[idx]}', fontsize=10)
    plt.axis('off')
plt.tight_layout(); plt.savefig('misclassified_samples.png', dpi=150); plt.close()
print("  -> misclassified_samples.png")

# ═══════════════════ 9. Prediction Samples ═══════════════════
plt.figure(figsize=(12, 6))
sample_indices = np.random.choice(len(x_test), 20, replace=False)
for i, idx in enumerate(sample_indices):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
    pred_label, true_label = y_pred_classes[idx], y_test[idx]
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f'预测:{pred_label} 真实:{true_label}', color=color, fontsize=9)
    plt.axis('off')
plt.tight_layout(); plt.savefig('prediction_samples.png', dpi=150); plt.close()
print("  -> prediction_samples.png")

# ═══════════════════ 10. Save Results ═══════════════════
model.save('cnn_mnist_model.h5')
print("\n  Model saved: cnn_mnist_model.h5")

with open('test_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"Test loss:    {test_loss:.4f}\n")
    for i in range(10):
        mask = (y_test == i)
        acc_i = np.sum(y_pred_classes[mask] == y_test[mask]) / np.sum(mask)
        f.write(f"Digit {i} accuracy: {acc_i:.4f}\n")
print("  -> test_results.txt")

print(f"\n{'='*60}")
print(f"  Done! Test accuracy: {test_acc*100:.2f}%")
print(f"  5 charts + report generated")
print(f"{'='*60}")

# ═══════════════════ 11. GUI Application ═══════════════════
def launch_gui():
    """Launch handwritten digit recognition GUI (Tkinter)"""
    try:
        import tkinter as tk
        from PIL import Image, ImageDraw
    except ImportError:
        print("Install: pip install pillow")
        return

    class App:
        def __init__(self, master):
            self.master = master
            master.title("手写数字识别系统 - CNN")
            master.geometry("500x550")
            master.resizable(False, False)

            tk.Label(master, text="手写数字识别系统",
                     font=("微软雅黑", 18, "bold")).pack(pady=10)

            self.canvas = tk.Canvas(master, width=280, height=280,
                                    bg='white', cursor='cross')
            self.canvas.pack(pady=5)
            self.canvas.bind("<B1-Motion>", self.paint)
            self.canvas.bind("<Button-1>", self.start_paint)
            self.canvas.bind("<ButtonRelease-1>", self.end_paint)

            self.image = Image.new('L', (280, 280), 255)
            self.draw = ImageDraw.Draw(self.image)
            self.last_x, self.last_y = None, None

            self.result_label = tk.Label(master, text="请绘制数字",
                                         font=("微软雅黑", 14))
            self.result_label.pack(pady=5)

            self.prob_label = tk.Label(master, text="置信度: ", font=("微软雅黑", 12))
            self.prob_label.pack()

            btn_frame = tk.Frame(master)
            btn_frame.pack(pady=10)
            tk.Button(btn_frame, text="识  别", command=self.recognize,
                      font=("微软雅黑", 12), bg="#4CAF50", fg="white",
                      width=8).pack(side=tk.LEFT, padx=10)
            tk.Button(btn_frame, text="清  除", command=self.clear,
                      font=("微软雅黑", 12), bg="#f44336", fg="white",
                      width=8).pack(side=tk.LEFT, padx=10)

        def start_paint(self, e):
            self.last_x, self.last_y = e.x, e.y

        def paint(self, e):
            x, y = e.x, e.y
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, x, y,
                                        width=20, fill='black',
                                        capstyle=tk.ROUND, smooth=True)
                self.draw.line([self.last_x, self.last_y, x, y], fill=0, width=20)
            self.last_x, self.last_y = x, y

        def end_paint(self, e):
            self.last_x, self.last_y = None, None

        def clear(self):
            self.canvas.delete("all")
            self.image = Image.new('L', (280, 280), 255)
            self.draw = ImageDraw.Draw(self.image)
            self.result_label.config(text="请绘制数字")
            self.prob_label.config(text="")

        def recognize(self):
            img = self.image.resize((28, 28), Image.LANCZOS)
            arr = (255 - np.array(img)).astype('float32') / 255.0
            arr = arr.reshape(1, 28, 28, 1)
            pred = model.predict(arr, verbose=0)
            cls = np.argmax(pred)
            conf = pred[0][cls]
            self.result_label.config(
                text=f"识别结果: {cls}",
                fg="#4CAF50" if conf > 0.9 else "#FF9800")
            self.prob_label.config(text=f"置信度: {conf*100:.2f}%")

    tk.Tk().mainloop()

if __name__ == '__main__':
    if '--gui' in sys.argv:
        launch_gui()
    else:
        print("\nRun: python cnn_digit_recognition.py --gui  to launch GUI")
