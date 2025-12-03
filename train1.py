import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, Callback

# 自定义回调类，用于打印训练进度
class ProgressLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch + 1}/{self.params["epochs"]} - Loss: {logs["loss"]:.4f} - Accuracy: {logs["accuracy"]:.4f}')

# 设置参数
data_dir = r'数据集'  # 数据集路径
batch_size = 32
img_height = 150
img_width = 150
num_classes = 5  # 类别数量
epochs = 5
initial_learning_rate = 0.01  # 固定学习率

# 数据预处理
#数据增强
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

#创建训练生成器
train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # 适合多分类问题
    shuffle=True
)

# 构建 ResNet50 模型
base_model = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=(img_height, img_width, 3))

# 添加自定义顶层
model = models.Sequential([
    base_model,
    #在最后一个卷积层之后添加全局平均池化层，防止过拟合
    layers.GlobalAveragePooling2D(),
    #添加一个全连接层，具有256个神经元，用于学习特征
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# 设置优化器和学习率
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)  # 直接设置固定学习率
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# 设置 CSV logger
csv_logger = CSVLogger('training_log1.csv', append=True, separator=',')

# 自定义训练进度打印
progress_logger = ProgressLogger()

# 训练模型
history = model.fit(train_generator,
                    epochs=epochs,
                    callbacks=[csv_logger, progress_logger])

# 保存模型权重
model.save('resnet_diabetic_retinopathy_32_5_0.01.h5')
print("Model weights saved as 'resnet_diabetic_retinopathy.h5'.")

# 绘制损失和准确率的曲线
plt.figure(figsize=(12, 4))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 保存曲线图像
plt.savefig('training_curve_32_5_0.01.png')
plt.show()