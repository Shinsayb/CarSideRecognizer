import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt


def plot_images(n_row:int, n_col:int, images:list[np.ndarray]) -> None:
    fig = plt.figure()
    (fig, ax) = plt.subplots(n_row, n_col, figsize = (n_col, n_row))
    for i in range(n_row):
        for j in range(n_col):
            axis = ax[i, j]
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
            axis.imshow(images[i * n_col + j])
    plt.show()
    return None

# 이미지를 읽고 처리하는 함수
def load_and_process_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지를 반전시키는 함수
def flip_image(image):
    return cv2.flip(image, 1)


left_car_images = list()
right_car_images = list()
# 왼쪽을 보는 자동차 이미지 로드 및 좌우 반전
for i in range(50):
    file = "./carLeft/after/CL{0:02d}.png".format(i + 1)
    img = load_and_process_image(file)
    if img is not None:
        left_car_images.append(img)
        flipped_img = flip_image(img)
        right_car_images.append(flipped_img)  # 반전된 이미지를 오른쪽 목록에 추가



# 오른쪽을 보는 자동차 이미지 로드 및 좌우 반전
for i in range(50):
    file = "./carRight/after/CR{0:02d}.png".format(i + 1)
    img = load_and_process_image(file)
    if img is not None:
        right_car_images.append(img)
        flipped_img = flip_image(img)
        left_car_images.append(flipped_img)  # 반전된 이미지를 왼쪽 목록에 추가

other_images = list()
for i in range(12):
    file = "./other/after/" + "other{0:02d}.png".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("missing file {0:02d}".format(i + 1))
        break

    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    other_images.append(img)


plot_images(n_row=10, n_col=10, images=left_car_images)
plot_images(n_row=10, n_col=10, images=right_car_images)
plot_images(n_row=4, n_col=3, images=other_images)


X =left_car_images + right_car_images + other_images
y = [[1, 0, 0]] * len(left_car_images) + [[0, 1, 0]] * len(right_car_images) + [[0, 0, 1]] * len(other_images)
print(y)

# 자동차 분류 모델
car_classifier = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(256, 256, 3), kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax'),  # 자동차 vs 비자동차
])

# 자동차 방향 분류 모델
car_direction_classifier = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(input_shape=(256, 256, 3), kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax'),  # 왼쪽 vs 오른쪽
])

car_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
car_direction_classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 데이터 레이블링 및 훈련 데이터 준비
car_labels = [[1, 0] if i < len(left_car_images) + len(right_car_images) else [0, 1] for i in range(len(X))]
direction_labels = [[1, 0] if i < len(left_car_images) else [0, 1] for i in range(len(left_car_images) + len(right_car_images))]

car_classifier.fit(x=np.array(X), y=np.array(car_labels), epochs=300)
car_direction_classifier.fit(x=np.array(X[:len(left_car_images) + len(right_car_images)]), y=np.array(direction_labels), epochs=300)


example_left_images = list()
for i in range(10):
    file = "./example/left/" + "CL{0:02d}.png".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("missing file {0:02d}".format(i + 1))
        break

    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    example_left_images.append(img)

example_right_images = list()
for i in range(10):
    file = "./example/right/" + "CR{0:02d}.png".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("missing file {0:02d}".format(i + 1))
        break

    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    example_right_images.append(img)

example_other_images = list()
for i in range(4):
    file = "./example/other/" + "other{0:02d}.png".format(i + 1)
    img = cv2.imread(file)
    if img is None:
        print("missing file {0:02d}".format(i + 1))
        break

    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    example_other_images.append(img)


test_images = example_left_images + example_right_images + example_other_images
plot_images(4, 6, test_images)

# 이미지 데이터 차원 확인 및 수정
test_images = np.array(test_images)  # 이미지 리스트를 numpy 배열로 변환
# 자동차 분류 모델을 사용하여 예측
car_predictions = car_classifier.predict(test_images)
car_predicted_labels = np.argmax(car_predictions, axis=1)  # 0이면 자동차, 1이면 비자동차

# 자동차 이미지만 추출하여 방향 분류 모델에 입력
car_images = test_images[car_predicted_labels == 0]

if car_images.size > 0:
    direction_predictions = car_direction_classifier.predict(car_images)
    direction_predicted_labels = np.argmax(direction_predictions, axis=1)  # 0이면 왼쪽, 1이면 오른쪽
else:
    print("No car images to predict direction.")
    direction_predicted_labels = []


# 자동차 분류 결과 시각화 및 방향 분류 결과 시각화
fig, ax = plt.subplots(4, 6, figsize=(12, 8))
car_index = 0  # 자동차 이미지에 대한 인덱스 추적
for i, img in enumerate(test_images):
    axis = ax[i // 6, i % 6]
    axis.imshow(img)
    axis.axis('off')

    if car_predicted_labels[i] == 0:  # 자동차인 경우
        if car_index < len(direction_predicted_labels):  # 방향 레이블 범위 확인
            if direction_predicted_labels[car_index] == 0:
                direction_labels = 'Left'
            else:
                'Right'
            axis.set_title(f"Car: {direction_labels}")
            car_index += 1
        else:
            axis.set_title("Car: Direction Unknown")
    else:
        axis.set_title("Not a Car")

plt.tight_layout()
plt.show()
