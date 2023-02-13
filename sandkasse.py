from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense, LeakyReLU

(X_train, y_train), (X_test, y_test) = mnist.load_data()

activation = LeakyReLU(alpha=0.01)

## PENIS


model = Sequential()
model.add(Dense(128, input_shape=(784,), activation=activation))
model.add(Dense(48, activation = activation))
model.add(Dense(128, activation = activation))
model.add(Dense(784, activation = activation))

model.compile(loss="mean_squared_error", optimizer="adam")

X_train = np.reshape(X_train, (X_train.shape[0], 28*28))
X_test = np.reshape(X_test, (X_test.shape[0], 28*28))

model.fit(X_train, X_train, epochs=5, batch_size=10)
#%%
preds = model(X_test)
fig, ax = plt.subplots(4,2)

for i in range(4):
    indx = np.random.randint(len(X_test))
    real = X_test[indx]
    generated = preds[indx]
    
    ax[i,0].imshow(np.reshape(real, (28,28)), cmap="gray")
    ax[i,1].imshow(np.reshape(generated, (28,28)), cmap="gray")


#%%

def draw(vec):
    plt.imshow(np.reshape(vec, (28,28)), cmap="gray")

#%%

indx = np.random.randint(len(X_test))
noise = np.random.rand(28*28) * 100
pic = X_test[indx]
pic_w_noise = pic + noise

plt.imshow(np.reshape(pic_w_noise, (28,28)), cmap="gray")
plt.show()

pic_denoised = model(np.reshape(pic_w_noise, (1,-1)))
plt.imshow(np.reshape(pic_denoised, (28,28)), cmap="gray")
plt.show()

#%%

encoder = Sequential(model.layers[:-2])
generator = Sequential(model.layers[-2:])

encodings = encoder(X_test)


#%%

xmin, xmax = min(encodings[:,0]), max(encodings[:,0])
ymin, ymax = min(encodings[:,1]), max(encodings[:,1])

fig, ax = plt.subplots(10, 10)

for i in range(10):
    for j in range(10):
        indx = (y_test == i) | (y_test == j)
        good_points = encodings[indx]
        
        x,y = good_points[:,0], good_points[:, 1]
        ax[i,j].scatter(x,y,c=y_test[indx], s=1)
        
        ax[i,j].axis("off")
        
        ax[i,j].set_xlim(xmin, xmax)
        ax[i,j].set_ylim(ymin, ymax)
        
fig.set_figheight(10)
fig.set_figwidth(10)

plt.savefig("fordeling af encodings", dpi=300)

#%%

which_feature = 1

average_encodings = np.mean(encodings, axis=0)
min_feature, max_feature = min(encodings[:,which_feature]), max(encodings[:,which_feature])
feature_space = np.linspace(min_feature, max_feature, 100)

for i in feature_space:
    curr_features = average_encodings
    curr_features[which_feature] = i
    generated_picture = generator(np.reshape(curr_features, (-1, 48)))
    draw(generated_picture)
    plt.show()