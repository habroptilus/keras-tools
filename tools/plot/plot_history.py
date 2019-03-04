import matplotlib.pyplot as plt


def plot_history_classificaion(history):
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()


def plot_history_regression(history):
    plt.plot(history['mean_squared_error'])
    plt.plot(history['val_mean_squared_error'])
    plt.title('model mse')
    plt.xlabel('epoch')
    plt.ylabel('mean_squared_error')
    plt.legend(['mse', 'val_mse'], loc='lower right')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()
