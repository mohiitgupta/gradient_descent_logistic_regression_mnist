import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (7,7)

def plot_learning_curves(x_axis_label, x_axis, y_axis_1, y_axis_2, type):
    plt.plot(x_axis, y_axis_1, marker='*')
    plt.plot(x_axis, y_axis_2, marker='*')
    plt.legend(['Train', 'Test'], loc='best')
    plt.xlabel(x_axis_label)
    plt.ylabel('Accuracy')
    plt.title('Accuracy v/s ' + x_axis_label)
    plt.savefig(type,dpi=300)
    plt.show() 