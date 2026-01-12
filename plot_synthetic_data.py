import pickle

import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


def plot_decision_boundary(model, X, y, title=None, ax=None, alpha=0.4, cmap='Blues', xlim=(-1, 1), ylim=(-1, 1),
                           save_path=None, x_hilight=None, plot_method='contourf'):

    disp = DecisionBoundaryDisplay.from_estimator(model, X,
                                                  response_method='predict',
                                                  alpha=alpha,
                                                  cmap=cmap,
                                                  ax=ax,
                                                  plot_method=plot_method)
    # Overlay the original data points
    colormap = plt.get_cmap('tab20') #'#006BA4'
    disp.ax_.scatter(X[y == -1, 0], X[y == -1, 1], color='#FF800E', alpha=1, marker='.', s=70)
    disp.ax_.scatter(X[y == 1, 0], X[y == 1, 1], c='#006BA4', alpha=1, marker='^',
                     s=70)
    if x_hilight is not None:
        disp.ax_.scatter(x_hilight[:,0], x_hilight[:,1], color='k', alpha=0.8, marker='s')
    disp.ax_.set_title(title)
    disp.ax_.set_xlim(xmin=xlim[0], xmax=xlim[1])
    disp.ax_.set_ylim(ymin=ylim[0], ymax=ylim[1])
    disp.ax_.set_xticks([-1, -0.5, 0, 0.5, 1])
    disp.ax_.set_yticks([-1, -0.5, 0, 0.5, 1])
    disp.ax_.tick_params(labelsize=FONTSIZE)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show(block=False)
    return disp

def plot_data(X, y, title=None, save_path=None, x_hilight=None, plot_method='contourf', xlim=(-1, 1), ylim=(-1, 1)):
    plt.figure()
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='#FF800E', alpha=1, marker='.', s=70)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='#006BA4', alpha=1, marker='^',
                     s=70)
    plt.xlim(xmin=xlim[0], xmax=xlim[1])
    plt.ylim(ymin=ylim[0], ymax=ylim[1])
    plt.xticks([-1, -0.5, 0, 0.5, 1], fontsize=FONTSIZE)
    plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=FONTSIZE)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show(block=False)


if __name__ == '__main__':
    from tensorlibrary.learning.transfer import CPKRR_LMPROJ, SVC_LMPROJ
    from tensorlibrary.learning import CPKRR
    from sklearn.svm import SVC
    from krr import KRR_LMPROJ_Classifier, KRRClassifier
    data = "./data/syn_data_9.pkl"
    with open(data, 'rb') as f:
        data = pickle.load(f)

    X_s = data['X_s']
    y_s = data['y_s']
    X_t = data['X_t']
    y_t = data['y_t']

    FONTSIZE = 20
    sig = 0.3
    M = 15
    Ld = 2.4
    R = 10
    reg_par = 1e-3
    C = 10
    RANDOM_STATE = 10
    gam = 1 / (sig ** 2)

    adapt = CPKRR_LMPROJ(
        feature_map='rbf',
        M=M,
        map_param=sig,
        num_sweeps=10,
        reg_par=reg_par,
        max_rank=R,
        Ld=Ld,
        random_init=True,
        train_loss_flag=False,
        mu=(len(y_s) + len(y_t)),
        random_state=RANDOM_STATE,
    )
    source = CPKRR(
        feature_map='rbf',
        M=M,
        map_param=sig,
        num_sweeps=10,
        reg_par=reg_par,
        max_rank=R,
        Ld=Ld,
        random_init=True,
        train_loss_flag=False,
        random_state=RANDOM_STATE,
    )

    source.fit(X_s, y_s)
    adapt.fit(X_s, y_s, x_target=X_t)
    plot_data(X_s, y_s, save_path="./plots/source_data.pdf")
    plot_data(X_t, y_t, save_path="./plots/target_data.pdf")
    plot_decision_boundary(adapt, X_t, y_t, save_path="./plots/T_LMPROJ_on_synthetic.pdf", plot_method="contour", cmap='Greys')
    plot_decision_boundary(source, X_t, y_t, save_path="./plots/T_KRR_synthetic.pdf", plot_method="contour", cmap='Greys')