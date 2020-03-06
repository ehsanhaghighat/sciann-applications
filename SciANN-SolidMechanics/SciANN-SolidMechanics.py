""" SciANN-SolidMechanics.py

Description:
    SciANN code for solution and discovery of solid mechanics from data.
    For additional details, please check our paper at: https://arxiv.org/abs/2003.02751
    
Created by Ehsan Haghighat on 2/14/20.
"""

import os, sys, time
import numpy as np
from sciann.utils.math import diff
from sciann import SciModel, Functional, Parameter
from sciann import Data, Tie
from sciann import Variable, Field

import matplotlib.pyplot as plt
import argparse
pi = np.pi 

# current file name. 
current_file_name = os.path.basename(__file__).split(".")[0]

# Lame paramters used in the paper. 
lmbd = 1.0
mu = 0.5
qload = 4.0

# Input interface for python. 
parser = argparse.ArgumentParser(description='''
        SciANN code for solution and discovery of solid mechanics from data. \n 
        For additional details, please check our paper at: https://arxiv.org/submit/3042511'''
)

# Define number of data points.
parser.add_argument('-l', '--layers', help='Num layers and neurons (default 4 layers each 40 neurons [40, 40, 40, 40])', type=int, nargs='+', default=[40]*4)
parser.add_argument('-af', '--actf', help='Activation function (default tanh)', type=str, nargs=1, default=['tanh'])
parser.add_argument('-nx', '--numx', help='Num Node in X (default 40)', type=int, nargs=1, default=[20])
parser.add_argument('-ny', '--numy', help='Num Node in Y (default 40)', type=int, nargs=1, default=[20])
parser.add_argument('-bs', '--batchsize', help='Batch size for Adam optimizer (default 32)', type=int, nargs=1, default=[32])
parser.add_argument('-e', '--epochs', help='Maximum number of epochs (default 2000)', type=int, nargs=1, default=[5000])
parser.add_argument('-lr', '--learningrate', help='Initial learning rate (default 0.001)', type=float, nargs=1, default=[0.001])
parser.add_argument('-in', '--independent_networks', help='Use independent networks for each var (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('-v', '--verbose', help='Show training progress (default 2) (check Keras.fit)', type=int, nargs=1, default=[2])

parser.add_argument('--shuffle', help='Shuffle data for training (default True)', type=bool, nargs=1, default=[True])
parser.add_argument('--stopafter', help='Patience argument from Keras (default 500)', type=int, nargs=1, default=[500])
parser.add_argument('--savefreq', help='Frequency to save weights (each n-epoch)', type=int, nargs=1, default=[100000])
parser.add_argument('--dtype', help='Data type for weights and biases (default float64)', type=str, nargs=1, default=['float64'])
parser.add_argument('--gpu', help='Use GPU if available (default False)', type=bool, nargs=1, default=[False])
parser.add_argument('-op', '--outputpath', help='Output path (default ./file_name)', type=str, nargs=1, default=['output'])
parser.add_argument('-of', '--outputprefix', help='Output path (default res**)', type=str, nargs=1, default=['res'])

parser.add_argument('-nxp', '--numxplot', help='Num Node in X for ploting final results (default 200)', type=int, nargs=1, default=[200])
parser.add_argument('-nyp', '--numyplot', help='Num Node in Y for ploting final results (default 200)', type=int, nargs=1, default=[200])
parser.add_argument('--plot', help='Plot the model', nargs='?', default=False)

args = parser.parse_args()

if not args.gpu[0]:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    

def load(xx):
    x, y = xx[0], xx[1]
    Q = qload
    return Q * np.sin(pi*x)


def bodyfx(xx):
    x, y = xx[0], xx[1]
    Q = qload
    frc = - lmbd*(4*pi**2*np.cos(2*pi*x)*np.sin(pi*y) - Q*y**3*pi*np.cos(pi*x)) \
          - mu*(pi**2*np.cos(2*pi*x)*np.sin(pi*y) - Q*y**3*pi*np.cos(pi*x)) \
          - 8*mu*pi**2*np.cos(2*pi*x)*np.sin(pi*y)
    return frc


def bodyfy(xx):
    x, y = xx[0], xx[1]
    Q = qload
    frc = lmbd*(3*Q*y**2*np.sin(pi*x) - 2*pi**2*np.cos(pi*y)*np.sin(2*pi*x)) \
          - mu*(2*pi**2*np.cos(pi*y)*np.sin(2*pi*x) + (Q*y**4*pi**2*np.sin(pi*x))/4) \
          + 6*Q*mu*y**2*np.sin(pi*x)
    return frc


def dispx(xx):
    x, y = xx[0], xx[1]
    return np.cos(2*pi*x) * np.sin(pi*y)


def dispy(xx):
    x, y = xx[0], xx[1]
    Q = qload
    return np.sin(pi*x) * Q * y**4/4


def strainxx(xx):
    x, y = xx[0], xx[1]
    Q = qload
    return -2*pi*np.sin(2*pi*x)*np.sin(pi*y)

def strainyy(xx):
    x, y = xx[0], xx[1]
    Q = qload
    return np.sin(pi*x)*Q*y**3

def strainxy(xx):
    x, y = xx[0], xx[1]
    Q = qload
    return 0.5*(pi*np.cos(2*pi*x)*np.cos(pi*y) + pi*np.cos(pi*x)*Q*y**4/4)

def stressxx(xx):
    return (lmbd+2*mu)*strainxx(xx) + lmbd*strainyy(xx)

def stressyy(xx):
    return (lmbd+2*mu)*strainyy(xx) + lmbd*strainxx(xx)

def stressxy(xx):
    return 2.0*mu*strainxy(xx)

def cust_pcolor(AX, X, Y, C, title):
    im = AX.pcolor(X, Y, C, cmap="jet")
    AX.axis("equal")
    AX.axis("off")
    AX.set_title(title, fontsize=14)
    plt.colorbar(im, ax=AX)

def cust_semilogx(AX, X, Y, xlabel, ylabel):
    if X is None:
        im = AX.semilogy(Y)
    else:
        im = AX.semilogy(X, Y)
    if xlabel is not None: AX.set_xlabel(xlabel)
    if ylabel is not None: AX.set_ylabel(ylabel)

def train():
    # define output folder. 
    if not os.path.isdir(args.outputpath[0]):
        os.mkdir(args.outputpath[0])
        
    output_file_name = os.path.join(args.outputpath[0], args.outputprefix[0])
    fname = output_file_name + "_{}_".format(args.actf[0]) + "x".join([str(x) for x in args.layers])
    
    # Neural Network Setup.
    x = Variable("x", dtype=args.dtype[0])
    y = Variable("y", dtype=args.dtype[0])

    if args.independent_networks[0]:
        Uxy = Functional("Uxy", [x, y], args.layers, args.actf[0])
        Vxy = Functional("Vxy", [x, y], args.layers, args.actf[0])
        Sxx = Functional("Sxx", [x, y], args.layers, args.actf[0])
        Syy = Functional("Syy", [x, y], args.layers, args.actf[0])
        Sxy = Functional("Sxy", [x, y], args.layers, args.actf[0])

    else:
        Uxy, Vxy, Sxx, Syy, Sxy = Functional(
            ["Uxy", "Vxy", "Sxx", "Syy", "Sxy"],
            [x, y],
            args.layers, args.actf[0]).split()
    
    lame1 = Parameter(2.0, inputs=[x,y], name="lame1")
    lame2 = Parameter(2.0, inputs=[x,y], name="lame2")

    C11 = (2*lame2 + lame1)
    C12 = lame1
    C33 = 2*lame2

    Exx = diff(Uxy, x)
    Eyy = diff(Vxy, y)
    Exy = (diff(Uxy, y) + diff(Vxy, x))*0.5

    # Define constraints 
    d1 = Data(Uxy)
    d2 = Data(Vxy)
    d3 = Data(Sxx)
    d4 = Data(Syy)
    d5 = Data(Sxy)
    
    c1 = Tie(Sxx, Exx*C11 + Eyy*C12)
    c2 = Tie(Syy, Eyy*C11 + Exx*C12)
    c3 = Tie(Sxy, Exy*C33)
    
    Lx = diff(Sxx, x) + diff(Sxy, y)
    Ly = diff(Sxy, x) + diff(Syy, y)
    
    # Define the optimization model (set of inputs and constraints)
    model = SciModel(
        inputs=[x, y],
        targets=[d1, d2, d3, d4, d5, c1, c2, c3, Lx, Ly],
        loss_func="mse"
    )
    with open("{}_summary".format(fname), "w") as fobj:
        model.summary(print_fn=lambda x: fobj.write(x + '\n'))
        
    # Prepare training data 
    ## Training grid 
    XMIN, XMAX = 0.0, 1.0
    YMIN, YMAX = 0.0, 1.0
    Xmesh = np.linspace(XMIN, XMAX, args.numx[0]).reshape((-1, 1))
    Ymesh = np.linspace(YMIN, YMAX, args.numy[0]).reshape((-1, 1))
    X, Y = np.meshgrid(Xmesh, Ymesh)

    input_data = [X.reshape(-1, 1), Y.reshape(-1, 1)]

    ## data associated to constrains defined earlier 
    # Define constraints 
    data_d1 = dispx(input_data)
    data_d2 = dispy(input_data)
    data_d3 = stressxx(input_data)
    data_d4 = stressyy(input_data)
    data_d5 = stressxy(input_data)
    data_c1 = 'zeros'
    data_c2 = 'zeros'
    data_c3 = 'zeros'
    data_Lx = bodyfx(input_data)
    data_Ly = bodyfy(input_data)
    
    target_data = [data_d1, data_d2, data_d3, data_d4, data_d5, 
                   data_c1, data_c2, data_c3,
                   data_Lx, data_Ly]

    # Train the model 
    training_time = time.time()
    history = model.train(
        x_true=input_data,
        y_true=target_data,
        epochs=args.epochs[0],
        batch_size=args.batchsize[0],
        shuffle=args.shuffle[0],
        learning_rate=args.learningrate[0],
        stop_after=args.stopafter[0],
        verbose=args.verbose[0],
        save_weights_to="{}_WEIGHTS".format(fname),
        save_weights_freq=args.savefreq[0]
    )
    training_time = time.time() - training_time

    for loss in history.history:
        np.savetxt(fname+"_{}".format("_".join(loss.split("/"))), 
                    np.array(history.history[loss]).reshape(-1, 1))
    
    time_steps = np.linspace(0, training_time, len(history.history["loss"]))
    np.savetxt(fname+"_Time", time_steps.reshape(-1,1))

    # Post process the trained model.
    Xmesh_plot = np.linspace(XMIN, XMAX, args.numxplot[0]).reshape((-1, 1))
    Ymesh_plot = np.linspace(YMIN, YMAX, args.numyplot[0]).reshape((-1, 1))
    X_plot, Y_plot = np.meshgrid(Xmesh_plot, Ymesh_plot)
    input_plot = [X_plot.reshape(-1, 1), Y_plot.reshape(-1, 1)]

    lame1_pred = lame1.eval(model, input_plot)
    lame2_pred = lame2.eval(model, input_plot)
    Uxy_pred = Uxy.eval(model, input_plot)
    Vxy_pred = Vxy.eval(model, input_plot)
    Exx_pred = Exx.eval(model, input_plot)
    Eyy_pred = Eyy.eval(model, input_plot)
    Exy_pred = Exy.eval(model, input_plot)
    Sxx_pred = Sxx.eval(model, input_plot)
    Syy_pred = Syy.eval(model, input_plot)
    Sxy_pred = Sxy.eval(model, input_plot)
        
    np.savetxt(fname+"_Xmesh", X_plot, delimiter=', ')
    np.savetxt(fname+"_Ymesh", Y_plot, delimiter=', ')
    np.savetxt(fname+"_lame1", lame1_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_lame2", lame2_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Uxy", Uxy_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Vxy", Vxy_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Exx", Exx_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Eyy", Eyy_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Exy", Exy_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Sxx", Sxx_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Syy", Syy_pred.reshape(X_plot.shape), delimiter=', ')
    np.savetxt(fname+"_Sxy", Sxy_pred.reshape(X_plot.shape), delimiter=', ')


def plot():
    output_file_name = os.path.join(args.outputpath[0], args.outputprefix[0])
    fname = output_file_name + "_{}_".format(args.actf[0]) + "x".join([str(x) for x in args.layers])
    
    loss = np.loadtxt(fname+"_loss")
    time = np.loadtxt(fname+"_Time")
    fig, ax = plt.subplots(1, 2, figsize=(7, 3), dpi=300)
    cust_semilogx(ax[0], None, loss/loss[0], "epochs", "L/L0")
    cust_semilogx(ax[1], time, loss/loss[0], "time(s)", None)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("{}_loss.png".format(output_file_name))
    
    Xmesh = np.loadtxt(fname+"_Xmesh", delimiter=',')
    Ymesh = np.loadtxt(fname+"_Ymesh", delimiter=',')
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
    cust_pcolor(ax[0, 0], Xmesh, Ymesh, np.ones_like(Xmesh)*lmbd, "L*={:.3f}".format(lmbd))
    cust_pcolor(ax[0, 1], Xmesh, Ymesh, np.ones_like(Xmesh)*mu, "G*={:.3f}".format(mu))
    lmbd_pred = np.loadtxt(fname+"_lame1", delimiter=',')
    mu_pred = np.loadtxt(fname+"_lame2", delimiter=',')
    cust_pcolor(ax[1, 0], Xmesh, Ymesh, lmbd_pred, "L={:.3f}".format(lmbd_pred.mean()))
    cust_pcolor(ax[1, 1], Xmesh, Ymesh, mu_pred, "G={:.3f}".format(mu_pred.mean()))
    plt.savefig("{}_Parameters.png".format(output_file_name))
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=300)
    cust_pcolor(ax[0, 0], Xmesh, Ymesh, dispx([Xmesh, Ymesh]), "Ux*")
    cust_pcolor(ax[0, 1], Xmesh, Ymesh, dispy([Xmesh, Ymesh]), "Uy*")
    cust_pcolor(ax[1, 0], Xmesh, Ymesh, np.loadtxt(fname+"_Uxy", delimiter=','), "Ux")
    cust_pcolor(ax[1, 1], Xmesh, Ymesh, np.loadtxt(fname+"_Vxy", delimiter=','), "Uy")
    plt.savefig("{}_Displacement.png".format(output_file_name))
    
    fig, ax = plt.subplots(2, 3, figsize=(11, 6), dpi=300)
    cust_pcolor(ax[0, 0], Xmesh, Ymesh, stressxx([Xmesh, Ymesh]), "Sxx*")
    cust_pcolor(ax[0, 1], Xmesh, Ymesh, stressyy([Xmesh, Ymesh]), "Syy*")
    cust_pcolor(ax[0, 2], Xmesh, Ymesh, stressxy([Xmesh, Ymesh]), "Sxy*")
    cust_pcolor(ax[1, 0], Xmesh, Ymesh, np.loadtxt(fname+"_Sxx", delimiter=','), "Sxx")
    cust_pcolor(ax[1, 1], Xmesh, Ymesh, np.loadtxt(fname+"_Syy", delimiter=','), "Syy")
    cust_pcolor(ax[1, 2], Xmesh, Ymesh, np.loadtxt(fname+"_Sxy", delimiter=','), "Sxy")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("{}_Stress.png".format(output_file_name))
    
    fig, ax = plt.subplots(2, 3, figsize=(11, 6), dpi=300)
    cust_pcolor(ax[0, 0], Xmesh, Ymesh, strainxx([Xmesh, Ymesh]), "Exx*")
    cust_pcolor(ax[0, 1], Xmesh, Ymesh, strainyy([Xmesh, Ymesh]), "Eyy*")
    cust_pcolor(ax[0, 2], Xmesh, Ymesh, strainxy([Xmesh, Ymesh]), "Exy*")
    cust_pcolor(ax[1, 0], Xmesh, Ymesh, np.loadtxt(fname+"_Exx", delimiter=','), "Exx")
    cust_pcolor(ax[1, 1], Xmesh, Ymesh, np.loadtxt(fname+"_Eyy", delimiter=','), "Eyy")
    cust_pcolor(ax[1, 2], Xmesh, Ymesh, np.loadtxt(fname+"_Exy", delimiter=','), "Exy")
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("{}_Strain.png".format(output_file_name))
    

if __name__ == "__main__":
    
    if args.plot==False:
        train()
        plot()
        
    else:
        plot()
    
