import matplotlib.pyplot as plt
import daft
from matplotlib import rc

rc("font", family="serif", size=12)
rc("text", usetex=False)
# plt.rcParams['figure.figsize'] = 12, 8


def daft_pooled():

    # create the PGM
    pgm = daft.PGM(shape=[4, 2.5], origin=[0, 0], grid_unit=4,
                   label_params={'fontsize':18})

    # priors
    pgm.add_node(daft.Node("beta", r"$\beta$", 1, 2, scale=2))

    # Latent variable.
    pgm.add_node(daft.Node("mu", r"$\beta X_{n}$", 1, 1, scale=2))

    # noise
    pgm.add_node(daft.Node("epsilon", r"$\epsilon$", 3, 1, scale=2))

    # observed data
    pgm.add_node(daft.Node("y", r"$y_n$", 2, 1, scale=2, observed=True))

    # edges
    pgm.add_edge("beta", "mu")
    pgm.add_edge("mu", "y")
    pgm.add_edge("epsilon", "y")

    # plate
    pgm.add_plate(daft.Plate([0.5, 0.6, 2, 0.9],
            label=r"$n \in 1:N$", shift=-0.1))

    pgm.render()
    plt.show()



def daft_unpooled():

    # create the PGM
    pgm = daft.PGM(shape=[5, 2.5], origin=[0, 0], grid_unit=4,
                   label_params={'fontsize':18})

    # priors
    pgm.add_node(daft.Node("beta_mfr", r"$\beta_{mfr}$", 1, 1, scale=2))
    pgm.add_node(daft.Node("beta", r"$\beta$", 2, 2, scale=2))

    # latent variable.
    pgm.add_node(daft.Node("mu", r"$\beta X_{n}$", 2, 1, scale=2))

    # noise
    pgm.add_node(daft.Node("epsilon", r"$\epsilon$", 4, 1, scale=2))

    # observed data
    pgm.add_node(daft.Node("y", r"$y_n$", 3, 1, scale=2, observed=True))

    # edges
    pgm.add_edge("beta_mfr", "mu")
    pgm.add_edge("beta", "mu")
    pgm.add_edge("mu", "y")
    pgm.add_edge("epsilon", "y")

    # plates
    pgm.add_plate(daft.Plate([1.5, 0.6, 2, 0.9],
            label=r"$n \in 1:N$", shift=-0.1))

    pgm.add_plate(daft.Plate([0.5, 0.5, 3.1, 1.1],
            label=r"$mfr \in 1:N_{mfr}$", shift=-0.1))

    pgm.render()
    plt.show()




def daft_partpooled():

    # create the PGM
    pgm = daft.PGM(shape=[6, 2.5], origin=[0, 0], grid_unit=4,
                   label_params={'fontsize':18})

    # priors
    pgm.add_node(daft.Node("beta_mfr_mu", r"$\mu_{mfr}$", 1, 1, scale=2))
    pgm.add_node(daft.Node("beta_mfr_sd", r"$\sigma_{mfr}$", 2, 2, scale=2))
    pgm.add_node(daft.Node("beta_mfr", r"$\beta_{mfr}$", 2, 1, scale=2))
    pgm.add_node(daft.Node("beta", r"$\beta$", 3, 2, scale=2))

    # latent variable.
    pgm.add_node(daft.Node("mu", r"$\beta X_{n}$", 3, 1, scale=2))

    # noise
    pgm.add_node(daft.Node("sigma", r"$\sigma$", 5, 1, scale=2))

    # observed data
    pgm.add_node(daft.Node("y", r"$y_n$", 4, 1, scale=2, observed=True))

    # edges
    pgm.add_edge("beta_mfr_mu", "beta_mfr")
    pgm.add_edge("beta_mfr_sd", "beta_mfr")
    pgm.add_edge("beta_mfr", "mu")
    pgm.add_edge("beta", "mu")
    pgm.add_edge("mu", "y")
    pgm.add_edge("sigma", "y")

    # plates
    pgm.add_plate(daft.Plate([2.5, 0.6, 2, 0.9],
            label=r"$n \in 1:N$", shift=-0.1))

    pgm.add_plate(daft.Plate([1.5, 0.5, 3.1, 1.1],
            label=r"$mfr \in 1:N_{mfr}$", shift=-0.1))

    pgm.render()
    plt.show()




def daft_hier():

    # create the PGM
    pgm = daft.PGM(shape=[7, 2.5], origin=[0, 0], grid_unit=4,
                   label_params={'fontsize':18})

    # priors
    pgm.add_node(daft.Node("beta_parent_mu", r"$\mu_{parent}$", 1, 1, scale=2))
    pgm.add_node(daft.Node("beta_parent_sd", r"$\sigma_{parent}$", 2, 2.2, scale=2))
    pgm.add_node(daft.Node("beta_mfr_mu", r"$\mu_{mfr}$", 2, 1, scale=2))
    pgm.add_node(daft.Node("beta_mfr_sd", r"$\sigma_{mfr}$", 3, 2.2, scale=2))
    pgm.add_node(daft.Node("beta_mfr", r"$\beta_{mfr}$", 3, 1, scale=2))
    pgm.add_node(daft.Node("beta", r"$\beta$", 4, 2.2, scale=2))

    # latent variable.
    pgm.add_node(daft.Node("mu", r"$\beta X_{n}$", 4, 1, scale=2))

    # noise
    pgm.add_node(daft.Node("epsilon", r"$\epsilon$", 6.2, 1, scale=2))

    # observed data
    pgm.add_node(daft.Node("y", r"$y_n$", 5, 1, scale=2, observed=True))

    # edges
    pgm.add_edge("beta_parent_mu", "beta_mfr_mu")
    pgm.add_edge("beta_parent_sd", "beta_mfr_mu")
    pgm.add_edge("beta_mfr_mu", "beta_mfr")
    pgm.add_edge("beta_mfr_sd", "beta_mfr")
    pgm.add_edge("beta_mfr", "mu")
    pgm.add_edge("beta", "mu")
    pgm.add_edge("mu", "y")
    pgm.add_edge("epsilon", "y")

    # plates
    pgm.add_plate(daft.Plate([3.5, 0.6, 2, 0.9],
            label=r"$n \in 1:N$", shift=-0.1))

    pgm.add_plate(daft.Plate([2.5, 0.5, 3.1, 1.1],
            label=r"$mfr \in 1:N_{mfr}$", shift=-0.1))

    pgm.add_plate(daft.Plate([1.5, 0.4, 4.2, 1.3],
            label=r"$parent \in 1:N_{parent}$", shift=-0.1))

    pgm.render()
    plt.show()
