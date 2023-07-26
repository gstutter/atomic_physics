import numpy as np
import typing
import atomic_physics as ap

class QOperators:
    def __init__(self, atom: ap.Atom):
        self.atom = atom

        if atom.ePole is None:
            atom.calc_Epole()
            
    def get_cavity_collapse(self):
        """Returns the cavity collapse operator"""

    def get_spont_collapse(self):
        """Returns the spontaneous decay ."""
        GammaRoot = np.abs(self.atom.ePole)
        for ii in range(GammaRoot.shape[0]):
            GammaRoot[ii, ii] = 0
        return GammaRoot
    
    def get_cavity_hamiltonian(self, lasers: [ap.Cavity]):
        """Returns the cavity Hamiltonian for a list of cavity modes"""

    def get_laser_hamiltonian(self, lasers: [ap.Laser]):
        """Returns the laser interaction Hamiltonian for a list of lasers."""
        Gamma = np.power(np.abs(self.atom.ePole), 2)
        GammaJ = self.atom.GammaJ
        HL = np.zeros(Gamma.shape)

        for transition in self.atom.transitions.keys():
            _lasers = [laser for laser in lasers if laser.transition == transition]
            if _lasers == []:
                continue

            lower = self.atom.transitions[transition].lower
            upper = self.atom.transitions[transition].upper
            lower_states = self.atom.slice(lower)
            upper_states = self.atom.slice(upper)
            n_lower = self.atom.levels[lower]._num_states
            n_upper = self.atom.levels[upper]._num_states

            dJ = upper.J - lower.J
            dL = upper.L - lower.L
            if dJ in [-1, 0, +1] and dL in [-1, 0, +1]:
                order = 1
            elif abs(dJ) in [0, 1, 2] and abs(dL) in [0, 1, 2]:
                order = 2
            else:
                raise ValueError(
                    "Unsupported transition order. \n"
                    "Only 1st and 2nd order transitions are "
                    "supported. [abs(dL) & abs(dJ) <2]\n"
                    "Got dJ={} and dL={}".format(dJ, dL)
                )

            Mu = self.atom.M[upper_states]
            Ml = self.atom.M[lower_states]
            Mu = np.repeat(Mu, n_lower).reshape(n_upper, n_lower).T
            Ml = np.repeat(Ml, n_upper).reshape(n_lower, n_upper)

            # Transition detunings
            El = self.atom.E[lower_states]
            Eu = self.atom.E[upper_states]
            El = np.repeat(El, n_upper).reshape(n_lower, n_upper)
            Eu = np.repeat(Eu, n_lower).reshape(n_upper, n_lower).T

            # Total scattering rate out of each state
            GammaJ_subs = GammaJ[upper_states]
            GammaJ_subs = np.repeat(GammaJ_subs, n_lower).reshape(n_upper, n_lower).T

            Gamma_subs = Gamma[lower_states, upper_states]
            HL1 = np.zeros((n_lower, n_upper))
            for q in range(-order, order + 1):
                Q = np.zeros((n_lower, n_upper))
                # q := Mu - Ml
                Q[Ml == (Mu - q)] = 1
                for laser in [laser for laser in _lasers if laser.q == q]:
                    I = laser.I
                    HL1 += np.sqrt(GammaJ_subs * I * Q * Gamma_subs)
                assert (HL1 >= 0).all()

            HL[lower_states, upper_states] = HL1
            HL[upper_states, lower_states] = HL1.T
        return HL

    def get_transitions(self, lasers: [ap.Laser]):
        """
        Returns the complete transitions matrix for a given set of lasers.
        """
        return self.get_spont() + self.get_stim(lasers)