# dealii-adapter
preCICE-adapter for the C++ finite element library deal.II

*This adapter is currently under development.*

The purpose of this adapter is to be used for partitioned FSI calculations steered by preCICE,
for which this program calculates the linear elastic solid mechanics.

The linear elastic model is based on the step-8 tutorial program of the deal.II library and has been extended to linear elasto-dynamics.
Currently, the solver is extended with additional features to enable the coupling by preCICE.
