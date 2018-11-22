# dealii-adapter
preCICE-adapter for the C++ finite element library deal.II

*This adapter is currently under development.*
The purpose of this adapter is to be used for partitioned FSI calculations steered by preCICE,
for which this program calculates the linear elastic solid mechanics.

The linear elastic model is based on the step-8 tutorial program of the deal.II library.
Currently, the solver is extended in order to capture dynamic solid mechanics, which are necessary for FSI. 
